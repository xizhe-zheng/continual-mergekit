"""Disk-backed OPCM execution; global normalization still happens every round."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import safetensors
import safetensors.torch
import torch
from torch import nn
from tqdm.auto import tqdm

from mergekit.io import ShardedTensorIndex, TensorWriter


class Layout:
    """Module semantics and parameter aliases, built without allocating weights."""

    def __init__(self, model):
        self.state = model.state_dict()
        parameters = dict(model.named_parameters(remove_duplicate=False))
        buffers = dict(model.named_buffers(remove_duplicate=False))
        by_id = {}
        self.canonical = {}
        self.aliases = {}
        for name in self.state:
            value = parameters.get(name, buffers.get(name))
            canonical = by_id.setdefault(id(value), name)
            self.canonical[name] = canonical
            self.aliases.setdefault(canonical, []).append(name)
        self.parameters = list(dict.fromkeys(self.canonical[n] for n in parameters))
        self.actions = []
        for prefix, module in model.named_modules():
            if list(module.children()):
                continue
            for name, _ in module.named_parameters():
                full_name = f"{prefix}.{name}" if prefix else name
                self.actions.append(
                    (
                        self.canonical[full_name],
                        isinstance(module, nn.Linear) and name == "weight",
                    )
                )


class Checkpoint:
    def __init__(self, path, layout, dtype):
        self.index = ShardedTensorIndex.from_disk(str(path))
        if not self.index.is_safetensors:
            raise ValueError("Streaming OPCM requires safetensors input checkpoints")
        self.layout, self.dtype = layout, dtype
        self.names = {}
        for canonical, aliases in layout.aliases.items():
            found = [n for n in aliases if n in self.index.tensor_paths]
            if not found:
                raise ValueError(f"OPCM checkpoint {path} is missing {canonical}")
            self.names[canonical] = found[0]
        unexpected = set(self.index.tensor_paths) - set(layout.state)
        if unexpected:
            raise ValueError(
                f"Unexpected OPCM checkpoint tensors: {sorted(unexpected)}"
            )
        # Check shapes using safetensors metadata without loading any weights.
        for canonical, name in self.names.items():
            with safetensors.safe_open(self._path(name), framework="pt") as f:
                if tuple(f.get_slice(name).get_shape()) != tuple(
                    layout.state[canonical].shape
                ):
                    raise ValueError(f"OPCM shape mismatch: {name}")

    def _path(self, name):
        return str(Path(self.index.base_path) / self.index.tensor_paths[name])

    def get(self, canonical):
        name = self.names[canonical]
        with safetensors.safe_open(self._path(name), framework="pt") as f:
            tensor = f.get_tensor(name)
        if tensor.dtype == torch.bool:
            raise ValueError(
                f"OPCM reference subtraction does not support bool: {name}"
            )
        return tensor.to(self.dtype) if tensor.is_floating_point() else tensor


class DiskState:
    def __init__(self, path, layout):
        self.path = Path(path)
        self.files = {
            name: self.path / f"{i}.safetensors"
            for i, name in enumerate(layout.aliases)
        }

    def put(self, name, tensor):
        path = self.files[name]
        temporary = path.with_suffix(".tmp")
        safetensors.torch.save_file({"weight": tensor.contiguous()}, str(temporary))
        os.replace(temporary, path)

    def get(self, name):
        with safetensors.safe_open(str(self.files[name]), framework="pt") as f:
            return f.get_tensor("weight")


def streamed_norm(source, base, layout, dtype, *, quiet=True, description="OPCM norm"):
    # Accumulate in FP64, one bounded chunk at a time. The result is cast back
    # before reference scaling; the sum's rounding differs from one giant norm.
    square_sum = torch.zeros((), dtype=torch.float64)
    for name in tqdm(sorted(layout.state), desc=description, disable=quiet):
        canonical = layout.canonical[name]
        left, right = source.get(canonical), base.get(canonical)
        left, right = left.reshape(-1), right.reshape(-1)
        for offset in range(0, left.numel(), 1024 * 1024):
            difference = (
                left[offset : offset + 1024 * 1024]
                - right[offset : offset + 1024 * 1024]
            ).double()
            square_sum += difference.square().sum()
        del left, right
    return square_sum.sqrt().to(dtype)


def project_update(previous, task, alpha):
    """Full-basis reference mask via subtraction, preserving null-space updates.

    A naive reduced-SVD reconstruction would discard rectangular complements.
    Here only the removed block requires singular vectors; unmodified
    components are retained in `task`, so square full U/V are never needed.
    """
    u, s, vh = torch.linalg.svd(
        previous, full_matrices=False, driver="gesvd" if previous.is_cuda else None
    )
    k = (s.cumsum(0) / s.sum() > alpha).float().argmax().item()
    v = vh.T
    cleaned = task.clone()
    if k:
        cleaned -= u[:, :k] @ (u[:, :k].T @ task @ v[:, :k]) @ vh[:k]
    # Reference projected.diag().fill_(0) modifies a copy, not projected.
    # Preserve that actual behavior: only the top-left block is removed.
    return cleaned


def _export(state, path, layout, model_config, options, out_dtype, consume=False):
    writer = TensorWriter(
        str(path),
        max_shard_size=options.out_shard_size,
        safe_serialization=options.safe_serialization,
    )
    for name in layout.aliases:
        tensor = state.get(name)
        if out_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(out_dtype)
        writer.save_tensor(name, tensor)
        if consume:
            # Unlink after handing the tensor to the writer. Existing mmap views
            # remain valid until the shard flushes; disk is reclaimed gradually.
            state.files[name].unlink()
        del tensor
    writer.finalize()
    cfg = model_config.to_dict()
    if out_dtype is not None:
        cfg["dtype"] = str(out_dtype).removeprefix("torch.")
        cfg.pop("torch_dtype", None)
    Path(path, "config.json").write_text(json.dumps(cfg, indent=2))


@torch.no_grad()
def run_streaming(
    base_ref, refs, model_class, base_config, params, output, options, dtype, out_dtype
):
    with torch.device("meta"):
        skeleton = model_class(base_config)
    layout = Layout(skeleton)
    del skeleton
    sizes = [
        layout.state[n].numel()
        * (
            torch.empty((), dtype=dtype).element_size()
            if layout.state[n].is_floating_point()
            else layout.state[n].element_size()
        )
        for n in layout.aliases
    ]
    # A shard's mmap-backed scratch files stay allocated until it is flushed.
    # Nonpositive shard sizes disable splitting in TensorWriter.
    shard_headroom = (
        min(options.out_shard_size, sum(sizes))
        if options.out_shard_size > 0
        else sum(sizes)
    )
    required = (
        sum(sizes) * (1 + len(refs) * int(params.save_on_every_step))
        + 3 * max(sizes)
        + 2 * shard_headroom
        + 512 * 1024**2
    )
    if shutil.disk_usage(output).free < required:
        raise ValueError(
            f"OPCM needs approximately {required / 1024**3:.1f} GiB free disk space; disable save_on_every_step or choose a larger output filesystem"
        )
    base = Checkpoint(base_ref.local_path(options.transformers_cache), layout, dtype)
    tasks = []
    for ref in refs:
        cfg = ref.config(options.trust_remote_code)
        if cfg.architectures != base_config.architectures:
            raise ValueError("OPCM requires identical model architectures")
        with torch.device("meta"):
            task_skeleton = model_class(cfg)
        task_layout = Layout(task_skeleton)
        if (
            task_layout.canonical != layout.canonical
            or task_layout.actions != layout.actions
            or task_layout.parameters != layout.parameters
        ):
            raise ValueError(
                "OPCM requires identical module structure and weight tying"
            )
        del task_skeleton, task_layout
        tasks.append(
            Checkpoint(ref.local_path(options.transformers_cache), layout, dtype)
        )

    reports, norms = [], []
    # One canonical FP32 model on disk, plus one tensor's atomic replacement.
    # TemporaryDirectory also removes intermediates on Python exceptions.
    with tempfile.TemporaryDirectory(prefix=".opcm-work-", dir=output) as scratch:
        state = DiskState(scratch, layout)
        for name in tqdm(layout.aliases, desc="OPCM initialize", disable=options.quiet):
            state.put(name, tasks[0].get(name))
        previous_lambda = 1
        for step, task in enumerate(tasks):
            norms.append(
                streamed_norm(
                    task,
                    base,
                    layout,
                    dtype,
                    quiet=options.quiet,
                    description=f"OPCM step {step} task norm",
                )
            )
            avg_norm = np.mean(norms)
            if step:
                for name, projected in tqdm(
                    layout.actions,
                    desc=f"OPCM step {step} projection",
                    disable=options.quiet,
                ):
                    old, reference, incoming = (
                        s.get(name).to(options.device) for s in (state, base, task)
                    )
                    previous, update = old - reference, incoming - reference
                    del old, incoming
                    if projected:
                        update = project_update(previous, update, params.alpha)
                    result = (
                        reference + (previous_lambda * previous + update) / 1
                    ).cpu()
                    state.put(name, result)
                    del previous, update, reference, result
                norm = streamed_norm(
                    state,
                    base,
                    layout,
                    dtype,
                    quiet=options.quiet,
                    description=f"OPCM step {step} merged norm",
                )
                if (
                    not torch.isfinite(norm)
                    or norm == 0
                    or not np.isfinite(avg_norm)
                    or avg_norm == 0
                ):
                    raise ValueError(
                        "OPCM normalization is undefined (zero or nonfinite norm)"
                    )
                previous_lambda = norm / avg_norm
                for name in tqdm(
                    layout.parameters,
                    desc=f"OPCM step {step} normalize",
                    disable=options.quiet,
                ):
                    reference, merged = base.get(name), state.get(name)
                    state.put(
                        name, reference + (merged - reference) * (avg_norm / norm)
                    )
                    del reference, merged
            reports.append(
                {
                    "step": step,
                    "lambda_t": float(previous_lambda),
                    "avg_task_vector_norm": float(avg_norm),
                }
            )
            (output / "opcm_steps.json").write_text(json.dumps(reports, indent=2))
            if params.save_on_every_step:
                _export(
                    state,
                    output / "checkpoints" / f"merged_model_{step}",
                    layout,
                    base_config,
                    options,
                    dtype,
                )
        _export(
            state,
            output,
            layout,
            base_config,
            options,
            out_dtype or dtype,
            consume=True,
        )
