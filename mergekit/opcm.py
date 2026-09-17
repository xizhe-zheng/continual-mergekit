"""Whole-model OPCM, following fusion_bench/method/opcm/opcm.py.

The sequential normalization and full SVD intentionally follow the reference.
This is not a GroupMergeMethod: each step requires a whole-model norm.
"""

import json
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import transformers
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from mergekit.common import dtype_from_name

LOG = logging.getLogger(__name__)


class OPCMParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    shuffle_order: bool = True
    seed: Optional[int] = None
    save_on_every_step: bool = True


@torch.no_grad()
def task_vector_norm(model: nn.Module, base: nn.Module) -> torch.Tensor:
    # The reference flattens the *sorted state_dict*, including buffers and
    # duplicate tied-weight entries. Do not replace this with named_parameters.
    state, reference = model.state_dict(), base.state_dict()
    return torch.linalg.norm(
        torch.cat([(state[k] - reference[k]).reshape(-1) for k in sorted(state)])
    )


def _validate_model(model: nn.Module, base: nn.Module) -> None:
    state, reference = model.state_dict(), base.state_dict()
    if state.keys() != reference.keys():
        raise ValueError("OPCM requires identical state_dict keys")
    for name, tensor in state.items():
        if (
            tensor.shape != reference[name].shape
            or tensor.dtype != reference[name].dtype
        ):
            raise ValueError(f"OPCM requires matching shape and dtype: {name}")
        if tensor.device.type != "cpu" or reference[name].device.type != "cpu":
            raise ValueError("OPCM stores models on CPU; use device for projection")
        if tensor.dtype == torch.bool:
            raise ValueError(
                f"OPCM reference subtraction does not support bool: {name}"
            )
        if tensor.is_floating_point() and tensor.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("OPCM requires float32 or float64 working weights for SVD")
    if [(n, type(m)) for n, m in model.named_modules()] != [
        (n, type(m)) for n, m in base.named_modules()
    ]:
        raise ValueError("OPCM requires identical module structure and types")


def _merge_parameter(merged, base, task, previous_lambda, alpha, device):
    original_device = merged.device
    merged, base, task = (x.to(device) for x in (merged, base, task))
    previous_tv, task_tv = merged - base, task - base
    if alpha is not None:
        u, s, vh = torch.linalg.svd(
            previous_tv,
            full_matrices=True,
            driver="gesvd" if previous_tv.is_cuda else None,
        )
        v = vh.T
        # Preserve the reference's strict comparison, zero-based split and
        # alpha=1/all-zero argmax behavior, rather than correcting its boundary.
        split_rank = (s.cumsum(dim=0) / s.sum() > alpha).float().argmax().item()
        projected = u.T @ task_tv @ v
        projected.diag().fill_(0)
        projected[:split_rank, :split_rank] = 0
        task_tv = u @ projected @ v.T
    # lambda_t is temporarily 1 in the reference's inner loop.
    return (base + (previous_lambda * previous_tv + task_tv) / 1).to(original_device)


@torch.no_grad()
def merge_opcm(
    base: nn.Module,
    tasks: Iterable[nn.Module],
    *,
    alpha: float = 0.5,
    device: str = "cpu",
    on_step: Optional[Callable[[int, nn.Module, dict], None]] = None,
) -> nn.Module:
    """Merge ordered CPU models without modifying the caller's models.

    Only the base, current task and accumulated model must remain resident.
    Callbacks see each normalized model, including the unchanged first task.
    """
    OPCMParameters(alpha=alpha)
    iterator = iter(tasks)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("OPCM needs at least one task model") from None
    _validate_model(first, base)
    merged = deepcopy(first)
    del first
    norms = [task_vector_norm(merged, base)]
    previous_lambda = 1
    if on_step:
        on_step(0, merged, {"lambda_t": 1.0, "avg_task_vector_norm": norms[0].item()})

    for step, task in enumerate(iterator, start=1):
        _validate_model(task, base)
        norms.append(task_vector_norm(task, base))
        avg_norm = np.mean(norms)
        for module_name, module in merged.named_modules():
            if len(list(module.children())) != 0:
                continue
            base_module = base.get_submodule(module_name)
            task_module = task.get_submodule(module_name)
            if isinstance(module, nn.Linear):
                module.weight.data = _merge_parameter(
                    module.weight,
                    base_module.weight,
                    task_module.weight,
                    previous_lambda,
                    alpha,
                    device,
                )
                if module.bias is not None:
                    module.bias.data = _merge_parameter(
                        module.bias,
                        base_module.bias,
                        task_module.bias,
                        previous_lambda,
                        None,
                        device,
                    )
            else:
                for name, param in module.named_parameters():
                    param.data = _merge_parameter(
                        param,
                        base_module.get_parameter(name),
                        task_module.get_parameter(name),
                        previous_lambda,
                        None,
                        device,
                    )
        norm = task_vector_norm(merged, base)
        if (
            not torch.isfinite(norm)
            or norm == 0
            or not np.isfinite(avg_norm)
            or avg_norm == 0
        ):
            raise ValueError("OPCM normalization is undefined (zero or nonfinite norm)")
        lambda_t = 1 * (norm / avg_norm)
        for name, param in merged.named_parameters():
            param.data = base.get_parameter(name) + (
                param - base.get_parameter(name)
            ) * (avg_norm / norm)
        previous_lambda = lambda_t
        if on_step:
            on_step(
                step,
                merged,
                {"lambda_t": lambda_t.item(), "avg_task_vector_norm": float(avg_norm)},
            )
        del task
    return merged


def run_opcm(config, out_path: str, options, config_source=None):
    """mergekit-yaml adapter for complete Hugging Face model checkpoints."""
    if (
        not config.models
        or config.base_model is None
        or config.slices
        or config.modules
    ):
        raise ValueError(
            "OPCM requires models and base_model; slices/modules are unsupported"
        )
    if config.tokenizer or config.tokenizer_source not in (None, "base"):
        raise ValueError(
            "OPCM requires matching vocabularies; tokenizer remapping is unsupported"
        )
    if (
        options.multi_gpu
        or options.read_to_gpu
        or options.low_cpu_memory
        or options.unsafe_truncate_embeddings
    ):
        raise ValueError(
            "OPCM streams tensors from CPU storage to one projection device; "
            "read_to_gpu, low_cpu_memory, multi_gpu and embedding truncation options are unsupported"
        )
    params = OPCMParameters.model_validate(config.parameters or {})
    refs = []
    for entry in config.models:
        if entry.parameters:
            raise ValueError("OPCM does not support per-model parameters")
        if entry.model == config.base_model:
            raise ValueError("OPCM models must list tasks only, excluding base_model")
        refs.append(entry.model)
    for ref in [config.base_model, *refs]:
        if ref.lora or ref.override_architecture:
            raise ValueError(
                "OPCM requires full checkpoints without architecture overrides"
            )
    dtype = dtype_from_name(config.dtype) or torch.float32
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "OPCM dtype must be float32 or float64; use out_dtype for output casting"
        )
    out_dtype = dtype_from_name(config.out_dtype)
    if out_dtype is not None and not out_dtype.is_floating_point:
        raise ValueError("OPCM out_dtype must be floating point")
    seed = params.seed if params.seed is not None else options.random_seed
    if seed is not None:
        transformers.trainer_utils.set_seed(seed)
    if params.shuffle_order:
        random.shuffle(refs)

    base_config = config.base_model.config(options.trust_remote_code)
    architectures = base_config.architectures or []
    if len(architectures) != 1:
        raise ValueError(
            "OPCM requires one explicit Hugging Face architecture in config.json"
        )
    model_class = getattr(transformers, architectures[0], None)
    if model_class is None or not issubclass(model_class, transformers.PreTrainedModel):
        raise ValueError("OPCM currently requires a built-in Transformers model class")

    output = Path(out_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_names.json").write_text(
        json.dumps([str(r) for r in refs], indent=2)
    )
    from mergekit.opcm_streaming import run_streaming

    run_streaming(
        config.base_model,
        refs,
        model_class,
        base_config,
        params,
        output,
        options,
        dtype,
        out_dtype,
    )
    (output / "mergekit_config.yml").write_text(config_source or config.to_yaml())
    if options.write_model_card:
        from mergekit.card import generate_card

        (output / "README.md").write_text(
            generate_card(
                config=config,
                config_yaml=config_source or config.to_yaml(),
                name=output.name,
            )
        )
    if options.copy_tokenizer:
        from mergekit.merge import _copy_tokenizer

        try:
            _copy_tokenizer(config, out_path, options=options)
        except Exception:
            LOG.warning(
                "Could not copy tokenizer; model weights were saved", exc_info=True
            )
