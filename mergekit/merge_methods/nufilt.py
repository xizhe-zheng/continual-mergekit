"""NUFILT following fusion_bench/models/filter_lora.py and method/nufilt.

Preserves the reference first-task projection and final fusion orientation.
Only selected linear weight names are updated; all other tensors keep the base.
"""

import hashlib
from typing import Annotated, Any

import torch
from pydantic import Field

from mergekit.merge_methods.base import (
    BasePolicy,
    GroupMergeMethod,
    InputContract,
    MergeMethodSpec,
    ParameterScope,
    ParameterSpec,
    TensorGroup,
)

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def _right_vectors(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svd(
        matrix, full_matrices=False, driver="gesvd" if matrix.is_cuda else None
    ).Vh.T


def _adapt(
    task: torch.Tensor,
    previous: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    pre_v: torch.Tensor,
    task_v: torch.Tensor | None,
    u: torch.Tensor | None,
    lr: float,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Each layer's objective and Adam state are independent in the reference.
    # Cache fixed residuals without changing its multiplication order.
    objectives = []
    for v, target in ((task_v, task), (pre_v, previous)):
        if v is not None:
            projected = v if u is None else v - u @ (u.T @ v)
            residual = target @ v - (previous @ v + task @ projected)
            objectives.append((v, residual))
    with torch.enable_grad():
        a.requires_grad_()
        b.requires_grad_()
        optimizer = torch.optim.Adam((a, b), lr=lr)
        for _ in range(steps):
            loss = sum(
                (residual - task @ (b @ (a @ v))).square().sum()
                for v, residual in objectives
            )
            # pre_v is always a Tensor, even when null_r=0. The official
            # implementation therefore retains this (possibly empty) objective.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return a.detach(), b.detach()


class NUFILTMerge(GroupMergeMethod):
    @property
    def spec(self) -> MergeMethodSpec:
        shared = ParameterScope.SHARED
        return MergeMethodSpec(
            name="nufilt",
            pretty_name="NUFILT",
            reference_url="https://openreview.net/forum?id=HDIf3fYqPP",
            parameters=(
                ParameterSpec("lora_r", NonNegativeInt, shared, default=64),
                ParameterSpec("null_r", NonNegativeInt, shared, default=128),
                ParameterSpec("grad_r", NonNegativeInt, shared, default=8),
                ParameterSpec("lr", PositiveFloat, shared, default=1e-3),
                ParameterSpec("max_steps", NonNegativeInt, shared, default=50),
                ParameterSpec("null_space", bool, shared, default=True),
                ParameterSpec("seed", NonNegativeInt, shared, default=42),
                # Comma-separated substrings: YAML lists denote schedules in
                # mergekit, so they cannot represent a literal module list.
                ParameterSpec("target_modules", str, shared, default="attn,fc1"),
            ),
            contract=InputContract(base=BasePolicy.REQUIRED, min_non_base=1),
        )

    @torch.no_grad()
    def merge_group(self, group: TensorGroup, /, **p: Any) -> torch.Tensor:
        base = group.base.tensor
        name = group.metadata.name
        patterns = [x.strip() for x in p["target_modules"].split(",") if x.strip()]
        if not patterns:
            raise ValueError("NUFILT target_modules must contain module substrings")
        if name is None:
            raise ValueError(
                "NUFILT requires a tensor name for target module selection"
            )
        if (
            base.ndim != 2
            or not name.endswith(".weight")
            or group.metadata.vocabulary_axis is not None
            or not any(pattern in name.removesuffix(".weight") for pattern in patterns)
        ):
            return base.clone()
        if not base.is_floating_point():
            raise TypeError("NUFILT requires floating-point weights")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("NUFILT optimization cannot run inside inference_mode")

        work_dtype = torch.float64 if base.dtype == torch.float64 else torch.float32
        base_w = base.detach().to(work_dtype)
        merged = base_w.clone()
        for task_index, entry in enumerate(group.non_base):
            task = entry.tensor.detach().to(work_dtype) - base_w
            previous = merged - base_w
            pre_v = _right_vectors(previous)[:, : p["null_r"]]
            task_v = _right_vectors(task)[:, : p["grad_r"]] if p["grad_r"] > 0 else None
            u = pre_v if p["null_space"] else None
            # Stable per-weight initialization avoids dependence on graph order
            # and never changes the caller's global RNG state. This intentionally
            # differs from FusionBench's whole-model RNG consumption sequence.
            digest = hashlib.sha256(
                f"{p['seed']}:{name}:{task_index}".encode()
            ).digest()
            generator = torch.Generator(device="cpu").manual_seed(
                int.from_bytes(digest[:8], "little")
            )
            rank = max(p["lora_r"], 1)
            a = torch.empty(rank, base.shape[1], dtype=work_dtype)
            a.normal_(std=0.02, generator=generator)
            a = a.to(base.device)
            b = torch.zeros(base.shape[1], rank, device=base.device, dtype=work_dtype)
            # Do not special-case the zero historical vector: the reference
            # filters the first task with its SVD basis, without training LoRA.
            if task_index > 0 and p["lora_r"] > 0 and p["max_steps"] > 0:
                a, b = _adapt(
                    task,
                    previous,
                    a,
                    b,
                    pre_v=pre_v,
                    task_v=task_v,
                    u=u,
                    lr=p["lr"],
                    steps=p["max_steps"],
                )
            # Preserve reference arithmetic order: tiny changes can rotate
            # degenerate historical SVD subspaces at the next task.
            gate = torch.eye(base.shape[1], device=base.device, dtype=work_dtype)
            if u is not None:
                gate = gate - u @ u.T
            merged = task @ (gate + b @ a) + merged
        return merged.to(base.dtype)


nufilt_merge = NUFILTMerge()
