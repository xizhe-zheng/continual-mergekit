# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""WUDI-Merging: data-free interference reduction guided by task vectors."""

import math
import re

import torch

from mergekit.merge_methods.base import BasePolicy, InputContract, TensorGroup
from mergekit.merge_methods.easy_define import merge_method

_DEFAULT_EXCLUDE = r"(?:embed|embedding|lm_head|classifier|score|wte|wpe|shared)"


@merge_method(
    name="wudi",
    pretty_name="WUDI-Merging",
    reference_url="https://arxiv.org/abs/2503.08099",
    contract=InputContract(base=BasePolicy.REQUIRED, min_non_base=1),
)
@torch.no_grad()
def wudi_merge(
    group: TensorGroup,
    learning_rate: float = 1e-5,
    iterations: int = 300,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    exclude_regex: str = _DEFAULT_EXCLUDE,
) -> torch.Tensor:
    """Merge linear-layer task vectors with the WUDI objective.

    This implements Algorithm 1 from the WUDI paper.  The reference code uses
    autograd to optimize each layer independently.  Here the same gradient and
    Adam update are evaluated explicitly, which avoids retaining a large
    autograd graph and makes checkpoint-scale, layerwise execution practical.

    Non-matrix tensors and embedding/output-head tensors are copied from the
    base model.  This matches the paper's policy of applying WUDI only to linear
    layers; ``exclude_regex`` can be overridden for other architectures.
    """
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and greater than zero")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("beta1 and beta2 must be in [0, 1)")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and greater than zero")

    base = group.base.tensor
    name = group.metadata.name or ""
    if base.ndim != 2 or (
        name and exclude_regex and re.search(exclude_regex, name, re.I)
    ):
        return base.clone()

    work_dtype = torch.float64 if base.dtype == torch.float64 else torch.float32
    work_base = base.to(dtype=work_dtype)
    deltas = [entry.tensor.to(dtype=work_dtype) - work_base for entry in group.non_base]

    active = []
    for delta in deltas:
        norm_sq = delta.square().sum()
        if norm_sq.item() > 0:
            active.append((delta, norm_sq))
    if not active:
        return base.clone()

    merged = sum((delta for delta, _ in active), start=torch.zeros_like(work_base))
    if iterations == 0:
        return (work_base + merged).to(dtype=base.dtype)

    # For tall/equal matrices, caching D^T D avoids one GEMM per expert and
    # iteration while keeping the temporary square matrix on the smaller axis.
    rows, columns = merged.shape
    right_grams = None
    if rows >= columns:
        right_grams = [
            torch.mm(delta.transpose(0, 1), delta).div_(norm_sq)
            for delta, norm_sq in active
        ]

    first_moment = torch.zeros_like(merged)
    second_moment = torch.zeros_like(merged)
    for step in range(1, iterations + 1):
        gradient = torch.zeros_like(merged)
        for index, (delta, norm_sq) in enumerate(active):
            residual = merged - delta
            if right_grams is not None:
                gradient.addmm_(residual, right_grams[index], beta=1.0, alpha=2.0)
            else:
                projected = torch.mm(residual, delta.transpose(0, 1))
                gradient.addmm_(
                    projected,
                    delta,
                    beta=1.0,
                    alpha=2.0 / norm_sq.item(),
                )

        first_moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
        second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        denominator = second_moment.sqrt().div_(math.sqrt(bias_correction2))
        denominator.add_(epsilon)
        merged.addcdiv_(
            first_moment,
            denominator,
            value=-learning_rate / bias_correction1,
        )

    return (work_base + merged).to(dtype=base.dtype)


__all__ = ["wudi_merge"]
