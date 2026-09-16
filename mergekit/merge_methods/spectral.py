# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Layerwise TSV-Merge and Iso-C, following the authors' reference algorithms.

References:
https://github.com/AntoAndGar/task_singular_vectors/blob/main/src/utils/TSVM_utils.py
https://github.com/danielm1405/iso-merging/blob/main/src/utils/iso.py
"""

from dataclasses import dataclass
from typing import Any

import torch

from mergekit.merge_methods.base import (
    BasePolicy,
    GroupMergeMethod,
    InputContract,
    MergeMethodSpec,
    ParameterScope,
    ParameterSpec,
    TensorGroup,
)


@dataclass(frozen=True)
class SpectralMerge(GroupMergeMethod):
    method_name: str

    @property
    def spec(self) -> MergeMethodSpec:
        return MergeMethodSpec(
            name=self.method_name,
            pretty_name="TSV-Merge" if self.method_name == "tsv" else "Iso-C",
            reference_url=(
                "https://arxiv.org/abs/2412.00081"
                if self.method_name == "tsv"
                else "https://arxiv.org/abs/2502.04959"
            ),
            parameters=(
                ParameterSpec("lambda", float, ParameterScope.SHARED, default=1.0),
            ),
            contract=InputContract(base=BasePolicy.REQUIRED, min_non_base=1),
        )

    def merge_group(self, group: TensorGroup, /, **parameters: Any) -> torch.Tensor:
        base = group.base.tensor
        # CPU/GPU SVD requires single or double precision. Promote before subtraction.
        dtype = torch.float64 if base.dtype == torch.float64 else torch.float32
        work_base = base.to(dtype)
        entries = group.non_base
        scale = parameters["lambda"]
        if scale == 0:
            return base.clone()
        is_matrix = base.ndim == 2 and "text_projection" not in (
            group.metadata.name or ""
        )

        if not is_matrix:
            # Preserve the reference implementations' incremental mean for TSV.
            if self.method_name == "tsv":
                merged = entries[0].tensor.to(dtype) - work_base
                for i, entry in enumerate(entries[1:], start=2):
                    merged = merged + (entry.tensor.to(dtype) - work_base - merged) / i
            else:
                merged = sum(entry.tensor.to(dtype) - work_base for entry in entries)
                merged = merged / len(entries)
        elif self.method_name == "iso_c":
            merged = sum(entry.tensor.to(dtype) - work_base for entry in entries)
            # Match the reference's mean-then-sum arithmetic order as well.
            merged = (merged / len(entries)) * len(entries)
            u, s, vh = torch.linalg.svd(merged, full_matrices=False)
            merged = torch.linalg.multi_dot(
                (u, torch.diag(torch.ones_like(s) * s.mean()), vh)
            )
        else:
            q = min(base.shape)
            rank = int(q * (1 / len(entries)))
            # The official implementation returns a zero update when rank is zero.
            if rank == 0:
                return base.clone()
            left = work_base.new_zeros((base.shape[0], q))
            values = work_base.new_zeros(q)
            right = work_base.new_zeros((q, base.shape[1]))
            for i, entry in enumerate(entries):
                delta = entry.tensor.to(dtype) - work_base
                u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                block = slice(i * rank, (i + 1) * rank)
                left[:, block] = u[:, :rank]
                values[block] = s[:rank]
                right[block, :] = vh[:rank, :]
            # Keep the zero padding when q is not divisible by the model count.
            lu, _, lvh = torch.linalg.svd(left, full_matrices=False)
            ru, _, rvh = torch.linalg.svd(right, full_matrices=False)
            merged = torch.linalg.multi_dot((lu, lvh, torch.diag(values), ru, rvh))

        return (work_base + scale * merged).to(base.dtype)


tsv_merge = SpectralMerge("tsv")
iso_c_merge = SpectralMerge("iso_c")
