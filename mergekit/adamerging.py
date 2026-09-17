# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Configuration and small reusable pieces for data-informed AdaMerging.

AdaMerging learns task-vector coefficients with a full model forward/backward
pass.  The learned coefficients can then be materialized by mergekit's existing
task-arithmetic implementation.  This module defines that boundary without
coupling the merge planner to datasets or model training.
"""

import json
import random
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field, model_validator

from mergekit.common import ModelReference
from mergekit.config import (
    ConditionalParameter,
    InputModelDefinition,
    MergeConfiguration,
)
from mergekit.tokenizer.config import TokenizerConfig

Coefficient = Annotated[float, Field(ge=0.0, le=1.0)]


class AdaMergingInputModel(BaseModel, frozen=True):
    """One task-specific expert participating in AdaMerging."""

    name: str = Field(min_length=1)
    model: ModelReference


class AdaMergingOptimizationConfig(BaseModel, frozen=True):
    """Hyperparameters used while learning task-wise coefficients."""

    variant: Literal["adamerging", "adamerging_plus_plus"] = "adamerging"
    mode: Literal["task_wise", "layer_wise"] = "task_wise"
    ties_density: float = Field(default=0.2, gt=0.0, le=1.0)
    initial_coefficient: Coefficient = 0.3
    learning_rate: float = Field(default=1e-3, gt=0.0)
    iterations: int = Field(default=500, gt=0)
    seed: int = 0
    max_sequence_length: int = Field(default=32768, gt=1)
    max_new_tokens: int = Field(default=512, gt=0)
    samples_per_task: int = Field(default=16, gt=0)


class AdaMergingDatasetConfig(BaseModel, frozen=True):
    """JSONL source containing frozen training-split first-turn prompts."""

    path: Path
    prompt_field: str = "prompt"
    episode_id_field: str = "episode_id"
    turn_field: str = "turn"


class AdaMergingConfiguration(BaseModel, frozen=True):
    """Top-level AdaMerging configuration shared by optimization and export."""

    base_model: ModelReference
    models: List[AdaMergingInputModel]
    optimization: AdaMergingOptimizationConfig = AdaMergingOptimizationConfig()
    data: Dict[str, AdaMergingDatasetConfig]
    dtype: str = "bfloat16"
    out_dtype: Optional[str] = None
    tokenizer: Optional[TokenizerConfig] = None

    @model_validator(mode="after")
    def validate_models(self):
        if len(self.models) < 2:
            raise ValueError("AdaMerging requires at least two expert models")

        names = [entry.name for entry in self.models]
        if len(set(names)) != len(names):
            raise ValueError("AdaMerging expert names must be unique")

        models = [str(entry.model) for entry in self.models]
        if len(set(models)) != len(models):
            raise ValueError("AdaMerging expert model references must be unique")
        if str(self.base_model) in models:
            raise ValueError("The AdaMerging base model cannot also be an expert")

        expected = set(names)
        supplied = set(self.data)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected: {', '.join(extra)}")
            raise ValueError(
                "AdaMerging data names do not match experts ("
                + "; ".join(details)
                + ")"
            )
        return self


class AdaMergingCoefficients(BaseModel, frozen=True):
    """Portable artifact produced by coefficient optimization."""

    variant: Literal["adamerging", "adamerging_plus_plus"] = "adamerging"
    mode: Literal["task_wise", "layer_wise"] = "task_wise"
    weights: Dict[str, Union[Coefficient, Dict[str, Coefficient]]]
    iterations: int = Field(ge=0)
    final_objective: Optional[float] = None
    ties_thresholds: Optional[Dict[str, float]] = None


def build_task_arithmetic_config(
    config: AdaMergingConfiguration,
    coefficients: AdaMergingCoefficients,
) -> MergeConfiguration:
    """Build the static merge that materializes learned AdaMerging weights."""

    if coefficients.variant != "adamerging":
        raise ValueError("AdaMerging++ is saved directly by mergekit-adamerging")

    expected = {entry.name for entry in config.models}
    supplied = set(coefficients.weights)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise ValueError(
            "Coefficient names do not match experts (" + "; ".join(details) + ")"
        )

    tokenizer = config.tokenizer or TokenizerConfig(source=config.base_model)
    input_models = []
    for entry in config.models:
        learned = coefficients.weights[entry.name]
        if isinstance(learned, dict):
            weight = [
                ConditionalParameter(filter=name, value=value)
                for name, value in learned.items()
            ]
            weight.append(ConditionalParameter(filter="*", value=0.0))
        else:
            weight = float(learned)
        input_models.append(
            InputModelDefinition(model=entry.model, parameters={"weight": weight})
        )
    return MergeConfiguration(
        merge_method="task_arithmetic",
        base_model=config.base_model,
        models=input_models,
        parameters={"normalize": False, "lambda": 1.0},
        dtype=config.dtype,
        out_dtype=config.out_dtype,
        tokenizer=tokenizer,
    )


def load_first_turn_prompts(
    source: AdaMergingDatasetConfig,
    limit: int,
    seed: int = 0,
) -> List[dict]:
    """Load one first-turn prompt per episode from a frozen JSONL split."""

    records: List[dict] = []
    seen = set()
    with source.path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get(source.turn_field, -1)) != 0:
                continue
            episode_id = str(row.get(source.episode_id_field, ""))
            prompt = row.get(source.prompt_field)
            if not episode_id or not isinstance(prompt, str) or not prompt:
                raise ValueError(
                    f"{source.path}:{line_number}: invalid episode ID or prompt"
                )
            if episode_id in seen:
                continue
            seen.add(episode_id)
            records.append({"episode_id": episode_id, "prompt": prompt})
    if not records:
        raise ValueError(f"No first-turn prompts found in {source.path}")
    random.Random(seed).shuffle(records)
    return records[:limit]


def response_token_entropy(
    logits: torch.Tensor,
    prompt_length: int,
    response_length: int,
) -> torch.Tensor:
    """Mean predictive entropy over response positions only."""

    if prompt_length < 1 or response_length < 1:
        raise ValueError("prompt and response must each contain at least one token")
    start = prompt_length - 1
    stop = start + response_length
    selected = logits[..., start:stop, :].float()
    if selected.shape[-2] != response_length:
        raise ValueError("logits do not cover the complete response")
    log_probabilities = selected.log_softmax(dim=-1)
    probabilities = log_probabilities.exp()
    return -(probabilities * log_probabilities).sum(dim=-1).mean()


__all__ = [
    "AdaMergingCoefficients",
    "AdaMergingConfiguration",
    "AdaMergingDatasetConfig",
    "AdaMergingInputModel",
    "AdaMergingOptimizationConfig",
    "build_task_arithmetic_config",
    "load_first_turn_prompts",
    "response_token_entropy",
]
