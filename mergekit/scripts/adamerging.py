# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Learn task-wise AdaMerging coefficients and materialize the final merge."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import click
import torch
import yaml
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from mergekit.adamerging import (
    AdaMergingCoefficients,
    AdaMergingConfiguration,
    build_task_arithmetic_config,
    load_first_turn_prompts,
    response_token_entropy,
)
from mergekit.common import ModelReference, dtype_from_name
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, PrettyPrintHelp


def _local_path(reference: ModelReference, cache_dir: str | None) -> str:
    if reference.lora is not None:
        raise ValueError("AdaMerging optimization does not accept unmerged LoRAs")
    return reference.local_path(cache_dir=cache_dir)


def _load_model(
    path: str,
    dtype: torch.dtype,
    device_map,
    trust_remote_code: bool,
):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )


@torch.no_grad()
def _write_weights(model, base, deltas, coefficients, task_names) -> None:
    coefficient_values = coefficients.detach().cpu()
    for parameter_index, (name, parameter) in enumerate(model.named_parameters()):
        merged = base[name].clone()
        for task_index, task_name in enumerate(task_names):
            index = (
                task_index
                if coefficient_values.ndim == 1
                else (
                    parameter_index,
                    task_index,
                )
            )
            merged.add_(deltas[task_name][name], alpha=float(coefficient_values[index]))
        parameter.copy_(merged)


@torch.no_grad()
def _coefficient_gradients(model, deltas, task_names, layer_wise=False) -> torch.Tensor:
    device = next(model.parameters()).device
    shape = (
        (sum(1 for _ in model.named_parameters()), len(task_names))
        if layer_wise
        else (len(task_names),)
    )
    result = torch.zeros(shape, dtype=torch.float64, device=device)
    for parameter_index, (name, parameter) in enumerate(model.named_parameters()):
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        for index, task_name in enumerate(task_names):
            value = (gradient * deltas[task_name][name].float()).sum(
                dtype=torch.float64
            )
            if layer_wise:
                result[parameter_index, index] = value.to(device)
            else:
                result[index] += value.to(device)
        parameter.grad = None
    return result.float()


@torch.no_grad()
def _magnitude_threshold(task_deltas, density: float) -> float:
    """Exact global top-density threshold at bfloat16 storage precision."""

    histogram = torch.zeros(32768, dtype=torch.int64)
    total = 0
    chunk_size = 16 * 1024 * 1024
    for tensor in task_deltas.values():
        flat = tensor.detach().abs().to(torch.bfloat16).reshape(-1)
        total += flat.numel()
        for start in range(0, flat.numel(), chunk_size):
            bits = flat[start : start + chunk_size].view(torch.int16).long()
            histogram += torch.bincount(bits, minlength=32768).cpu()
    keep = int(total * density)
    rank = total - keep
    threshold_bits = int(
        torch.searchsorted(
            histogram.cumsum(0), torch.tensor(max(rank, 1), dtype=torch.int64)
        )
    )
    return float(
        torch.tensor([threshold_bits], dtype=torch.int16).view(torch.bfloat16)[0]
    )


@torch.no_grad()
def _apply_ties_preprocessing(deltas, task_names, density: float) -> dict[str, float]:
    """Apply the global trim/elect/disjoint steps used by AdaMerging++."""

    thresholds = {
        task_name: _magnitude_threshold(deltas[task_name], density)
        for task_name in task_names
    }
    for task_name in task_names:
        threshold = thresholds[task_name]
        for tensor in deltas[task_name].values():
            tensor.mul_(tensor.abs() >= threshold)

    majority_balance = 0
    for name in deltas[task_names[0]]:
        summed = torch.zeros_like(deltas[task_names[0]][name], dtype=torch.float32)
        for task_name in task_names:
            summed.add_(deltas[task_name][name].float())
        majority_balance += int(summed.sign().sum())
    majority_sign = 1 if majority_balance > 0 else -1 if majority_balance < 0 else 0

    for name in deltas[task_names[0]]:
        summed = torch.zeros_like(deltas[task_names[0]][name], dtype=torch.float32)
        for task_name in task_names:
            summed.add_(deltas[task_name][name].float())
        elected = summed.sign()
        if majority_sign:
            elected[elected == 0] = majority_sign
        for task_name in task_names:
            delta = deltas[task_name][name]
            keep = torch.where(elected > 0, delta > 0, delta < 0)
            delta.mul_(keep)
    return thresholds


def _prompt_fingerprint(task_name: str, record: dict) -> str:
    source = f"{task_name}\0{record['episode_id']}\0{record['prompt']}"
    return hashlib.sha256(source.encode()).hexdigest()


def _read_response_cache(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output[str(row["fingerprint"])] = [int(token) for token in row["token_ids"]]
    return output


def _append_response(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def _fixed_responses(model, tokenizer, datasets, optimization, cache_path: Path):
    cached = _read_response_cache(cache_path)
    input_device = model.get_input_embeddings().weight.device
    prepared = {}
    for task_name, records in datasets.items():
        prepared[task_name] = []
        for record in records:
            fingerprint = _prompt_fingerprint(task_name, record)
            prompt_ids = tokenizer.encode(record["prompt"], add_special_tokens=False)
            if (
                len(prompt_ids) + optimization.max_new_tokens
                > optimization.max_sequence_length
            ):
                raise ValueError(
                    f"{task_name}/{record['episode_id']}: prompt is too long for "
                    "max_sequence_length + max_new_tokens"
                )
            response_ids = cached.get(fingerprint)
            if response_ids is None:
                inputs = torch.tensor(
                    [prompt_ids], dtype=torch.long, device=input_device
                )
                generated = model.generate(
                    inputs,
                    do_sample=False,
                    max_new_tokens=optimization.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                response_ids = generated[0, len(prompt_ids) :].tolist()
                if not response_ids:
                    raise RuntimeError(
                        f"{task_name}/{record['episode_id']}: generated an empty response"
                    )
                _append_response(
                    cache_path,
                    {
                        "task": task_name,
                        "episode_id": record["episode_id"],
                        "fingerprint": fingerprint,
                        "token_ids": response_ids,
                    },
                )
            prepared[task_name].append((prompt_ids, response_ids))
    return prepared


def _entropy_loss(model, prompt_ids, response_ids) -> torch.Tensor:
    input_device = model.get_input_embeddings().weight.device
    tokens = torch.tensor(
        [prompt_ids + response_ids], dtype=torch.long, device=input_device
    )
    # The final input position predicts a token beyond the fixed response and
    # is not part of the objective. Requesting R+1 trailing positions gives
    # logits [P-1, ..., P+R-1]; the first R cover exactly the response.
    logits = model(
        input_ids=tokens,
        use_cache=False,
        logits_to_keep=len(response_ids) + 1,
    ).logits
    if logits.shape[-2] == len(response_ids) + 1:
        return response_token_entropy(logits, 1, len(response_ids))
    return response_token_entropy(logits, len(prompt_ids), len(response_ids))


def optimize_coefficients(
    config: AdaMergingConfiguration,
    artifact_dir: Path,
    cache_dir: str | None,
    trust_remote_code: bool,
    device_map: str,
    direct_output: Path | None = None,
) -> AdaMergingCoefficients:
    optimization = config.optimization
    torch.manual_seed(optimization.seed)
    dtype = dtype_from_name(config.dtype)
    base_path = _local_path(config.base_model, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = {
        task.name: load_first_turn_prompts(
            config.data[task.name],
            optimization.samples_per_task,
            optimization.seed + index,
        )
        for index, task in enumerate(config.models)
    }
    model = _load_model(base_path, dtype, device_map, trust_remote_code)
    model.config.use_cache = False
    model.eval()
    resolved_device_map = getattr(model, "hf_device_map", device_map)
    base = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    deltas = {}
    base_names = set(base)
    for task in config.models:
        expert = _load_model(
            _local_path(task.model, cache_dir),
            dtype,
            resolved_device_map,
            trust_remote_code,
        )
        expert_parameters = dict(expert.named_parameters())
        if set(expert_parameters) != base_names:
            raise ValueError(f"{task.name}: expert parameter names do not match base")
        deltas[task.name] = {}
        for name, base_parameter in base.items():
            expert_parameter = expert_parameters[name]
            if expert_parameter.shape != base_parameter.shape:
                raise ValueError(f"{task.name}/{name}: parameter shape mismatch")
            deltas[task.name][name] = (
                expert_parameter.detach().to(base_parameter.device) - base_parameter
            )
        del expert, expert_parameters
        gc.collect()
        torch.cuda.empty_cache()

    task_names = [task.name for task in config.models]
    ties_thresholds = None
    if optimization.variant == "adamerging_plus_plus":
        ties_thresholds = _apply_ties_preprocessing(
            deltas, task_names, optimization.ties_density
        )
    parameter_names = [name for name, _ in model.named_parameters()]
    coefficient_shape = (
        (len(parameter_names), len(task_names))
        if optimization.mode == "layer_wise"
        else (len(task_names),)
    )
    raw_coefficients = torch.nn.Parameter(
        torch.full(
            coefficient_shape,
            optimization.initial_coefficient,
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
    )
    optimizer = torch.optim.Adam(
        [raw_coefficients], lr=optimization.learning_rate, weight_decay=0.0
    )

    def coefficient_dict():
        values = raw_coefficients.detach().cpu()
        if optimization.mode == "task_wise":
            return {name: float(values[index]) for index, name in enumerate(task_names)}
        return {
            task_name: {
                parameter_name: float(values[parameter_index, task_index])
                for parameter_index, parameter_name in enumerate(parameter_names)
            }
            for task_index, task_name in enumerate(task_names)
        }

    def coefficient_summary():
        values = raw_coefficients.detach().float().cpu()
        if values.ndim == 1:
            return coefficient_dict()
        return {
            task_name: {
                "min": float(values[:, task_index].min()),
                "mean": float(values[:, task_index].mean()),
                "max": float(values[:, task_index].max()),
            }
            for task_index, task_name in enumerate(task_names)
        }

    _write_weights(model, base, deltas, raw_coefficients, task_names)
    fixed = _fixed_responses(
        model,
        tokenizer,
        datasets,
        optimization,
        artifact_dir / "fixed_responses.jsonl",
    )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    log_path = artifact_dir / "training.jsonl"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final_objective = None
    with log_path.open("w", encoding="utf-8") as log:
        for step in trange(optimization.iterations, desc="AdaMerging"):
            model.zero_grad(set_to_none=True)
            task_losses = {}
            for task_name in task_names:
                samples = fixed[task_name]
                prompt_ids, response_ids = samples[step % len(samples)]
                loss = _entropy_loss(model, prompt_ids, response_ids)
                (loss / len(task_names)).backward()
                task_losses[task_name] = float(loss.detach())
            scalar_gradient = _coefficient_gradients(
                model,
                deltas,
                task_names,
                layer_wise=optimization.mode == "layer_wise",
            )
            optimizer.zero_grad(set_to_none=True)
            raw_coefficients.grad = scalar_gradient.to(raw_coefficients.device)
            optimizer.step()
            with torch.no_grad():
                raw_coefficients.clamp_(0.0, 1.0)
            _write_weights(model, base, deltas, raw_coefficients, task_names)
            final_objective = sum(task_losses.values()) / len(task_names)
            row = {
                "iteration": step + 1,
                "objective": final_objective,
                "task_entropy": task_losses,
                "coefficients": coefficient_summary(),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()

    coefficients = AdaMergingCoefficients(
        variant=optimization.variant,
        mode=optimization.mode,
        weights=coefficient_dict(),
        iterations=optimization.iterations,
        final_objective=final_objective,
        ties_thresholds=ties_thresholds,
    )
    if direct_output is not None:
        direct_output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(
            direct_output,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer.save_pretrained(direct_output)
    del model, base, deltas, optimizer, raw_coefficients
    gc.collect()
    torch.cuda.empty_cache()
    return coefficients


@click.command("mergekit-adamerging", cls=PrettyPrintHelp)
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=click.Path())
@click.option("--device-map", default="balanced", show_default=True)
@click.option("--cache-dir", default=None, type=click.Path(file_okay=False))
@click.option("--trust-remote-code", is_flag=True)
@click.option("--optimize-only", is_flag=True)
def main(
    config_file: str,
    out_path: str,
    device_map: str,
    cache_dir: str | None,
    trust_remote_code: bool,
    optimize_only: bool,
) -> None:
    source = Path(config_file).read_text(encoding="utf-8")
    config = AdaMergingConfiguration.model_validate(yaml.safe_load(source))
    artifact_dir = Path(str(out_path) + ".adamerging")
    direct_output = (
        Path(out_path)
        if config.optimization.variant == "adamerging_plus_plus" and not optimize_only
        else None
    )
    coefficients = optimize_coefficients(
        config,
        artifact_dir,
        cache_dir,
        trust_remote_code,
        device_map,
        direct_output=direct_output,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    coefficient_path = artifact_dir / "coefficients.json"
    coefficient_path.write_text(
        coefficients.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    click.echo(coefficients.model_dump_json(indent=2))
    if optimize_only:
        return
    if config.optimization.variant == "adamerging":
        merge_config = build_task_arithmetic_config(config, coefficients)
        merge_yaml = merge_config.to_yaml()
        (artifact_dir / "merge.yml").write_text(merge_yaml + "\n", encoding="utf-8")
        run_merge(
            merge_config,
            out_path,
            options=MergeOptions(cuda=True, trust_remote_code=trust_remote_code),
            config_source=merge_yaml,
        )
    else:
        (Path(out_path) / "adamerging_config.yml").write_text(source, encoding="utf-8")
    (Path(out_path) / "adamerging_coefficients.json").write_text(
        coefficients.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
