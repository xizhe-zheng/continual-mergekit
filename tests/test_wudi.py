import math

import pytest
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.io import LazyTensorLoader
from mergekit.merge import run_merge
from mergekit.merge_methods import merge_state_dicts, merge_tensors
from mergekit.options import MergeOptions


def _reference_wudi(base, experts, learning_rate, iterations):
    deltas = [expert - base for expert in experts]
    active = [delta for delta in deltas if delta.square().sum().item() > 0]
    merged = torch.nn.Parameter(sum(active, start=torch.zeros_like(base)))
    optimizer = torch.optim.Adam([merged], lr=learning_rate)
    norms = [delta.square().sum() for delta in active]
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = sum(
            torch.mm(merged - delta, delta.transpose(0, 1)).square().sum() / norm
            for delta, norm in zip(active, norms)
        )
        loss.backward()
        optimizer.step()
    return base + merged.detach()


@pytest.mark.parametrize("shape", [(4, 3), (3, 4), (4, 4)])
def test_matches_reference_adam(shape):
    generator = torch.Generator().manual_seed(123)
    base = torch.randn(shape, generator=generator)
    experts = [base + torch.randn(shape, generator=generator) * 0.1 for _ in range(3)]
    expected = _reference_wudi(base, experts, learning_rate=2e-4, iterations=7)
    actual = merge_tensors(
        [base, *experts],
        "wudi",
        base_index=0,
        name="model.layers.0.self_attn.q_proj.weight",
        parameters={"learning_rate": 2e-4, "iterations": 7},
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_zero_iterations_is_task_vector_sum():
    base = torch.ones(2, 2)
    first = base + torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    second = base + torch.tensor([[0.5, -1.0], [2.0, 0.0]])
    actual = merge_tensors(
        [base, first, second],
        "wudi",
        base_index=0,
        name="linear.weight",
        parameters={"iterations": 0},
    )
    torch.testing.assert_close(actual, base + (first - base) + (second - base))


@pytest.mark.parametrize(
    "name,shape",
    [
        ("model.layers.0.input_layernorm.weight", (4,)),
        ("model.embed_tokens.weight", (8, 4)),
        ("lm_head.weight", (8, 4)),
    ],
)
def test_non_linear_and_excluded_weights_keep_base(name, shape):
    base = torch.randn(shape)
    actual = merge_tensors(
        [base, base + 1], "wudi", base_index=0, name=name, parameters={"iterations": 1}
    )
    torch.testing.assert_close(actual, base)
    assert actual.data_ptr() != base.data_ptr()


def test_zero_delta_expert_is_ignored():
    base = torch.zeros(3, 2)
    expert = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    result = merge_tensors(
        [base, base.clone(), expert],
        "wudi",
        base_index=0,
        name="linear.weight",
        parameters={"iterations": 2},
    )
    assert result.isfinite().all()


@pytest.mark.parametrize(
    "parameters",
    [
        {"learning_rate": 0.0},
        {"iterations": -1},
        {"beta1": 1.0},
        {"beta2": -0.1},
        {"epsilon": math.inf},
    ],
)
def test_rejects_invalid_parameters(parameters):
    base = torch.zeros(2, 2)
    with pytest.raises(ValueError):
        merge_tensors(
            [base, base + 1],
            "wudi",
            base_index=0,
            name="linear.weight",
            parameters=parameters,
        )


def test_yaml_checkpoint_merge(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(42)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
        )
    )
    states = []
    for index in range(3):
        state = {
            name: tensor.detach().clone() + index * torch.randn_like(tensor) * 0.01
            for name, tensor in model.state_dict().items()
        }
        states.append(state)
        model.load_state_dict(state)
        model.save_pretrained(tmp_path / str(index))

    parameters = {"learning_rate": 1e-4, "iterations": 2}
    config = MergeConfiguration.model_validate(
        yaml.safe_load(
            f"""
merge_method: wudi
base_model: {tmp_path / "0"}
models:
  - model: {tmp_path / "1"}
  - model: {tmp_path / "2"}
parameters:
  learning_rate: 1.0e-4
  iterations: 2
"""
        )
    )
    expected = merge_state_dicts(states, "wudi", base=0, parameters=parameters)
    run_merge(
        config, str(tmp_path / "out"), MergeOptions(copy_tokenizer=False, quiet=True)
    )
    loader = LazyTensorLoader.from_disk(str(tmp_path / "out"))
    for key, tensor in expected.items():
        torch.testing.assert_close(loader.get_tensor(key), tensor)
