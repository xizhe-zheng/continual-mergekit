import pytest
import torch
import yaml
from pydantic import ValidationError

from mergekit.adamerging import (
    AdaMergingCoefficients,
    AdaMergingConfiguration,
    AdaMergingDatasetConfig,
    build_task_arithmetic_config,
    load_first_turn_prompts,
    response_token_entropy,
)
from mergekit.scripts.adamerging import (
    _apply_ties_preprocessing,
    _coefficient_gradients,
    _write_weights,
)

CONFIG = """
base_model: base
models:
  - name: tool
    model: tool-expert
  - name: search
    model: search-expert
optimization:
  mode: task_wise
  initial_coefficient: 0.3
  learning_rate: 0.001
  iterations: 500
data:
  tool:
    path: tool.jsonl
  search:
    path: search.jsonl
dtype: bfloat16
"""


def test_build_task_arithmetic_config_from_learned_coefficients():
    config = AdaMergingConfiguration.model_validate(yaml.safe_load(CONFIG))
    coefficients = AdaMergingCoefficients(
        weights={"tool": 0.2, "search": 0.7},
        iterations=500,
        final_objective=1.25,
    )

    merge_config = build_task_arithmetic_config(config, coefficients)

    assert merge_config.merge_method == "task_arithmetic"
    assert str(merge_config.base_model) == "base"
    assert merge_config.parameters == {"normalize": False, "lambda": 1.0}
    assert [model.parameters["weight"] for model in merge_config.models] == [
        pytest.approx(0.2),
        pytest.approx(0.7),
    ]
    assert str(merge_config.tokenizer.source) == "base"


def test_coefficient_names_must_match_experts():
    config = AdaMergingConfiguration.model_validate(yaml.safe_load(CONFIG))
    coefficients = AdaMergingCoefficients(
        weights={"tool": 0.2, "other": 0.7},
        iterations=1,
    )

    with pytest.raises(ValueError, match="missing: search; unexpected: other"):
        build_task_arithmetic_config(config, coefficients)


@pytest.mark.parametrize("coefficient", [-0.01, 1.01])
def test_coefficients_are_clamped_to_paper_range(coefficient):
    with pytest.raises(ValidationError):
        AdaMergingCoefficients(weights={"tool": coefficient}, iterations=1)


def test_expert_names_must_be_unique():
    raw = yaml.safe_load(CONFIG)
    raw["models"][1]["name"] = "tool"

    with pytest.raises(ValidationError, match="expert names must be unique"):
        AdaMergingConfiguration.model_validate(raw)


def test_data_names_must_match_experts():
    raw = yaml.safe_load(CONFIG)
    raw["data"]["other"] = raw["data"].pop("search")

    with pytest.raises(ValidationError, match="missing: search; unexpected: other"):
        AdaMergingConfiguration.model_validate(raw)


@pytest.mark.parametrize(
    ("variant", "mode"),
    [
        ("adamerging", "task_wise"),
        ("adamerging", "layer_wise"),
        ("adamerging_plus_plus", "task_wise"),
        ("adamerging_plus_plus", "layer_wise"),
    ],
)
def test_official_variant_and_coefficient_granularity_combinations(variant, mode):
    raw = yaml.safe_load(CONFIG)
    raw["optimization"]["variant"] = variant
    raw["optimization"]["mode"] = mode

    config = AdaMergingConfiguration.model_validate(raw)

    assert config.optimization.variant == variant
    assert config.optimization.mode == mode


def test_loads_one_shuffled_first_turn_per_episode(tmp_path):
    path = tmp_path / "states.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"episode_id":"a","turn":0,"prompt":"pa"}',
                '{"episode_id":"a","turn":0,"prompt":"duplicate"}',
                '{"episode_id":"a","turn":1,"prompt":"later"}',
                '{"episode_id":"b","turn":0,"prompt":"pb"}',
            ]
        )
        + "\n"
    )

    rows = load_first_turn_prompts(AdaMergingDatasetConfig(path=path), 10, seed=3)

    assert {row["episode_id"] for row in rows} == {"a", "b"}
    assert next(row for row in rows if row["episode_id"] == "a")["prompt"] == "pa"


def test_entropy_uses_all_and_only_response_positions():
    logits = torch.zeros(1, 5, 2)
    logits[:, 0, :] = torch.tensor([100.0, -100.0])
    logits[:, 1:3, :] = 0.0
    logits[:, 3:, :] = torch.tensor([100.0, -100.0])

    loss = response_token_entropy(logits, prompt_length=2, response_length=2)

    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_manual_task_vector_gradient_matches_chain_rule():
    model = torch.nn.Linear(2, 1, bias=False)
    base = {"weight": torch.tensor([[1.0, 2.0]])}
    deltas = {
        "a": {"weight": torch.tensor([[0.5, -1.0]])},
        "b": {"weight": torch.tensor([[2.0, 3.0]])},
    }
    weights = torch.tensor([0.2, 0.4])
    _write_weights(model, base, deltas, weights, ["a", "b"])
    model(torch.tensor([[3.0, -2.0]])).square().sum().backward()

    actual = _coefficient_gradients(model, deltas, ["a", "b"])

    coefficients = torch.tensor([0.2, 0.4], requires_grad=True)
    merged = base["weight"] + coefficients[0] * deltas["a"]["weight"]
    merged = merged + coefficients[1] * deltas["b"]["weight"]
    (torch.tensor([[3.0, -2.0]]) @ merged.T).square().sum().backward()
    assert actual.tolist() == pytest.approx(coefficients.grad.tolist())


def test_layer_wise_coefficients_export_as_tensor_filters():
    config = AdaMergingConfiguration.model_validate(yaml.safe_load(CONFIG))
    coefficients = AdaMergingCoefficients(
        mode="layer_wise",
        weights={
            "tool": {"model.layer.weight": 0.2},
            "search": {"model.layer.weight": 0.7},
        },
        iterations=1,
    )

    merge_config = build_task_arithmetic_config(config, coefficients)

    tool_weights = merge_config.models[0].parameters["weight"]
    assert tool_weights[0].filter == "model.layer.weight"
    assert tool_weights[0].value == pytest.approx(0.2)
    assert tool_weights[-1].filter == "*"
    assert tool_weights[-1].value == 0.0


def test_layer_wise_manual_gradient_keeps_one_value_per_tensor_and_task():
    model = torch.nn.Linear(2, 1, bias=False)
    base = {"weight": torch.tensor([[1.0, 2.0]])}
    deltas = {
        "a": {"weight": torch.tensor([[0.5, -1.0]])},
        "b": {"weight": torch.tensor([[2.0, 3.0]])},
    }
    coefficients = torch.tensor([[0.2, 0.4]])
    _write_weights(model, base, deltas, coefficients, ["a", "b"])
    model(torch.tensor([[3.0, -2.0]])).square().sum().backward()

    actual = _coefficient_gradients(model, deltas, ["a", "b"], layer_wise=True)

    assert actual.shape == (1, 2)
    scalar_coefficients = torch.tensor([0.2, 0.4], requires_grad=True)
    merged = base["weight"] + scalar_coefficients[0] * deltas["a"]["weight"]
    merged = merged + scalar_coefficients[1] * deltas["b"]["weight"]
    (torch.tensor([[3.0, -2.0]]) @ merged.T).square().sum().backward()
    assert actual[0].tolist() == pytest.approx(scalar_coefficients.grad.tolist())


def test_adamerging_plus_plus_ties_preprocessing():
    deltas = {
        "a": {"weight": torch.tensor([10.0, 1.0, -8.0, 0.5]).bfloat16()},
        "b": {"weight": torch.tensor([-9.0, 7.0, -2.0, 0.1]).bfloat16()},
    }

    thresholds = _apply_ties_preprocessing(deltas, ["a", "b"], density=0.5)

    # This matches the reference implementation's kth-value boundary,
    # including values equal to the threshold.
    assert thresholds == {"a": 1.0, "b": 2.0}
    assert deltas["a"]["weight"].float().tolist() == [10.0, 1.0, -8.0, 0.0]
    assert deltas["b"]["weight"].float().tolist() == [0.0, 7.0, -2.0, 0.0]
