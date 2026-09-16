import ast
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from mergekit.config import MergeConfiguration
from mergekit.io import LazyTensorLoader
from mergekit.io.tasks import LoaderCache
from mergekit.merge import run_merge
from mergekit.merge_methods import merge_tensors
from mergekit.options import MergeOptions
from mergekit.ramplus import prepare_ramplus, unique_scales


def test_scale_edge_cases():
    assert unique_scales([0, 4, 6], [0, 4, 2], 1.2) == [1.0, 1.2, 1.1]
    assert unique_scales([4], [4], 1.0) == [1.0]
    assert unique_scales([4], [2], 0.5) == [1.0]


def test_unique_scaling_and_overlap_cancellation():
    base = torch.full((4,), 10.0)
    a = base + torch.tensor([2.0, 4.0, 0.0, 0.0])
    b = base + torch.tensor([0.0, -4.0, 6.0, 0.0])
    result = merge_tensors(
        [base, a, b], "ramplus", base_index=0, parameters={"unique_scale": [1.1, 1.2]}
    )
    torch.testing.assert_close(result, torch.tensor([12.2, 10.0, 17.2, 10.0]))
    torch.testing.assert_close(base, torch.full((4,), 10.0))


@pytest.mark.parametrize("r", [1.0, 1.2])
def test_full_checkpoint_global_statistics(tmp_path, r):
    cfg = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
    )
    model = LlamaForCausalLM(cfg)
    base = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    inputs = []
    for i in range(3):
        state = {}
        for j, (key, tensor) in enumerate(base.items()):
            delta = tensor.clone().flatten()
            if i:
                delta[(i - 1) :: (3 + j % 2)] = 0.01 * i
                delta[0] = 0.03 * i
            state[key] = delta.reshape_as(tensor)
        inputs.append(state)
        model.load_state_dict(state)
        model.save_pretrained(tmp_path / str(i))
    changed = [0, 0]
    overlapping = [0, 0]
    for key in base:
        masks = torch.stack([state[key].abs() > 1e-5 for state in inputs[1:]])
        shared = masks.sum(0) > 1
        for i in range(2):
            changed[i] += int(masks[i].sum())
            overlapping[i] += int((masks[i] & shared).sum())
    scales = [1 + (r - 1) * min(1, o / (c - o)) for c, o in zip(changed, overlapping)]
    expected = {}
    for key in base:
        ds = torch.stack([state[key] for state in inputs[1:]])
        masks = ds.abs() > 1e-5
        counts = masks.sum(0)
        avg = (ds * masks).sum(0) / counts.clamp(min=1)
        scaled = sum(ds[i] * masks[i] * scales[i] for i in range(2))
        expected[key] = torch.where(counts == 1, scaled, avg)
    config = MergeConfiguration.model_validate(
        {
            "merge_method": "ramplus",
            "base_model": str(tmp_path / "0"),
            "models": [{"model": str(tmp_path / str(i))} for i in [1, 2]],
            "parameters": {"rescale_factor": r},
        }
    )
    run_merge(
        config,
        str(tmp_path / "out"),
        MergeOptions(copy_tokenizer=False, quiet=True, write_model_card=True),
    )
    loader = LazyTensorLoader.from_disk(str(tmp_path / "out"))
    for key in expected:
        torch.testing.assert_close(loader.get_tensor(key), expected[key])
    assert config.models[0].parameters is None
    saved = yaml.safe_load((tmp_path / "out" / "mergekit_config.yml").read_text())
    assert saved["parameters"] == {"rescale_factor": r}


def test_against_actual_mrl_function(tmp_path):
    source = Path(__file__).resolve().parents[2] / "mrl" / "ram-main.py"
    if not source.exists():
        pytest.skip("Optional sibling mrl checkout unavailable")
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "agentic_reinforcement_merge_rescale_v2"
    ]
    namespace = {"torch": torch}
    exec(compile(tree, str(source), "exec"), namespace)
    base = {"x": torch.zeros(4), "y": torch.zeros(4)}
    tasks = [
        {
            "x": torch.tensor([1.0, 2.0, 0.0, 0.0]),
            "y": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        },
        {
            "x": torch.tensor([0.0, 3.0, 4.0, 0.0]),
            "y": torch.tensor([2.0, 0.0, 0.0, 1.0]),
        },
    ]
    paths = []
    for i, state in enumerate([base, *tasks]):
        path = tmp_path / str(i)
        path.mkdir()
        save_file(state, path / "model.safetensors")
        paths.append(str(path))
    config = MergeConfiguration.model_validate(
        {
            "merge_method": "ramplus",
            "base_model": paths[0],
            "models": [{"model": path} for path in paths[1:]],
            "parameters": {"rescale_factor": 1.2},
        }
    )
    resolved = prepare_ramplus(config, LoaderCache(), quiet=True)
    scales = [m.parameters["unique_scale"] for m in resolved.models]
    expected = namespace["agentic_reinforcement_merge_rescale_v2"](base, tasks, r=1.2)
    for key in base:
        result = merge_tensors(
            [base[key], tasks[0][key], tasks[1][key]],
            "ramplus",
            base_index=0,
            parameters={"unique_scale": scales},
        )
        torch.testing.assert_close(result, expected[key])


@pytest.mark.parametrize(
    "update",
    [
        {"dtype": "bfloat16"},
        {"tokenizer_source": "base"},
        {"parameters": {"epsilon": -1.0}},
        {"parameters": {"epsilon": [0.0, 1.0]}},
        {"parameters": {"rescale_factor": float("inf")}},
    ],
)
def test_reject_unsupported_settings(update):
    config = MergeConfiguration.model_validate(
        {
            "merge_method": "ramplus",
            "base_model": "/base",
            "models": [{"model": "/task"}],
            **update,
        }
    )
    with pytest.raises(ValueError):
        prepare_ramplus(config, None)


def test_all_overlap_and_unchanged_task(tmp_path):
    paths = []
    for i, value in enumerate([0.0, 1.0, 2.0, 0.0]):
        path = tmp_path / str(i)
        path.mkdir()
        save_file({"w": torch.full((3,), value)}, path / "model.safetensors")
        paths.append(str(path))
    config = MergeConfiguration.model_validate(
        {
            "merge_method": "ramplus",
            "base_model": paths[0],
            "models": [{"model": path} for path in paths[1:]],
        }
    )
    result = prepare_ramplus(config, LoaderCache(), quiet=True)
    assert [m.parameters["unique_scale"] for m in result.models] == [1.2, 1.2, 1.0]
