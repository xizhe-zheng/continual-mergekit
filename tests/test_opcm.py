"""Compare every step against the actual optional OPCM sibling source."""

import ast
import json
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.opcm import merge_opcm
from mergekit.options import MergeOptions


@pytest.fixture
def official():
    root = Path(__file__).resolve().parents[2] / "opcm/fusion_bench"
    if not root.exists():
        pytest.skip("Official OPCM sibling checkout unavailable")
    ns = dict(
        torch=torch,
        nn=nn,
        Tensor=Tensor,
        Tuple=tuple,
        Optional=Optional,
        random=random,
        np=np,
        deepcopy=deepcopy,
        copy=SimpleNamespace(deepcopy=deepcopy),
        OrderedDict=OrderedDict,
        List=list,
        StateDictType=dict,
        cast=cast,
        CLIPVisionModelTaskPool=object,
        CLIPVisionModel=nn.Module,
        BaseModelPool=object,
        tqdm=lambda iterable, **kwargs: iterable,
    )
    for filename, names in [
        ("utils/parameters.py", {"state_dict_to_vector"}),
        ("utils/state_dict_arithmetic.py", {"state_dict_sub"}),
        (
            "method/opcm/utils.py",
            {"_svd", "svd", "get_task_vector_norm", "is_leaf_module"},
        ),
    ]:
        path = root / filename
        tree = ast.parse(path.read_text())
        tree.body = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        exec(compile(tree, str(path), "exec"), ns)
    path = root / "method/opcm/opcm.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.bases = []
    tree.body = [cls]
    exec(compile(tree, str(path), "exec"), ns)
    return ns["OPCMForCLIP"]


class ToyModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.root_parameter = nn.Parameter(torch.randn(2, dtype=dtype))
        self.embedding = nn.Embedding(7, 4, dtype=dtype)
        self.up = nn.Linear(4, 7, dtype=dtype)
        self.down = nn.Linear(7, 4, bias=False, dtype=dtype)
        self.norm = nn.LayerNorm(4, dtype=dtype)
        self.register_buffer("counter", torch.tensor(2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_every_step_matches_original(official, dtype, alpha, device):
    torch.manual_seed(12)
    base = ToyModel(dtype)
    tasks = [deepcopy(base) for _ in range(4)]
    for task in tasks:
        for param in task.parameters():
            param.data.add_(torch.randn_like(param) * 0.1)
    originals = [deepcopy(m.state_dict()) for m in [base, *tasks]]
    expected = []
    runner = official(alpha=alpha, shuffle_order=False)
    runner.fabric = SimpleNamespace(device=device, log=lambda *args, **kwargs: None)
    runner._program = SimpleNamespace(taskpool=SimpleNamespace(_test_datasets={}))
    runner.log_dir = None
    runner.save_merged_model = lambda model, step: expected.append(
        deepcopy(model.state_dict())
    )
    pool = SimpleNamespace(
        model_names=list(range(len(tasks))),
        load_pretrained_model=lambda: deepcopy(base),
        load_model=lambda name: deepcopy(tasks[name]),
    )
    runner.run(pool)
    actual = []
    merge_opcm(
        base,
        iter(tasks),
        alpha=alpha,
        device=device,
        on_step=lambda step, model, stats: actual.append(deepcopy(model.state_dict())),
    )
    assert len(actual) == len(expected) == 4
    for got, want in zip(actual, expected):
        for name in want:
            torch.testing.assert_close(got[name], want[name], rtol=0, atol=0)
    for model, original in zip([base, *tasks], originals):
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


def test_invalid_and_zero_norm():
    base = nn.Linear(3, 4)
    with pytest.raises(ValueError, match="at least one"):
        merge_opcm(base, [])
    with pytest.raises(ValueError, match="matching shape"):
        merge_opcm(base, [nn.Linear(4, 4)])
    zero = deepcopy(base)
    with pytest.raises(ValueError, match="undefined"):
        merge_opcm(base, [zero, zero])
    got = merge_opcm(base, [zero])
    torch.testing.assert_close(got.weight, zero.weight, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["clip", "llama"])
def test_yaml_roundtrip(tmp_path, architecture):
    from transformers import (
        CLIPVisionConfig,
        CLIPVisionModel,
        LlamaConfig,
        LlamaForCausalLM,
    )

    torch.manual_seed(9)
    cfg = CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    if architecture == "llama":
        cfg = LlamaConfig(
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=16,
            tie_word_embeddings=True,
        )
        model_class = LlamaForCausalLM
    else:
        model_class = CLIPVisionModel
    base = model_class(cfg)
    tasks = [deepcopy(base) for _ in range(3)]
    for task in tasks:
        for param in task.parameters():
            param.data.add_(0.01 * torch.randn_like(param))
    paths = [tmp_path / name for name in ("base", "one", "two", "three")]
    for path, model in zip(paths, [base, *tasks]):
        model.save_pretrained(path)
    config = MergeConfiguration.model_validate(
        dict(
            merge_method="opcm",
            base_model=str(paths[0]),
            models=[{"model": str(p)} for p in paths[1:]],
            dtype="float32",
            out_dtype="bfloat16" if architecture == "llama" else None,
            parameters=dict(
                alpha=0.5, shuffle_order=True, seed=42, save_on_every_step=True
            ),
        )
    )
    output = tmp_path / "out"
    run_merge(config, str(output), MergeOptions(copy_tokenizer=False))
    result = model_class.from_pretrained(output, dtype="auto")
    order = list(range(len(tasks)))
    random.Random(42).shuffle(order)
    assert json.loads((output / "model_names.json").read_text()) == [
        str(paths[i + 1]) for i in order
    ]
    expected = merge_opcm(base, [tasks[i] for i in order])
    if architecture == "llama":
        expected.to(torch.bfloat16)
    for name, tensor in expected.state_dict().items():
        # Streaming changes norm reduction and projection multiplication order.
        torch.testing.assert_close(
            result.state_dict()[name], tensor, rtol=1e-5, atol=1e-7
        )
    assert len(json.loads((output / "opcm_steps.json").read_text())) == 3
    assert (output / "checkpoints/merged_model_2/config.json").exists()
    assert (output / "README.md").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"parameters": {"evaluate_on_every_step": True}},
        {"parameters": {"alpha": -1}},
        {"dtype": "bfloat16"},
        {"tokenizer_source": "union"},
    ],
)
def test_unsupported_config_fails_before_loading(tmp_path, overrides):
    config = MergeConfiguration.model_validate(
        dict(
            merge_method="opcm",
            base_model="absent/base",
            models=[{"model": "absent/task"}],
            **overrides,
        )
    )
    with pytest.raises(ValueError):
        run_merge(config, str(tmp_path / "out"), MergeOptions(copy_tokenizer=False))


@pytest.mark.parametrize("shape", [(7, 4), (4, 7), (5, 5)])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_memory_bounded_projection_matches_source(official, shape, alpha, device):
    from mergekit.opcm_streaming import project_update

    torch.manual_seed(81)
    previous, task = [
        torch.randn(shape, dtype=torch.float64, device=device) for _ in range(2)
    ]
    runner = official(alpha=alpha)
    runner.previous_lambda_t, runner.lambda_t = 0, 1
    want = runner.merge_linear_weights(
        previous, torch.zeros_like(previous), task, "weight", alpha, accelerator=device
    )
    got = project_update(previous, task, alpha)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_streamed_norm_aliases_buffers_and_chunks(tmp_path):
    from mergekit.opcm_streaming import DiskState, Layout, streamed_norm

    torch.manual_seed(42)
    base = nn.Module()
    base.first = nn.Embedding(1025, 1024)
    base.second = nn.Linear(1024, 1025, bias=False)
    base.second.weight = base.first.weight
    base.register_buffer("counter", torch.tensor(2))
    task = deepcopy(base)
    task.first.weight.data.add_(torch.randn_like(task.first.weight))
    task.counter.add_(1)
    layout = Layout(base)
    stores = []
    for i, model in enumerate((base, task)):
        path = tmp_path / str(i)
        path.mkdir()
        store = DiskState(path, layout)
        for name in layout.aliases:
            store.put(name, model.state_dict()[name])
        stores.append(store)
    got = streamed_norm(stores[1], stores[0], layout, torch.float32)
    # A monolithic FP32 norm itself loses precision at this size. Compare to
    # a dense FP64 reduction of the same FP32 differences, including both aliases.
    reference = torch.linalg.vector_norm(
        torch.cat(
            [
                (task.state_dict()[name] - base.state_dict()[name]).reshape(-1).double()
                for name in sorted(base.state_dict())
            ]
        )
    ).float()
    torch.testing.assert_close(got, reference, rtol=1e-7, atol=0)


def test_no_full_checkpoint_load_or_leftover_scratch(tmp_path, monkeypatch):
    from transformers import CLIPVisionConfig, CLIPVisionModel

    model = CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
        )
    )
    base = tmp_path / "base"
    task = tmp_path / "task"
    model.save_pretrained(base)
    model.save_pretrained(task)

    def forbidden(*args, **kwargs):
        raise AssertionError("Streaming must not load a full model")

    monkeypatch.setattr(CLIPVisionModel, "from_pretrained", forbidden)
    config = MergeConfiguration.model_validate(
        dict(
            merge_method="opcm",
            base_model=str(base),
            models=[{"model": str(task)}],
            parameters={"save_on_every_step": False},
        )
    )
    output = tmp_path / "out"
    run_merge(
        config,
        str(output),
        MergeOptions(copy_tokenizer=False, write_model_card=False, quiet=True),
    )
    assert not list(output.glob(".opcm-work-*"))
    assert not (output / "checkpoints").exists()
    assert list(output.glob("*.safetensors"))
    # A failing normalization must also clean up disk intermediates.
    config.models.append(config.models[0])
    with pytest.raises(ValueError, match="undefined"):
        run_merge(
            config,
            str(tmp_path / "failed"),
            MergeOptions(copy_tokenizer=False, quiet=True),
        )
    assert not list((tmp_path / "failed").glob(".opcm-work-*"))
