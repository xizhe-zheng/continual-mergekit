import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge_methods import merge_state_dicts, merge_tensors


@pytest.fixture(params=["tsv", "iso_c"])
def method(request):
    return request.param


@pytest.fixture
def official(method):
    sibling = Path(__file__).resolve().parents[2]
    if method == "tsv":
        path = sibling / "task_singular_vectors/src/utils/TSVM_utils.py"
        function = "compute_and_sum_svd_mem_reduction"
    else:
        path = sibling / "iso-merging/src/utils/iso.py"
        function = "iso_c"
    if not path.exists():
        pytest.skip("Optional official sibling checkout unavailable")
    spec = importlib.util.spec_from_file_location("official_spectral", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function)


@pytest.mark.parametrize(
    "shape", [(8, 8), (7, 11), (11, 7), (2, 2), (8,), (), (2, 3, 4)]
)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
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
def test_matches_official(method, official, shape, dtype, device):
    generator = torch.Generator().manual_seed(123)
    inputs = [
        torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
        for _ in range(4)
    ]
    originals = [t.clone() for t in inputs]
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    base = inputs[0].to(work_dtype)
    tvs = [
        SimpleNamespace(vector={"weight": t.to(work_dtype) - base}) for t in inputs[1:]
    ]
    config = SimpleNamespace(device=device, DATASETS=list(range(3)))
    expected = (base + 0.7 * official(tvs, config)["weight"]).to(dtype)
    actual = merge_tensors(inputs, method, base_index=0, parameters={"lambda": 0.7})
    torch.testing.assert_close(actual, expected)
    for tensor, original in zip(inputs, originals):
        torch.testing.assert_close(tensor, original)


def test_projection_and_vector_use_mean(method):
    base = {"text_projection.weight": torch.ones(4, 4), "norm.weight": torch.ones(4)}
    models = [base] + [{k: t + delta for k, t in base.items()} for delta in [2.0, 6.0]]
    result = merge_state_dicts(models, method, base=0, parameters={"lambda": 0.5})
    for tensor in result.values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 3.0))


def test_iso_c_uniform_spectrum_including_zero_singular_values():
    base = torch.zeros(3, 3)
    delta = torch.diag(torch.tensor([9.0, 3.0, 0.0]))
    result = merge_tensors([base, delta], "iso_c", base_index=0)
    torch.testing.assert_close(torch.linalg.svdvals(result), torch.full((3,), 4.0))


def test_tsv_retains_each_tasks_top_direction():
    base = torch.zeros(4, 4)
    a = torch.diag(torch.tensor([9.0, 8.0, 0.2, 0.1]))
    b = torch.diag(torch.tensor([0.1, 0.2, 7.0, 6.0]))
    result = merge_tensors([base, a, b], "tsv", base_index=0)
    torch.testing.assert_close(result, torch.diag(torch.tensor([9.0, 8.0, 7.0, 6.0])))


def test_tsv_zero_rank_keeps_base():
    base = torch.ones(2, 2)
    result = merge_tensors([base, base + 1, base + 2, base + 3], "tsv", base_index=0)
    torch.testing.assert_close(result, base)


@pytest.mark.parametrize("kind", ["zero", "identical", "opposite", "rank_one"])
def test_degenerate_matrices_match_official(method, official, kind):
    base = torch.zeros(6, 6)
    delta = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    if kind == "zero":
        deltas = [base.clone(), base.clone()]
    elif kind == "identical":
        deltas = [delta, delta.clone()]
    elif kind == "opposite":
        deltas = [delta, -delta]
    else:
        deltas = [torch.ones_like(base), torch.ones_like(base) * 2]
    expected = official(
        [SimpleNamespace(vector={"w": t.clone()}) for t in deltas],
        SimpleNamespace(device="cpu", DATASETS=[0, 1]),
    )["w"]
    result = merge_tensors([base, *deltas], method, base_index=0)
    assert result.isfinite().all()
    torch.testing.assert_close(result, expected)


def test_contract_and_zero_scale(method):
    base = torch.ones(4, 4)
    with pytest.raises(ValueError, match="base"):
        merge_tensors([base, base + 1], method)
    with pytest.raises(ValueError):
        merge_tensors([base], method, base_index=0)
    result = merge_tensors(
        [base, base + 1], method, base_index=0, parameters={"lambda": 0}
    )
    torch.testing.assert_close(result, base)


def test_yaml_checkpoint_merge(tmp_path, method):
    from transformers import LlamaConfig, LlamaForCausalLM

    from mergekit.io import LazyTensorLoader
    from mergekit.io.tasks import LoaderCache
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions

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
    for i in range(3):
        state = {k: torch.randn_like(v) * 0.1 for k, v in model.state_dict().items()}
        states.append(state)
        model.load_state_dict(state)
        model.save_pretrained(tmp_path / str(i))
    config = MergeConfiguration.model_validate(
        yaml.safe_load(f"""
merge_method: {method}
base_model: {tmp_path / "0"}
models:
  - model: {tmp_path / "1"}
  - model: {tmp_path / "2"}
parameters:
  lambda: 0.6
""")
    )
    expected = merge_state_dicts(states, method, base=0, parameters={"lambda": 0.6})
    run_merge(
        config, str(tmp_path / "out"), MergeOptions(copy_tokenizer=False, quiet=True)
    )
    loader = LazyTensorLoader.from_disk(str(tmp_path / "out"))
    for key, tensor in expected.items():
        torch.testing.assert_close(loader.get_tensor(key), tensor)
    LoaderCache().flush_all()
