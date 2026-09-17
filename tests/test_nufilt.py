"""Numerical checks against the optional authors' sibling checkout."""

import ast
import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from mergekit.config import MergeConfiguration, evaluate_setting
from mergekit.merge_methods import get, merge_state_dicts, merge_tensors


@pytest.fixture
def official():
    path = Path(__file__).resolve().parents[2] / "NUFILT/fusion_bench/models/filter_lora.py"
    if not path.exists():
        pytest.skip("Official NUFILT sibling checkout unavailable")
    # Load the actual classes without importing unrelated FusionBench datasets,
    # Lightning, or model packages. Only the SVD wrapper is supplied here.
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    def svd(w, full_matrices=True):
        u, s, vh = torch.linalg.svd(
            w, full_matrices=full_matrices, driver="gesvd" if w.is_cuda else None
        )
        return u, s, vh.T

    namespace = {"torch": torch, "nn": torch.nn, "Tensor": torch.Tensor, "svd": svd}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["FilterLoRA"]


@pytest.mark.parametrize("shape", [(5, 7), (7, 5), (6, 6)])
@pytest.mark.parametrize("null_space,lora_r,null_r,grad_r", [
    (True, 2, 2, 2), (False, 2, 2, 2), (True, 0, 2, 2),
    (True, 2, 0, 0), (True, 2, 30, 0),
])
def test_official_continual_result(official, shape, null_space, lora_r, null_r, grad_r):
    torch.manual_seed(100)
    inputs = [torch.randn(shape) for _ in range(4)]
    originals = [x.clone() for x in inputs]
    name = "encoder.attn.q_proj.weight"
    params = dict(null_space=null_space, lora_r=lora_r, null_r=null_r,
                  grad_r=grad_r, max_steps=5, seed=42)
    base = inputs[0]
    layer = torch.nn.Linear(shape[1], shape[0], bias=False)
    layer.weight.data.copy_(base)
    for i, task in enumerate(inputs[1:]):
        module = official(layer, task - base, null_space, layer.weight.detach() - base,
                          lora_r, null_r, grad_r, "cpu")
        # Same initialization isolates the algorithm from graph traversal RNG.
        seed = int.from_bytes(hashlib.sha256(f"42:{name}:{i}".encode()).digest()[:8], "little")
        with torch.no_grad():
            module.gate.A.normal_(std=0.02, generator=torch.Generator().manual_seed(seed))
        if i > 0 and lora_r > 0:
            optimizer = torch.optim.Adam(module.gate.parameters(), lr=1e-3)
            for _ in range(5):
                optimizer.zero_grad()
                module.solve_lora().backward()
                optimizer.step()
        layer.weight.data.copy_(module.merge_to_base())
    with torch.no_grad():
        result = merge_tensors(inputs, "nufilt", base_index=0, name=name, parameters=params)
    torch.testing.assert_close(result, layer.weight, atol=3e-5, rtol=3e-5)
    for x, original in zip(inputs, originals):
        torch.testing.assert_close(x, original)


def test_selection_and_rng():
    torch.manual_seed(9)
    base = {"encoder.attn.q.weight": torch.randn(6, 6),
            "encoder.attn.q.bias": torch.randn(6),
            "encoder.fc2.weight": torch.randn(6, 6),
            "embedding.weight": torch.randn(6, 6)}
    models = [base, {k: v + torch.randn_like(v) for k, v in base.items()},
              {k: v + torch.randn_like(v) for k, v in base.items()}]
    state = torch.random.get_rng_state()
    kwargs = dict(method="nufilt", base=0,
                  parameters=dict(null_r=2, grad_r=2, lora_r=2, max_steps=3))
    output = merge_state_dicts(models, **kwargs)
    assert torch.equal(state, torch.random.get_rng_state())
    repeat = merge_state_dicts(models, **kwargs)
    for name in base:
        torch.testing.assert_close(output[name], repeat[name], rtol=0, atol=0)
        if name != "encoder.attn.q.weight":
            torch.testing.assert_close(output[name], base[name], rtol=0, atol=0)
    assert not torch.equal(output["encoder.attn.q.weight"], base["encoder.attn.q.weight"])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_and_first_task(dtype):
    base = torch.zeros(4, 6, dtype=dtype)
    task = torch.arange(24).reshape(4, 6).to(dtype)
    output = merge_tensors([base, task], "nufilt", base_index=0,
                           name="attn.weight", parameters={"null_r": 2})
    work = base.double() if dtype == torch.float64 else base.float()
    v = torch.linalg.svd(work, full_matrices=False).Vh.T[:, :2]
    expected = task.to(work.dtype) @ (torch.eye(6, dtype=work.dtype) - v @ v.T)
    assert output.dtype == dtype
    torch.testing.assert_close(output, expected.to(dtype))
    assert not torch.equal(output, task)


def test_example_parameters_resolve():
    path = Path(__file__).resolve().parents[1] / "examples/nufilt.yml"
    config = MergeConfiguration.model_validate(yaml.safe_load(path.read_text()))
    specs = {p.name: p for p in get("nufilt").spec.parameters}
    for name, value in config.parameters.items():
        assert evaluate_setting("attn.weight", value, validate=specs[name].validate) == value


@pytest.mark.parametrize("parameter,value", [("lr", -1), ("null_r", -1), ("max_steps", -1)])
def test_invalid_parameters(parameter, value):
    with pytest.raises(ValueError):
        merge_tensors([torch.zeros(2, 2)] * 2, "nufilt", base_index=0,
                      name="attn.weight", parameters={parameter: value})
