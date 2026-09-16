"""Streaming whole-checkpoint statistics for mrl's arm-r-v2 RAM+."""

import logging
import math

import torch
import tqdm

from mergekit.config import MergeConfiguration
from mergekit.io.tasks import LoaderCache

LOG = logging.getLogger(__name__)


def unique_scales(changed, overlapping, rescale_factor):
    r = max(1.0, rescale_factor)
    return [
        1.0 + (r - 1.0) * (min(1.0, o / (c - o)) if c > o else float(o > 0))
        for c, o in zip(changed, overlapping)
    ]


def prepare_ramplus(
    config: MergeConfiguration, cache: LoaderCache, *, quiet: bool = False
) -> MergeConfiguration:
    """Resolve global scales on a copy of an aligned whole-model configuration.

    Statistics and subsequent merging use float32, as in mrl. One corresponding
    tensor is processed at a time; loaders may retain their current shard.
    """
    if not config.models or config.base_model is None:
        raise ValueError(
            "ramplus requires models and base_model; slices/modules are unsupported"
        )
    if config.tokenizer is not None or config.tokenizer_source is not None:
        raise ValueError(
            "ramplus requires aligned vocabularies; omit tokenizer remapping settings"
        )
    if config.dtype not in (None, "float32"):
        raise ValueError(
            "ramplus requires dtype: float32; use out_dtype for saved precision"
        )
    config = config.model_copy(deep=True)
    params = dict(config.parameters or {})
    if set(params) - {"epsilon", "rescale_factor"}:
        raise ValueError("ramplus accepts only global epsilon and rescale_factor")
    epsilon = params.get("epsilon", 1e-5)
    r = params.get("rescale_factor", 1.2)
    for name, value in (("epsilon", epsilon), ("rescale_factor", r)):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"ramplus {name} must be a finite scalar")
    if epsilon < 0:
        raise ValueError("ramplus epsilon must be nonnegative")
    if any(model.parameters for model in config.models):
        raise ValueError(
            "ramplus computes scales automatically; per-model parameters are unsupported"
        )
    tasks = [model for model in config.models if model.model != config.base_model]
    if not tasks or len({model.model for model in tasks}) != len(tasks):
        raise ValueError("ramplus requires distinct non-base task models")
    base_loader = cache.get(config.base_model)
    loaders = [cache.get(model.model) for model in tasks]
    keys = set(base_loader.index.tensor_paths)
    if any(set(loader.index.tensor_paths) != keys for loader in loaders):
        raise ValueError("ramplus requires matching checkpoint tensor names")
    changed = [0] * len(tasks)
    overlapping = [0] * len(tasks)
    try:
        for name in tqdm.tqdm(
            sorted(keys), desc="RAM+ global statistics", disable=quiet
        ):
            base = base_loader.get_tensor(name).to(torch.float32)
            masks = []
            for loader in loaders:
                tensor = loader.get_tensor(name).to(torch.float32)
                if tensor.shape != base.shape:
                    raise ValueError(f"ramplus shape mismatch for {name}")
                masks.append((tensor - base).abs() > epsilon)
            active = torch.stack(masks)
            overlap = active.sum(dim=0) >= 2
            for i, mask in enumerate(masks):
                changed[i] += int(mask.sum())
                overlapping[i] += int((mask & overlap).sum())
    finally:
        for loader in [base_loader, *loaders]:
            loader.flush()
    scales = unique_scales(changed, overlapping, r)
    for model, scale in zip(tasks, scales):
        model.parameters = {"unique_scale": scale}
        LOG.info("RAM+ %s: unique_scale=%s", model.model, scale)
    config.parameters = {"epsilon": epsilon}
    config.dtype = "float32"
    return config
