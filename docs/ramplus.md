# Global RAM+ (mrl arm-r-v2)

Use `merge_method: ramplus` with `models` and `base_model`. See
[the four-model example](../examples/ramplus.yml).

```bash
mergekit-yaml examples/ramplus.yml /path/to/output
```

The YAML/run_merge entry point scans all checkpoint tensors in float32. For each
task it counts overlapping updates O and unique updates U, using
`abs(task - base) > epsilon`. It computes one scale per task:
`1 + (max(1, rescale_factor) - 1) * min(1, O/U)`. The regular merge pass applies
that scale only to unique updates; overlapping updates are averaged. Defaults
are `epsilon: 0.00001` and `rescale_factor: 1.2` (the mrl README example).
`rescale_factor: 1.0` reduces to RAM. Unlike `ramplus_tl`, the scale is shared
across every tensor of a task model.

An unchanged task gets scale 1. An entirely overlapping task gets the maximum
scale, which has no effect because it has no unique positions. This fixes the
reference implementation's division by zero when U=0.

The prepass processes corresponding tensors on CPU without loading all full
models; checkpoint loaders may retain a shard. It adds a second checkpoint read.
Use `dtype: float32` (also the default for this method) for mrl-compatible
subtraction and thresholding, and `out_dtype: bfloat16` for compact saved weights.
The original recipe is retained in the output mergekit_config.yml.

Supported: whole-model merges with matching checkpoint tensor names/shapes and
already aligned vocabularies. Slices/modules, tokenizer remapping, per-model
overrides, and parameter filters/gradients are rejected. Omit `tokenizer` and
`tokenizer_source`; the standard output process copies the base tokenizer.
Different checkpoint key sets are rejected rather than taking mrl's intersection.
Statistics count serialized tensors: tied aliases omitted from a checkpoint are
not counted twice, unlike a fully materialized state_dict. Exact comparison with
mrl requires the same set of counted tensors (e.g. aligned, untied checkpoints).

Direct `merge_tensors`, `merge_state_dicts`, and raw-PyTorch callers do not run
the checkpoint prepass: supply the required per-task `unique_scale` values
calculated across the entire model. Only the YAML/run_merge path computes these
automatically. Tensor-local estimates are not equivalent.

Validation: `python -m pytest tests/test_ramplus.py -q`. Includes full miniature
checkpoint merges, boundaries, and comparison with the original mrl function
when a sibling mrl checkout is available.
