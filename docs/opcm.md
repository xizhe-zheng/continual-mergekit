# OPCM

`mergekit-yaml` supports sequential OPCM based on
`fusion_bench/method/opcm/opcm.py` and `utils.py` in the reference repository.
Use [examples/opcm.yml](../examples/opcm.yml):

```sh
mergekit-yaml examples/opcm.yml ./merged-opcm --cuda
```

Replace model paths as needed. The YAML runner streams safetensors checkpoints;
`--device cpu` also works. It constructs only meta-device model skeletons to
identify actual leaf linear modules and tied parameters. It never loads a full
model's weights into an `nn.Module` or copies a full model in RAM.

## Reference semantics and numerical differences

The first task initializes the merged model unchanged. Subsequent tasks use the
previous merged task vector's SVD and the original strict cumulative
**singular-value sum** threshold and zero-based split index. Only leaf
`nn.Linear.weight` parameters receive projection. Other leaf parameters use
additive updates; parameters on non-leaf modules skip that update, exactly as
in the source. Tied parameter updates retain module traversal order.

The reference calls `projected.diag().fill_(0)`, but `diag()` returns a copy, so
that operation does **not** clear the projected matrix diagonal. Only its
subsequent top-left block assignment changes the matrix. This implementation
preserves the actual source behavior.

Every round computes the global norm over the sorted whole state dictionary
(including buffers and duplicate tied entries), computes the mean task norm,
then scales all named parameters once. It retains `previous_lambda_t` and does
not defer normalization. Buffers remain those of the first task.

To keep memory bounded, the YAML path differs in floating-point execution:

- Norms use bounded chunks and FP64 sum-of-squares accumulation, then cast to
  working precision, instead of allocating a flattened model-sized vector.
- Projection uses a reduced SVD and subtracts the removed top-left component
  from the incoming update: `D - U_k @ (U_k.T @ D @ V_k) @ V_k.T`.
  This is algebraically equivalent to the reference full-basis mask and retains
  rectangular null-space components. It is not a truncated reconstruction.

These preserve the real-arithmetic algorithm but cannot promise bitwise equality.
Threshold boundaries and degenerate SVD subspaces can amplify rounding changes;
model quality still needs evaluation. The full-matrix Python reference path
`mergekit.opcm.merge_opcm(base, ordered_tasks, ...)` remains available for small
in-memory comparisons and is not used by `mergekit-yaml`.

## Configuration and resources

Working precision is `float32` (default) or `float64`; `out_dtype` casts only the
final output. Defaults: `alpha: 0.5`, `shuffle_order: true`, `seed: null`,
`save_on_every_step: true`. The example fixes seed 42 and disables step saves.
Parameter `seed` overrides the CLI seed; `shuffle_order: false` follows YAML order.
Undefined zero/nonfinite normalization raises an error instead of saving NaNs.

The runner keeps one working checkpoint in `.opcm-work-*` under the output
folder. Each tensor is replaced atomically after projection and normalization.
The temporary folder is removed after success or a Python exception. A hard kill
can leave this folder behind; remove stale folders only when no merge is running.

RAM scales with a few copies of the largest tensor, norm chunks and a bounded
output-shard buffer, rather than three full models. The output buffer follows
`--out-shard-size` without an additional OPCM cap; larger shards require more
memory and temporary disk headroom. GPU memory still depends on
the largest matrix and reduced-SVD workspace. Exact SVD remains computationally
expensive for large language-model matrices; this fix is not a speed guarantee.

Scratch disk requires roughly one model in working precision plus tensor-sized
replacement headroom. Final export progressively releases scratch tensors to
avoid retaining two full FP32 copies. `save_on_every_step: true` additionally
retains one model per task under `checkpoints/merged_model_N`; disk space is
checked before computation. `false` saves only final weights plus recipe,
`model_names.json` and `opcm_steps.json`. Progress bars identify norm, projection
and normalization stages.

## Scope

Inputs must be safetensors checkpoints with matching architecture, vocabulary,
module structure, state shapes and weight tying. A built-in Transformers class
must be named in `config.json`. Supported examples include CLIP vision, Llama and
Qwen2. Slices/modules, parameter schedules, tokenizer remapping, unmerged LoRAs,
architecture overrides and multi-GPU execution are unsupported. Tensor-only APIs
and `mergekit-pytorch` cannot express this global normalization.

FusionBench evaluation and TensorBoard logging are not included; evaluate saved
checkpoints separately. Reference experiments concern CLIP, not LLM quality.

## Validation

Tests execute actual reference methods from the optional sibling `../opcm`
checkout, compare full-matrix and memory-bounded projection on square and
rectangular matrices, and compare end-to-end CLIP and tied-weight Llama outputs.
They cover CPU/CUDA, normalization aliases/buffers, saved intermediates, output
casting, no full checkpoint loading, and scratch cleanup on success/failure.
