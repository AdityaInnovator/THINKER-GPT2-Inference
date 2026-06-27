# Optimization catalog

Every technique lives under [`variants/`](../variants/) with a **descriptive folder name**.
See [variants/README.md](../variants/README.md) for the full index.

The default **`kernels/`** build composes the best general-purpose combination.

## Default optimized stack

| Active file | Variant folder | Technique |
|-------------|----------------|-----------|
| `matmul.cuh` | `matmul-cp-async-pipeline` | Tiled GEMM + **`cp.async`** double buffering |
| `attention.cuh` | `attention-kv-cache` | **KV-cache** prefill + decode |
| `layernorm.cuh`, `softmax.cuh` | `layernorm-softmax-reduction` | Block reductions |
| `encoder.cuh`, `gelu.cuh`, `residual.cuh` | _(M1 baseline)_ | Embeddings, GELU, residuals |

### Why this composition?

- **Matmul** dominates FLOPs; async pipelines hide memory latency.
- **Attention** is \(O(T^2)\) without cache; KV-cache makes decode \(O(T)\) per step.
- **LayerNorm + softmax** need correct reductions before exotic attention variants.

## All variants (by category)

### Matrix multiply

| Folder | Technique |
|--------|-----------|
| `matmul-shared-register-tiling` | Shared-memory + register tiling |
| `matmul-tensor-core-wmma` | WMMA / Tensor Core (TF32) |
| `matmul-cublas` | NVIDIA cuBLAS `sgemm` |
| `matmul-split-k` | Split-K parallel reduction |
| `matmul-block-size-sweep` | Autotuned tile sizes (`benchmarks/sweep_matmul.py`) |
| `matmul-cp-async-pipeline` | `cp.async` pipelined loads (**in default build**) |
| `matmul-block-rasterization` | Thread-block rasterization |
| `matmul-register-tiling-snapshot` | Reference register-tiling implementation |

### Attention

| Folder | Technique |
|--------|-----------|
| `attention-flash` | Flash Attention–style blocking |
| `attention-local-window` | Local / windowed attention |
| `attention-kv-cache` | KV-cache + decode kernel (**in default build**) |
| `attention-constant-memory` | Constant-memory metadata |
| `attention-restrict-ptr` | `__restrict__` qualifiers |
| `attention-memory-swizzle` | Shared-memory swizzling |
| `attention-shared-memory-padding` | Padded tiles (bank conflict mitigation) |

### Normalization & softmax

| Folder | Technique |
|--------|-----------|
| `layernorm-softmax-reduction` | Parallel reductions (**in default build**) |

## Switching kernels

```bash
# Restore composed optimized set
./scripts/use-kernel-set.sh optimized
make clean all

# Milestone-1 baseline
./scripts/use-kernel-set.sh baseline
make clean all

# Try one technique
./scripts/select-kernels.sh attention-flash apply
make clean all
./scripts/select-kernels.sh attention-flash revert
```

Legacy course IDs (`req_4`, `op_17`, …) still work as aliases in `select-kernels.sh`.

## Benchmark table (fill from your runs)

| Configuration | Tokens/s (decode) | Prefill ms | Notes |
|---------------|-------------------|------------|-------|
| baseline (M1) | _TBD_ | _TBD_ | `make baseline` |
| optimized (default) | _TBD_ | _TBD_ | |
| KV-cache enabled | _TBD_ | _TBD_ | `verify_kv_cache` |
| Flash attention only | _TBD_ | _TBD_ | `select-kernels.sh attention-flash` |
