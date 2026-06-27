# Kernel optimization variants

Each folder is a **self-contained experiment** you can install into active `kernels/` for
build and profiling. Names describe the technique—not course milestone IDs.

## Quick index

| Folder | What it implements | Primary files |
|--------|-------------------|---------------|
| [matmul-shared-register-tiling](matmul-shared-register-tiling/) | Shared + register tiled GEMM (M2) | `matmul.cuh`, `attention.cuh` |
| [matmul-tensor-core-wmma](matmul-tensor-core-wmma/) | WMMA / Tensor Core TF32 matmul | `matmul.cuh`, `attention.cuh` |
| [matmul-cublas](matmul-cublas/) | cuBLAS `sgemm` matmul | `matmul.cuh`, `attention.cuh` |
| [layernorm-softmax-reduction](layernorm-softmax-reduction/) | Block reductions in norm + softmax | `layernorm.cuh`, `softmax.cuh` |
| [attention-flash](attention-flash/) | Flash Attention–style blocking | `attention.cuh` |
| [attention-local-window](attention-local-window/) | Local / windowed attention | `local_attention.cuh` |
| [attention-kv-cache](attention-kv-cache/) | KV-cache prefill + decode | `attention.cuh` |
| [matmul-block-size-sweep](matmul-block-size-sweep/) | Tunable tile sizes + sweep script | `matmul.cuh`, `sweep_matmul.py` |
| [attention-constant-memory](attention-constant-memory/) | `__constant__` shape metadata | `attention.cuh` |
| [attention-restrict-ptr](attention-restrict-ptr/) | `__restrict__` pointer hints | `attention.cuh` |
| [matmul-split-k](matmul-split-k/) | Split-K parallel reduction along K | `matmul.cuh` |
| [attention-memory-swizzle](attention-memory-swizzle/) | Shared-memory access swizzling | `attention.cuh` |
| [attention-shared-memory-padding](attention-shared-memory-padding/) | Padded shared tiles (bank conflicts) | `attention.cuh` |
| [matmul-cp-async-pipeline](matmul-cp-async-pipeline/) | `cp.async` double-buffered GEMM | `matmul.cuh`, `cp_async_utils.cuh` |
| [matmul-block-rasterization](matmul-block-rasterization/) | Block scheduling / raster order | `matmul.cuh`, `attention.cuh` |
| [matmul-register-tiling-snapshot](matmul-register-tiling-snapshot/) | Reference register-tiling matmul | `matmul.cuh`, `attention.cuh` |

### Course ID crosswalk (for your records only)

| Old name | New folder |
|----------|------------|
| `kernels_req_0` | `matmul-shared-register-tiling` |
| `kernels_req_1` | `matmul-tensor-core-wmma` |
| `kernels_req_2` | `matmul-cublas` |
| `kernels_req_3` | `layernorm-softmax-reduction` |
| `kernels_req_4` | `attention-flash` |
| `kernels_req_5` | `attention-local-window` |
| `kernels_req_6` | `attention-kv-cache` |
| `kernels_op_7` | `matmul-block-size-sweep` |
| `kernels_op_8` | `attention-constant-memory` |
| `kernels_op_9` | `attention-restrict-ptr` |
| `kernels_op_10` | `matmul-split-k` |
| `kernels_op_11` | `attention-memory-swizzle` |
| `kernels_op_12` | `attention-shared-memory-padding` |
| `kernels_op_17` | `matmul-cp-async-pipeline` |
| `kernels_op_18` | `matmul-block-rasterization` |

## Try a variant

```bash
./scripts/select-kernels.sh attention-flash apply
make clean all
./scripts/select-kernels.sh attention-flash revert
```

Default **optimized** build uses `matmul-cp-async-pipeline` + `attention-kv-cache` +
`layernorm-softmax-reduction` — see [docs/optimizations.md](../docs/optimizations.md).
