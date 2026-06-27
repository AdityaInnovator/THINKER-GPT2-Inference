# Profiling guide

## Tools

| Tool | Use case |
|------|----------|
| **Nsight Systems** (`nsys`) | Timeline: kernel order, CPU launch gaps, memcpy |
| **Nsight Compute** (`ncu`) | Kernel deep dive: memory throughput, occupancy, warp stalls |
| **compute-sanitizer** | Race / OOB checks before trusting timings |

## Recommended workflow

1. `make optimized && ./test_gpt2` — confirm correctness.
2. Profile **`test_gpt2`** or **`verify_kv_cache`** on an **exclusive GPU** node.
3. Compare **baseline** vs **optimized** with identical `B`, `T`.

### Slurm (Delta-style)

```bash
#SBATCH --constraint=perf,nvperf   # exclusive node for stable NCU
srun ncu --set full -o reports/matmul_opt ./test_gpt2
```

### Local

```bash
ncu --set full -k regex:matmul_forward_kernel -o ncu_matmul ./test_gpt2
nsys profile -o nsys_gpt2 ./verify_kv_cache --batch 1 --prompt 64 --gen 32
```

## What to look for

### Matmul (`kernels/matmul.cuh`, `matmul-cp-async-pipeline`)

- **DRAM throughput** vs theoretical peak (A40 ~ 696 GB/s)
- **`cp.async`** — memory pipeline should hide latency across K-tiles
- **Occupancy** vs register/shared memory usage (`BM`, `BN`, `TN`)

### Attention

- Prefill: `qk_matmul` + softmax often dominate at large `T`
- Decode with KV-cache: compare per-token time vs full re-computation

### End-to-end

- `verify_kv_cache` prints prefill vs decode timing — ideal for a portfolio chart

## Matmul autotune

```bash
cd benchmarks
python sweep_matmul.py   # requires built test_gpt2 + Delta paths adjusted
```

Copy the winning `BLOCK_SIZE` / `U_TILE` into a custom `variants/` matmul if needed.

## Publishing results

Full write-up templates and checklists:

**[gpu-profiling-notes](../../gpu-profiling-notes)** (sibling repo) — `notes/03-gpt2-inference-timeline.md`, `04-kv-cache-decode.md`, `CAPTURE_CHECKLIST.md`.

Export **PNG screenshots** from Nsight UI into that repo’s `figures/03-gpt2/`. Do **not** commit multi-MB `.ncu-rep` files here.
