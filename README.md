# GPT-2 CUDA Inference

CUDA implementation of the **GPT-2 (124M) forward pass** with a full optimization
ladder: tiled GEMM, Tensor Cores, cuBLAS, Flash Attention, **KV-cache decode**, and
`cp.async` memory pipelines. Built as the ECE 408 final project (UIUC); packaged here
as an open portfolio repo.

> **Based on [llm.c](https://github.com/karpathy/llm.c)** (Andrej Karpathy, MIT). See
> [NOTICE](NOTICE) and [LICENSE.llm.c](LICENSE.llm.c).

---

## What this demonstrates

| Skill | Evidence in repo |
|-------|------------------|
| **GPU kernel design** | Custom matmul, attention, layernorm, softmax, GELU |
| **ML systems** | Full transformer inference, autoregressive generation |
| **Performance engineering** | 15+ optimization variants, sweep script, profiling docs |
| **Production patterns** | KV-cache prefill/decode, cuBLAS integration, env-based paths |

---

## Quick start

```bash
# 1. Clone and add checkpoints (not in git — see checkpoints/README.md)
export GPT2_CHECKPOINT=checkpoints/gpt2_124M.bin
export GPT2_TOKENIZER=checkpoints/gpt2_tokenizer.bin

# 2. Build (optimized kernels by default)
make all

# 3. Correctness
./test_gpt2_kernels
./test_gpt2

# 4. Talk to your GPT-2
./next_token_generation "The meaning of life is"
```

Example output is saved under [samples/generation_outputs.txt](samples/generation_outputs.txt).

---

## Repository layout

```
gpt2-cuda-inference/
├── kernels/                 # Active GPU kernels (default = optimized)
├── kernels_baseline/        # Milestone 1 implementations
├── kernels_optimized/       # Frozen optimized snapshot
├── variants/                # Named optimizations (e.g. attention-flash, matmul-cublas)
├── cpu_kernels/             # CPU reference kernels for tests
├── gpt2.cuh                 # Model + forward pass orchestration
├── next_token_generation.cu # Text completion CLI
├── test_gpt2*.cu            # Correctness harnesses
├── verify_kv_cache.cu       # KV-cache benchmark
├── benchmarks/sweep_matmul.py
├── docs/                    # Architecture, optimizations, profiling
└── scripts/                 # Kernel switching utilities
```

---

## Default optimized stack

The checked-in `kernels/` directory composes the strongest general-purpose set from the
project milestones:

| File | Source variant | Highlights |
|------|----------------|------------|
| `matmul.cuh` | [`matmul-cp-async-pipeline`](variants/matmul-cp-async-pipeline/) | Tiled GEMM + **`cp.async`** |
| `attention.cuh` | [`attention-kv-cache`](variants/attention-kv-cache/) | **KV-cache** prefill + decode |
| `layernorm.cuh`, `softmax.cuh` | [`layernorm-softmax-reduction`](variants/layernorm-softmax-reduction/) | Block reductions |
| `encoder.cuh`, `gelu.cuh`, `residual.cuh` | M1 | Embeddings, activation, residuals |

Switch sets:

```bash
make optimized    # restore composed stack (recommended)
make baseline     # Milestone-1 only
./scripts/select-kernels.sh attention-flash apply   # try Flash Attention alone
```

Details: [docs/optimizations.md](docs/optimizations.md).

---

## Architecture

Decoder-only GPT-2: embedding → 12×(LayerNorm → Attention → Residual → FFN) → LM head.

```mermaid
flowchart LR
  A[Tokens] --> B[Encoder]
  B --> C[Transformer x12]
  C --> D[LayerNorm]
  D --> E[Logits]
```

Full diagram and tensor shapes: [docs/architecture.md](docs/architecture.md).

---

## Binaries

| Target | Purpose |
|--------|---------|
| `test_gpt2_kernels` | Per-kernel numerical check vs gold |
| `test_gpt2` | Full forward pass vs golden activations |
| `verify_kv_cache` | Benchmark prefill vs KV-cached decode |
| `next_token_generation` | Sample text from a prompt |
| `output_verification_rand` | Randomized kernel tests vs CPU |
| `local_attn_verify` | Windowed attention correctness |

---

## Profiling

Use **Nsight Compute** on matmul/attention kernels and **Nsight Systems** for end-to-end
timeline. Step-by-step: [docs/profiling.md](docs/profiling.md).

Suggested portfolio metrics (fill from your runs):

- Prefill latency vs sequence length
- Decode tokens/sec with KV-cache on/off
- NCU: DRAM throughput before/after `matmul-cp-async-pipeline`

---

## Configuration

| Variable | Default |
|----------|---------|
| `CUDA_ARCH` | `sm_86` |
| `GPT2_CHECKPOINT` | `checkpoints/gpt2_124M.bin` |
| `GPT2_TOKENIZER` | `checkpoints/gpt2_tokenizer.bin` |

See [docs/setup.md](docs/setup.md).

---

## Team & attribution

Developed by team **ClosedAI** for ECE 408. Edit [CONTRIBUTORS.md](CONTRIBUTORS.md)
with accurate credit before publishing.

**Before making the repo public:** get teammate approval; do not upload unreleased
course autograder assets; keep checkpoint files out of git.

---

## Related portfolio repo

[CUDA Parallel Portfolio](https://github.com/<you>/cuda-parallel-portfolio) — labs
(GEMM tiling, scan, SpMV) and a standalone GEMM profiling benchmark.

---

## License

MIT — [LICENSE](LICENSE). Upstream llm.c: [LICENSE.llm.c](LICENSE.llm.c).
