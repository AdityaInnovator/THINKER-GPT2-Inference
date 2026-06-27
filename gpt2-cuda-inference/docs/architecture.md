# GPT-2 inference architecture

## High-level flow

GPT-2 (124M) is a **decoder-only** transformer. This repository implements the
**forward pass (inference)** in CUDA.

```mermaid
flowchart TB
  subgraph input [Input]
    TOK[Token IDs B x T]
  end

  subgraph embed [Embedding]
    ENC[encoder_forward: wte + wpe]
  end

  subgraph block [Transformer block x L]
    LN1[layernorm_forward]
    QKV[matmul_forward QKV projection]
    ATT[attention_forward]
    PROJ[matmul_forward output proj]
    RES1[residual_forward]
    LN2[layernorm_forward]
    FC[matmul_forward FFN up]
    GELU[gelu_forward]
    FC2[matmul_forward FFN down]
    RES2[residual_forward]
  end

  subgraph head [LM head]
    LNF[layernorm_forward]
    LM[matmul_forward to vocab]
  end

  TOK --> ENC --> LN1 --> QKV --> ATT --> PROJ --> RES1
  RES1 --> LN2 --> FC --> GELU --> FC2 --> RES2
  RES2 --> LNF --> LM
```

## Active kernel set (`kernels/`)

| Kernel | Variant source | Role |
|--------|----------------|------|
| `encoder.cuh` | M1 baseline | Token + position embedding |
| `layernorm.cuh` | `layernorm-softmax-reduction` | Layer normalization |
| `matmul.cuh` | `matmul-cp-async-pipeline` | FFN + projection GEMM |
| `attention.cuh` | `attention-kv-cache` | Multi-head attention + cache |
| `softmax.cuh` | `layernorm-softmax-reduction` | Causal softmax |
| `gelu.cuh`, `residual.cuh` | M1 baseline | Activation + skip connection |

## KV-cache (`attention-kv-cache`)

When `g_enable_kv_cache = true` (see `verify_kv_cache`):

1. **Prefill** — full prompt; store K/V per layer.
2. **Decode** — one token; attend over cached K/V via `decode_attention_kernel`.

Globals: `src/kv_cache_globals.cu`.

## Code map

| Path | Description |
|------|-------------|
| `gpt2.cuh` | Model struct, `gpt2_forward` |
| `kernels/` | Active GPU code (compiled) |
| `kernels_baseline/` | Milestone-1 reference |
| `kernels_optimized/` | Frozen optimized snapshot |
| [`variants/`](../variants/) | Named optimization experiments |
| `cpu_kernels/` | CPU reference for tests |
