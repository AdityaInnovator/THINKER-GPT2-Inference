# Setup

## Requirements

- Linux or WSL2 with NVIDIA driver + **CUDA 12.x** toolkit
- GPU with compute capability ≥ **8.0** for `cp.async` (default `sm_86` = A40)
- `gcc`, `make`
- GPT-2 checkpoint files (see [checkpoints/README.md](../checkpoints/README.md))

## Build

```bash
git clone https://github.com/<you>/gpt2-cuda-inference.git
cd gpt2-cuda-inference

# Place weights in checkpoints/ (see checkpoints/README.md)

make all
```

### GPU architecture

```bash
make CUDA_ARCH=sm_89 all    # e.g. RTX 4090
```

## Environment variables

| Variable | Default | Used by |
|----------|---------|---------|
| `GPT2_CHECKPOINT` | `checkpoints/gpt2_124M.bin` | All inference binaries |
| `GPT2_TOKENIZER` | `checkpoints/gpt2_tokenizer.bin` | `next_token_generation` |
| `GPT2_INFERENCE_VERIF` | `checkpoints/gpt2_inference_verif.bin` | `test_gpt2` |
| `GPT2_KERNEL_VERIF` | `checkpoints/gpt2_kernel_verif.bin` | `test_gpt2_kernels` |
| `GPT2_LOCAL_ATTN_DIR` | `checkpoints/local_attn_examples` | `local_attn_verify` |

## Run tests

```bash
./test_gpt2_kernels      # per-kernel vs gold
./test_gpt2              # full forward vs gold
./verify_kv_cache --batch 1 --prompt 32 --gen 16
```

## Generate text

```bash
./next_token_generation "The meaning of life is"
# Appends to samples/generation_outputs.txt
```

## Kernel sets

```bash
make baseline     # M1 kernels
make optimized    # default stack (recommended)
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `libwb` / missing checkpoint | Copy `.bin` files into `checkpoints/` |
| Wrong GPU arch | Set `CUDA_ARCH` |
| `cp.async` errors on old GPU | Use `make baseline` or `select-kernels.sh matmul-shared-register-tiling` |
| Duplicate symbol for KV globals | Only link one of `kv_cache_globals.cu` / local defs |
