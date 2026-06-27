# Model checkpoints (not in git)

Place GPT-2 124M assets here (files are **not** committed — too large for GitHub):

| File | Purpose |
|------|---------|
| `gpt2_124M.bin` | Model weights (`GPT2_CHECKPOINT`) |
| `gpt2_tokenizer.bin` | BPE tokenizer (`GPT2_TOKENIZER`) |
| `gpt2_inference_verif.bin` | End-to-end golden activations for `test_gpt2` |
| `gpt2_kernel_verif.bin` | Per-kernel gold outputs for `test_gpt2_kernels` |
| `local_attn_examples/` | `example_N.tcin` / `.tcout` for `local_attn_verify` |

## Obtain files

Copy from your course cluster path (e.g. Delta `Project_GPT/`) or export from the
[llm.c](https://github.com/karpathy/llm.c) / course release instructions.

## Environment overrides

```bash
export GPT2_CHECKPOINT=/path/to/gpt2_124M.bin
export GPT2_TOKENIZER=/path/to/gpt2_tokenizer.bin
export GPT2_INFERENCE_VERIF=/path/to/gpt2_inference_verif.bin
export GPT2_KERNEL_VERIF=/path/to/gpt2_kernel_verif.bin
export GPT2_LOCAL_ATTN_DIR=/path/to/local_attn_examples
```
