# Benchmarks

## `sweep_matmul.py`

Autotunes `BLOCK_SIZE` and `U_TILE` for the
[`matmul-block-size-sweep`](../variants/matmul-block-size-sweep/) kernel by copying
its `matmul.cuh` into `kernels/` and rebuilding `test_gpt2`.

```bash
# From repo root, on a machine with GPU + checkpoints
python3 benchmarks/sweep_matmul.py
```

Adjust `REPO_ROOT` detection if running from another cwd.
