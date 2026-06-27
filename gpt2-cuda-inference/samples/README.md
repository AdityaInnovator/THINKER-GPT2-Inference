# Generation samples

`generation_outputs.txt` contains example completions from `next_token_generation`
(run on course GPU hardware).

Regenerate:

```bash
./next_token_generation "Your prompt here"
```

New runs append to `generation_outputs.txt` (see `log_generation_output` in
`next_token_generation.cu`).
