#include "paths.h"
#include <stdlib.h>

static const char* env_or_default(const char* key, const char* fallback) {
    const char* v = getenv(key);
    return (v && v[0]) ? v : fallback;
}

const char* portfolio_gpt2_checkpoint(void) {
    return env_or_default("GPT2_CHECKPOINT", "checkpoints/gpt2_124M.bin");
}

const char* portfolio_gpt2_tokenizer(void) {
    return env_or_default("GPT2_TOKENIZER", "checkpoints/gpt2_tokenizer.bin");
}

const char* portfolio_gpt2_inference_verif(void) {
    return env_or_default("GPT2_INFERENCE_VERIF", "checkpoints/gpt2_inference_verif.bin");
}

const char* portfolio_gpt2_kernel_verif(void) {
    return env_or_default("GPT2_KERNEL_VERIF", "checkpoints/gpt2_kernel_verif.bin");
}

const char* portfolio_local_attn_dir(void) {
    return env_or_default("GPT2_LOCAL_ATTN_DIR", "checkpoints/local_attn_examples");
}
