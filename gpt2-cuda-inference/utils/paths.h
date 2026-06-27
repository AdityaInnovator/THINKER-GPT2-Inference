#ifndef PORTFOLIO_PATHS_H
#define PORTFOLIO_PATHS_H

#ifdef __cplusplus
extern "C" {
#endif

const char* portfolio_gpt2_checkpoint(void);
const char* portfolio_gpt2_tokenizer(void);
const char* portfolio_gpt2_inference_verif(void);
const char* portfolio_gpt2_kernel_verif(void);
const char* portfolio_local_attn_dir(void);

#ifdef __cplusplus
}
#endif

#endif
