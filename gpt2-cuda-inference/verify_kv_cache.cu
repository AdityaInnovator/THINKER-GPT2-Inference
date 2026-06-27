#define BENCH
#include "gpt2.cuh"
#include "utils/paths.h"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// KV-cache globals: src/kv_cache_globals.cu

struct Args {
    int batch = 1;
    int prompt_len = 32;
    int gen_len = 32;
    unsigned int seed = 0;
    bool random_mode = false;
};

static void usage(const char* prog) {
    fprintf(stderr,
    "Usage: %s [--batch N] [--prompt-len L] [--gen-len G] [--seed S] [--random]\n"
    "Defaults: --batch 1 --prompt-len 32 --gen-len 32 --seed 0\n",
        prog);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](int i){ if (i+1 >= argc) { usage(argv[0]); exit(EXIT_FAILURE);} };
        if (k == "--batch")      { need(i); a.batch      = atoi(argv[++i]); }
        else if (k == "--prompt-len"){ need(i); a.prompt_len = atoi(argv[++i]); }
        else if (k == "--gen-len"){ need(i); a.gen_len   = atoi(argv[++i]); }
        else if (k == "--seed")   { need(i); a.seed      = (unsigned)strtoul(argv[++i], nullptr, 10); }
        else if (k == "--random") { a.random_mode = true; }
        else { usage(argv[0]); exit(EXIT_FAILURE); }
    }
    return a;
}

static void fill_random_prompts(int* buf, int B, int stride, int prompt_len, int vocab_size) {
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < prompt_len; ++t) {
            buf[b * stride + t] = rand() % vocab_size;
        }
    }
}

static int sample_token_argmax(const float* logits, int vocab_size) {
    float max_logit = -FLT_MAX;
    int best = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            best = i;
        }
    }
    return best;
}

static void copy_last_logits(float* dest, const float* device_logits, int B, int cur_len, size_t Vp) {
    for (int b = 0; b < B; ++b) {
        size_t offset = (b * cur_len + (cur_len - 1)) * Vp;
        cudaCheck(cudaMemcpy(dest + b * Vp, device_logits + offset, Vp * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

double benchmark_gpt2_forward(GPT2 *model, int* inputs, int B, int T) {
    if (model->params_memory == NULL) {
        printf("GPT2 - Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
    }

    if(model->acts_memory == NULL || B != model->batch_size || T != model->seq_len) {
        model->batch_size = B;
        model->seq_len = T;
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        model->num_activations = num_activations;

        if(model->acts_memory != NULL) {
            cudaCheck(cudaFree(model->acts_memory));
            cudaCheck(cudaFreeHost(model->cpu_losses));
            cudaCheck(cudaFree(model->inputs));
        }
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));
    }

    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));

    cudaCheck(cudaDeviceSynchronize());
    auto compute_start = std::chrono::high_resolution_clock::now();

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    float* residual;

    // When using KV cache in decode mode (T=1), we must use the correct
    // absolute position for positional embeddings, not position 0.
    if (g_enable_kv_cache && !g_is_prefill) {
        // Decode: T=1, actual position is g_current_pos
        // Use wpe offset to point to the correct position embedding
        encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe + g_current_pos * C, B, T, C);
    } else {
        encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);
    }

    for (int l = 0; l < L; l++) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean;
        float* l_ln1_rstd = acts.ln1_rstd;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_qkvr = acts.qkvr + l * B * T * 3*C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean;
        float* l_ln2_rstd = acts.ln2_rstd;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;
        float* scratch = acts.output;

        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = acts.residual3 + (L-1) * B * T * C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    cudaCheck(cudaDeviceSynchronize());
    auto compute_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
}

static void copy_single_token_logits(float* dest, const float* device_logits, int B, size_t Vp) {
    cudaCheck(cudaMemcpy(dest, device_logits, B * Vp * sizeof(float), cudaMemcpyDeviceToHost));
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (args.seed == 0) args.seed = time(nullptr);
    srand(args.seed);

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cublasCheck(cublasCreate(&cublas_handle));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, portfolio_gpt2_checkpoint());

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, portfolio_gpt2_tokenizer());

    int B = args.batch;
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;
    int maxT = model.config.max_seq_len;
    int prompt_len = args.prompt_len;
    int gen_len = args.gen_len;

    int longest_seq = prompt_len + gen_len;
    if (longest_seq > maxT) {
        fprintf(stderr, "Exceeds max sequence length %d\n", maxT);
        return 1;
    }

    int* sequence_buffer = (int*)mallocCheck(B * longest_seq * sizeof(int));
    if (args.random_mode) {
        printf("[Prompt] Generating Random Prompts\n");
        fill_random_prompts(sequence_buffer, B, longest_seq, prompt_len, V);
    } else {
        printf("[Prompt] Loading Deterministic Prompts\n");
        FILE *state_file = fopenCheck(portfolio_gpt2_inference_verif(), "rb");
        int state_header[256];
        freadCheck(state_header, sizeof(int), 256, state_file);
        int file_B = state_header[2];
        int file_T = state_header[3];
        B = file_B; 
        prompt_len = file_T;
        longest_seq = prompt_len + gen_len;
        
        free(sequence_buffer);
        sequence_buffer = (int*)mallocCheck(B * longest_seq * sizeof(int));
        int* input_tokens = (int*)mallocCheck(B * prompt_len * sizeof(int));
        freadCheck(input_tokens, sizeof(int), B * prompt_len, state_file);
        
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < prompt_len; ++t) {
                sequence_buffer[b * longest_seq + t] = input_tokens[b * prompt_len + t];
            }
        }
        free(input_tokens);
        fcloseCheck(state_file);
    }

    printf("[Config] B=%d, Prompt=%d, Gen=%d\n", B, prompt_len, gen_len);

    int* forward_inputs = (int*)mallocCheck(B * longest_seq * sizeof(int));
    float* baseline_logits = (float*)mallocCheck(gen_len * B * Vp * sizeof(float));
    std::vector<double> baseline_latencies;
    
    // ================= BASELINE RUN =================
    printf("\n=== Running Baseline (No KV Cache) ===\n");
    g_enable_kv_cache = false;
    int current_len = prompt_len;
    
    // Copy prefill prompt
    for (int b = 0; b < B; ++b) {
        memcpy(forward_inputs + b * current_len, sequence_buffer + b * longest_seq, current_len * sizeof(int));
    }

    // Prefill Baseline
    double prefill_time = benchmark_gpt2_forward(&model, forward_inputs, B, current_len);
    printf("Prefill latency: %.3f ms\n", prefill_time);

    for (int t = 0; t < gen_len; ++t) {
        float* cur_logits = baseline_logits + t * B * Vp;
        copy_last_logits(cur_logits, model.acts.output, B, current_len, Vp);
        
        for (int b = 0; b < B; ++b) {
            sequence_buffer[b * longest_seq + current_len] = sample_token_argmax(cur_logits + b * Vp, V);
        }
        current_len++;

        // Baseline step
        for (int b = 0; b < B; ++b) {
            memcpy(forward_inputs + b * current_len, sequence_buffer + b * longest_seq, current_len * sizeof(int));
        }

        double step_time = benchmark_gpt2_forward(&model, forward_inputs, B, current_len);
        baseline_latencies.push_back(step_time);
    }


    // ================= KV CACHE RUN =================
    printf("\n=== Running KV Cache ===\n");
    g_enable_kv_cache = true;
    if (g_kv_cache) { cudaFree(g_kv_cache); g_kv_cache = nullptr; }
    g_is_prefill = true;
    g_layer_idx = 0;
    current_len = prompt_len;

    float* kv_logits = (float*)mallocCheck(B * Vp * sizeof(float));
    std::vector<double> kv_latencies;
    int* kv_gen_tokens = (int*)mallocCheck(B * gen_len * sizeof(int)); // track KV-sampled tokens
    
    for (int b = 0; b < B; ++b) {
        memcpy(forward_inputs + b * current_len, sequence_buffer + b * longest_seq, current_len * sizeof(int));
    }

    // Prefill KV
    double kv_prefill_time = benchmark_gpt2_forward(&model, forward_inputs, B, current_len);
    printf("KV Prefill latency: %.3f ms\n", kv_prefill_time);

    int decode_input[1024]; // max batch size

    bool ok = true;
    for (int t = 0; t < gen_len; ++t) {
        float* ref_logits = baseline_logits + t * B * Vp;

        // After prefill (t==0), output has shape (B, prompt_len, Vp) — extract last token.
        // After decode steps (t>0), output has shape (B, 1, Vp) — just copy directly.
        if (t == 0) {
            copy_last_logits(kv_logits, model.acts.output, B, current_len, Vp);
        } else {
            copy_single_token_logits(kv_logits, model.acts.output, B, Vp);
        }
        
        // ACCURACY CHECK
        double step_maxerr = 0;
        double rmse = 0;
        for (int b = 0; b < B; ++b) {
            for (int v = 0; v < V; ++v) {
                double expected = ref_logits[b * Vp + v];
                double actual = kv_logits[b * Vp + v];
                double abserr = abs(actual - expected);
                double relerr = abs(actual - expected) / fmaxf(1e-3, abs(expected));
                double err = fmin(abserr, relerr);
                step_maxerr = fmax(step_maxerr, err);
                rmse += err * err;
                
                if (err > 0.1) {
                    printf("=========================MISMATCH AT TOKEN %d BATCH %d VOCAB %d: ", t, b, v);
                    printf("%f %f (err=%f)=========================\n", expected, actual, err);
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
        if (!ok) break;

        // Record what KV cache would have sampled (for debugging output)
        for (int b = 0; b < B; ++b) {
            kv_gen_tokens[t * B + b] = sample_token_argmax(kv_logits + b * Vp, V);
        }

        // Use the BASELINE's sampled token for both paths to prevent sequence
        // divergence from tiny numerical differences that pass the error tolerance.
        // sequence_buffer already contains the baseline-sampled tokens from the baseline run.
        for (int b = 0; b < B; ++b) {
            decode_input[b] = sequence_buffer[b * longest_seq + current_len];
        }
        
        g_current_pos = current_len;
        current_len++;

        // KV Step
        double step_time = benchmark_gpt2_forward(&model, decode_input, B, 1);
        kv_latencies.push_back(step_time);
    }

    if (ok) {
        printf("\n=================== KV CACHE PASSED ALL ACCURACY TESTS ===================\n");
        printf("Speedup Analysis:\n");
        double avg_baseline = 0;
        double avg_kv = 0;
        for (int t = 0; t < gen_len; ++t) {
            double b_ms = baseline_latencies[t];
            double k_ms = kv_latencies[t];
            avg_baseline += b_ms;
            avg_kv += k_ms;
            if (t < 5 || t == gen_len - 1) {
                printf("Token %3d: Baseline = %7.3f ms | KV Cache = %7.3f ms | Speedup = %5.2fx\n", 
                       t, b_ms, k_ms, b_ms / k_ms);
            }
            if (t == 5 && gen_len > 6) printf("...\n");
        }
        avg_baseline /= gen_len;
        avg_kv /= gen_len;
        printf("---------------------------------------------------------------------------\n");
        printf("AVERAGE:    Baseline = %7.3f ms | KV Cache = %7.3f ms | Speedup = %5.2fx\n", 
               avg_baseline, avg_kv, avg_baseline / avg_kv);

        // Print generated text for deterministic prompts
        if (!args.random_mode && tokenizer.init_ok) {
            printf("\nGenerated Text:\n");
            for (int b = 0; b < B; ++b) {
                printf("--- Batch %d ---\n", b);
                printf("[Prompt]       ");
                for (int t = 0; t < prompt_len; ++t) {
                    safe_printf(tokenizer_decode(&tokenizer, sequence_buffer[b * longest_seq + t]));
                }
                printf("\n[Baseline]     ");
                for (int t = 0; t < gen_len; ++t) {
                    safe_printf(tokenizer_decode(&tokenizer, sequence_buffer[b * longest_seq + prompt_len + t]));
                }
                printf("\n[KV Cache]     ");
                for (int t = 0; t < gen_len; ++t) {
                    safe_printf(tokenizer_decode(&tokenizer, kv_gen_tokens[t * B + b]));
                }
                printf("\n\n");
            }
        }
    }

    free(sequence_buffer);
    free(forward_inputs);
    free(baseline_logits);
    free(kv_logits);
    free(kv_gen_tokens);
    if (g_kv_cache) cudaFree(g_kv_cache);
    gpt2_free(&model);
    tokenizer_free(&tokenizer);

    return ok ? 0 : 1;
}
