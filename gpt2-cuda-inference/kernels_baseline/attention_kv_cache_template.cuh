#ifndef __ATTENTION_KV_CACHE_TEMPLATE_CUH__
#define __ATTENTION_KV_CACHE_TEMPLATE_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

extern bool g_enable_kv_cache; // Toggles KV-caching logic
extern float* g_kv_cache;      // Pointer to the GPU-allocated KV cache buffer
extern int g_layer_idx;        // Tracks the current transformer layer, useful for knowing when to switch to decode
extern bool g_is_prefill;      // True during the initial prompt, False during token generation
extern int g_current_pos;      // Tracks the absolute sequence position during Decode, helps index into the KV cache

// --- Suggested Kernel Placeholders (TODO: Add your regular attention kernel here, and any other kernels you may need) ---

/**
 * Kernel to store the newly computed Key and Value into the global KV cache. 
 * CAN BE FUSED WITH QKV MATMUL
 * In Prefill: Stores T tokens for the current layer.
 * In Decode: Stores 1 token at g_current_pos for the current layer.
 */
__global__ void store_kv_kernel(/* args */) {
    // Implement logic to copy K and V from the input QKV buffer into g_kv_cache
}

/**
 * Kernel to compute the attention for the Decode phase (T=1).
 * This involves:
 * 1. Extracting the current Query (Q)
 * 2. Gathering all previous Keys (K) and Values (V) from the cache
 * 3. Computing attention using the new Q and cached KV values.
 */
__global__ void decode_attention_kernel(/* args */) {
    // Implement T=1 attention logic using the KV cache
}

void attention_forward(float* out, float* qkvr, float* att, float* inp, int B, int T, int C, int NH) {
    // 1. Check if KV-caching is enabled, used by the verification script
    if (!g_enable_kv_cache) {
        // TODO: Call your standard Baseline Attention here (Milestone 1/2)
        return;
    }

    // 2. Initialize KV Cache if necessary
    if (g_kv_cache == nullptr) {
        // TODO: Allocate the KV cache, based accordingly to the model parameters
        // Option 1: Allocate a very big kv cache to store all the tokens
        // Option 2: Dynamically resize the kv cache repeatedly for each new token
    }

    if (g_is_prefill) {
        // --- PREFILL PHASE ---
        // During prefill, we process the entire input sequence (T > 1).
        
        // A. Store the K and V results from this layer into the cache.
        //    The projections are available in 'inp' (the QKV matmul output).
        
        // B. Run standard baseline attention for the initial prompt tokens.

        // C. Update the global layer index to track progress through the model.
        g_layer_idx++;
        if (g_layer_idx == 12) {
            g_layer_idx = 0;
            g_is_prefill = false; // Transition to Decode phase for the next token
        }
    } else {
        // --- DECODE PHASE ---
        // During decode, we only process the single newest token (T = 1).
        
        // A. Store the single new Key and Value for this step into the cache at g_current_pos.
        
        // B. Run specialized Decode Attention.
        //    This requires reading the history of Keys and Values from the cache
        //    to compute the output for the current token.
        //    TODO: Don't forget to scale your attention scores
        //    TODO: Ensure the output shapes match the next layer
        
        // C. Update the global state.
        g_layer_idx++;
        if (g_layer_idx == 12) {
            g_layer_idx = 0;
            g_current_pos++; // Advance the sequence position
        }
    }
}

#endif // __ATTENTION_KV_CACHE_TEMPLATE_CUH__
