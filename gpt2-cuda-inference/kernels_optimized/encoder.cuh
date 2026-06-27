#ifndef ENCODER_FORWARD_KERNEL_CUH
#define ENCODER_FORWARD_KERNEL_CUH

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void encoder_forward_kernel(float* out, const int* inp, const float* wte, const float* wpe,
                                       int B, int T, int C) {
    // Implement this
    // Idea is each thread computes each output element in the output of shape (B,T,C)
    // Thinking of one block as responsible for one token position across all batches, having 768 threads in the block,
    // and each thread within the block is responsible for one channel/value of the output embedding for that token position across all batches

    int b = blockIdx.x; // batch index
    int token_pos = blockIdx.y; // token position index
    int c = threadIdx.x; // channel index

    if (b<B && token_pos<T && c<C) {
        int token = inp[b*T + token_pos]; // Get the token ID
        float wte_val = wte[token*C + c]; // Get the word token embedding value
        float wpe_val = wpe[token_pos*C + c]; // Get the word position embedding value
        out[b*T*C + token_pos*C + c] = wte_val + wpe_val; // Compute the output embedding
    }
}

// Launch kernel here
void encoder_forward(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C) {
    // Implement this
    //block and grid dimensions
    dim3 blockDim(C); // 768 threads for each channel
    dim3 gridDim(B,T); // B blocks for batch, T blocks for token positions, so total B*T blocks and one blocks works on one token doing work for all channles (768)
    encoder_forward_kernel<<<gridDim, blockDim>>>(out, inp, wte, wpe, B, T, C);
}


#endif // ENCODER_FORWARD_KERNEL_CUH