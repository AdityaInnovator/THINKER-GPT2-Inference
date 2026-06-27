#ifndef __LAYERNORM_KERNEL_CUH__
#define __LAYERNORM_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "../utils/cuda_utils.cuh"

__global__ void layernorm_forward_kernel(float* out, float* mean, float* rstd, const float* inp, const float* weight,
                                         const float* bias, int B, int T, int C) {

    // Dynamically allocated shared memory
    extern __shared__ float shared_mem[];
    float *shared_inp = shared_mem;
    float *shared_weight = shared_mem + C;
    float *shared_bias = shared_mem + 2 * C;

    // Implement this
    int b = blockIdx.x; // batch index
    int t = blockIdx.y; // token index
    int c = threadIdx.x; //channel index

    shared_inp[c] = inp[b*T*C + t*C + c]; // load input into shared memory
    shared_weight[c] = weight[c]; // load weight into shared memory
    shared_bias[c] = bias[c]; // load bias into shared memory
    __syncthreads(); //synchronize to make sure all threads have the data

    if (b<B && t<T && c<C) {
        //Compute the mean and rstd for the current token across the channel dimension
        
        // TODO
        // One issue is every thread will compute the same mean and rstd for the current token, 
        // which is redundant, but later we will optimize this by only letting one thread 
        // compute the mean and rstd and then share it with other threads using shared memory

        // mean calculation
        float sum = 0.0f;
        for (int i=0;i<C; i++) {
            sum += shared_inp[i];
        }
        float mean_val = sum/C;

        // rstd calcultion
        float sum_sq_diff = 0.0f;
        for (int i=0; i<C; i++) {
            float diff = shared_inp[i] - mean_val;
            sum_sq_diff += diff * diff;
        }
        float rstd_val = rsqrtf(sum_sq_diff/C + 1e-5f); // reciprocal of std dev with epsilon for numerical stability

        // Normalize the input and apply the transformation using the weight and bias
        float norm_val = (shared_inp[c] - mean_val) * rstd_val; // normalize the input
        out[b*T*C + t*C + c] = norm_val * shared_weight[c] + shared_bias[c]; //apply the ransformation and store the output in the global memory
    }
}

// Launch kernel here
void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // Implement this

    //block and grid dimensions
    dim3 blockDim(C); // 768 threads for each channel
    dim3 gridDim(B, T); // B blocks for batch, T blocks for token positions, so total B*T blocks and one blocks works on one token doing work for all channles (768)
    size_t shared_mem_size = 3 * C * sizeof(float); // for shared_inp, shared_weight, shared_bias
    layernorm_forward_kernel<<<gridDim, blockDim, shared_mem_size>>>(out, mean, rstd, inp, weight, bias, B, T, C);
}

#endif // __LAYERNORM_KERNEL_CUH__