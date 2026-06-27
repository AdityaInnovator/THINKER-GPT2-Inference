#ifndef GELU_KERNEL_CUH_
#define GELU_KERNEL_CUH_

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

#define GELU_SCALING_FACTOR sqrtf(2.0f/M_PI)

__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    // Implement this
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = inp[idx];
        float cube = 0.044715f * val*val*val;

        // lets approxiate the gelu suig the tanh approximation
        out[idx] = 0.5f * val * (1.0f + tanhf(GELU_SCALING_FACTOR * (val + cube)));
    }
}

// Launch kernel here
void gelu_forward(float* out, const float* inp, int N) {
    // Implement this
    int numThreadsPerBlock = 256;
    int numBlocks = (N + numThreadsPerBlock -1) / numThreadsPerBlock;
    gelu_forward_kernel<<<numBlocks, numThreadsPerBlock>>>(out, inp, N);
    
}

#endif // GELU_KERNEL_CUH_