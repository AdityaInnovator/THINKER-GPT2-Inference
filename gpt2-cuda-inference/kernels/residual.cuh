#ifndef RESIDUAL_KERNEL_CUH_
#define RESIDUAL_KERNEL_CUH_

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    // Implement this
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = inp1[idx] + inp2[idx];
    }

}

// Launch kernel here
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    // Implement this
    dim3 block(256);                            
    dim3 grid((N + block.x - 1) / block.x);     
    residual_forward_kernel<<<grid, block>>>(out, inp1, inp2, N); 
}

#endif // RESIDUAL_KERNEL_CUH_
