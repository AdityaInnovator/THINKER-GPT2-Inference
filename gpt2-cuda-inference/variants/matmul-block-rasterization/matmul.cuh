#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__device__ __forceinline__ int rasterize_x_matmul(int bx, int by, int grid_x) {
    return (by & 1) ? (grid_x - 1 - bx) : bx;
}

__global__ void matmul_forward_kernel(float* out, const float* inp, const float* weight, 
                                      const float* bias, int B, int T, int C, int OC) {
    int bx = rasterize_x_matmul((int)blockIdx.x, (int)blockIdx.y, (int)gridDim.x);
    int by = (int)blockIdx.y;

    int col = bx * (int)blockDim.x + (int)threadIdx.x;
    int row = by * (int)blockDim.y + (int)threadIdx.y;
    
    if (row < B * T && col < OC) {
        float acc = (bias != NULL) ? bias[col] : 0.0f;
        for (int k = 0; k < C; k++) {
            acc += inp[row * C + k] * weight[col * C + k];
        }
        out[row * OC + col] = acc;
    }
}

// Launch kernel here
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    dim3 block(32, 32);
    dim3 grid((OC + block.x - 1) / block.x, (B * T + block.y - 1) / block.y);
    matmul_forward_kernel<<<grid, block>>>(out, inp, weight, bias, B, T, C, OC);
}

#endif // __MATMUL_KERNEL_CUH__