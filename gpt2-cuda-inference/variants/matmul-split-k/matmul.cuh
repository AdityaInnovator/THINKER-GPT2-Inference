#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

constexpr int SPLIT_K = 2;

__global__ void init_out_kernel(float* out, const float* bias, int BT, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BT * OC;
    if (idx >= total) return;
    int col = idx % OC;
    out[idx] = (bias != nullptr) ? bias[col] : 0.0f;
}

__global__ void matmul_forward_splitk_kernel(float* out, const float* inp, const float* weight,
                                             int BT, int C, int OC) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int split = (int)blockIdx.z;

    if (row >= BT || col >= OC) return;

    int k0 = (split * C) / SPLIT_K;
    int k1 = ((split + 1) * C) / SPLIT_K;

    float acc = 0.0f;
    for (int k = k0; k < k1; k++) {
        acc += inp[row * C + k] * weight[col * C + k];
    }
    atomicAdd(&out[row * OC + col], acc);
}

// Launch kernel here
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    int BT = B * T;

    {
        int threads = 256;
        int blocks = (BT * OC + threads - 1) / threads;
        init_out_kernel<<<blocks, threads>>>(out, bias, BT, OC);
        cudaCheck(cudaGetLastError());
    }

    {
        dim3 block(16, 16);
        dim3 grid((OC + block.x - 1) / block.x, (BT + block.y - 1) / block.y, SPLIT_K);
        matmul_forward_splitk_kernel<<<grid, block>>>(out, inp, weight, BT, C, OC);
        cudaCheck(cudaGetLastError());
    }
}

#endif // __MATMUL_KERNEL_CUH__