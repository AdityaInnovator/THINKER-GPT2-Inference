#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

// req_0: joint shared-memory + register tiling
// C = A(MxK) * B(NxK)^T + bias(N)
// A is inp (B*T, C), B is weight (OC, C)
// Each block computes a BM x BN output tile.
// Each thread computes one row and TN columns in registers.
constexpr int BM = 16;
constexpr int BK = 16;
constexpr int BN = 64;
constexpr int TN = 4;

__global__ void matmul_forward_kernel(float* out, const float* inp, const float* weight, 
                                      const float* bias, int B, int T, int C, int OC) {
    const int row = blockIdx.y * BM + threadIdx.y;
    const int col_base = blockIdx.x * BN + threadIdx.x * TN;

    float acc[TN] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (bias != nullptr) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            const int col = col_base + n;
            if (col < OC) {
                acc[n] = bias[col];
            }
        }
    }

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int k0 = 0; k0 < C; k0 += BK) {
        const int a_col = k0 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < B * T && a_col < C) ? inp[row * C + a_col] : 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int linear = tid + i * BM * BK;
            const int bk = linear / BN;
            const int bn = linear % BN;
            const int w_col = blockIdx.x * BN + bn;
            const int w_k = k0 + bk;
            Bs[bk][bn] = (w_col < OC && w_k < C) ? weight[w_col * C + w_k] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            const float a = As[threadIdx.y][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                acc[n] += a * Bs[k][threadIdx.x * TN + n];
            }
        }

        __syncthreads();
    }

    if (row < B * T) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            const int col = col_base + n;
            if (col < OC) {
                out[row * OC + col] = acc[n];
            }
        }
    }
}

// Launch kernel here
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    dim3 block(BM, BM);
    dim3 grid((OC + BN - 1) / BN, (B * T + BM - 1) / BM);

    matmul_forward_kernel<<<grid, block>>>(out, inp, weight, bias, B, T, C, OC);
}

#endif // __MATMUL_KERNEL_CUH__