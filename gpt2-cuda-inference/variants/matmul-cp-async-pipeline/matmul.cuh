#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"
#include "cp_async_utils.cuh"

constexpr int BM = 16;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TN = 4;

__global__ void matmul_forward_kernel(float* out, const float* inp, const float* weight,
                                      const float* bias, int BT, int C, int OC) {
    const int row = blockIdx.y * BM + threadIdx.y;
    const int col_base = blockIdx.x * BN + threadIdx.x * TN;

    float acc[TN] = {0.f, 0.f, 0.f, 0.f};
    if (bias != nullptr) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int col = col_base + n;
            acc[n] = (col < OC) ? bias[col] : 0.0f;
        }
    }

    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255

    auto stage_load = [&](int stage, int k0) {
        // A tile (BM x BK): each thread loads one element
        {
            int a_col = k0 + threadIdx.x;
            if (row < BT && a_col < C) {
                const float* src = inp + (size_t)row * C + a_col;
                cp_async_4(&As[stage][threadIdx.y][threadIdx.x], src);
            } else {
                As[stage][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        // B tile (BK x BN): each thread loads 4 elements (total BK*BN=1024)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int linear = tid + i * (BM * BK);
            int bk = linear / BN; // 0..15
            int bn = linear % BN; // 0..63
            int w_col = blockIdx.x * BN + bn;
            int w_k = k0 + bk;
            if (w_col < OC && w_k < C) {
                const float* src = weight + (size_t)w_col * C + w_k;
                cp_async_4(&Bs[stage][bk][bn], src);
            } else {
                Bs[stage][bk][bn] = 0.0f;
            }
        }
        cp_async_commit();
    };

    // Prefetch first K tile
    stage_load(0, 0);
    cp_async_wait_all();
    __syncthreads();

    int stage = 0;
    for (int k0 = 0; k0 < C; k0 += BK) {
        int next_k0 = k0 + BK;
        int next_stage = stage ^ 1;
        if (next_k0 < C) {
            stage_load(next_stage, next_k0);
        }

        if (row < BT) {
            #pragma unroll
            for (int kk = 0; kk < BK; kk++) {
                float a = As[stage][threadIdx.y][kk];
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    acc[n] += a * Bs[stage][kk][threadIdx.x * TN + n];
                }
            }
        }

        cp_async_wait_all();
        __syncthreads();
        stage = next_stage;
    }

    if (row < BT) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int col = col_base + n;
            if (col < OC) {
                out[(size_t)row * OC + col] = acc[n];
            }
        }
    }
}

// Launch kernel here
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    int BT = B * T;

    {
        dim3 block(BM, BM);
        dim3 grid((OC + BN - 1) / BN, (BT + BM - 1) / BM);
        matmul_forward_kernel<<<grid, block>>>(out, inp, weight, bias, BT, C, OC);
        cudaCheck(cudaGetLastError());
    }
}

#endif // __MATMUL_KERNEL_CUH__