#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

#define BLOCK_SIZE 128
#define U 8
#define S (BLOCK_SIZE / U)

__global__ void matmul_forward_kernel(float* out, const float* inp, const float* weight, 
                                      const float* bias, int B, int T, int C, int OC) {
    int tid = threadIdx.x;
    int global_row = blockIdx.y * BLOCK_SIZE + tid;
    int global_col_start = blockIdx.x * U;

    __shared__ float N_tile[S][U];

    float acc[U];
    
    if (bias != NULL) {
        #pragma unroll
        for (int u = 0; u < U; u++) {
            if (global_col_start + u < OC) {
                acc[u] = bias[global_col_start + u];
            } else {
                acc[u] = 0.0f;
            }
        }
    } else {
        #pragma unroll
        for (int u = 0; u < U; u++) {
            acc[u] = 0.0f;
        }
    }

    int s_row = tid / U;
    int s_col = tid % U;
    int weight_col = global_col_start + s_col;
    bool valid_row = global_row < (B * T);

    for (int k_step = 0; k_step < (C + S - 1) / S; k_step++) {
        int k_n = k_step * S + s_row;
        if (weight_col < OC && k_n < C) {
            N_tile[s_row][s_col] = weight[weight_col * C + k_n];
        } else {
            N_tile[s_row][s_col] = 0.0f;
        }

        float M_reg[S];
        
        #pragma unroll
        for (int s = 0; s < S; s++) {
            int k_m = k_step * S + s;
            if (valid_row && k_m < C) {
                M_reg[s] = inp[global_row * C + k_m];
            } else {
                M_reg[s] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int s = 0; s < S; s++) {
            #pragma unroll
            for (int u = 0; u < U; u++) {
                acc[u] += M_reg[s] * N_tile[s][u];
            }
        }

        __syncthreads();
    }

    if (valid_row) {
        #pragma unroll
        for (int u = 0; u < U; u++) {
            if (global_col_start + u < OC) {
                out[global_row * OC + global_col_start + u] = acc[u];
            }
        }
    }
}

void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {

    dim3 block(BLOCK_SIZE);
    dim3 grid((OC + U - 1) / U, (B * T + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_forward_kernel<<<grid, block>>>(out, inp, weight, bias, B, T, C, OC);
}

#endif