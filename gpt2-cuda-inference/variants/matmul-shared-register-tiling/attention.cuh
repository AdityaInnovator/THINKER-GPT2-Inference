#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"
#include "softmax.cuh"

#define ATTN_TILE_M 128
#define ATTN_U 8
#define ATTN_S (ATTN_TILE_M / ATTN_U)

__global__ void permute_kernel(float* q, float* k, float* v, const float* inp, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * N * d;
    if (idx >= total)
        return;

    int hs = idx % d;
    int t = (idx / d) % N;
    int nh = (idx / (d * N)) % NH;
    int b = idx / (d * N * NH);

    int inp_base = ((b * N + t) * 3 * NH + nh) * d + hs;

    q[idx] = inp[inp_base + 0 * NH * d];
    k[idx] = inp[inp_base + 1 * NH * d];
    v[idx] = inp[inp_base + 2 * NH * d];
}

__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * N * d;
    if (idx >= total)
        return;
    
    int hs = idx % d;
    int t = (idx / d) % N;
    int nh = (idx / (d * N)) % NH;
    int b = idx / (d * N * NH);

    int out_base = (b * N + t) * (NH * d) + (nh * d) + hs;

    out[out_base] = inp[idx];
}

__global__ void qk_matmul_kernel(float *preatt, const float *q, const float *k,
                                 int B, int T, int NH, int HS) {
    int tid = threadIdx.x;
    int b_nh = blockIdx.z;
    int t1 = blockIdx.y * ATTN_TILE_M + tid;
    int t2_start = blockIdx.x * ATTN_U;

    __shared__ float N_tile[ATTN_S][ATTN_U];

    float acc[ATTN_U];
    #pragma unroll
    for (int u = 0; u < ATTN_U; u++) {
        acc[u] = 0.0f;
    }

    int s_row = tid / ATTN_U;
    int s_col = tid % ATTN_U;
    int t2 = t2_start + s_col;
    bool valid_row = t1 < T;

    int q_head = b_nh * T * HS;
    int k_head = b_nh * T * HS;

    for (int step = 0; step < (HS + ATTN_S - 1) / ATTN_S; step++) {
        int hs_n = step * ATTN_S + s_row;
        if (t2 < T && hs_n < HS) {
            N_tile[s_row][s_col] = k[k_head + t2 * HS + hs_n];
        } else {
            N_tile[s_row][s_col] = 0.0f;
        }

        float M_reg[ATTN_S];
        
        #pragma unroll
        for (int s = 0; s < ATTN_S; s++) {
            int hs_m = step * ATTN_S + s;
            if (valid_row && hs_m < HS) {
                M_reg[s] = q[q_head + t1 * HS + hs_m];
            } else {
                M_reg[s] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int s = 0; s < ATTN_S; s++) {
            #pragma unroll
            for (int u = 0; u < ATTN_U; u++) {
                acc[u] += M_reg[s] * N_tile[s][u];
            }
        }

        __syncthreads();
    }

    if (valid_row) {
        int preatt_head = b_nh * T * T;
        #pragma unroll
        for (int u = 0; u < ATTN_U; u++) {
            if (t2_start + u < T) {
                preatt[preatt_head + t1 * T + t2_start + u] = acc[u];
            }
        }
    }
}

__global__ void pv_matmul_kernel(float *vaccum, const float *att, const float *v,
                                 int B, int T, int NH, int HS) {
    int tid = threadIdx.x;
    int b_nh = blockIdx.z;
    int t1 = blockIdx.y * ATTN_TILE_M + tid;
    int hs_start = blockIdx.x * ATTN_U;

    __shared__ float N_tile[ATTN_S][ATTN_U];

    float acc[ATTN_U];
    #pragma unroll
    for (int u = 0; u < ATTN_U; u++) {
        acc[u] = 0.0f;
    }

    int s_row = tid / ATTN_U;
    int s_col = tid % ATTN_U;
    int hs_out = hs_start + s_col;
    bool valid_row = t1 < T;

    int att_head = b_nh * T * T;
    int v_head = b_nh * T * HS;

    for (int step = 0; step < (T + ATTN_S - 1) / ATTN_S; step++) {
        int t2_n = step * ATTN_S + s_row;
        if (hs_out < HS && t2_n < T) {
            N_tile[s_row][s_col] = v[v_head + t2_n * HS + hs_out];
        } else {
            N_tile[s_row][s_col] = 0.0f;
        }

        float M_reg[ATTN_S];
        
        #pragma unroll
        for (int s = 0; s < ATTN_S; s++) {
            int t2_m = step * ATTN_S + s;
            if (valid_row && t2_m < T) {
                M_reg[s] = att[att_head + t1 * T + t2_m];
            } else {
                M_reg[s] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int s = 0; s < ATTN_S; s++) {
            #pragma unroll
            for (int u = 0; u < ATTN_U; u++) {
                acc[u] += M_reg[s] * N_tile[s][u];
            }
        }

        __syncthreads();
    }

    if (valid_row) {
        int vaccum_head = b_nh * T * HS;
        #pragma unroll
        for (int u = 0; u < ATTN_U; u++) {
            if (hs_start + u < HS) {
                vaccum[vaccum_head + t1 * HS + hs_start + u] = acc[u];
            }
        }
    }
}


void attention_forward(float *out, float *qkvr, float *att, const float *inp,
                       int B, int T, int C, int NH)
{
    int HS = C / NH;

    float *q = qkvr + 0 * B * T * C;
    float *k = qkvr + 1 * B * T * C;
    float *v = qkvr + 2 * B * T * C;

    float *vaccum = nullptr;
    cudaCheck(cudaMalloc(&vaccum, (size_t)B * T * C * sizeof(float)));

    {
        int threads = 256;
        int elems = B * NH * T * HS;
        int blocks = (elems + threads - 1) / threads;
        permute_kernel<<<blocks, threads>>>(q, k, v, inp, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }

    {
        dim3 block(ATTN_TILE_M);
        dim3 grid((T + ATTN_U - 1) / ATTN_U, (T + ATTN_TILE_M - 1) / ATTN_TILE_M, B * NH);
        qk_matmul_kernel<<<grid, block>>>(att, q, k, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }

    {
        float scale = 1.0f / sqrtf((float)HS);
        int rows = B * NH * T;
        int threads = 256;
        int blocks = (rows + threads - 1) / threads;
        softmax_forward_kernel<<<blocks, threads>>>(att, scale, att, B * NH, T);
        cudaCheck(cudaGetLastError());
    }

    {
        dim3 block(ATTN_TILE_M);
        dim3 grid((HS + ATTN_U - 1) / ATTN_U, (T + ATTN_TILE_M - 1) / ATTN_TILE_M, B * NH);
        pv_matmul_kernel<<<grid, block>>>(vaccum, att, v, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }

    {
        int threads = 256;
        int elems = B * NH * T * HS;
        int blocks = (elems + threads - 1) / threads;
        unpermute_kernel<<<blocks, threads>>>(vaccum, out, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }

    cudaCheck(cudaDeviceSynchronize());
    cudaFree(vaccum);
}

#endif