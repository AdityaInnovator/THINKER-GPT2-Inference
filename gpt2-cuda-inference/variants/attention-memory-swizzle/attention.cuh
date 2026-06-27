#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"
#include "../kernels/softmax.cuh"

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

__device__ __forceinline__ int swizzle_col(int row, int col)
{
    return col ^ ((row & 3) << 2);
}

__global__ void permute_kernel(float *q, float *k, float *v, const float *inp,
                               int B, int N, int NH, int d)
{
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

__global__ void unpermute_kernel(float *inp, float *out, int B, int N, int NH, int d)
{
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
                                 int B, int T, int NH, int HS)
{
    const int bh = blockIdx.z;
    const int b = bh / NH;
    const int nh = bh % NH;

    const int t1 = blockIdx.y * TILE_M + threadIdx.y;
    const int t2 = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float Qs[TILE_M][TILE_K + 1];
    __shared__ float Ks[TILE_N][TILE_K + 1];

    float acc = 0.0f;

    for (int k0 = 0; k0 < HS; k0 += TILE_K)
    {
        const int hs_q = k0 + threadIdx.x;
        const int hs_k = k0 + threadIdx.y;

        Qs[threadIdx.y][swizzle_col(threadIdx.y, threadIdx.x)] =
            (t1 < T && hs_q < HS) ? q[((b * NH + nh) * T + t1) * HS + hs_q] : 0.0f;

        Ks[threadIdx.x][swizzle_col(threadIdx.x, threadIdx.y)] =
            (t2 < T && hs_k < HS) ? k[((b * NH + nh) * T + t2) * HS + hs_k] : 0.0f;

        __syncthreads();

        if (t1 < T && t2 < T)
        {
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk)
            {
                acc += Qs[threadIdx.y][swizzle_col(threadIdx.y, kk)] *
                       Ks[threadIdx.x][swizzle_col(threadIdx.x, kk)];
            }
        }

        __syncthreads();
    }

    if (t1 < T && t2 < T)
    {
        preatt[((b * NH + nh) * T + t1) * T + t2] = acc;
    }
}

__global__ void pv_matmul_kernel(float *vaccum, const float *att, const float *v,
                                 int B, int T, int NH, int HS)
{
    const int bh = blockIdx.z;
    const int b = bh / NH;
    const int nh = bh % NH;

    const int t = blockIdx.y * TILE_M + threadIdx.y;
    const int hs = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    float acc = 0.0f;

    for (int t0 = 0; t0 < T; t0 += TILE_K)
    {
        const int k_att = t0 + threadIdx.x;
        const int k_v = t0 + threadIdx.y;

        As[threadIdx.y][swizzle_col(threadIdx.y, threadIdx.x)] =
            (t < T && k_att < T) ? att[((b * NH + nh) * T + t) * T + k_att] : 0.0f;

        Bs[threadIdx.y][swizzle_col(threadIdx.y, threadIdx.x)] =
            (k_v < T && hs < HS) ? v[((b * NH + nh) * T + k_v) * HS + hs] : 0.0f;

        __syncthreads();

        if (t < T && hs < HS)
        {
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk)
            {
                acc += As[threadIdx.y][swizzle_col(threadIdx.y, kk)] *
                       Bs[kk][swizzle_col(kk, threadIdx.x)];
            }
        }

        __syncthreads();
    }

    if (t < T && hs < HS)
    {
        vaccum[((b * NH + nh) * T + t) * HS + hs] = acc;
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
        dim3 threads(TILE_N, TILE_M);
        dim3 blocks((T + TILE_N - 1) / TILE_N,
                    (T + TILE_M - 1) / TILE_M,
                    B * NH);
        qk_matmul_kernel<<<blocks, threads>>>(att, q, k, B, T, NH, HS);
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
        dim3 threads(TILE_N, TILE_M);
        dim3 blocks((HS + TILE_N - 1) / TILE_N,
                    (T + TILE_M - 1) / TILE_M,
                    B * NH);
        pv_matmul_kernel<<<blocks, threads>>>(vaccum, att, v, B, T, NH, HS);
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

#endif // __ATTENTION_CUH__
