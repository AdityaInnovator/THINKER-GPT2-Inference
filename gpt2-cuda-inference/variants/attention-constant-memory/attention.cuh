#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"
#include "softmax.cuh"

// op_8: frequently-read attention shape metadata staged in constant memory.
__constant__ int c_att_B;
__constant__ int c_att_T;
__constant__ int c_att_NH;
__constant__ int c_att_HS;

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = c_att_B * c_att_NH * c_att_T * c_att_T;
    if (idx >= total)
        return;

    int t2 = idx % c_att_T;
    int t1 = (idx / c_att_T) % c_att_T;
    int nh = (idx / (c_att_T * c_att_T)) % c_att_NH;
    int b = idx / (c_att_NH * c_att_T * c_att_T);

    float sum = 0.0f;
    int q_base = ((b * c_att_NH + nh) * c_att_T + t1) * c_att_HS;
    int k_base = ((b * c_att_NH + nh) * c_att_T + t2) * c_att_HS;
    for (int hs = 0; hs < c_att_HS; ++hs)
    {
        sum += q[q_base + hs] * k[k_base + hs];
    }
    preatt[idx] = sum;
}

__global__ void pv_matmul_kernel(float *vaccum, const float *att, const float *v,
                                 int B, int T, int NH, int HS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = c_att_B * c_att_NH * c_att_T * c_att_HS;
    if (idx >= total)
        return;

    int hs = idx % c_att_HS;
    int t = (idx / c_att_HS) % c_att_T;
    int nh = (idx / (c_att_HS * c_att_T)) % c_att_NH;
    int b = idx / (c_att_NH * c_att_T * c_att_HS);

    float sum = 0.0f;
    int att_base = ((b * c_att_NH + nh) * c_att_T + t) * c_att_T;
    int v_base0 = ((b * c_att_NH + nh) * c_att_T) * c_att_HS;
    for (int t2 = 0; t2 < c_att_T; ++t2)
    {
        sum += att[att_base + t2] * v[v_base0 + t2 * c_att_HS + hs];
    }
    vaccum[idx] = sum;
}

void attention_forward(float *out, float *qkvr, float *att, const float *inp,
                       int B, int T, int C, int NH)
{
    int HS = C / NH;

    cudaCheck(cudaMemcpyToSymbol(c_att_B, &B, sizeof(int)));
    cudaCheck(cudaMemcpyToSymbol(c_att_T, &T, sizeof(int)));
    cudaCheck(cudaMemcpyToSymbol(c_att_NH, &NH, sizeof(int)));
    cudaCheck(cudaMemcpyToSymbol(c_att_HS, &HS, sizeof(int)));

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
        int threads = 256;
        int elems = B * NH * T * T;
        int blocks = (elems + threads - 1) / threads;
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
        int threads = 256;
        int elems = B * NH * T * HS;
        int blocks = (elems + threads - 1) / threads;
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
