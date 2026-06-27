#ifndef __LOCAL_ATTENTION_CUH__
#define __LOCAL_ATTENTION_CUH__

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#include "../utils/cuda_utils.cuh"

#define WINDOW_SIZE 128
#define LOCAL_WINDOW_TOKENS (WINDOW_SIZE + 1)

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

__global__ void unpermute_kernel(float *inp, float *out, int B, int N, int NH,
                                 int d)
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

__global__ void local_qk_matmul_kernel(float *scores, const float *q,
                                       const float *k, int B, int T, int NH,
                                       int HS)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = B * NH * T;
    if (row >= total_rows)
        return;

    int t = row % T;
    int nh = (row / T) % NH;
    int b = row / (T * NH);
    int k_start = max(0, t - WINDOW_SIZE);
    int window_len = t - k_start + 1;

    const float *q_row = q + ((b * NH + nh) * T + t) * HS;
    float *score_row = scores + row * LOCAL_WINDOW_TOKENS;

    for (int offset = 0; offset < window_len; ++offset)
    {
        int k_t = k_start + offset;
        const float *k_row = k + ((b * NH + nh) * T + k_t) * HS;
        float sum = 0.0f;
        for (int hs = 0; hs < HS; ++hs)
            sum += q_row[hs] * k_row[hs];
        score_row[offset] = sum;
    }

    for (int offset = window_len; offset < LOCAL_WINDOW_TOKENS; ++offset)
        score_row[offset] = 0.0f;
}

__global__ void local_softmax_kernel(float *out, const float *inp, int B, int T,
                                     int NH, float inv_temperature)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = B * NH * T;
    if (row >= total_rows)
        return;

    int t = row % T;
    int k_start = max(0, t - WINDOW_SIZE);
    int window_len = t - k_start + 1;

    const float *x = inp + row * LOCAL_WINDOW_TOKENS;
    float *y = out + row * LOCAL_WINDOW_TOKENS;

    float maxval = -FLT_MAX;
    for (int offset = 0; offset < window_len; ++offset)
        maxval = fmaxf(maxval, x[offset]);

    float sumval = 0.0f;
    for (int offset = 0; offset < window_len; ++offset)
    {
        float ev = expf(inv_temperature * (x[offset] - maxval));
        sumval += ev;
        y[offset] = ev;
    }

    for (int offset = window_len; offset < LOCAL_WINDOW_TOKENS; ++offset)
        y[offset] = 0.0f;

    float norm = 1.0f / sumval;
    for (int offset = 0; offset < window_len; ++offset)
        y[offset] *= norm;
}

__global__ void local_pv_matmul_kernel(float *vaccum, const float *att,
                                       const float *v, int B, int T, int NH,
                                       int HS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * HS;
    if (idx >= total)
        return;

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int nh = (idx / (HS * T)) % NH;
    int b = idx / (NH * T * HS);
    int k_start = max(0, t - WINDOW_SIZE);
    int window_len = t - k_start + 1;

    const float *att_row = att + ((b * NH + nh) * T + t) * LOCAL_WINDOW_TOKENS;
    const float *v_row = v + ((b * NH + nh) * T + k_start) * HS;

    float sum = 0.0f;
    for (int offset = 0; offset < window_len; ++offset)
        sum += att_row[offset] * v_row[offset * HS + hs];

    vaccum[idx] = sum;
}

void local_attention_forward_gpu(float *out, float *qkvr, float *att,
                                 const float *inp, int B, int T, int C,
                                 int NH)
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
        int threads = 256;
        int rows = B * NH * T;
        int blocks = (rows + threads - 1) / threads;
        local_qk_matmul_kernel<<<blocks, threads>>>(att, q, k, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }

    {
        float scale = 1.0f / sqrtf((float)HS);
        int threads = 256;
        int rows = B * NH * T;
        int blocks = (rows + threads - 1) / threads;
        local_softmax_kernel<<<blocks, threads>>>(att, att, B, T, NH, scale);
        cudaCheck(cudaGetLastError());
    }

    {
        int threads = 256;
        int elems = B * NH * T * HS;
        int blocks = (elems + threads - 1) / threads;
        local_pv_matmul_kernel<<<blocks, threads>>>(vaccum, att, v, B, T, NH,
                                                     HS);
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

#endif // __LOCAL_ATTENTION_CUH__