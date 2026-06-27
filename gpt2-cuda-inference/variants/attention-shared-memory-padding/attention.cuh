#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"
#include "../kernels/softmax.cuh"

constexpr int ATTN_BM = 16;
constexpr int ATTN_BK = 16;
constexpr int ATTN_BN = 64;
constexpr int ATTN_TN = 4;
constexpr int ATTN_PAD = 1;

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
    const int bh = blockIdx.z;
    const int b = bh / NH;
    const int nh = bh % NH;
    const int t1 = blockIdx.y * ATTN_BM + threadIdx.y;
    const int t2_base = blockIdx.x * ATTN_BN + threadIdx.x * ATTN_TN;
    float acc[ATTN_TN] = {0.0f, 0.0f, 0.0f, 0.0f};
    __shared__ float Qs[ATTN_BM][ATTN_BK];
    __shared__ float Ks[ATTN_BN][ATTN_BK + ATTN_PAD];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int k0 = 0; k0 < HS; k0 += ATTN_BK) {
        const int hs_q = k0 + threadIdx.x;
        Qs[threadIdx.y][threadIdx.x] =
            (t1 < T && hs_q < HS) ? q[((b * NH + nh) * T + t1) * HS + hs_q] : 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int linear = tid + i * ATTN_BM * ATTN_BK;
            const int row = linear / ATTN_BK;
            const int col = linear % ATTN_BK;
            const int t2 = blockIdx.x * ATTN_BN + row;
            const int hs_k = k0 + col;
            Ks[row][col] =
                (t2 < T && hs_k < HS) ? k[((b * NH + nh) * T + t2) * HS + hs_k] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < ATTN_BK; kk++) {
            const float qv = Qs[threadIdx.y][kk];
            #pragma unroll
            for (int n = 0; n < ATTN_TN; n++) {
                acc[n] += qv * Ks[threadIdx.x * ATTN_TN + n][kk];
            }
        }
        __syncthreads();
    }
    if (t1 < T) {
        const int out_row = ((b * NH + nh) * T + t1) * T;
        #pragma unroll
        for (int n = 0; n < ATTN_TN; n++) {
            const int t2 = t2_base + n;
            if (t2 < T) {
                preatt[out_row + t2] = acc[n];
            }
        }
    }
}
__global__ void pv_matmul_kernel(float *vaccum, const float *att, const float *v,
                                 int B, int T, int NH, int HS) {
    const int bh = blockIdx.z;
    const int b = bh / NH;
    const int nh = bh % NH;
    const int t = blockIdx.y * ATTN_BM + threadIdx.y;
    const int hs_base = blockIdx.x * ATTN_BN + threadIdx.x * ATTN_TN;
    float acc[ATTN_TN] = {0.0f, 0.0f, 0.0f, 0.0f};
    __shared__ float As[ATTN_BM][ATTN_BK + ATTN_PAD];
    __shared__ float Bs[ATTN_BK][ATTN_BN + ATTN_PAD];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int k0 = 0; k0 < T; k0 += ATTN_BK) {
        const int t2 = k0 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (t < T && t2 < T) ? att[((b * NH + nh) * T + t) * T + t2] : 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int linear = tid + i * ATTN_BM * ATTN_BK;
            const int row = linear / ATTN_BN;
            const int col = linear % ATTN_BN;
            const int vt = k0 + row;
            const int hs = blockIdx.x * ATTN_BN + col;
            Bs[row][col] =
                (vt < T && hs < HS) ? v[((b * NH + nh) * T + vt) * HS + hs] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < ATTN_BK; kk++) {
            const float av = As[threadIdx.y][kk];
            #pragma unroll
            for (int n = 0; n < ATTN_TN; n++) {
                acc[n] += av * Bs[kk][threadIdx.x * ATTN_TN + n];
            }
        }
        __syncthreads();
    }
    if (t < T) {
        const int out_row = ((b * NH + nh) * T + t) * HS;
        #pragma unroll
        for (int n = 0; n < ATTN_TN; n++) {
            const int hs = hs_base + n;
            if (hs < HS) {
                vaccum[out_row + hs] = acc[n];
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
        int elems = B * NH * T * HS; // == B*T*C
        int blocks = (elems + threads - 1) / threads;
        permute_kernel<<<blocks, threads>>>(q, k, v, inp, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }
    {
        dim3 block(ATTN_BM, ATTN_BM);
        dim3 grid((T + ATTN_BN - 1) / ATTN_BN,
                  (T + ATTN_BM - 1) / ATTN_BM,
                  B * NH);
        qk_matmul_kernel<<<grid, block>>>(att, q, k, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }
    {
        float scale = 1.0f / sqrtf((float)HS);
        int rows = B * NH * T; // number of (b,nh,t1) rows
        int threads = 256;
        int blocks = (rows + threads - 1) / threads;
        softmax_forward_kernel<<<blocks, threads>>>(att, scale, att, B * NH, T); // inp==out allowed
        cudaCheck(cudaGetLastError());
    }
    {
        dim3 block(ATTN_BM, ATTN_BM);
        dim3 grid((HS + ATTN_BN - 1) / ATTN_BN,
                  (T + ATTN_BM - 1) / ATTN_BM,
                  B * NH);
        pv_matmul_kernel<<<grid, block>>>(vaccum, att, v, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }
    {
        int threads = 256;
        int elems = B * NH * T * HS; // == B*T*C
        int blocks = (elems + threads - 1) / threads;
        unpermute_kernel<<<blocks, threads>>>(vaccum, out, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }

    cudaCheck(cudaDeviceSynchronize());
    cudaFree(vaccum);
}

#endif // __ATTENTION_CUH__
