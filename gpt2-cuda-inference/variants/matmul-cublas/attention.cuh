#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils/cuda_utils.cuh"
#include "softmax.cuh"

extern cublasHandle_t cublas_handle;

__global__ void permute_kernel(float* q, float* k, float* v, const float* inp, int B, int N, int NH, int d) {
    // Implement this
    // input dimensions: (B, N, 3, NH, d) in that order ( N is sequence length(T), not batch size)
    // output dimensions: 3X (B, NH, N, d) in that order
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * N * d;
    if (idx >= total)
        return;

    int hs = idx % d;
    int t = (idx / d) % N;
    int nh = (idx / (d * N)) % NH;
    int b = idx / (d * N * NH);

    // base for inp[b, t, 0, 0, 0] in (B,N,3,NH,d)
    int inp_base = ((b * N + t) * 3 * NH + nh) * d + hs;

    // write outputs in (B,NH,N,d) using idx directly
    q[idx] = inp[inp_base + 0 * NH * d]; // qkv=0
    k[idx] = inp[inp_base + 1 * NH * d]; // qkv=1
    v[idx] = inp[inp_base + 2 * NH * d]; // qkv=2
}

__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {
    // Implement this
    // input dimensions: 3X (B, NH, N, d) in that order ( N is sequence length(T), not batch size)
    // output dimensions: (B, N, 3, NH, d) in that order
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * N * d;
    if (idx >= total)
        return;
    
    int hs = idx % d;
    int t = (idx / d) % N;
    int nh = (idx / (d * N)) % NH;
    int b = idx / (d * N * NH);

    // base for out[b, t, 0, 0, 0] in (B,N,C)
    int out_base = (b * N + t) * (NH * d) + (nh * d) + hs;

    out[out_base] = inp[idx];
}

// qk_matmul using cuBLAS: computes Q @ K^T for each batch and head
// Q: (B, NH, T, HS), K: (B, NH, T, HS) -> preatt: (B, NH, T, T)
void qk_matmul_cublas(float *preatt, const float *q, const float *k,
                      int B, int T, int NH, int HS, cublasHandle_t handle) {
    // Keep QK GEMM unscaled; softmax_forward_kernel applies the standard
    // 1/sqrt(HS) scaling via its inv_temperature argument.
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // For each batch and head, do: preatt = alpha * Q @ K^T
    // Q is (T, HS), K is (T, HS), preatt is (T, T)
    for (int b = 0; b < B; b++) {
        for (int nh = 0; nh < NH; nh++) {
            float *q_ptr = (float*)(q + b * NH * T * HS + nh * T * HS);
            float *k_ptr = (float*)(k + b * NH * T * HS + nh * T * HS);
            float *preatt_ptr = preatt + b * NH * T * T + nh * T * T;
            
            // preatt = alpha * Q @ K^T
            // Q: (T, HS), K^T: (HS, T) -> preatt: (T, T)
            cublasStatus_t status = cublasSgemm(handle,
                                                CUBLAS_OP_T,      // K is transposed
                                                CUBLAS_OP_N,      // Q is not transposed
                                                T, T, HS,         // output: T x T
                                                &alpha,
                                                k_ptr, HS,        // K: HS x T (leading dim HS)
                                                q_ptr, HS,        // Q: T x HS (leading dim HS)
                                                &beta,
                                                preatt_ptr, T);   // preatt: T x T (leading dim T)

            cublasCheck(status);
        }
    }
}

__global__ void qk_matmul_kernel(float *preatt, const float *q, const float *k,
                                 int B, int T, int NH, int HS) {
    // Dummy kernel - actual work done in qk_matmul_cublas
}

// pv_matmul using cuBLAS: computes Attention @ V for each batch and head
// att: (B, NH, T, T), V: (B, NH, T, HS) -> vaccum: (B, NH, T, HS)
void pv_matmul_cublas(float *vaccum, const float *att, const float *v,
                      int B, int T, int NH, int HS, cublasHandle_t handle) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // For each batch and head, do: vaccum = att @ V
    // att is (T, T), V is (T, HS), vaccum is (T, HS)
    for (int b = 0; b < B; b++) {
        for (int nh = 0; nh < NH; nh++) {
            float *att_ptr = (float*)(att + b * NH * T * T + nh * T * T);
            float *v_ptr = (float*)(v + b * NH * T * HS + nh * T * HS);
            float *vaccum_ptr = vaccum + b * NH * T * HS + nh * T * HS;
            
            // vaccum = att @ V
            // att: (T, T), V: (T, HS) -> vaccum: (T, HS)
            cublasStatus_t status = cublasSgemm(handle,
                                                CUBLAS_OP_N,      // V is not transposed
                                                CUBLAS_OP_N,      // att is not transposed
                                                HS, T, T,         // output: T x HS
                                                &alpha,
                                                v_ptr, HS,        // V: T x HS (leading dim HS)
                                                att_ptr, T,       // att: T x T (leading dim T)
                                                &beta,
                                                vaccum_ptr, HS);  // vaccum: T x HS (leading dim HS)

            cublasCheck(status);
        }
    }
}

__global__ void pv_matmul_kernel(float *vaccum, const float *att, const float *v,
                                 int B, int T, int NH, int HS) {
    // Dummy kernel - actual work done in pv_matmul_cublas
}

void attention_forward(float *out, float *qkvr, float *att, const float *inp,
                       int B, int T, int C, int NH)
{

    int HS = C / NH;

    // qkvr holds Q,K,V in (B, NH, T, HS) each, contiguous chunks
    float *q = qkvr + 0 * B * T * C;
    float *k = qkvr + 1 * B * T * C;
    float *v = qkvr + 2 * B * T * C;

    // Allocate vaccum (B, NH, T, HS) == B*T*C floats
    float *vaccum = nullptr;
    cudaCheck(cudaMalloc(&vaccum, (size_t)B * T * C * sizeof(float)));

    // 1) Permute packed QKV inp (B,T,3C) -> q,k,v each (B,NH,T,HS)
    {
        int threads = 256;
        int elems = B * NH * T * HS; // == B*T*C
        int blocks = (elems + threads - 1) / threads;
        permute_kernel<<<blocks, threads>>>(q, k, v, inp, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }

    // 2) Compute preatt = Q @ K^T into att buffer (size B*NH*T*T) using cuBLAS
    {
        qk_matmul_cublas(att, q, k, B, T, NH, HS, cublas_handle);
    }

    // 3) Softmax in-place: att = softmax(att) (causal)
    {
        float scale = 1.0f / sqrtf((float)HS);
        int rows = B * NH * T; // number of (b,nh,t1) rows
        int threads = 256;
        int blocks = (rows + threads - 1) / threads;
        softmax_forward_kernel<<<blocks, threads>>>(att, scale, att, B * NH, T); // inp==out allowed
        cudaCheck(cudaGetLastError());
    }

    // 4) vaccum = att @ V using cuBLAS -> (B,NH,T,HS)
    {
        pv_matmul_cublas(vaccum, att, v, B, T, NH, HS, cublas_handle);
    }

    // 5) Unpermute vaccum (B,NH,T,HS) -> out (B,T,C)
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