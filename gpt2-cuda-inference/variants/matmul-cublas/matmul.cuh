#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils/cuda_utils.cuh"

// Kernel to add bias
__global__ void add_bias_kernel(float* out, const float* bias, int rows, int cols, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * OC;
    if (idx < total) {
        int col = idx % OC;
        out[idx] += bias[col];
    }
}

// Launch kernel here - uses cuBLAS for matmul
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // Use cuBLAS for matrix multiplication
    // inp: (B*T, C), weight: (OC, C) transposed -> (C, OC)
    // out = inp @ weight^T = (B*T, C) @ (C, OC) = (B*T, OC)
    
    int M = B * T;  // number of rows in inp
    int N = OC;     // number of columns in weight (after transpose)
    int K = C;      // inner dimension
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    if (cublas_handle == nullptr) {
        cublasCheck(cublasCreate(&cublas_handle));
    }
    
    // cublasSgemm: C = alpha * op(A) * op(B) + beta * C
    // We want: out = inp @ weight^T
    // A = weight (K x N), op(A) = weight^T (N x K)
    // B = inp (M x K), op(B) = inp (M x K)
    // C = out (M x N)
    cublasStatus_t status = cublasSgemm(cublas_handle,
                                        CUBLAS_OP_T,      // weight is transposed
                                        CUBLAS_OP_N,      // inp is not transposed
                                        N, M, K,          // dimensions: out is M x N
                                        &alpha,
                                        weight, K,        // weight: K x N, leading dim K
                                        inp, K,           // inp: M x K, leading dim K
                                        &beta,
                                        out, N);          // out: M x N, leading dim N

    cublasCheck(status);
    
    // Add bias if provided
    if (bias != NULL) {
        int numThreads = 256;
        int numBlocks = (M * OC + numThreads - 1) / numThreads;
        add_bias_kernel<<<numBlocks, numThreads>>>(out, bias, M, OC, OC);
        cudaCheck(cudaGetLastError());
    }
}

#endif // __MATMUL_KERNEL_CUH__