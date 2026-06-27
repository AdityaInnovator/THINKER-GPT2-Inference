#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"
#include "softmax.cuh"

__device__ __forceinline__ int rasterize_x(int bx, int by, int grid_x) {
    return (by & 1) ? (grid_x - 1 - bx) : bx;
}

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

__global__ void qk_matmul_kernel(float *preatt, const float *q, const float *k,
                                 int B, int T, int NH, int HS) {
    int bn = (int)blockIdx.z;
    int b = bn / NH;
    int nh = bn - b * NH;

    int bx = rasterize_x((int)blockIdx.x, (int)blockIdx.y, (int)gridDim.x);
    int by = (int)blockIdx.y;

    int t2 = bx * (int)blockDim.x + (int)threadIdx.x;
    int t1 = by * (int)blockDim.y + (int)threadIdx.y;
    if (b >= B || t1 >= T || t2 >= T) return;

    float sum = 0.0f;
    int q_base = ((b * NH + nh) * T + t1) * HS;
    int k_base = ((b * NH + nh) * T + t2) * HS;
    for (int hs = 0; hs < HS; ++hs) sum += q[q_base + hs] * k[k_base + hs];
    preatt[((b * NH + nh) * T + t1) * T + t2] = sum;
}

__global__ void pv_matmul_kernel(float *vaccum, const float *att, const float *v,
                                 int B, int T, int NH, int HS) {
    int bn = (int)blockIdx.z;
    int b = bn / NH;
    int nh = bn - b * NH;

    int bx = rasterize_x((int)blockIdx.x, (int)blockIdx.y, (int)gridDim.x);
    int by = (int)blockIdx.y;

    int hs = bx * (int)blockDim.x + (int)threadIdx.x;
    int t  = by * (int)blockDim.y + (int)threadIdx.y;
    if (b >= B || t >= T || hs >= HS) return;

    float sum = 0.0f;
    int att_base = ((b * NH + nh) * T + t) * T; // row of length T
    int v_base0  = ((b * NH + nh) * T) * HS;    // start of V for this head
    for (int t2 = 0; t2 < T; ++t2) sum += att[att_base + t2] * v[v_base0 + t2 * HS + hs];
    vaccum[((b * NH + nh) * T + t) * HS + hs] = sum;
}

// // Launch all kernels related to attention here 
// void attention_forward(float* out, float* qkvr, float* att, float* inp, int B, int T, int C, int NH) {
//     // Implement this

//     // Note: `inp` is re-used as a scratch buffer.
//     // Its contents will be overwritten by this function.

//     // inp is (B, T, 3C) QKV
//     // preatt, att are (B, NH, T, T)
//     // output is (B, T, C)
//     int HS = C / NH; // head size

//     // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
//     float *q, *k, *v;
//     q = qkvr + 0 * B * T * C;
//     k = qkvr + 1 * B * T * C;
//     v = qkvr + 2 * B * T * C;

//     // permute and separate inp from (B, T, 3C) to 3X (B, NH, T, HS)
//     // permute_kernel<<<...>>>(q, k, v, inp, B, T, NH, HS);
//     const int numThreadsPermute = 256;
//     int numBlocksPermute = (B * T * C + numThreadsPermute - 1) / numThreadsPermute;
//     permute_kernel<<<numBlocksPermute, numThreadsPermute>>>(q, k, v, inp, B, T, NH, HS);
//     cudaCheck(cudaGetLastError());
//     cudaCheck(cudaDeviceSynchronize());

//     // Attention matmul: Q @ K^T
//     // Compute pre-attention scores (B, NH, T, T)
//     float* preatt = inp;
//     for (int b = 0; b < B; b++) {
//         for (int nh = 0; nh < NH; nh++) {
//             for (int t1 = 0; t1 < T; t1++) {
//                 for (int t2 = 0; t2 < T; t2++) {
//                     float sum = 0.0f;
//                     for (int hs = 0; hs < HS; hs++) {
//                        sum += k[b * NH * T * HS + nh * T * HS + t2 * HS + hs] * 
//                                q[b * NH * T * HS + nh * T * HS + t1 * HS + hs];
//                     }
//                     preatt[b * NH * T * T + nh * T * T + t1 * T + t2] = sum;
//                 }
//             }
//         }
//     }

//     // Compute the softmax
//     float scale = 1.0 / sqrtf(HS);
//     const int numThreadsSoftMax = 256;
//     int numBlocksSoftmax = (B * NH * T + numThreadsSoftMax - 1) / numThreadsSoftMax;
//     softmax_forward_kernel<<<numBlocksSoftmax, numThreadsSoftMax>>>(att, scale, preatt, B * NH, T);
//     cudaCheck(cudaGetLastError());
//     cudaCheck(cudaDeviceSynchronize());

//     float *vaccum = inp;
//     // Attention matmul: P @ V, where P holds the attention probabilities
//     // (B, NH, T, T) @ (B, NH, T, HS) -> (B, NH, T, HS)
//     for (int b = 0; b < B; ++b){
//         for (int nh = 0; nh < NH; ++nh){
//             for (int t = 0; t < T; ++t){
//                 for (int hs = 0; hs < HS; ++hs){
//                     float sum = 0.0f;
//                     for (int t2 = 0; t2 < T; ++t2){
//                         sum += att[b * NH * T * T + nh * T * T + t * T + t2] *
//                                v[b * NH * T * HS + nh * T * HS + t2 * HS + hs];
//                     }
//                     vaccum[b * NH * T * HS + nh * T * HS + t * HS + hs] = sum;
//                 }
//             }
//         }
//     }


//     // unpermute from (B, NH, T, HS) to (B, T, C)
//     // unpermute_kernel<<<...>>>(vaccum, out, B, T, NH, HS);
//     const int numThreadsUnpermute = 256;
//     int numBlocksUnpermute = (B * T * C + numThreadsUnpermute - 1) / numThreadsUnpermute;
//     unpermute_kernel<<<numBlocksUnpermute, numThreadsUnpermute>>>(vaccum, out, B, T, NH, HS);
//     cudaCheck(cudaGetLastError());
//     cudaCheck(cudaDeviceSynchronize());
// }

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

    // 2) Compute preatt = Q @ K^T into att buffer (size B*NH*T*T)
    //    (You must have a CUDA kernel for this; CPU loops will segfault.)
    {
        dim3 block(16, 16);
        dim3 grid((T + block.x - 1) / block.x,
                  (T + block.y - 1) / block.y,
                  B * NH);
        qk_matmul_kernel<<<grid, block>>>(att, q, k, B, T, NH, HS); // att = preatt
        cudaCheck(cudaGetLastError());
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

    // 4) vaccum = att @ V  -> (B,NH,T,HS)
    {
        dim3 block(16, 16);
        dim3 grid((HS + block.x - 1) / block.x,
                  (T  + block.y - 1) / block.y,
                  B * NH);
        pv_matmul_kernel<<<grid, block>>>(vaccum, att, v, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
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