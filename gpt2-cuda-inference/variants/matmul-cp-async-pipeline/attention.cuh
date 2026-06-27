#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include "../utils/cuda_utils.cuh"
#include "softmax.cuh"
#include "cp_async_utils.cuh"

// Tile sizes
constexpr int ATTN_BM = 16;
constexpr int ATTN_BN = 64;
constexpr int ATTN_BK = 16;
constexpr int ATTN_TN = 4;

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
    const int bh = blockIdx.z;
    const int b = bh / NH;
    const int nh = bh % NH;

    const int t1 = blockIdx.y * ATTN_BM + threadIdx.y;
    const int t2_base = blockIdx.x * ATTN_BN + threadIdx.x * ATTN_TN;

    float acc[ATTN_TN] = {0.f, 0.f, 0.f, 0.f};

    __shared__ float Qs[2][ATTN_BM][ATTN_BK];
    __shared__ float Ks[2][ATTN_BN][ATTN_BK];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255

    auto stage_qk = [&](int stage, int k0) {
        // Q tile: ATTN_BM x ATTN_BK
        {
            int q_row = threadIdx.y;
            int q_col = threadIdx.x; // 0..15
            int hs = k0 + q_col;
            if (t1 < T && hs < HS) {
                const float* src = q + (((b * NH + nh) * T + t1) * HS + hs);
                cp_async_4(&Qs[stage][q_row][q_col], src);
            } else {
                Qs[stage][q_row][q_col] = 0.0f;
            }
        }

        // K tile: ATTN_BN x ATTN_BK, each thread loads 4 elements (total 1024)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int linear = tid + i * (ATTN_BM * ATTN_BK);
            int row = linear / ATTN_BK; // 0..63
            int col = linear % ATTN_BK; // 0..15
            int t2 = blockIdx.x * ATTN_BN + row;
            int hs = k0 + col;
            if (t2 < T && hs < HS) {
                const float* src = k + (((b * NH + nh) * T + t2) * HS + hs);
                cp_async_4(&Ks[stage][row][col], src);
            } else {
                Ks[stage][row][col] = 0.0f;
            }
        }
        cp_async_commit();
    };

    // Prefetch stage 0
    stage_qk(0, 0);
    cp_async_wait_all();
    __syncthreads();

    int stage = 0;
    for (int k0 = 0; k0 < HS; k0 += ATTN_BK) {
        int next_k0 = k0 + ATTN_BK;
        int next_stage = stage ^ 1;
        if (next_k0 < HS) {
            stage_qk(next_stage, next_k0);
        }

        if (t1 < T) {
            #pragma unroll
            for (int kk = 0; kk < ATTN_BK; kk++) {
                float qv = Qs[stage][threadIdx.y][kk];
                #pragma unroll
                for (int n = 0; n < ATTN_TN; n++) {
                    acc[n] += qv * Ks[stage][threadIdx.x * ATTN_TN + n][kk];
                }
            }
        }

        cp_async_wait_all();
        __syncthreads();
        stage = next_stage;
    }

    if (t1 < T) {
        int out_row = ((b * NH + nh) * T + t1) * T;
        #pragma unroll
        for (int n = 0; n < ATTN_TN; n++) {
            int t2 = t2_base + n;
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

    float acc[ATTN_TN] = {0.f, 0.f, 0.f, 0.f};

    __shared__ float As[2][ATTN_BM][ATTN_BK];
    __shared__ float Bs[2][ATTN_BK][ATTN_BN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    auto stage_pv = [&](int stage, int k0) {
        // A tile from att: (ATTN_BM x ATTN_BK), where K dimension is t2
        {
            int t2 = k0 + threadIdx.x;
            if (t < T && t2 < T) {
                const float* src = att + (((b * NH + nh) * T + t) * T + t2);
                cp_async_4(&As[stage][threadIdx.y][threadIdx.x], src);
            } else {
                As[stage][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        // B tile from V: (ATTN_BK x ATTN_BN)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int linear = tid + i * (ATTN_BM * ATTN_BK);
            int row = linear / ATTN_BN; // 0..15
            int col = linear % ATTN_BN; // 0..63
            int vt = k0 + row;
            int hs = blockIdx.x * ATTN_BN + col;
            if (vt < T && hs < HS) {
                const float* src = v + (((b * NH + nh) * T + vt) * HS + hs);
                cp_async_4(&Bs[stage][row][col], src);
            } else {
                Bs[stage][row][col] = 0.0f;
            }
        }
        cp_async_commit();
    };

    stage_pv(0, 0);
    cp_async_wait_all();
    __syncthreads();

    int stage = 0;
    for (int k0 = 0; k0 < T; k0 += ATTN_BK) {
        int next_k0 = k0 + ATTN_BK;
        int next_stage = stage ^ 1;
        if (next_k0 < T) {
            stage_pv(next_stage, next_k0);
        }

        if (t < T) {
            #pragma unroll
            for (int kk = 0; kk < ATTN_BK; kk++) {
                float av = As[stage][threadIdx.y][kk];
                #pragma unroll
                for (int n = 0; n < ATTN_TN; n++) {
                    acc[n] += av * Bs[stage][kk][threadIdx.x * ATTN_TN + n];
                }
            }
        }

        cp_async_wait_all();
        __syncthreads();
        stage = next_stage;
    }

    if (t < T) {
        int out_row = ((b * NH + nh) * T + t) * HS;
        #pragma unroll
        for (int n = 0; n < ATTN_TN; n++) {
            int hs = hs_base + n;
            if (hs < HS) {
                vaccum[out_row + hs] = acc[n];
            }
        }
    }
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
        dim3 block(ATTN_BM, ATTN_BM);
        dim3 grid((T + ATTN_BN - 1) / ATTN_BN,
                  (T + ATTN_BM - 1) / ATTN_BM,
                  B * NH);
        qk_matmul_kernel<<<grid, block>>>(att, q, k, B, T, NH, HS);
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
        dim3 block(ATTN_BM, ATTN_BM);
        dim3 grid((HS + ATTN_BN - 1) / ATTN_BN,
                  (T + ATTN_BM - 1) / ATTN_BM,
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