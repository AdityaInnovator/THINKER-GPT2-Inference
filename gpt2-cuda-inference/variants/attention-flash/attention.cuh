#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include "../utils/cuda_utils.cuh"
#include "softmax.cuh"

constexpr int FA_BM = 16;  // queries per CTA
constexpr int FA_BN = 32;  // keys per iteration
constexpr int FA_WARP_Q = 16; // threads per query row (sub-warp width)

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * T;
    if (idx >= total)
        return;

    int t2 = idx % T;
    int t1 = (idx / T) % T;
    int nh = (idx / (T * T)) % NH;
    int b = idx / (NH * T * T);

    float sum = 0.0f;
    int q_base = ((b * NH + nh) * T + t1) * HS;
    int k_base = ((b * NH + nh) * T + t2) * HS;
    for (int hs = 0; hs < HS; ++hs)
    {
        sum += q[q_base + hs] * k[k_base + hs];
    }
    preatt[idx] = sum;
}

__global__ void pv_matmul_kernel(float *vaccum, const float *att, const float *v,
                                 int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * HS;
    if (idx >= total)
        return;

    int hs = idx % HS;
    int t = (idx / HS) % T;
    int nh = (idx / (HS * T)) % NH;
    int b = idx / (NH * T * HS);

    float sum = 0.0f;
    int att_base = ((b * NH + nh) * T + t) * T; // row of length T
    int v_base0 = ((b * NH + nh) * T) * HS;     // start of V for this head
    for (int t2 = 0; t2 < T; ++t2)
    {
        sum += att[att_base + t2] * v[v_base0 + t2 * HS + hs];
    }
    vaccum[idx] = sum;
}

__global__ void flash_attention_kernel(float *vaccum, const float *q, const float *k, const float *v,
                                       int T, int HS) {
    const int tid = threadIdx.x;
    const int qi = tid / FA_WARP_Q;
    const int lane = tid % FA_WARP_Q;

    const int bh = blockIdx.z;
    const int tq0 = blockIdx.y * FA_BM;
    const int tq = tq0 + qi;
    if (qi >= FA_BM) return;

    __shared__ float Qs[FA_BM][64];
    __shared__ float Ks[FA_BN][64];
    __shared__ float Vs[FA_BN][64];

    for (int d = lane; d < 64; d += FA_WARP_Q) {
        float qv = 0.0f;
        if (tq < T && d < HS) {
            qv = q[((size_t)bh * T + tq) * HS + d];
        }
        if (d < 64) Qs[qi][d] = qv;
    }
    __syncthreads();

    float m = -FLT_MAX;
    float l = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const float inv_sqrt_hs = rsqrtf((float)HS);

    for (int tk0 = 0; tk0 < T; tk0 += FA_BN) {
        for (int idx = tid; idx < FA_BN * 64; idx += blockDim.x) {
            const int kk = idx / 64;
            const int d = idx % 64;
            float kv = 0.0f, vv = 0.0f;
            const int tk = tk0 + kk;
            if (tk < T && d < HS) {
                kv = k[((size_t)bh * T + tk) * HS + d];
                vv = v[((size_t)bh * T + tk) * HS + d];
            }
            Ks[kk][d] = kv;
            Vs[kk][d] = vv;
        }
        __syncthreads();

        if (tq < T) {
            float scores[FA_BN];
            float tile_max = -FLT_MAX;

            if (lane == 0) {
                #pragma unroll
                for (int kk = 0; kk < FA_BN; kk++) {
                    const int tk = tk0 + kk;
                    if (tk >= T || tk > tq) {
                        scores[kk] = -FLT_MAX;
                        continue;
                    }

                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < 64; d++) {
                        if (d < HS) {
                            dot += Qs[qi][d] * Ks[kk][d];
                        }
                    }
                    float s = dot * inv_sqrt_hs;
                    scores[kk] = s;
                    tile_max = fmaxf(tile_max, s);
                }
            }

            float m_new = __shfl_sync(0xFFFFFFFF, tile_max, 0, FA_WARP_Q);
            m_new = fmaxf(m_new, m);
            m_new = __shfl_sync(0xFFFFFFFF, m_new, 0, FA_WARP_Q);

            const float scale_old = expf(m - m_new);
            const float scale_bcast = __shfl_sync(0xFFFFFFFF, scale_old, 0, FA_WARP_Q);
            l *= scale_bcast;
            o0 *= scale_bcast; o1 *= scale_bcast; o2 *= scale_bcast; o3 *= scale_bcast;

            float l_add = 0.0f;
            if (lane == 0) {
                #pragma unroll
                for (int kk = 0; kk < FA_BN; kk++) {
                    float s = scores[kk];
                    float p = (s <= -FLT_MAX / 2) ? 0.0f : expf(s - m_new);
                    scores[kk] = p; // reuse as probabilities for broadcast
                    l_add += p;
                }
            }
            l_add = __shfl_sync(0xFFFFFFFF, l_add, 0, FA_WARP_Q);

            #pragma unroll
            for (int kk = 0; kk < FA_BN; kk++) {
                float p = 0.0f;
                if (lane == 0) p = scores[kk];
                p = __shfl_sync(0xFFFFFFFF, p, 0, FA_WARP_Q);

                int d0 = lane;
                int d1 = lane + 16;
                int d2 = lane + 32;
                int d3 = lane + 48;
                if (d0 < HS) o0 += p * Vs[kk][d0];
                if (d1 < HS) o1 += p * Vs[kk][d1];
                if (d2 < HS) o2 += p * Vs[kk][d2];
                if (d3 < HS) o3 += p * Vs[kk][d3];
            }

            l += l_add;
            m = m_new;
        }

        __syncthreads();
    }

    if (tq < T) {
        float inv_l = 1.0f / l;
        inv_l = __shfl_sync(0xFFFFFFFF, inv_l, 0, FA_WARP_Q);

        int d0 = lane;
        int d1 = lane + 16;
        int d2 = lane + 32;
        int d3 = lane + 48;
        const size_t out_base = ((size_t)bh * T + tq) * HS;
        if (d0 < HS) vaccum[out_base + d0] = o0 * inv_l;
        if (d1 < HS) vaccum[out_base + d1] = o1 * inv_l;
        if (d2 < HS) vaccum[out_base + d2] = o2 * inv_l;
        if (d3 < HS) vaccum[out_base + d3] = o3 * inv_l;
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
        dim3 block(256);
        dim3 grid(1, (T + FA_BM - 1) / FA_BM, B * NH);
        flash_attention_kernel<<<grid, block>>>(vaccum, q, k, v, T, HS);
        cudaCheck(cudaGetLastError());
    }

    // Unpermute vaccum (B,NH,T,HS) -> out (B,T,C)
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