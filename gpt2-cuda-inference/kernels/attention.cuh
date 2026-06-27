#ifndef __ATTENTION_KV_CACHE_CUH__
#define __ATTENTION_KV_CACHE_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"
#include "softmax.cuh"

extern bool   g_enable_kv_cache;
extern float* g_kv_cache;
extern int    g_layer_idx;
extern bool   g_is_prefill;
extern int    g_current_pos;

static int kv_L    = 12;
static int kv_maxT = 1024;

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kv_permute_kernel(float* q, float* k, float* v,
                                  const float* inp,
                                  int B, int N, int NH, int d)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * N * d;
    if (idx >= total) return;

    int hs = idx % d;
    int t  = (idx / d) % N;
    int nh = (idx / (d * N)) % NH;
    int b  = idx / (d * N * NH);

    int inp_base = ((b * N + t) * 3 * NH + nh) * d + hs;
    q[idx] = inp[inp_base + 0 * NH * d];
    k[idx] = inp[inp_base + 1 * NH * d];
    v[idx] = inp[inp_base + 2 * NH * d];
}

__global__ void kv_unpermute_kernel(const float* inp, float* out,
                                    int B, int N, int NH, int d)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * N * d;
    if (idx >= total) return;

    int hs = idx % d;
    int t  = (idx / d) % N;
    int nh = (idx / (d * N)) % NH;
    int b  = idx / (d * N * NH);

    out[(b * N + t) * (NH * d) + nh * d + hs] = inp[idx];
}

__global__ void kv_qk_matmul_kernel(float* preatt,
                                     const float* q, const float* k,
                                     int B, int T, int NH, int HS)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * T;
    if (idx >= total) return;

    int t2 = idx % T;
    int t1 = (idx / T) % T;
    int nh = (idx / (T * T)) % NH;
    int b  = idx / (NH * T * T);

    float sum = 0.f;
    int q_base = ((b * NH + nh) * T + t1) * HS;
    int k_base = ((b * NH + nh) * T + t2) * HS;
    for (int hs = 0; hs < HS; ++hs)
        sum += q[q_base + hs] * k[k_base + hs];
    preatt[idx] = sum;
}

__global__ void kv_pv_matmul_kernel(float* vaccum,
                                     const float* att, const float* v,
                                     int B, int T, int NH, int HS)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * T * HS;
    if (idx >= total) return;

    int hs = idx % HS;
    int t  = (idx / HS) % T;
    int nh = (idx / (HS * T)) % NH;
    int b  = idx / (NH * T * HS);

    float sum = 0.f;
    int att_base = ((b * NH + nh) * T + t) * T;
    int v_base0  = ((b * NH + nh) * T) * HS;
    for (int t2 = 0; t2 < T; ++t2)
        sum += att[att_base + t2] * v[v_base0 + t2 * HS + hs];
    vaccum[idx] = sum;
}

__global__ void store_kv_kernel(const float* k_src, const float* v_src,
                                 float* kv_cache,
                                 int layer, int seq_offset,
                                 int B, int T, int NH, int HS, int maxT)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * NH * HS;
    if (idx >= total) return;

    int hs = idx % HS;
    int nh = (idx / HS) % NH;
    int t  = (idx / (HS * NH)) % T;
    int b  = idx / (HS * NH * T);

    int src_idx = ((b * NH + nh) * T + t) * HS + hs;

    int stride_L  = 2 * B * maxT * NH * HS;
    int stride_kv =     B * maxT * NH * HS;
    int base = layer * stride_L
             + b * (maxT * NH * HS)
             + (seq_offset + t) * (NH * HS)
             + nh * HS
             + hs;

    kv_cache[0 * stride_kv + base] = k_src[src_idx];
    kv_cache[1 * stride_kv + base] = v_src[src_idx];
}

__global__ void decode_attention_kernel(
    float*       out,
    const float* q_new,
    const float* kv_cache,
    int layer, int cur_pos,
    int B, int NH, int HS, int maxT,
    float scale)
{
    int b        = blockIdx.x;
    int nh       = blockIdx.y;
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;
    int hist     = cur_pos + 1;
    int nwarps   = nthreads / 32;

    extern __shared__ float smem[];
    float* scores    = smem;
    float* q_smem    = smem + hist;
    float* out_smem  = smem + hist + HS;
    float* warp_max  = smem + hist + 2 * HS;
    float* warp_sum  = smem + hist + 2 * HS + nwarps;

    int stride_L  = 2 * B * maxT * NH * HS;
    int stride_kv =     B * maxT * NH * HS;
    int stride_t  = NH * HS;
    int batch_base = layer * stride_L + b * (maxT * NH * HS);

    int q_base = (b * NH + nh) * HS;
    for (int hs = tid; hs < HS; hs += nthreads)
        q_smem[hs] = q_new[q_base + hs];
    __syncthreads();

    for (int t2 = tid; t2 < hist; t2 += nthreads) {
        float dot = 0.f;
        int k_base = 0 * stride_kv + batch_base + t2 * stride_t + nh * HS;
        for (int hs = 0; hs < HS; ++hs)
            dot += q_smem[hs] * kv_cache[k_base + hs];
        scores[t2] = dot * scale;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int t2 = tid; t2 < hist; t2 += nthreads)
        local_max = fmaxf(local_max, scores[t2]);
    local_max = warp_reduce_max(local_max);
    if ((tid & 31) == 0) warp_max[tid / 32] = local_max;
    __syncthreads();

    float global_max;
    if (tid < 32) {
        float v = (tid < nwarps) ? warp_max[tid] : -1e30f;
        v = warp_reduce_max(v);
        if (tid == 0) warp_max[0] = v;
    }
    __syncthreads();
    global_max = warp_max[0];

    float local_sum = 0.f;
    for (int t2 = tid; t2 < hist; t2 += nthreads) {
        scores[t2] = expf(scores[t2] - global_max);
        local_sum += scores[t2];
    }
    local_sum = warp_reduce_sum(local_sum);
    if ((tid & 31) == 0) warp_sum[tid / 32] = local_sum;
    __syncthreads();

    float global_sum;
    if (tid < 32) {
        float v = (tid < nwarps) ? warp_sum[tid] : 0.f;
        v = warp_reduce_sum(v);
        if (tid == 0) warp_sum[0] = v;
    }
    __syncthreads();
    global_sum = warp_sum[0];

    float inv_sum = 1.f / global_sum;
    for (int t2 = tid; t2 < hist; t2 += nthreads)
        scores[t2] *= inv_sum;
    __syncthreads();

    for (int hs = tid; hs < HS; hs += nthreads) {
        float acc = 0.f;
        for (int t2 = 0; t2 < hist; ++t2) {
            int v_idx = 1 * stride_kv + batch_base + t2 * stride_t + nh * HS + hs;
            acc += scores[t2] * kv_cache[v_idx];
        }
        out_smem[hs] = acc;
    }
    __syncthreads();

    int out_base = (b * NH + nh) * HS;
    for (int hs = tid; hs < HS; hs += nthreads)
        out[out_base + hs] = out_smem[hs];
}

static void standard_attention_forward(float* out, float* qkvr, float* att,
                                        const float* inp,
                                        int B, int T, int C, int NH)
{
    int HS      = C / NH;
    int threads = 256;

    float* q = qkvr + 0 * B * T * C;
    float* k = qkvr + 1 * B * T * C;
    float* v = qkvr + 2 * B * T * C;

    float* vaccum = nullptr;
    cudaCheck(cudaMalloc(&vaccum, (size_t)B * T * C * sizeof(float)));

    kv_permute_kernel<<<(B*NH*T*HS+threads-1)/threads, threads>>>(
        q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    kv_qk_matmul_kernel<<<(B*NH*T*T+threads-1)/threads, threads>>>(
        att, q, k, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    {
        float scale = 1.f / sqrtf((float)HS);
        softmax_forward_kernel<<<(B*NH*T+threads-1)/threads, threads>>>(
            att, scale, att, B * NH, T);
        cudaCheck(cudaGetLastError());
    }

    kv_pv_matmul_kernel<<<(B*NH*T*HS+threads-1)/threads, threads>>>(
        vaccum, att, v, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    kv_unpermute_kernel<<<(B*NH*T*HS+threads-1)/threads, threads>>>(
        vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaDeviceSynchronize());
    cudaFree(vaccum);
}

void attention_forward(float* out, float* qkvr, float* att,
                       float* inp,
                       int B, int T, int C, int NH)
{
    if (!g_enable_kv_cache) {
        standard_attention_forward(out, qkvr, att, inp, B, T, C, NH);
        return;
    }

    int HS      = C / NH;
    int threads = 256;

    if (g_kv_cache == nullptr) {
        size_t elems = (size_t)kv_L * 2 * B * kv_maxT * NH * HS;
        cudaCheck(cudaMalloc(&g_kv_cache, elems * sizeof(float)));
        cudaCheck(cudaMemset(g_kv_cache,  0,     elems * sizeof(float)));
        printf("[KV-Cache] Allocated %.1f MiB  "
               "(L=%d, B=%d, maxT=%d, NH=%d, HS=%d)\n",
               (double)(elems * sizeof(float)) / (1 << 20),
               kv_L, B, kv_maxT, NH, HS);
    }

    int layer = g_layer_idx;

    if (g_is_prefill) {
        standard_attention_forward(out, qkvr, att, inp, B, T, C, NH);

        float* k_perm = qkvr + 1 * B * T * C;
        float* v_perm = qkvr + 2 * B * T * C;

        int total  = B * T * NH * HS;
        int blocks = (total + threads - 1) / threads;
        store_kv_kernel<<<blocks, threads>>>(
            k_perm, v_perm, g_kv_cache,
            layer, 0,
            B, T, NH, HS, kv_maxT);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        g_layer_idx++;
        if (g_layer_idx == kv_L) {
            g_layer_idx  = 0;
            g_is_prefill = false;
        }
        return;
    }

    {
        float* q = qkvr + 0 * B * 1 * C;
        float* k = qkvr + 1 * B * 1 * C;
        float* v = qkvr + 2 * B * 1 * C;

        kv_permute_kernel<<<(B*NH*1*HS+threads-1)/threads, threads>>>(
            q, k, v, inp, B, 1, NH, HS);
        cudaCheck(cudaGetLastError());

        {
            int total  = B * 1 * NH * HS;
            int blocks = (total + threads - 1) / threads;
            store_kv_kernel<<<blocks, threads>>>(
                k, v, g_kv_cache,
                layer, g_current_pos,
                B, 1, NH, HS, kv_maxT);
            cudaCheck(cudaGetLastError());
        }

        float* vaccum = nullptr;
        cudaCheck(cudaMalloc(&vaccum, (size_t)B * 1 * C * sizeof(float)));

        {
            int hist       = g_current_pos + 1;
            int block_T    = 128;
            int nwarps     = block_T / 32;
            int smem_bytes = (hist + 2 * HS + 2 * nwarps) * (int)sizeof(float);
            float scale    = 1.f / sqrtf((float)HS);

            dim3 grid(B, NH);
            dim3 block(block_T);
            decode_attention_kernel<<<grid, block, smem_bytes>>>(
                vaccum, q, g_kv_cache,
                layer, g_current_pos,
                B, NH, HS, kv_maxT, scale);
            cudaCheck(cudaGetLastError());
        }

        kv_unpermute_kernel<<<(B*NH*1*HS+threads-1)/threads, threads>>>(
            vaccum, out, B, 1, NH, HS);
        cudaCheck(cudaGetLastError());

        cudaCheck(cudaDeviceSynchronize());
        cudaFree(vaccum);

        g_layer_idx++;
        if (g_layer_idx == kv_L) {
            g_layer_idx = 0;
        }
    }
}

#endif // __ATTENTION_KV_CACHE_CUH__
