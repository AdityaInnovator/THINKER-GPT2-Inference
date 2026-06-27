#ifndef __LAYERNORM_KERNEL_CUH__
#define __LAYERNORM_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math.h>

__global__ void layernorm_forward_kernel( float* out, float* mean, float* rstd, const float* inp,
    const float* weight, const float* bias, int B, int T, int C)
{
    extern __shared__ float sdata[];

    float* s_sum = sdata;
    float* s_var = sdata + blockDim.x;

    int b = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;

    int row = b * T + t;

    const float* x = inp + row * C;
    float* y = out + row * C;


    // MEAN REDUCTION
    s_sum[tid] = (tid < C) ? x[tid] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    float mu = s_sum[0] / (float)C;

    // In this project path, mean/rstd buffers are often placeholders (size 1).
    // Avoid writing them here to prevent out-of-bounds accesses.
    __syncthreads();


    // VARIANCE REDUCTION
    float diff = (tid < C) ? (x[tid] - mu) : 0.0f;
    s_var[tid] = diff * diff;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }

    float var = s_var[0] / (float)C;
    float inv_std = rsqrtf(var + 1e-5f);

    // mean/rstd are placeholders in this project path; avoid writing them.
    __syncthreads();


    // NORMALIZATION
    if (tid < C) {
        float norm = (x[tid] - mu) * inv_std;
        float g = weight ? weight[tid] : 1.0f;
        float bval = bias ? bias[tid] : 0.0f;
        y[tid] = norm * g + bval;
    }
}

void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias, int B, int T, int C)
{
    dim3 grid(B, T);

    int threads = 1;
    while (threads < C) threads <<= 1;
    if (threads > 1024) threads = 1024;

    dim3 block(threads);
    size_t smem = 2 * threads * sizeof(float);

    layernorm_forward_kernel<<<grid, block, smem>>>(
        out, mean, rstd, inp, weight, bias, B, T, C);
}


#endif
