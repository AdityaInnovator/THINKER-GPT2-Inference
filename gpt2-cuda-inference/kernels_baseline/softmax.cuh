#ifndef __SOFTMAX_KERNEL_CUH__
#define __SOFTMAX_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "../utils/cuda_utils.cuh"

// __global__ void softmax_forward_kernel(float* out, float inv_temperature, const float* inp, int N, int T) {
//     // Implement this
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < N * T) {
//         int own_pos = idx % T;
//         const float* x = inp + idx * T;

//         float maxval = -FLT_MAX;
//         for (int i = 0; i <= own_pos; ++i) {
//             maxval = fmaxf(maxval, x[i]);
//         }

//         float sumval = 0.0f;
//         for (int i = 0; i <= own_pos; ++i) {
//             float ev = expf(inv_temperature * (x[i] - maxval));
//             sumval += ev;
//             out[idx * T + i] = ev;
//         }

//         float norm = 1.0f / sumval;
//         for (int i = 0; i <= own_pos; ++i) {
//             out[idx * T + i] *= norm;
//         }
//     }

// }

__global__ void softmax_forward_kernel(float *out, float inv_temperature, const float *inp, int N, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * T)
        return;

    int own_pos = idx % T;          // query position t1
    const float *x = inp + idx * T; // logits for this (b,nh,t1) over t2
    float *y = out + idx * T;

    float maxval = -FLT_MAX;
    for (int i = 0; i <= own_pos; ++i)
        maxval = fmaxf(maxval, x[i]);

    float sumval = 0.0f;
    for (int i = 0; i <= own_pos; ++i)
    {
        float ev = expf(inv_temperature * (x[i] - maxval));
        sumval += ev;
        y[i] = ev;
    }
    for (int i = own_pos + 1; i < T; ++i)
        y[i] = 0.0f; // mask future keys

    float norm = 1.0f / sumval;
    for (int i = 0; i <= own_pos; ++i)
        y[i] *= norm;
}

#endif // __SOFTMAX_KERNEL_CUH__