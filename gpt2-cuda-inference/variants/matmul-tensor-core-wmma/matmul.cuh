#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <mma.h>
#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

namespace wmma = nvcuda::wmma;

// Tensor Core tile sizes for TF32 WMMA.
static const int TM = 16;
static const int TN = 16;
static const int TK = 8;

// Convert FP32 to TF32-like value (10-bit mantissa) for WMMA inputs.
__device__ __forceinline__ float to_tf32(float x) {
  unsigned int u = __float_as_uint(x);
  unsigned int exp = u & 0x7F800000u;
  // Preserve NaN/Inf as-is.
  if (exp == 0x7F800000u) {
    return x;
  }
  // Round-to-nearest then truncate lower 13 mantissa bits.
  u += 0x00001000u;
  u &= 0xFFFFE000u;
  return __uint_as_float(u);
}

// One warp computes one 16x16 output tile with boundary-safe staging.
__global__ void matmul_tf32_wmma_kernel(float* out, const float* inp, const float* weight,
                                        const float* bias, int BT, int C, int OC) {
  __shared__ __align__(16) float As[TM * TK];      // row-major A tile (16x8)
  __shared__ __align__(16) float Bs[TK * TN];      // col-major B tile (8x16)
  __shared__ __align__(16) float Cs[TM * TN];      // row-major output tile

  const int lane = threadIdx.x;                    // 0..31 (one warp per block)
  const int row0 = blockIdx.y * TM;
  const int col0 = blockIdx.x * TN;

  wmma::fragment<wmma::accumulator, TM, TN, TK, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k0 = 0; k0 < C; k0 += TK) {
    // Stage A tile with zero padding for out-of-bounds
    for (int idx = lane; idx < TM * TK; idx += 32) {
      int r = idx / TK;
      int c = idx % TK;
      int gr = row0 + r;
      int gk = k0 + c;
      float a = (gr < BT && gk < C) ? inp[(size_t)gr * C + gk] : 0.0f;
      As[idx] = to_tf32(a);
    }

    // Stage B tile in COL-MAJOR layout (row + col*ld, ld=TK)
    for (int idx = lane; idx < TK * TN; idx += 32) {
      int col = idx / TK;
      int row = idx % TK;
      int gc = col0 + col;
      int gk = k0 + row;
      float b = (gc < OC && gk < C) ? weight[(size_t)gc * C + gk] : 0.0f;
      Bs[idx] = to_tf32(b);
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_a, TM, TN, TK, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TM, TN, TK, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, As, TK);
    wmma::load_matrix_sync(b_frag, Bs, TK);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncthreads();
  }

  wmma::store_matrix_sync(Cs, acc, TN, wmma::mem_row_major);
  __syncthreads();

  for (int idx = lane; idx < TM * TN; idx += 32) {
    int dr = idx / TN;
    int dc = idx % TN;
    int gr = row0 + dr;
    int gc = col0 + dc;
    if (gr < BT && gc < OC) {
      float v = Cs[idx];
      if (bias != NULL) v += bias[gc];
      out[(size_t)gr * OC + gc] = v;
    }
  }
}

void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
  int BT = B * T;
  dim3 block(32); // one warp
  dim3 grid(CEIL_DIV(OC, TN), CEIL_DIV(BT, TM));
  matmul_tf32_wmma_kernel<<<grid, block>>>(out, inp, weight, bias, BT, C, OC);
  cudaCheck(cudaGetLastError());
}

#endif // __MATMUL_KERNEL_CUH__
