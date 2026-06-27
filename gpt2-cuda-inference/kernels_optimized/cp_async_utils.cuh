#ifndef __CP_ASYNC_UTILS_CUH__
#define __CP_ASYNC_UTILS_CUH__

#include <cuda_runtime.h>

static __device__ __forceinline__ void cp_async_4(void* smem_ptr, const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    unsigned long long gmem = reinterpret_cast<unsigned long long>(gmem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem), "l"(gmem));
#else
    *reinterpret_cast<int*>(smem_ptr) = *reinterpret_cast<const int*>(gmem_ptr);
#endif
}

static __device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

static __device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

#endif // __CP_ASYNC_UTILS_CUH__

