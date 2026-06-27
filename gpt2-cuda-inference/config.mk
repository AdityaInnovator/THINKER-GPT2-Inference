# Build configuration — override: make CUDA_ARCH=sm_89

CUDA_ARCH     ?= sm_86
NVCC          ?= nvcc
CC            ?= gcc
NVCC_STD      := -std=c++17
NVCC_OPT      := -O3
NVCC_DEBUG    := -g -lineinfo
NVCC_ARCH     := -arch=$(CUDA_ARCH)
NVCC_FLAGS    := $(NVCC_STD) $(NVCC_OPT) $(NVCC_ARCH) $(NVCC_DEBUG) -rdc=true

CUDA_HOME     ?= /usr/local/cuda
CUBLAS_INC    := -I$(CUDA_HOME)/include
CUBLAS_LIB    := -L$(CUDA_HOME)/lib64 -lcublas -lcublasLt

# Optional CUTLASS (only needed for op_14–op_16 experiments)
CUTLASS_ROOT  ?=
ifneq ($(CUTLASS_ROOT),)
  CUTLASS_INC := -I$(CUTLASS_ROOT)/include
endif

KERNEL_DIR    := kernels
KERNEL_CUH    := $(wildcard $(KERNEL_DIR)/*.cuh)
CPU_KERNELS   := $(wildcard cpu_kernels/*.cuh)
COMMON_SRC    := src/kv_cache_globals.cu
