include config.mk

BINS := output_verification_rand test_gpt2 test_gpt2_kernels next_token_generation \
        local_attn_verify verify_kv_cache

COMMON_OBJS := kv_cache_globals.o paths.o

.PHONY: all baseline optimized clean help $(BINS)

all: $(BINS)

baseline:
	@./scripts/use-kernel-set.sh baseline
	@$(MAKE) clean all

optimized:
	@./scripts/use-kernel-set.sh optimized
	@$(MAKE) clean all

kv_cache_globals.o: src/kv_cache_globals.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

paths.o: utils/paths.c
	$(CC) -c utils/paths.c -o $@

define BUILD_BIN
$(1): $(1).cu $(KERNEL_CUH) $(CPU_KERNELS) gpt2.cuh $(COMMON_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(CUBLAS_INC) $(CUTLASS_INC) -c $(1).cu -o $(1).o
	$(NVCC) $(NVCC_FLAGS) -o $(1) $(1).o $(COMMON_OBJS) $(CUBLAS_LIB)
endef

$(foreach b,$(BINS),$(eval $(BUILD_BIN)))

clean:
	rm -f *.o $(COMMON_OBJS) $(BINS)

help:
	@echo "Targets:"
	@echo "  make all          Build all binaries (optimized kernels/)"
	@echo "  make baseline     Switch to M1 kernels and rebuild"
	@echo "  make optimized    Switch to optimized kernels and rebuild"
	@echo "  make clean"
	@echo ""
	@echo "Environment:"
	@echo "  GPT2_CHECKPOINT, GPT2_TOKENIZER, GPT2_*_VERIF, CUDA_ARCH"
