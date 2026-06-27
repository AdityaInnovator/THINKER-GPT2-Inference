#!/bin/bash
#SBATCH --job-name=gpt2-cuda
#SBATCH --output=gpt2-%j.out
#SBATCH --error=gpt2-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT

set -euo pipefail
module reset 2>/dev/null || true

export GPT2_CHECKPOINT="${GPT2_CHECKPOINT:-$PWD/checkpoints/gpt2_124M.bin}"
export GPT2_TOKENIZER="${GPT2_TOKENIZER:-$PWD/checkpoints/gpt2_tokenizer.bin}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
make clean all

srun ./test_gpt2
srun ./next_token_generation "The meaning of life is"
