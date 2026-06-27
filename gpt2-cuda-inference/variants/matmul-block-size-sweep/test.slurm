#!/bin/bash
#SBATCH --job-name="op7_matmul_sweep"
#SBATCH --output="kernels_op_7/sweep_slurm.out"
#SBATCH --error="kernels_op_7/sweep_slurm.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:45:00

# Submit from repo root (sp26_ece408_ClosedAI), e.g.:
#   cd /path/to/sp26_ece408_ClosedAI && sbatch kernels_op_7/sweep_matmul.slurm

module reset

echo "job ${SLURM_JOB_ID} on $(hostname)"
cd "${SLURM_SUBMIT_DIR}" || exit 1

python3 kernels_op_7/sweep_matmul.py | tee kernels_op_7/sweep_results.txt
