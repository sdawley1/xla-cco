#!/bin/bash
#SBATCH --job-name=xla-cco-w2-exact
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Exact periodic Wasserstein-2 distances (Hungarian algorithm) on ~200
# log-spaced frames of every POS/GAU trajectory (CPU job; no GPU requested).
# Submit from the repository root:
#   sbatch --export=ALL,CHI=0.1 experiments/slurm/wasserstein_exact.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR=/scratch/gpfs/TORQUATO/sd2402/.uv-cache
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CHI="${CHI:-0.1}"
uv run --no-sync python -I experiments/scripts/wasserstein_exact.py \
    --traj "experiments/trajectories/chi${CHI}" --workers "${SLURM_CPUS_PER_TASK}"
echo "Done."
