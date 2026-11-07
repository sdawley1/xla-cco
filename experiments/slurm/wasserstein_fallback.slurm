#!/bin/bash
#SBATCH --job-name=xla-cco-w2-fallback
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Exact (Hungarian) Wasserstein-2 at every frame where the Sinkhorn rounding
# was not a bijection, after all Sinkhorn slices of a chi have finished
# (CPU only). Submit from the repository root:
#   sbatch --export=ALL,CHI=0.4 experiments/slurm/wasserstein_fallback.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CHI="${CHI:-0.1}"
uv run --no-sync python -I experiments/scripts/wasserstein_fallback.py \
    --traj "experiments/trajectories/chi${CHI}" --workers "$(( SLURM_CPUS_PER_TASK - 1 ))"
echo "Done."
