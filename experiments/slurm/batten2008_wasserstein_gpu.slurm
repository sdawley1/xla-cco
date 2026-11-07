#!/bin/bash
#SBATCH --job-name=xla-cco-batten2008-w2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Exact Wasserstein-2 analysis of the stored N = 3 and N = 1000 flows (run
# after the flows and slices). GPU job (energies along the geodesics); the
# login node's per-user limits kill it. Submit from the repository root:
#   sbatch experiments/slurm/batten2008_wasserstein_gpu.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cuda"
unset JAX_PLATFORMS  # an inherited JAX_PLATFORMS=cpu (sbatch --export=ALL) would hide the GPU
uv run --no-sync python -I scripts/gpu_assert.py || { echo "ERROR: no GPU visible; aborting." >&2; exit 1; }

uv run --no-sync python -I experiments/scripts/batten2008_wasserstein.py
echo "Done."
