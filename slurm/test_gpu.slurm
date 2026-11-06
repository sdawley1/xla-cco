#!/bin/bash
#SBATCH --job-name=xla-cco-test-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
# GPU test suite on Della. Submit from the repository root:
#   sbatch slurm/test_gpu.slurm
# Compute nodes are offline: sync the GPU env once on a login node first:
#   UV_PROJECT_ENVIRONMENT=.venv-cuda uv sync --dev --extra cuda --locked
# Samuel J. Dawley, 11/2026
set -euo pipefail
export UV_CACHE_DIR=/scratch/gpfs/TORQUATO/sd2402/.uv-cache
export UV_PROJECT_ENVIRONMENT=.venv-cuda
export JAX_PLATFORMS=cuda,cpu  # cuda first (default device); cpu for the CPU-vs-GPU test
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"

if [ ! -x .venv-cuda/bin/python ]; then
    echo "ERROR: .venv-cuda missing; sync it on a login node (see header)." >&2
    exit 1
fi

# ----- fail fast: abort before any long step if no GPU is visible -----
.venv-cuda/bin/python scripts/gpu_assert.py || {
    echo "ERROR: GPU guard failed; aborting." >&2
    exit 1
}

# ============================================================
# RUN
# ============================================================
uv run --no-sync pytest -m "not della" --hypothesis-profile=ci

echo "Done."
