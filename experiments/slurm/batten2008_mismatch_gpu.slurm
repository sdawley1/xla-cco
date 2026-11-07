#!/bin/bash
#SBATCH --job-name=xla-cco-batten2008-mismatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Energy under the projected N = 1000 flows on each landscape plane (run
# after the flows). GPU job (about 2e5 energy evaluations at N = 1000); the
# login node's per-user limits kill it. Submit from the repository root:
#   sbatch experiments/slurm/batten2008_mismatch_gpu.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cuda"
unset JAX_PLATFORMS  # an inherited JAX_PLATFORMS=cpu (sbatch --export=ALL) would hide the GPU
uv run --no-sync python -I scripts/gpu_assert.py || { echo "ERROR: no GPU visible; aborting." >&2; exit 1; }

uv run --no-sync python -I experiments/scripts/batten2008_mismatch.py
echo "Done."
