#!/bin/bash
#SBATCH --job-name=xla-cco-batten2008-slices
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Energy landscapes on the start-end, PCA and two-particle slices of every
# N = 1000 flow (run after batten2008_flows_gpu.slurm). GPU job; guarded.
# Submit from the repository root:
#   sbatch --export=ALL,CHI=1/3 experiments/slurm/batten2008_slices_gpu.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cuda"
unset JAX_PLATFORMS  # an inherited JAX_PLATFORMS=cpu (sbatch --export=ALL) would hide the GPU
uv run --no-sync python -I scripts/gpu_assert.py || { echo "ERROR: no GPU visible; aborting." >&2; exit 1; }

CHI="${CHI:-1/3}"
N="${N:-1000}"
uv run --no-sync python -I experiments/scripts/batten2008_slices.py \
    --dir "experiments/trajectories/batten2008/N${N}" --chi "${CHI}" --device gpu
echo "Done."
