#!/bin/bash
#SBATCH --job-name=xla-cco-batten2008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Batten, Stillinger & Torquato (2008) appendix setting at N = 1000: gradient
# flows from Poisson starts to 1D stealthy ground states (GPU job; guarded,
# fails fast without a GPU). Submit from the repository root:
#   sbatch --export=ALL,CHI=1/3 experiments/slurm/batten2008_flows_gpu.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cuda"
unset JAX_PLATFORMS  # an inherited JAX_PLATFORMS=cpu (sbatch --export=ALL) would hide the GPU
uv run --no-sync python -I scripts/gpu_assert.py || { echo "ERROR: no GPU visible; aborting." >&2; exit 1; }

CHI="${CHI:-1/3}"
N="${N:-1000}"
NTRAJ="${NTRAJ:-20}"
uv run --no-sync python -I experiments/scripts/batten2008_flows.py \
    --n "${N}" --chi "${CHI}" --n-traj "${NTRAJ}" \
    --out "experiments/trajectories/batten2008/N${N}" --device gpu ${FLOW_ARGS:-}
echo "Done."
