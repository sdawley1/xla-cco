#!/bin/bash
#SBATCH --job-name=xla-cco-flows2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# 2D gradient flows from the PRE Poisson starts (flows2d.py). One chi per job;
# resubmit the same command to resume until the log prints the final status.
# From the repository root:
#   sbatch --export=ALL,CHI=0.1 experiments/slurm/flows2d_gpu.slurm
#   sbatch --export=ALL,CHI=0.4 experiments/slurm/flows2d_gpu.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cuda"
unset JAX_PLATFORMS  # an inherited JAX_PLATFORMS=cpu (sbatch --export=ALL) would hide the GPU
uv run --no-sync python -I scripts/gpu_assert.py || { echo "ERROR: no GPU visible; aborting." >&2; exit 1; }

uv run --no-sync python -I experiments/scripts/flows2d.py --chi "${CHI:?set CHI}" ${FLOW_ARGS:-}
echo "Done."
