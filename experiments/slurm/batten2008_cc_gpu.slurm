#!/bin/bash
#SBATCH --job-name=xla-cco-batten2008-cc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Landscape analyses of stored flows (1D Batten flows or 2D flows2d). STEP
# picks experiments/scripts/batten2008_<STEP>.py: slices, mismatch, tail,
# wasserstein, modes, hessian or floor; STEP_ARGS is passed through (e.g.
# --root experiments/trajectories/flows2d --sizes 1024 --chi 0.1 0.4).
# GPU job; guarded. From the repository root:
#   sbatch --export=ALL,STEP=modes experiments/slurm/batten2008_cc_gpu.slurm
#   sbatch --export=ALL,STEP=floor,STEP_ARGS="--chi 1/3" experiments/slurm/batten2008_cc_gpu.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
case "${STEP:?set STEP (slices, mismatch, tail, wasserstein, modes, hessian, floor)}" in
  slices|mismatch|tail|wasserstein|modes|hessian|floor) ;;
  *) echo "ERROR: unknown STEP '$STEP'" >&2; exit 1 ;;
esac
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cuda"
unset JAX_PLATFORMS  # an inherited JAX_PLATFORMS=cpu (sbatch --export=ALL) would hide the GPU
uv run --no-sync python -I scripts/gpu_assert.py || { echo "ERROR: no GPU visible; aborting." >&2; exit 1; }

# shellcheck disable=SC2086  # STEP_ARGS is a list of flags
uv run --no-sync python -I "experiments/scripts/batten2008_${STEP}.py" ${STEP_ARGS:-}
echo "Done."
