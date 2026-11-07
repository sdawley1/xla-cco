#!/bin/bash
#SBATCH --job-name=xla-cco-dt2025-traj-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Dawley & Torquato (2025) reproduction: record descent
# trajectories for the POS and GAU ensembles (GPU job; guarded, fails fast without a GPU).
# Submit from the repository root:  sbatch experiments/slurm/run_trajectories_gpu.slurm
# Prerequisite (login node): UV_PROJECT_ENVIRONMENT=.venv-cuda uv sync --dev --extra cuda --locked; initial conditions made
# by experiments/scripts/make_initial_conditions.py.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"

export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cuda"
uv run --no-sync python -I scripts/gpu_assert.py || { echo "ERROR: no GPU visible; aborting." >&2; exit 1; }

CHI="${CHI:-0.1}"  # override: sbatch --export=ALL,CHI=0.4 <script>
IC=experiments/initial_conditions
for spec in "POS:${IC}/POS-2D-N1024" "GAU:${IC}/GAU-2D-N1000"; do
    ens="${spec%%:*}"
    pack="${spec#*:}"
    test -f "${pack}.ConfigPack" || { echo "ERROR: missing ${pack}.ConfigPack" >&2; exit 1; }
    uv run --no-sync python -I experiments/scripts/run_trajectories.py \
        --ensemble "${ens}" --load "${pack}" --chi "${CHI}" --device gpu
done
echo "Done."
