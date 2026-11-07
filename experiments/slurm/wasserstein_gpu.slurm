#!/bin/bash
#SBATCH --job-name=xla-cco-w2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=00:59:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Periodic Wasserstein-2 distances along the recorded trajectories:
# Sinkhorn at every stored frame on the GPU, while the exact (Hungarian)
# assignment on ~200 log-spaced frames runs on the node's CPU cores.
# Guarded: fails fast without a GPU. Submit from the repository root:
#   sbatch --export=ALL,CHI=0.1 experiments/slurm/wasserstein_gpu.slurm
# Optional: ENSEMBLES="POS", START/LIMIT (trajectory slice; the Sinkhorn part
# is written as sinkhorn_<ENS>_<START>.npz), EXACT=0 to skip the exact pass,
# SINKHORN_ARGS="--max-iters 100" for extra wasserstein_sinkhorn.py options.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR=/scratch/gpfs/TORQUATO/sd2402/.uv-cache
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cuda"
uv run --no-sync python -I scripts/gpu_assert.py || { echo "ERROR: no GPU visible; aborting." >&2; exit 1; }

CHI="${CHI:-0.1}"
ENSEMBLES="${ENSEMBLES:-POS GAU}"
LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARG=(--limit "${LIMIT}"); fi
SLICE_ARG=(--start "${START:-0}" "${LIMIT_ARG[@]}")
# A START slice still runs the exact pass on the whole ensemble; without
# START, LIMIT applies to both passes (smoke tests).
EXACT_ARG=("${LIMIT_ARG[@]}")
if [[ -n "${START:-}" ]]; then EXACT_ARG=(); fi
TRAJ="experiments/trajectories/chi${CHI}"
WORKERS=$(( SLURM_CPUS_PER_TASK - 2 ))

exact_pid=""
if [[ "${EXACT:-1}" == "1" ]]; then
    # shellcheck disable=SC2086
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        uv run --no-sync python -I experiments/scripts/wasserstein_exact.py \
        --traj "${TRAJ}" --ensembles ${ENSEMBLES} --workers "${WORKERS}" "${EXACT_ARG[@]}" &
    exact_pid=$!
fi

# shellcheck disable=SC2086
uv run --no-sync python -I experiments/scripts/wasserstein_sinkhorn.py \
    --traj "${TRAJ}" --ensembles ${ENSEMBLES} --device gpu "${SLICE_ARG[@]}" ${SINKHORN_ARGS:-}

if [[ -n "${exact_pid}" ]]; then wait "${exact_pid}"; fi
echo "Done."
