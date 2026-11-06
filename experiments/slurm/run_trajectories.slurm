#!/bin/bash
#SBATCH --job-name=xla-cco-dt2025-traj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Dawley & Torquato (2025) reproduction: record descent
# trajectories for the POS and GAU ensembles (CPU job; no GPU requested).
# Submit from the repository root:  sbatch experiments/slurm/run_trajectories.slurm
# Prerequisite (login node): uv sync --dev --locked; initial conditions made
# by experiments/scripts/make_initial_conditions.py.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR=/scratch/gpfs/TORQUATO/sd2402/.uv-cache
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"

CHI="${CHI:-0.1}"  # override: sbatch --export=ALL,CHI=0.4 <script>
IC=experiments/initial_conditions
for spec in "POS:${IC}/POS-2D-N1024" "GAU:${IC}/GAU-2D-N1000"; do
    ens="${spec%%:*}"
    pack="${spec#*:}"
    test -f "${pack}.ConfigPack" || { echo "ERROR: missing ${pack}.ConfigPack" >&2; exit 1; }
    uv run --no-sync python -I experiments/scripts/run_trajectories.py \
        --ensemble "${ens}" --load "${pack}" --chi "${CHI}" --device cpu
done
echo "Done."
