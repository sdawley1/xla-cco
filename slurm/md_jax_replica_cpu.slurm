#!/bin/bash
#SBATCH --job-name=mdjaxrep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=experiments/trajectories/slurm-%x-%A_%a.out
# JAX replica of the C++ MD reference protocol (plan V2): one array task per
# 8 vmapped lanes (4 at T = 1e-5, 4 at 1.1 T), 20 samples per lane, the full
# C++ protocol (L-BFGS start, 20 x 2 x [MB redraw, NVE 5e4], 20 x [Andersen
# 5e4, NVE 5e4]). CPU only. Submit from the repository root:
#   mkdir -p experiments/trajectories
#   env -u JAX_PLATFORMS sbatch slurm/md_jax_replica_cpu.slurm
# Afterwards, on a login node:
#   python tools/cpp_reference/md_jax_replica.py collect
# Samuel J. Dawley, 11/2026
set -euo pipefail
export JAX_PLATFORMS=cpu
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"

VENV_PY="${VENV_PY:-/scratch/gpfs/TORQUATO/sd2402/xla-cco/.venv/bin/python}"
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"

# ============================================================
# RUN
# ============================================================
PYTHONPATH="${PWD}/src" "${VENV_PY}" tools/cpp_reference/md_jax_replica.py run \
    --task "${TASK}"

echo "Done."
