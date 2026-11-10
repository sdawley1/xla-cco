#!/bin/bash
#SBATCH --job-name=mdcxx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=07:45:00
#SBATCH --array=1-3
#SBATCH --output=experiments/trajectories/slurm-%x-%A_%a.out
# C++ MD reference runs (plan T4): one array task per seed (seed = task id).
# Runs the UNMODIFIED soft_core_stealthy2.out in random ... MD step 1e-5 mode,
# d = 2, N = 256, M = 50, Nc = 200, Nc_equi = 20 (~4 h serial per seed).
# Submit from the repository root (CPU only, no GPU):
#   mkdir -p experiments/trajectories  # job logs (relative to the submit dir)
#   env -u JAX_PLATFORMS sbatch slurm/md_cxx_reference_cpu.slurm
# Afterwards, on a login node:
#   python tools/cpp_reference/md_reference.py collect --seeds 1 2 3
# Samuel J. Dawley, 11/2026
set -euo pipefail
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"

VENV_PY="${VENV_PY:-/scratch/gpfs/TORQUATO/sd2402/xla-cco/.venv/bin/python}"
SEED="${SLURM_ARRAY_TASK_ID:?run as an array job}"

# ============================================================
# RUN
# ============================================================
# argv[1] = 7 h: the binary saves an .MDDump and stops 5 min before that
PYTHONPATH="${PWD}/src" "${VENV_PY}" tools/cpp_reference/md_reference.py run \
    --seed "${SEED}" --hours 7

echo "Done."
