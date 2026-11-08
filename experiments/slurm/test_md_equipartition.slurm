#!/bin/bash
#SBATCH --job-name=mdw_v1_equipartition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/v1/slurm-%x-%j.out
# V1 statistical validation of the MD library on a CPU node. Submit from the
# repository root (worktree) with
#   env -u JAX_PLATFORMS sbatch experiments/slurm/test_md_equipartition.slurm
# Samuel J. Dawley, 11/2026
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"
export PYTHONPATH="$PWD/src"
export UV_PROJECT_ENVIRONMENT=/scratch/gpfs/TORQUATO/sd2402/xla-cco/.venv
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"
export JAX_PLATFORMS=cpu
hostname; lscpu | grep "Model name"; git rev-parse HEAD
/usr/bin/time -v uv run --no-sync pytest -s -v -p no:cacheprovider \
    tests/integration/test_md_equipartition.py --durations=0
echo "Done."
