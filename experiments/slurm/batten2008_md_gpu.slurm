#!/bin/bash
#SBATCH --job-name=mdw_md
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/md_winding/logs/slurm-%x-%j.out
#
# Low-temperature MD jobs (plan "MD winding"): one full A100, guarded (nomig:
# a MIG 3g.40gb slice is billed as one GPU but is much slower). STEP picks the
# program, STEP_ARGS is passed through:
#   bench     -> benchmarks/md_bench.py (V3: throughput grid, calib, omega)
#   run       -> experiments/scripts/batten2008_md_run.py --step run (E1)
#   langevin  -> experiments/scripts/batten2008_md_run.py --step langevin (E4)
#   spectra   -> experiments/scripts/batten2008_md_run.py --step spectra (E1b)
# The code is taken from the submit directory (PYTHONPATH=$PWD/src, so a git
# worktree runs its own package); the CUDA environment is the main checkout's
# .venv-cuda (override with MD_VENV). Job names start with mdw_ (budget ledger:
# sacct --name=mdw_%). Logs go to experiments/md_winding/logs/ (the directory is
# tracked through .gitkeep, so it exists in every checkout; slurm-*.out files are
# git-ignored). From the repository (or worktree) root:
#   env -u JAX_PLATFORMS sbatch -J mdw_bench --time=00:30:00 \
#     --export=ALL,STEP=bench,STEP_ARGS="--mode calib --chi 1-10 --out X.json" \
#     experiments/slurm/batten2008_md_gpu.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
case "${STEP:?set STEP (bench, run, langevin, spectra)}" in
  bench) PROG=(benchmarks/md_bench.py) ;;
  run|langevin|spectra) PROG=(experiments/scripts/batten2008_md_run.py --step "$STEP") ;;
  *) echo "ERROR: unknown STEP '$STEP' (bench, run, langevin, spectra)" >&2; exit 1 ;;
esac
[[ -f "${PROG[0]}" ]] || { echo "ERROR: ${PROG[0]} not found in $PWD" >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"
export UV_PROJECT_ENVIRONMENT="${MD_VENV:-/scratch/gpfs/TORQUATO/sd2402/xla-cco/.venv-cuda}"
export PYTHONPATH="$PWD/src"
unset JAX_PLATFORMS  # an inherited JAX_PLATFORMS=cpu (sbatch --export=ALL) would hide the GPU
echo "job ${SLURM_JOB_ID} ${SLURM_JOB_NAME} STEP=${STEP} on $(hostname) at $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
uv run --no-sync python scripts/gpu_assert.py || { echo "ERROR: no GPU visible; aborting." >&2; exit 1; }

# shellcheck disable=SC2086  # STEP_ARGS is a list of flags
uv run --no-sync python "${PROG[@]}" ${STEP_ARGS:-}
echo "Done at $(date -Is)."
