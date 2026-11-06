#!/bin/bash
#SBATCH --job-name=xla-cco-parity-ensemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
# Ensemble-level parity vs the C++ code (L-BFGS statistics) on a GPU.
# Submit from the repository root: sbatch slurm/parity_ensemble.slurm
# Needs .venv-cuda synced on a login node (see slurm/test_gpu.slurm header).
# Report: parity_reports/ensemble_<date>.md (pytest log) plus the tables the test
# writes to parity_reports/ensemble_<date>_tables.md (appended below).
# Env: XLA_CCO_ENSEMBLE_LIMIT (configs/task, default full), XLA_CCO_ENSEMBLE_DEVICE.
# Samuel J. Dawley, 11/2026
set -euo pipefail
export UV_CACHE_DIR=/scratch/gpfs/TORQUATO/sd2402/.uv-cache
export UV_PROJECT_ENVIRONMENT=.venv-cuda
export JAX_PLATFORMS=cuda
export XLA_CCO_ENSEMBLE_DEVICE=gpu
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"

if [ ! -x .venv-cuda/bin/python ]; then
    echo "ERROR: .venv-cuda missing; sync it on a login node (see header)." >&2
    exit 1
fi

# ----- fail fast: abort before any long step if no GPU is visible -----
.venv-cuda/bin/python scripts/gpu_assert.py || {
    echo "ERROR: GPU guard failed; aborting." >&2
    exit 1
}

TEST_FILE=tests/parity/test_ensemble_parity.py
if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: $TEST_FILE does not exist (task T12 has not landed)." >&2
    exit 1
fi

# ============================================================
# RUN
# ============================================================
mkdir -p parity_reports
REPORT="parity_reports/ensemble_$(date +%Y%m%d).md"
status=0
uv run --no-sync pytest -m "della and slow" "$TEST_FILE" -v \
    --hypothesis-profile=ci 2>&1 | tee parity_reports/.ensemble_raw.log \
    || status=$?
{
    echo "# Ensemble parity report $(date +%Y-%m-%d)"
    echo
    echo "- host: $(hostname)"
    echo "- commit: $(git rev-parse --short HEAD)"
    echo "- pytest exit status: ${status}"
    echo
    echo "- SLURM job: ${SLURM_JOB_ID:-none}"
    echo
    cat "parity_reports/ensemble_$(date +%Y%m%d)_tables.md" 2>/dev/null || true
    echo
    echo '## pytest log'
    echo '```'
    cat parity_reports/.ensemble_raw.log
    echo '```'
} > "$REPORT"
rm -f parity_reports/.ensemble_raw.log
echo "Report: $REPORT"
exit "$status"
