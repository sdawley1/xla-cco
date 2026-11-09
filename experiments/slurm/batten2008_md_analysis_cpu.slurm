#!/bin/bash
#SBATCH --job-name=mdw_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Analysis of the low-temperature MD winding runs (plan E2 + E3), CPU only:
#   1. batten2008_md_analysis.py on every md_chi*, md_spectra_chi* and
#      md_langevin_g*_chi* output (E1, E1b, E4) under DATA, PAR files at a
#      time; one md_analysis*_chi<tag>.npz per input in OUT/<system>/N<N>/;
#   2. batten2008_md_stats.py: H1-H8 verdicts (+ the post hoc exploratory
#      section) -> OUT/verdicts.{json,md}; batten2008_md_diagnostics.py: post
#      hoc descriptive diagnostics -> OUT/diagnostics.{json,md};
#   3. batten2008_md_plot.py (plottime env): F1-F10 and X1 -> FIGS (PNG).
# Submit from the repository (or worktree) root:
#   mkdir -p experiments/trajectories
#   env -u JAX_PLATFORMS sbatch experiments/slurm/batten2008_md_analysis_cpu.slurm
# Override DATA, OUT, FIGS, PAR or SKIP_ANALYSIS=1 (stats and figures only)
# through --export=ALL,VAR=value.
# Samuel J. Dawley, 11/2026

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"
export JAX_PLATFORMS=cpu
export PYTHONPATH="${PWD}/src"
VENV_PY="${VENV_PY:-/scratch/gpfs/TORQUATO/sd2402/xla-cco/.venv/bin/python}"
PLOT_PY="${PLOT_PY:-/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python}"
MAIN=/scratch/gpfs/TORQUATO/sd2402/xla-cco/experiments
DATA="${DATA:-${MAIN}/trajectories}"
OUT="${OUT:-${MAIN}/trajectories/md_results}"
FIGS="${FIGS:-${MAIN}/figures}"
PAR="${PAR:-3}"
SCRIPTS="${PWD}/experiments/scripts"
echo "job ${SLURM_JOB_ID} on $(hostname) at $(date -Is); DATA=${DATA} OUT=${OUT}"

# ============================================================
# 1. PER-FILE ANALYSIS
# ============================================================
if [[ "${SKIP_ANALYSIS:-0}" != "1" ]]; then
  find "${DATA}" -mindepth 3 -maxdepth 3 -name 'md_*chi*.npz' \
      ! -name '*.ckpt.npz' ! -name 'md_analysis*' -path '*/N*/*' | sort \
    | xargs -P "${PAR}" -I{} "${VENV_PY}" "${SCRIPTS}/batten2008_md_analysis.py" \
        --md {} --out-dir "${OUT}"   # -> OUT/<system>/N<N>/ (D-089)
fi

# ============================================================
# 2. VERDICTS
# ============================================================
"${VENV_PY}" "${SCRIPTS}/batten2008_md_stats.py" --analysis "${OUT}" \
    --out-json "${OUT}/verdicts.json"
"${VENV_PY}" "${SCRIPTS}/batten2008_md_diagnostics.py" --results "${OUT}"

# ============================================================
# 3. FIGURES
# ============================================================
"${PLOT_PY}" -I "${SCRIPTS}/batten2008_md_plot.py" --root "${OUT}" --out "${FIGS}" \
    --exploratory
echo "Done at $(date -Is)."
