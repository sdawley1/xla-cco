#!/bin/bash
#SBATCH --job-name=xla-cco-coupling-loops
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/trajectories/slurm-%x-%j.out
#
# Coupling and loop analyses of the 2D ground states of one start (CPU, NumPy):
# computes the caches mp_chi, mp_shoulder_chi, phonon_chi, loops_chi,
# loopstats_chi and loopphase_chi<tag>.npz in ROOT and renders
# <OUT>/<PREFIX>_{coupling_mp_compare, coupling_mp_shoulder, coupling_phonon_gap,
# coupling_phonon_dispersion, loop_moments, loop_statistics}.png, with the
# state and surrogate counts of the Poisson runs (README). Submit from the
# repository root (the main checkout holding experiments/trajectories):
#   env -u JAX_PLATFORMS sbatch --export=ALL,ROOT=experiments/trajectories/hip2d/N1000,PREFIX=hip2d \
#       experiments/slurm/coupling_loops_cpu.slurm
# Optional: TAGS="1-10 2-5" (default: every chi tag present in ROOT; the
# coupling_mp_* figures then show 0.1, 0.25, 0.4 and 0.45, as for Poisson),
# OUT (default experiments/figures), SCRIPTS (default
# $SLURM_SUBMIT_DIR/experiments/scripts; point it at a worktree to run that
# branch's scripts), PY (default: the plottime interpreter, which has
# NumPy and matplotlib; the project .venv has no matplotlib).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${ROOT:?set ROOT, e.g. experiments/trajectories/hip2d/N1000}"
: "${PREFIX:?set PREFIX, e.g. hip2d}"
OUT="${OUT:-experiments/figures}"
SCRIPTS="${SCRIPTS:-$SLURM_SUBMIT_DIR/experiments/scripts}"
PY="${PY:-/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python}"
[ -d "$ROOT" ] || { echo "ERROR: no directory $ROOT" >&2; exit 1; }
[ -f "$SCRIPTS/batten2008_lattice.py" ] || { echo "ERROR: no scripts in $SCRIPTS" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"

# shellcheck disable=SC2206  # TAGS is a list of chi tags
tags=(${TAGS:-})
tag_args=()
all_args=(--tags all)
if [ "${#tags[@]}" -gt 0 ]; then
  tag_args=(--tags "${tags[@]}")
  all_args=(--tags "${tags[@]}")
fi
fig_args=(--prefix "$PREFIX" --out "$OUT")
echo "ROOT=$ROOT PREFIX=$PREFIX TAGS=${TAGS:-all} SCRIPTS=$SCRIPTS"

step() {  # name, then the command; prints the wall time of each step
  local name="$1" t0=$SECONDS
  shift
  echo "=== $name"
  "$@"
  echo "=== $name: $((SECONDS - t0)) s"
}

# ----- coupling spectra (caches for every tag, figures as for Poisson) -----
step mp_cache "$PY" -I "$SCRIPTS/batten2008_mp_plot.py" --root "$ROOT" "${all_args[@]}" --flows 20 --cache-only
step mp_compare "$PY" -I "$SCRIPTS/batten2008_mp_plot.py" --root "$ROOT" "${tag_args[@]}" --flows 20 "${fig_args[@]}"
step shoulder_cache "$PY" -I "$SCRIPTS/batten2008_mp_shoulder_plot.py" --root "$ROOT" "${all_args[@]}" --states 3 --cache-only
step shoulder "$PY" -I "$SCRIPTS/batten2008_mp_shoulder_plot.py" --root "$ROOT" "${tag_args[@]}" --states 3 "${fig_args[@]}"
step phonon "$PY" -I "$SCRIPTS/batten2008_phonon_gap_plot.py" --root "$ROOT" "${tag_args[@]}" --states 6 "${fig_args[@]}"

# ----- loops -----
step loop_moments "$PY" -I "$SCRIPTS/batten2008_loop_moments.py" --roots "$ROOT" "${tag_args[@]}" --states 6 --surrogates 8
step loop_moments_plot "$PY" -I "$SCRIPTS/batten2008_loop_moments_plot.py" --roots "$ROOT" "${tag_args[@]}" "${fig_args[@]}"
step loop_statistics "$PY" -I "$SCRIPTS/batten2008_loop_statistics_plot.py" --root "$ROOT" "${tag_args[@]}" --states 4 "${fig_args[@]}"
echo "Done."
