#!/bin/bash
#SBATCH --job-name=make-cpp-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/make_cpp_reference-%j.out
# CPU only. Builds the C++ golden-data harness and regenerates tests/data/cpp/.
# Submit from the repository root: sbatch slurm/make_cpp_reference.slurm
# Samuel J. Dawley, 11/2026
set -euo pipefail
export UV_CACHE_DIR=/scratch/gpfs/TORQUATO/sd2402/.uv-cache
export OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
cd "${SLURM_SUBMIT_DIR:-/scratch/gpfs/TORQUATO/sd2402/xla-cco}"
make -C tools/cpp_reference
.venv/bin/python -I tools/cpp_reference/make_goldens.py
.venv/bin/python -I tools/cpp_reference/make_goldens.py --check
