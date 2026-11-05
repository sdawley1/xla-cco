# xla_cco

Pure-JAX collective-coordinate optimization for stealthy hyperuniform ground
states; a port of the group's C++ CCO code.

## Development

Set the uv cache on scratch (the home quota is too small), in every shell and
in every SLURM script:

```bash
export UV_CACHE_DIR=/scratch/gpfs/TORQUATO/sd2402/.uv-cache
```

### Environments

Two separate environments; the GitHub CI uses only the first.

```bash
# CPU dev env -> .venv
uv sync --dev --locked

# GPU env -> .venv-cuda; sync ONCE on a login node (compute nodes are offline)
UV_PROJECT_ENVIRONMENT=.venv-cuda uv sync --dev --extra cuda --locked
```

### Tests

```bash
export JAX_PLATFORMS=cpu
uv run pytest -m "not gpu and not della"                       # dev profile
uv run pytest -m "not gpu and not della" --hypothesis-profile=ci   # as in CI
uv run pytest --hypothesis-profile=thorough tests/property     # 5000 examples
uv run pytest -m "not gpu and not della" --cov=xla_cco --cov-fail-under=90
```

Hypothesis profiles (`tests/conftest.py`): `dev` (100 examples, default),
`ci` (300, derandomized), `thorough` (5000). Markers: `slow` (> 30 s), `gpu`,
`della` (needs `/scratch` data or the C++ binary).

### Static checks and pre-commit

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src
uv run pre-commit install        # hooks run ruff, ruff format --check, mypy
uv run pre-commit run --all-files
```

### Della jobs

Submit from the repository root. GPU jobs fail fast through
`scripts/gpu_assert.py` (no `--gres=gpu`, wrong environment, or missing CUDA
libraries; never a silent CPU fallback).

```bash
sbatch slurm/test_gpu.slurm          # pytest -m "not della" on a GPU (.venv-cuda)
sbatch slurm/parity_ensemble.slurm   # ensemble parity -> parity_reports/ensemble_<date>.md
sbatch slurm/make_cpp_reference.slurm  # regenerate C++ goldens (CPU only)
```
