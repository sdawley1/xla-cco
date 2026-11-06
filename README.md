# xla_cco

Pure-JAX collective-coordinate optimization for stealthy hyperuniform ground
states; a port of the group's C++ CCO code.

## Usage

Relax a batch of configurations to stealthy ground states (C++ ground/input
mode; chi = 0.05 smoke case, d = 3, phi = 0.05):

```bash
python -m xla_cco.cli.relax --dim 3 --k1a 0 --ka 0.5963271031078479 --phi 0.05 \
    --load /path/to/ic/chunk_0000 --save out/smoke --method lbfgs \
    --tol 9.99999968e-21 --device gpu
```

- **Input** is a ConfigPack prefix (`--load X` reads `X.ConfigPack` and
  `X.ConfigPackIndex`) holding unit-cell configurations: basis with
  |det| = 1 (e.g. the `ic/chunk_*` packs). C++ `*_InitConfig` packs store a
  basis of side N^(1/d) (|det| = N) and are rejected with an error.
- **Parity tolerance:** C++ parses `1e-20` as float32, so use
  `--tol 9.99999968e-21` to reproduce its success counts (defect D-002).
- **Outputs** (prefix `--save S`): `S.ConfigPack` (all relaxed configurations),
  `S_Success.ConfigPack` (energy < `--tol`), `S_InitConfig.ConfigPack`,
  `S_energies.npy`, `S_log.json` (summary, `n_success`), and the ensemble
  tables `S_g2.txt`, `S_Hp.txt`, `S_Sk.txt`. Success is judged only by
  energy < `--tol`; an L-BFGS line-search end far below tol is normal.
- **Device:** `--device gpu|cpu` (default gpu). `gpu` fails fast with the likely
  cause (no `--gres=gpu`, wrong env, missing CUDA libraries) and never falls
  back to the CPU.

Parity evidence: [docs/parity.md](docs/parity.md). Timings and the kernel
choice rule: [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

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
