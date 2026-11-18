# xla_cco

Pure-JAX collective-coordinate optimization for stealthy hyperuniform ground
states; a port of the group's C++ CCO code.

## Install

On Della, from a login node (compute nodes are offline):

```bash
git clone https://github.com/sdawley1/xla-cco.git && cd xla-cco
# uv's cache must live on scratch (the home quota is too small); set it in
# every shell and SLURM script. Any directory you can write works.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"

uv sync --dev --locked                                  # CPU env -> .venv (what CI uses)
UV_PROJECT_ENVIRONMENT=.venv-cuda uv sync --dev --extra cuda --locked   # GPU env -> .venv-cuda
uv sync --dev --extra plot --locked                     # optional: matplotlib for the experiment figures
```

Run commands through the project environment, `uv run --no-sync ...` (or
`.venv/bin/python`); the system `python` on Della does not have `xla_cco`.

## Quickstart: command line

Relax the 5 bundled Poisson configurations (d = 3, N = 100) to chi ~ 0.05
stealthy ground states on the CPU, in about 10 s:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python -m xla_cco.cli.relax \
    --dim 3 --ka 0.5963271031078479 --phi 0.05 \
    --load tests/data/inputs/poisson_d3_n100 --save out/smoke --device cpu
```

`--ka` is the exclusion radius K times the C++ length unit a; it equals
`ka_from_chi(0.05, 0.05, 3)` here (see the Python quickstart). For real runs,
use a GPU node (`--device gpu`, the default) with `UV_PROJECT_ENVIRONMENT=.venv-cuda`.

- **Input** is a ConfigPack prefix (`--load X` reads `X.ConfigPack` and
  `X.ConfigPackIndex`). `--input-basis auto` (default) accepts a unit-volume
  basis (|det| = 1, C++ input mode) or a basis already scaled by N^(1/d)
  (|det| = N, as this CLI writes its outputs), so outputs can be relaxed
  again; `unit` or `scaled` enforce one convention.
- **Parity tolerance:** C++ parses `1e-20` as float32, so use
  `--tol 9.99999968e-21` to reproduce its success counts (defect D-002).
- **Outputs** (prefix `--save S`): `S.ConfigPack` (all relaxed configurations),
  `S_Success.ConfigPack` (energy < `--tol`), `S_InitConfig.ConfigPack`,
  `S_energies.npy`, `S_log.json` (summary, `n_success`), and the ensemble
  tables `S_g2.txt`, `S_Hp.txt`, `S_Sk.txt`. Success is judged only by
  energy < `--tol`; an L-BFGS line-search end far below tol is normal.
- **Device:** `--device gpu|cpu` (default gpu). `gpu` fails fast with the likely
  cause (`JAX_PLATFORMS` hiding the GPU, no `--gres=gpu`, wrong environment,
  missing CUDA libraries) and never falls back to the CPU.

## Quickstart: Python

```python
import numpy as np
import jax.numpy as jnp
from xla_cco import (
    DescentSettings,
    chi_from_count,
    cubic_box,
    enumerate_kvectors,
    k_outer_from_ka,
    ka_from_chi,
    make_stealthy_energy,
    relax,
)

n, d, chi = 64, 2, 0.25  # particles, dimension, target chi
box = cubic_box(n, d)  # density 1: side n**(1/d)
K = k_outer_from_ka(ka_from_chi(chi, 0.05, d), 0.05, d)  # exclusion radius
kset = enumerate_kvectors(np.asarray(box.basis), 0.0, K)  # half-space k, 0 < |k| < K
print("M =", kset.n.shape[0], "chi =", chi_from_count(kset.n.shape[0], n, d))
energy = make_stealthy_energy(box, kset)

s0 = jnp.asarray(np.random.default_rng(0).uniform(size=(8, n, d)))  # fractional
res = relax(energy, s0, DescentSettings(grow=2.0, max_evals=200_000, energy_tol=1e-20))
print("final energies:", np.asarray(res.energy))
```

Conventions: positions are **fractional** coordinates s in [0, 1) of the box
basis (rows = lattice vectors); the density is 1, so the box side is
N^(1/d) and V = N; Phi = (1/V) sum_k |rho(k)|^2 over the half-space k-set;
everything is float64 (float32 input raises `TypeError`).

## Where next

- `experiments/` (local only, not tracked by git): reproduction of Dawley &
  Torquato, PRE 112, 045310 (2025) trajectories, the Wasserstein analysis
  and the collective-coordinate studies; its tests live in
  `experiments/tests/`.
- [DEFECTS.md](DEFECTS.md): every defect found, with its regression test.
- [docs/library-evaluation.md](docs/library-evaluation.md): library choices.

Validated against the group's C++ code at kernel, trajectory and ensemble
level during development. Timings and the kernel
choice rule: [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Development

Use the environments from [Install](#install).

### Tests

```bash
export JAX_PLATFORMS=cpu
uv run pytest -m "not gpu"                       # dev profile
uv run pytest -m "not gpu" --hypothesis-profile=ci   # as in CI
uv run pytest --hypothesis-profile=thorough tests/property     # 5000 examples
uv run pytest -m "not gpu" --cov=xla_cco --cov-fail-under=90
```

The full suite with coverage needs more memory and time than a Della login
node allows (it is killed); there, run `-m "not gpu and not slow"` or one test
directory at a time, and leave the full run to CI or a compute node.

Hypothesis profiles (`tests/conftest.py`): `dev` (100 examples, default),
`ci` (300, derandomized), `thorough` (5000). Markers: `slow` (> 30 s), `gpu`.

#### Priority tiers

Every test carries exactly one tier marker; a test with none, or two, stops
collection with a usage error. `--priority=<tier>` runs that tier and every
higher one (`tests/tiers.py`):

| Tier | Covers | Tests | Cumulative CI time* |
|---|---|---|---|
| `critical` | Core numbers: energy and forces vs oracle and the C++ closed form, k-set enumeration and shell boundaries, χ/K conversions, minimizer agreement with the transcribed C++, batched `relax` (lane freezing, compaction order), ConfigPack round trip, CLI end to end and its success mask | 59 | 110 s |
| `high` | Secondary numerical paths and invariants: chunking, padding, phase tables, Hessian, gradient flow, S(k)/g₂/NN analysis, transport and exact 1D matching, trajectory tools, and regressions of defects that changed results | 157 | 717 s (404 s with `-m "not slow"`) |
| `medium` | Error contracts and input validation, dtypes, recording and status bookkeeping, file-format edge cases, the test oracles' own checks | 129 | 746 s |
| `low` | Tooling and messages: CLI help, GPU diagnostics, Hypothesis profiles, benchmark smoke, export list | 22 | 758 s (full suite) |

\* Hypothesis `ci` profile, 4 CPU cores on Della; GitHub runners are slower.
Tiers rank importance, not cost: eight `slow` Hypothesis sweeps (313 s) sit in
`high`. Combine the two when time matters, e.g.
`--priority high -m "not gpu and not slow"`.

```bash
uv run pytest -m "not gpu" --priority critical    # critical only
uv run pytest -m "not gpu" --priority high        # critical + high
uv run pytest -m "critical or high"               # same, as a marker expression
uv run pytest --priority high -m "not gpu and not slow"   # ~half the full time
```

CI (`.github/workflows/ci.yml`) picks the tiers per event: pushes to feature
branches run `critical` (or the repository variable `CI_PRIORITY`); pull requests
and pushes to `main` run every tier with the coverage gate, unless the pull
request carries a `ci:<tier>` label; manual runs (Actions → CI → Run
workflow) take the tier as an input. The coverage gate applies only when every
tier runs. New tests: mark the function, e.g. `@pytest.mark.high`, above its
other decorators.

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
sbatch slurm/test_gpu.slurm          # pytest on a GPU (.venv-cuda)
```
