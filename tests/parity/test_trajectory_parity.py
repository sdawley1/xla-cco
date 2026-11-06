#!/usr/bin/env python3

"""
LocalGradientDescent trajectory parity against the C++ goldens.

``adaptive_descent`` (grow 1, step 1e-3, rel_tol 0.01) must follow the C++
``RelaxStructure_LocalGradientDescent`` path step for step over the trustworthy
prefix P = min(horizon // 2, 200): identical accept bitstring and step sizes,
energies within ``TRAJ_RTOL_E`` and minimum-image positions within
``TRAJ_ATOL_S``.

Step-recording convention: ``Trajectory.step[m]`` is the step AFTER iteration
m's update (the recorder runs on the post-step state; record 0 holds
``step_init``). Hence the expected step is 1e-3 * 0.5**(rejects in 1..m), built
from the golden accept bits. Halving is exact in floating point.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional

# ----- numerics -----
import jax.numpy as jnp
import numpy as np
import pytest

# ----- project -----
import golden
from tolerances import TRAJ_ATOL_S, TRAJ_RTOL_E
from xla_cco.box import cubic_box
from xla_cco.energy import make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors
from xla_cco.minimize.descent import DescentSettings, adaptive_descent

# ============================================================
# CONSTANTS
# ============================================================

FILES = ("traj_d2_n64_chi0.25", "traj_d3_n100_chi0.05")
N_CONFIGS = 5
REJECT_CONFIG = 4
MAX_PREFIX = 200
MIN_HORIZON = 20
STEP_INIT = 1e-3

CASES = [(f, b) for f in FILES for b in range(N_CONFIGS)]
IDS = [f"{f}-{b}" for f, b in CASES]

# ============================================================
# HELPERS
# ============================================================


class Report(NamedTuple):
    """Comparison of one config over iterations 1..p."""

    first_fail: Optional[str]
    max_ds: float
    max_rel_de: float
    accept_ok: bool
    step_ok: bool
    rejects: int


@lru_cache(maxsize=None)
def _golden(name: str) -> Dict[str, Any]:
    return golden.load_golden(name)


@lru_cache(maxsize=None)
def _energy(name: str):  # noqa: ANN202
    p = _golden(name)["params"]
    box = cubic_box(p["N"], p["d"])
    kset = enumerate_kvectors(
        np.asarray(box.basis),
        float.fromhex(p["k_inner_hex"]),
        float.fromhex(p["k_outer_hex"]),
    )
    return make_stealthy_energy(box, kset)


def prefix_length(horizon: int) -> int:
    """Trustworthy prefix: half the measured accept-stable horizon, capped."""
    return min(horizon // 2, MAX_PREFIX)


def expected_steps(accepted: np.ndarray) -> np.ndarray:
    """Step after iteration m (m = 1..len) from the C++ rule: halve on reject."""
    return STEP_INIT * 0.5 ** np.cumsum(~accepted)


@lru_cache(maxsize=None)
def _run(name: str, b: int, p: int) -> Any:
    g = _golden(name)
    settings = DescentSettings(
        grow=1.0, record_every=1, max_records=p + 1, max_evals=2 * p
    )
    res = adaptive_descent(_energy(name), jnp.asarray(g["s0"][b]), settings)
    return res.trajectory


def compare(name: str, b: int, lo: int, hi: int) -> Report:
    """Compare iterations lo..hi (1-based, inclusive) of config ``b``."""
    g = _golden(name)
    traj = _run(name, b, hi)
    assert int(traj.count) >= hi + 1, f"only {int(traj.count)} records for hi={hi}"
    s = np.asarray(traj.s)[: hi + 1]
    acc_g = g["accepted"][:hi, b].astype(bool)
    step_g = expected_steps(acc_g)
    acc_j = np.array([np.any(s[m] != s[m - 1]) for m in range(1, hi + 1)])
    step_j = np.asarray(traj.step)[1 : hi + 1]
    e_j = np.asarray(traj.energy)[1 : hi + 1]
    e_g = g["energy"][:hi, b]
    d = s[1:] - g["s"][:hi, b]
    ds = np.abs(d - np.round(d)).reshape(hi, -1).max(axis=1)
    rel = np.abs(e_j - e_g) / e_g
    fails: List[str] = []
    for m in range(lo, hi + 1):
        i = m - 1
        checks = (
            ("accept", acc_j[i] == acc_g[i], acc_j[i], acc_g[i]),
            ("step", step_j[i] == step_g[i], step_j[i], step_g[i]),
            ("energy", rel[i] <= TRAJ_RTOL_E, e_j[i], e_g[i]),
            ("position", ds[i] <= TRAJ_ATOL_S, ds[i], TRAJ_ATOL_S),
        )
        for q, ok, vj, vg in checks:
            if not ok:
                fails.append(f"m={m} {q}: jax={vj!r} golden/limit={vg!r}")
    sl = slice(lo - 1, hi)
    return Report(
        fails[0] if fails else None,
        float(ds[sl].max()),
        float(rel[sl].max()),
        bool(np.all(acc_j[sl] == acc_g[sl])),
        bool(np.all(step_j[sl] == step_g[sl])),
        int(np.sum(~acc_g[sl])),
    )


# ============================================================
# TESTS
# ============================================================


@pytest.mark.parametrize(("name", "b"), CASES, ids=IDS)
def test_trajectory_parity(name: str, b: int) -> None:
    """JAX descent follows the C++ path over m = 1..P (first divergence reported)."""
    p = prefix_length(int(_golden(name)["horizon"][b]))
    rep = compare(name, b, 1, p)
    assert rep.first_fail is None, f"{name}[{b}] P={p}: {rep.first_fail}"


@pytest.mark.parametrize("name", FILES)
def test_horizon_sane(name: str) -> None:
    """Every accept-stable horizon is long enough for the golden to be useful."""
    h = _golden(name)["horizon"]
    assert np.all(h >= MIN_HORIZON), f"{name}: horizons {h.tolist()}"


def test_reject_branch_exercised() -> None:
    """The compared prefixes contain a reject, so step halving is compared."""
    counts = []
    for name in FILES:
        g = _golden(name)
        p = prefix_length(int(g["horizon"][REJECT_CONFIG]))
        counts.append(int(np.sum(~g["accepted"][:p, REJECT_CONFIG].astype(bool))))
    assert sum(counts) >= 1, f"no rejects within the prefix: {counts}"


@pytest.mark.slow
@pytest.mark.parametrize(("name", "b"), CASES, ids=IDS)
def test_trajectory_parity_tail_informational(name: str, b: int) -> None:
    """Same criteria over m = P+1..200 (kept apart from the prefix test)."""
    p = prefix_length(int(_golden(name)["horizon"][b]))
    rep = compare(name, b, p + 1, MAX_PREFIX)
    assert rep.first_fail is None, f"{name}[{b}] tail P={p}: {rep.first_fail}"
