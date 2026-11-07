#!/usr/bin/env python3

"""
Tests for the Sinkhorn certification and exact fallback in
experiments/scripts/analyze_wasserstein.py.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import sys
from pathlib import Path
from typing import Dict

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
from analyze_wasserstein import apply_fallback, certified  # noqa: E402

# ============================================================
# HELPERS
# ============================================================


def _sinkhorn(is_perm: np.ndarray, err: np.ndarray, lengths: list) -> Dict:
    """Minimal merged-Sinkhorn record with (frames, 2) diagnostics."""
    n = int(sum(lengths))
    assert is_perm.shape == err.shape == (n, 2), "shape mismatch"
    return {
        "is_perm": is_perm,
        "err": err,
        "tol": np.asarray(1e-4),
        "w_round": np.arange(2 * n, dtype=float).reshape(n, 2),
        "offsets": np.concatenate([[0], np.cumsum(lengths)]),
        "names": np.array([f"traj_{i:04d}" for i in range(len(lengths))]),
    }


# ============================================================
# TESTS
# ============================================================


def test_regression_D014_converged_collision_not_certified() -> None:
    """A converged plan (err < tol) whose rounding collides is not trusted.

    The greedy repair at such frames was off by up to 2.8e-3 relative at
    chi = 0.4 (D-014); only a bijective rounding certifies a frame.
    """
    is_perm = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    err = np.array([[1e-6, 1e-6], [1e-6, 1e-6], [0.5, 1e-6]])
    sk = _sinkhorn(is_perm, err, [3])
    np.testing.assert_array_equal(certified(sk), is_perm == 1.0)


def test_apply_fallback_substitutes_exact_values() -> None:
    """Uncertified entries take the fallback value; certified ones are kept."""
    is_perm = np.ones((5, 2))
    is_perm[1, 0] = is_perm[3, 1] = 0.0  # traj 0 frame 1 col 0; traj 1 frame 1 col 1
    sk = _sinkhorn(is_perm, np.zeros((5, 2)), [2, 3])
    fb = {
        "traj": np.array([0, 1]),
        "frame": np.array([1, 1]),
        "col": np.array([0, 1]),
        "w": np.array([-1.0, -2.0]),
        "names": sk["names"],
    }
    w = apply_fallback(sk, fb)
    expected = sk["w_round"].copy()
    expected[1, 0], expected[3, 1] = -1.0, -2.0
    np.testing.assert_array_equal(w, expected)


def test_apply_fallback_requires_full_coverage() -> None:
    """A missing fallback for an uncertified entry is an error, not a gap."""
    is_perm = np.ones((3, 2))
    is_perm[2, 1] = 0.0
    sk = _sinkhorn(is_perm, np.zeros((3, 2)), [3])
    empty = {
        "traj": np.empty(0, int),
        "frame": np.empty(0, int),
        "col": np.empty(0, int),
        "w": np.empty(0),
        "names": sk["names"],
    }
    with pytest.raises(SystemExit, match="lack an exact fallback"):
        apply_fallback(sk, empty)


@settings(deadline=None, max_examples=50)
@given(seed=st.integers(0, 2**31 - 1), n_traj=st.integers(1, 4))
def test_apply_fallback_covers_exactly_uncertified(seed: int, n_traj: int) -> None:
    """With a fallback for every uncertified entry, those and only those change."""
    rng = np.random.default_rng(seed)
    lengths = list(rng.integers(1, 6, n_traj))
    n = int(sum(lengths))
    is_perm = (rng.uniform(size=(n, 2)) > 0.3).astype(float)
    sk = _sinkhorn(is_perm, rng.uniform(size=(n, 2)), lengths)
    off = sk["offsets"]
    rows, cols = np.nonzero(is_perm != 1.0)
    traj = np.searchsorted(off, rows, side="right") - 1
    fb = {
        "traj": traj,
        "frame": rows - off[traj],
        "col": cols,
        "w": np.full(rows.size, -7.0),
        "names": sk["names"],
    }
    w = apply_fallback(sk, fb)
    changed = w != sk["w_round"]
    np.testing.assert_array_equal(changed, is_perm != 1.0)
