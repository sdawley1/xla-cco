#!/usr/bin/env python3

"""
Property tests of the half-space k-set enumeration.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

# ----- project -----
import oracle
from strategies import boxes
from xla_cco.kvectors import enumerate_kvectors

# ============================================================
# STRATEGIES
# ============================================================


@st.composite
def kset_cases(draw: st.DrawFn) -> tuple[np.ndarray, float, float]:
    """Draw (basis, k_inner, k_outer) with small scan boxes."""
    basis = draw(boxes())
    k_outer = draw(st.floats(0.5, 9.0, allow_nan=False))
    frac = draw(st.floats(0.0, 0.95, allow_nan=False))
    return basis, frac * k_outer, k_outer


# ============================================================
# TESTS
# ============================================================


def _rows(n: np.ndarray) -> set:
    return {tuple(int(c) for c in row) for row in n}


@settings(deadline=None)
@given(kset_cases())
def test_halfspace_rules(case: tuple[np.ndarray, float, float]) -> None:
    basis, k1, k2 = case
    kset = enumerate_kvectors(basis, k1, k2)
    rows = _rows(kset.n)
    assert len(rows) == kset.n.shape[0]  # no duplicates
    for row in rows:
        nz = [c for c in row if c != 0]
        assert nz and nz[0] > 0  # n != 0 and first nonzero positive
        assert tuple(-c for c in row) not in rows  # no +-n pair
    assert np.all(k1 * k1 < kset.k_sq) and np.all(kset.k_sq < k2 * k2)


@settings(deadline=None)
@given(kset_cases())
def test_kset_complete_vs_oracle(case: tuple[np.ndarray, float, float]) -> None:
    basis, k1, k2 = case
    kset = enumerate_kvectors(basis, k1, k2)
    n_ref, ksq_ref = oracle.halfspace_kvectors(basis, k1, k2)
    assert _rows(kset.n) == _rows(n_ref)
    assert kset.n.shape == n_ref.shape
    np.testing.assert_allclose(np.sort(kset.k_sq), np.sort(ksq_ref), rtol=1e-13)


@settings(deadline=None)
@given(kset_cases())
def test_sorted_by_k_sq(case: tuple[np.ndarray, float, float]) -> None:
    basis, k1, k2 = case
    kset = enumerate_kvectors(basis, k1, k2)
    keys = list(
        zip(kset.k_sq.tolist(), [tuple(r) for r in kset.n.tolist()], strict=True)
    )
    assert keys == sorted(keys)
