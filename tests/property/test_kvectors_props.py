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
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ----- project -----
import oracle
from strategies import boxes
from xla_cco.kvectors import (
    chi_from_count,
    enumerate_kvectors,
    k_outer_from_ka,
    ka_from_chi,
    unit_ball_volume,
)

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


@pytest.mark.critical
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


@pytest.mark.critical
@settings(deadline=None)
@given(kset_cases())
def test_kset_complete_vs_oracle(case: tuple[np.ndarray, float, float]) -> None:
    basis, k1, k2 = case
    kset = enumerate_kvectors(basis, k1, k2)
    n_ref, ksq_ref = oracle.halfspace_kvectors(basis, k1, k2)
    assert _rows(kset.n) == _rows(n_ref)
    assert kset.n.shape == n_ref.shape
    np.testing.assert_allclose(np.sort(kset.k_sq), np.sort(ksq_ref), rtol=1e-13)


@pytest.mark.medium
@settings(deadline=None)
@given(kset_cases())
def test_sorted_by_k_sq(case: tuple[np.ndarray, float, float]) -> None:
    basis, k1, k2 = case
    kset = enumerate_kvectors(basis, k1, k2)
    keys = list(
        zip(kset.k_sq.tolist(), [tuple(r) for r in kset.n.tolist()], strict=True)
    )
    assert keys == sorted(keys)


@pytest.mark.high
@settings(deadline=None)
@given(boxes(), st.integers(1, 20), st.floats(1.0, 3.0, allow_nan=False))
def test_kset_vs_oracle_scaled_boxes(
    basis: np.ndarray, scale: int, mult: float
) -> None:
    """Large boxes (side up to ~40): scan bound must grow with |a_i|."""
    big = scale * basis
    k_outer = mult * 2.0 * math.pi / scale
    kset = enumerate_kvectors(big, 0.0, k_outer)
    n_ref, _ = oracle.halfspace_kvectors(big, 0.0, k_outer)
    assert _rows(kset.n) == _rows(n_ref)


# ============================================================
# CONVERSION PROPERTIES
# ============================================================

_PHI = st.floats(0.01, 0.4, allow_nan=False)
_CHI = st.floats(0.001, 0.5, allow_nan=False)
_DIM = st.sampled_from((1, 2, 3))


@pytest.mark.medium
@given(_DIM, st.floats(0.1, 10.0, allow_nan=False))
def test_unit_ball_volume_scaling_and_recursion(dim: int, r: float) -> None:
    ball = {1: 2.0 * r, 2: math.pi * r**2, 3: 4.0 * math.pi * r**3 / 3.0}
    np.testing.assert_allclose(unit_ball_volume(dim) * r**dim, ball[dim], rtol=1e-14)
    np.testing.assert_allclose(
        unit_ball_volume(dim + 2),
        2.0 * math.pi / (dim + 2) * unit_ball_volume(dim),
        rtol=1e-13,
    )


@pytest.mark.high
@given(st.floats(0.05, 5.0, allow_nan=False), _PHI, _DIM)
def test_k_outer_from_ka_inverts_a(ka: float, phi: float, dim: int) -> None:
    a = (phi * math.gamma(dim / 2.0 + 1.0) / math.pi ** (dim / 2.0)) ** (1.0 / dim)
    np.testing.assert_allclose(k_outer_from_ka(ka, phi, dim) * a, ka, rtol=1e-12)


@pytest.mark.high
@given(_CHI, _PHI, _DIM)
def test_ka_from_chi_formula_and_monotone(chi: float, phi: float, dim: int) -> None:
    ka = ka_from_chi(chi, phi, dim)
    expected = 2.0 ** (dim + 1) * dim * chi * math.gamma(dim / 2.0 + 1.0) ** 2 * phi
    np.testing.assert_allclose(ka**dim, expected, rtol=1e-12)
    assert ka_from_chi(chi * 1.5, phi, dim) > ka
    assert ka_from_chi(chi, phi * 1.5, dim) > ka


@pytest.mark.medium
@given(st.integers(0, 5000), st.integers(2, 10000), _DIM, st.integers(1, 7))
def test_chi_from_count_linear(m: int, n: int, dim: int, factor: int) -> None:
    chi = chi_from_count(m, n, dim)
    np.testing.assert_allclose(chi, m / (dim * (n - 1)), rtol=1e-15)
    np.testing.assert_allclose(
        chi_from_count(factor * m, n, dim), factor * chi, rtol=1e-14
    )


_CPP_BALL = {1: 2.0, 2: math.pi, 3: 4.1887902047863909846, 4: 4.9348022005446793094}


@pytest.mark.high
@given(st.integers(1, 8))
def test_unit_ball_volume_closed_form_and_cpp_constants(dim: int) -> None:
    closed = math.pi ** (dim / 2.0) / math.gamma(dim / 2.0 + 1.0)
    np.testing.assert_allclose(unit_ball_volume(dim), closed, rtol=1e-15)
    if dim in _CPP_BALL:
        assert unit_ball_volume(dim) == _CPP_BALL[dim]
