#!/usr/bin/env python3

"""
Example tests of k-set construction and the chi / cutoff conversions.

Reference numbers were recorded from a C++ run (engine.log) during development.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math

# ----- numerics -----
import numpy as np
import pytest

# ----- project -----
import oracle
from xla_cco.box import cubic_box
from xla_cco.kvectors import (
    chi_from_count,
    enumerate_kvectors,
    k_outer_from_ka,
    ka_from_chi,
    unit_ball_volume,
)

# ============================================================
# TESTS
# ============================================================


def test_unit_ball_volume_values() -> None:
    assert unit_ball_volume(1) == pytest.approx(2.0, rel=1e-15)
    assert unit_ball_volume(2) == pytest.approx(math.pi, rel=1e-15)
    assert unit_ball_volume(3) == pytest.approx(4.0 * math.pi / 3.0, rel=1e-15)


def test_ka_from_chi_matches_slurm() -> None:
    assert ka_from_chi(0.05, 0.05, 3) == pytest.approx(0.5963271031078479, rel=1e-14)


def test_k_outer_from_ka_reference() -> None:
    ka = ka_from_chi(0.05, 0.05, 3)
    a = (0.05 / unit_ball_volume(3)) ** (1.0 / 3.0)
    assert a == pytest.approx(0.228539, rel=1e-6)  # engine.log "a"
    assert k_outer_from_ka(ka, 0.05, 3) == pytest.approx(2.6093, rel=1e-4)


def test_chi_from_count() -> None:
    assert chi_from_count(13, 100, 3) == pytest.approx(13.0 / 297.0, rel=1e-15)


def test_kset_n100_d3_chi005() -> None:
    box = cubic_box(100, 3)
    k_outer = k_outer_from_ka(ka_from_chi(0.05, 0.05, 3), 0.05, 3)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k_outer)
    assert kset.n.shape == (13, 3)
    assert chi_from_count(kset.n.shape[0], 100, 3) == pytest.approx(0.043771, abs=5e-7)
    assert kset.n.dtype == np.int64 and kset.k.dtype == np.float64
    np.testing.assert_array_equal(kset.weights, np.ones(13))
    assert kset.k_inner == 0.0 and kset.k_outer == k_outer


def test_kset_fields_consistent_with_oracle_conventions() -> None:
    basis = np.array([[2.0, 0.0], [0.7, 1.5]])
    kset = enumerate_kvectors(basis, 0.5, 6.0)
    recip = oracle.reciprocal_basis(basis)
    np.testing.assert_allclose(kset.k, kset.n @ recip, rtol=1e-14)
    expected = np.array([oracle.modulus2(k) for k in kset.k])
    np.testing.assert_array_equal(kset.k_sq, expected)


@pytest.mark.parametrize("side", [1.0, 2.0, 3.7])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_kset_shell_boundary_ulp(side: float, dim: int) -> None:
    basis = side * np.eye(dim)
    unit = float(oracle.reciprocal_basis(basis)[0, 0])  # k of n = e_1
    assert unit * unit == oracle.modulus2(np.array([unit] + [0.0] * (dim - 1)))
    up = float(np.nextafter(unit, np.inf))
    down = float(np.nextafter(unit, 0.0))
    # outer cutoff exactly on the first shell: shell excluded; one ulp above: included
    assert enumerate_kvectors(basis, 0.0, unit).n.shape[0] == 0
    assert enumerate_kvectors(basis, 0.0, up).n.shape[0] == dim
    # inner cutoff exactly on the first shell: excluded; one ulp below: included
    assert enumerate_kvectors(basis, unit, 1.2 * unit).n.shape[0] == 0
    assert enumerate_kvectors(basis, down, 1.2 * unit).n.shape[0] == dim


def test_enumerate_rejects_invalid_input() -> None:
    eye = np.eye(2)
    with pytest.raises(ValueError):
        enumerate_kvectors(np.ones((2, 3)), 0.0, 5.0)  # not square
    with pytest.raises(ValueError):
        enumerate_kvectors(np.ones((2, 2)), 0.0, 5.0)  # singular
    with pytest.raises(ValueError):
        enumerate_kvectors(eye, 5.0, 5.0)  # k_inner >= k_outer
    with pytest.raises(ValueError):
        enumerate_kvectors(eye, -1.0, 5.0)  # negative k_inner
    with pytest.raises(ValueError):
        enumerate_kvectors(eye, 0.0, float("inf"))  # non-finite
    with pytest.raises(ValueError):
        enumerate_kvectors(np.array([[1.0, np.nan], [0.0, 1.0]]), 0.0, 5.0)


def _same_set(basis: np.ndarray, k_outer: float) -> None:
    kset = enumerate_kvectors(basis, 0.0, k_outer)
    n_ref, _ = oracle.halfspace_kvectors(basis, 0.0, k_outer)
    assert {tuple(r) for r in kset.n.tolist()} == {tuple(r) for r in n_ref.tolist()}
    assert kset.n.shape[0] == n_ref.shape[0] > 0


def test_kset_strongly_sheared_3d_matches_oracle() -> None:
    _same_set(np.array([[1.0, 0, 0], [8.0, 1, 0], [-6.0, 7, 1]]), 9.0)


@pytest.mark.parametrize("k_outer", [3.0, 5.0, 8.0])
def test_kset_scaled_sheared_3d_matches_oracle(k_outer: float) -> None:
    base = np.array([[1.0, 0, 0], [0.9, 1, 0], [0.7, 0.8, 1]])
    _same_set(4.0 * base, k_outer)


@pytest.mark.parametrize(("n", "dim"), [(100, 2), (1000, 3), (100, 3)])
def test_kset_real_cubic_boxes_match_oracle(n: int, dim: int) -> None:
    _same_set(np.asarray(cubic_box(n, dim).basis), 2.6093)
