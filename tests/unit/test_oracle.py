#!/usr/bin/env python3

"""
Self-tests of the brute-force oracles in tests/oracle.py.

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
from hypothesis import given, settings
from hypothesis import strategies as st

# ----- project -----
import oracle

# ============================================================
# TESTS
# ============================================================


def test_energy_single_particle_is_sum_weights_over_volume() -> None:
    n_vec = np.array([[1, 0], [1, 2], [0, 1]], dtype=np.int64)
    w = np.array([1.0, 1.0, 2.0])
    s = np.array([[0.31, 0.77]])
    assert math.isclose(oracle.energy(s, n_vec, w, 1.0), w.sum(), rel_tol=1e-14)


@settings(deadline=None, max_examples=30)
@given(
    st.integers(1, 6),
    st.integers(0, 2**31),
)
def test_energy_n1_property(m: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_vec = rng.integers(-3, 4, size=(m, 3))
    w = rng.uniform(0.5, 2.0, size=m)
    s = rng.uniform(size=(1, 3))
    np.testing.assert_allclose(oracle.energy(s, n_vec, w, 1.0), w.sum(), rtol=1e-12)


def test_energy_translation_invariant() -> None:
    rng = np.random.default_rng(1)
    s = rng.uniform(size=(5, 2))
    n_vec = np.array([[1, 0], [0, 1], [1, -1]])
    w = np.ones(3)
    e0 = oracle.energy(s, n_vec, w, 5.0)
    e1 = oracle.energy(s + np.array([0.2, 0.7]), n_vec, w, 5.0)
    np.testing.assert_allclose(e0, e1, rtol=1e-12)


@settings(deadline=None, max_examples=20)
@given(st.integers(0, 2**31))
def test_forces_match_finite_difference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    basis = np.array([[2.0, 0.0], [0.5, 1.5]])
    vol = abs(np.linalg.det(basis))
    s = rng.uniform(size=(4, 2))
    n_vec = np.array([[1, 0], [0, 1], [1, 1], [1, -2]])
    w = np.ones(4)
    f = oracle.cartesian_forces(s, n_vec, w, basis, vol)
    h = 1e-6
    binv = np.linalg.inv(basis)
    for i in range(2):
        for c in range(2):
            dr = np.zeros(2)
            dr[c] = h
            sp = s.copy()
            sm = s.copy()
            sp[i] += dr @ binv
            sm[i] -= dr @ binv
            fd = -(
                oracle.energy(sp, n_vec, w, vol) - oracle.energy(sm, n_vec, w, vol)
            ) / (2 * h)
            assert math.isclose(f[i, c], fd, rel_tol=1e-6, abs_tol=1e-6)


def test_halfspace_scan_has_no_pm_pairs() -> None:
    basis = np.array([[1.0, 0.0], [0.3, 1.1]])
    n, ksq = oracle.halfspace_kvectors(basis, 0.0, 12.0)
    assert n.shape[0] > 0 and ksq.shape == (n.shape[0],)
    seen = {tuple(r) for r in n.tolist()}
    assert len(seen) == n.shape[0]
    for r in seen:
        assert tuple(-c for c in r) not in seen
        assert tuple(c for c in r if c != 0)[0] > 0


def test_halfspace_scan_strict_bounds_and_count() -> None:
    basis = np.eye(2) * 2.0 * math.pi  # reciprocal = identity
    # shell |n|^2 = 1 sits on k_inner (excluded, strict); shell 2 is beyond 1.4
    n, _ = oracle.halfspace_kvectors(basis, 1.0, 1.4)
    assert n.shape[0] == 0
    n, _ = oracle.halfspace_kvectors(basis, 0.5, 1.5)
    # |n|^2 in {1, 2}: (1,0),(0,1),(1,1),(1,-1)
    assert n.shape[0] == 4


def test_nn_distance_two_points_unit_cube() -> None:
    s = np.array([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1]])
    d = oracle.nearest_neighbor_distances(s, np.eye(3))
    np.testing.assert_allclose(d, [0.2, 0.2], atol=1e-14)


def test_nn_distance_single_particle_sees_own_image() -> None:
    d = oracle.nearest_neighbor_distances(np.array([[0.3, 0.3]]), np.eye(2))
    np.testing.assert_allclose(d, [1.0])


def test_structure_factor_single_particle_is_one() -> None:
    sk = oracle.structure_factor(np.array([[0.2, 0.9]]), np.array([[1, 2], [3, 1]]))
    np.testing.assert_allclose(sk, [1.0, 1.0], rtol=1e-12)


# ----- boundary strictness (exactly representable: reciprocal = identity) -----


def _shell_counts(k_inner: float, k_outer: float) -> int:
    basis = np.eye(2) * 2.0 * math.pi
    n, _ = oracle.halfspace_kvectors(basis, k_inner, k_outer)
    return int(n.shape[0])


def test_k_outer_boundary_is_strict() -> None:
    up = float(np.nextafter(1.0, np.inf))
    assert _shell_counts(0.0, 1.0) == 0  # |n|^2 = 1 excluded at k_outer = 1
    assert _shell_counts(0.0, up) == 2  # (1,0), (0,1) enter just above
    up2 = float(np.nextafter(2.0, np.inf))
    assert _shell_counts(0.0, 2.0) == 4  # |n|^2 = 4 shell (2,0),(0,2) excluded
    assert _shell_counts(0.0, up2) == 6


def test_k_inner_boundary_is_strict() -> None:
    basis = np.eye(2) * 2.0 * math.pi
    n, ksq = oracle.halfspace_kvectors(basis, 1.0, 1.5)
    assert not np.any(ksq == 1.0)  # shell on k_inner excluded
    assert n.shape[0] == 2  # (1,1), (1,-1)
    below = float(np.nextafter(1.0, 0.0))
    assert _shell_counts(below, 1.5) == 4  # shell 1 enters just below k_inner


# ----- 1/N normalisation of S(n) -----


def test_structure_factor_two_particles() -> None:
    s = np.array([[0.0, 0.0], [0.5, 0.0]])
    sk = oracle.structure_factor(s, np.array([[1, 0], [2, 0]]))
    np.testing.assert_allclose(sk, [0.0, 2.0], atol=1e-12)


def test_structure_factor_three_particles_hand_value() -> None:
    # s = 0, 1/3, 2/3 in 1-D: n=1 -> rho = 0 -> S = 0; n=3 -> rho = 3 -> S = 3
    s = np.array([[0.0], [1.0 / 3.0], [2.0 / 3.0]])
    sk = oracle.structure_factor(s, np.array([[1], [3]]))
    np.testing.assert_allclose(sk, [0.0, 3.0], atol=1e-12)
    # s = 0, 0, 1/2, n=1 -> rho = 1 => S = 1/3
    s2 = np.array([[0.0], [0.0], [0.5]])
    np.testing.assert_allclose(
        oracle.structure_factor(s2, np.array([[1]])), [1.0 / 3.0], rtol=1e-12
    )


@settings(deadline=None, max_examples=50)
@given(st.sampled_from((1, 2, 7, 16)), st.integers(1, 3), st.integers(0, 2**31))
def test_structure_factor_matches_complex_exponential(
    n: int, d: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    s = rng.uniform(size=(n, d))
    n_vec = rng.integers(-3, 4, size=(5, d))
    rho = np.exp(2j * np.pi * (n_vec @ s.T)).sum(axis=1)
    np.testing.assert_allclose(
        oracle.structure_factor(s, n_vec), np.abs(rho) ** 2 / n, rtol=1e-10, atol=1e-12
    )
