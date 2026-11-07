#!/usr/bin/env python3

"""
Tests for xla_cco.analysis.transport.periodic_assignment_1d (exact W_2 on a
circle).

Oracle: the Hungarian algorithm (scipy.optimize.linear_sum_assignment) on the
minimum-image squared cost matrix, which shares no code with the cyclic-shift
solver.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from scipy.optimize import linear_sum_assignment

# ----- project -----
from xla_cco.analysis.transport import periodic_assignment_1d

# ============================================================
# HELPERS
# ============================================================


def _min_image(d: np.ndarray, length: float) -> np.ndarray:
    return d - length * np.round(d / length)


def _hungarian(a: np.ndarray, b: np.ndarray, length: float) -> float:
    cost = _min_image(a[:, None] - b[None, :], length) ** 2
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum())


# ============================================================
# TESTS
# ============================================================


@pytest.mark.high
@settings(deadline=None, max_examples=300)
@given(
    n=st.integers(1, 14),
    length=st.floats(0.5, 50.0),
    seed=st.integers(0, 2**31 - 1),
    clump=st.booleans(),
)
@example(n=3, length=3.0, seed=0, clump=True)
def test_matches_hungarian(n: int, length: float, seed: int, clump: bool) -> None:
    """Optimal cost equals the Hungarian optimum; the matching is consistent."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2 * length, 2 * length, size=n)  # any reals: taken mod L
    b = rng.uniform(0.0, length, size=(3, n))
    if clump:  # near-ties: points packed into a small arc
        b[1] = rng.uniform(0.0, 0.05 * length, size=n)
        b[2] = a + rng.normal(scale=1e-3 * length, size=n)
    res = periodic_assignment_1d(a, b, length)
    for f in range(3):
        ref = _hungarian(a, b[f], length)
        assert float(res.cost[f]) == pytest.approx(
            ref, rel=1e-10, abs=1e-12 * length**2
        )
        perm = res.perm[f]
        assert sorted(perm.tolist()) == list(range(n))
        d = res.disp[f]
        assert np.all(np.abs(d) <= 0.5 * length * (1 + 1e-12))
        np.testing.assert_allclose(
            _min_image(b[f, perm] - a, length), d, atol=1e-9 * length
        )
        assert float(np.sum(d**2)) == pytest.approx(
            float(res.cost[f]), rel=1e-12, abs=1e-14
        )


@pytest.mark.medium
def test_identical_configurations_give_identity() -> None:
    """W_2(a, a) = 0 with the identity matching (distinct points)."""
    a = np.array([0.1, 2.5, 0.7, 3.9])
    res = periodic_assignment_1d(a, a[None, :], 4.0)
    assert float(res.cost[0]) == 0.0
    np.testing.assert_array_equal(res.perm[0], np.arange(4))


@pytest.mark.high
def test_invariant_under_relabelling_and_translation() -> None:
    """Permuting b leaves the cost; a common translation leaves cost and pairs."""
    rng = np.random.default_rng(3)
    a, b, length = rng.uniform(0, 7, 20), rng.uniform(0, 7, (1, 20)), 7.0
    base = periodic_assignment_1d(a, b, length)
    p = rng.permutation(20)
    perm_b = periodic_assignment_1d(a, b[:, p], length)
    assert float(perm_b.cost[0]) == pytest.approx(float(base.cost[0]), rel=1e-12)
    shifted = periodic_assignment_1d(a + 2.3, b + 2.3, length)
    assert float(shifted.cost[0]) == pytest.approx(float(base.cost[0]), rel=1e-10)
    np.testing.assert_allclose(shifted.disp[0], base.disp[0], atol=1e-10)


@pytest.mark.high
def test_symmetric_in_its_arguments() -> None:
    """W_2(a, b) = W_2(b, a), with inverse matchings."""
    rng = np.random.default_rng(4)
    a, b, length = rng.uniform(0, 10, 30), rng.uniform(0, 10, 30), 10.0
    ab = periodic_assignment_1d(a, b[None], length)
    ba = periodic_assignment_1d(b, a[None], length)
    assert float(ab.cost[0]) == pytest.approx(float(ba.cost[0]), rel=1e-12)
    np.testing.assert_array_equal(ba.perm[0][ab.perm[0]], np.arange(30))


@pytest.mark.high
def test_large_n_matches_hungarian() -> None:
    """N = 300, frames near a lattice (the stealthy regime) and random ones."""
    rng = np.random.default_rng(5)
    n, length = 300, 300.0
    a = rng.uniform(0, length, n)
    lattice = (np.arange(n) + 0.5 + rng.normal(scale=0.2, size=n)) % length
    b = np.stack([lattice, rng.uniform(0, length, n)])
    res = periodic_assignment_1d(a, b, length)
    for f in range(2):
        assert float(res.cost[f]) == pytest.approx(
            _hungarian(a, b[f], length), rel=1e-10
        )


@pytest.mark.medium
def test_rejects_bad_inputs() -> None:
    """Shape mismatches, a non-positive period and non-finite values raise."""
    with pytest.raises(ValueError):
        periodic_assignment_1d(np.zeros(3), np.zeros((2, 4)), 1.0)
    with pytest.raises(ValueError):
        periodic_assignment_1d(np.zeros(3), np.zeros((2, 3)), 0.0)
    with pytest.raises(ValueError):
        periodic_assignment_1d(np.array([0.0, np.nan]), np.zeros((1, 2)), 1.0)


@pytest.mark.high
@settings(deadline=None, max_examples=5)
@given(seed=st.integers(0, 2**31 - 1), shells=st.integers(2, 10))
def test_1d_stealthy_flow_keeps_the_identity_matching(seed: int, shells: int) -> None:
    """In 1D, dPhi/dx_i depends on i only through x_i: all particles move in one
    velocity field, so trajectories cannot cross and the optimal matching to
    r(0) stays the identity (W_0 = p_0) along the gradient flow."""
    import jax.numpy as jnp

    from xla_cco.box import cubic_box
    from xla_cco.energy import make_stealthy_energy
    from xla_cco.kvectors import enumerate_kvectors
    from xla_cco.minimize.flow import FlowSettings, gradient_flow

    n = 24
    box = cubic_box(n, 1)
    basis = np.asarray(box.basis)
    kset = enumerate_kvectors(basis, 0.0, 2 * np.pi / n * (shells + 0.5))
    energy = make_stealthy_energy(box, kset)
    s0 = jnp.asarray(np.random.default_rng(seed).uniform(size=(n, 1)))
    fs = FlowSettings(energy_tol=1e-16, record_every=1, max_records=20_000)
    res = gradient_flow(energy, s0, jnp.asarray(basis), fs)
    cnt = int(res.trajectory.count)
    x = np.asarray(res.trajectory.s[:cnt, :, 0]) * n
    match = periodic_assignment_1d(x[0], x, float(n))
    np.testing.assert_array_equal(
        match.perm, np.broadcast_to(np.arange(n), match.perm.shape)
    )
    p0 = np.linalg.norm(_min_image(x - x[0], float(n)), axis=1)
    np.testing.assert_allclose(np.sqrt(match.cost), p0, rtol=1e-12, atol=1e-12)
