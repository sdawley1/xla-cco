#!/usr/bin/env python3

"""
Tests for xla_cco.analysis.transport (periodic Wasserstein-2 via Sinkhorn).

Oracles: brute-force 5^d image scan for the cost matrix and SciPy's exact
linear_sum_assignment for the optimal matching.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import itertools

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from scipy.optimize import linear_sum_assignment

# ----- project -----
from tests.strategies import boxes, fractional_configs
from xla_cco.analysis.transport import (
    entropic_distance,
    matching_distance,
    repair_matching,
    required_images,
    round_to_matching,
    sinkhorn,
    sinkhorn_annealed,
    torus_sq_cost,
)

# ============================================================
# HELPERS
# ============================================================

EPS_SCHEDULE = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)


def _oracle_cost(
    a: np.ndarray, b: np.ndarray, basis: np.ndarray, radius: int = 5
) -> np.ndarray:
    """Brute-force squared periodic distances over images in [-radius, radius]^d.

    No rounding: raw differences in (-1, 1) plus every translate in range.
    """
    n, d = a.shape
    out = np.full((n, n), np.inf)
    for image in itertools.product(range(-radius, radius + 1), repeat=d):
        r = (a[:, None, :] - b[None, :, :] + np.asarray(image, float)) @ basis
        out = np.minimum(out, np.sum(r * r, axis=-1))
    return out


def _exact_w2(cost: np.ndarray, density: float) -> float:
    """Exact rho^{1/2} [min_sigma sum C]^{1/2} via the Hungarian algorithm."""
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(density * cost[rows, cols].sum()))


def _jittered_lattice(n_side: int, noise: float, seed: int) -> np.ndarray:
    """2-D square lattice in fractional coordinates with uniform jitter."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side)), -1) + 0.5
    s = g.reshape(-1, 2) / n_side
    return np.asarray(np.mod(s + rng.uniform(-noise, noise, s.shape) / n_side, 1.0))


# ============================================================
# COST MATRIX
# ============================================================


@settings(deadline=None)
@given(data=st.data())
def test_cost_matches_bruteforce_images(data: st.DataObject) -> None:
    """Minimum-image cost equals a 5^d brute-force scan, sheared cells included."""
    basis = data.draw(boxes())
    d = basis.shape[0]
    a = data.draw(fractional_configs(n_grid=(1, 2, 7), dim=d))
    b = data.draw(fractional_configs(n_grid=(a.shape[0],), dim=d))
    got = np.asarray(torus_sq_cost(jnp.asarray(a), jnp.asarray(b), jnp.asarray(basis)))
    np.testing.assert_allclose(got, _oracle_cost(a, b, basis), rtol=1e-12, atol=1e-14)


@settings(deadline=None)
@given(data=st.data())
def test_cost_single_image_exact_for_orthogonal_cells(data: st.DataObject) -> None:
    """images=False equals the full image scan when the cell is orthogonal."""
    d = data.draw(st.sampled_from((1, 2, 3)))
    sides = data.draw(st.lists(st.floats(0.5, 3.0), min_size=d, max_size=d))
    basis = jnp.asarray(np.diag(sides))
    a = jnp.asarray(data.draw(fractional_configs(n_grid=(1, 7), dim=d)))
    b = jnp.asarray(data.draw(fractional_configs(n_grid=(a.shape[0],), dim=d)))
    np.testing.assert_allclose(
        torus_sq_cost(a, b, basis, images=False),
        torus_sq_cost(a, b, basis),
        rtol=1e-12,
        atol=1e-14,
    )


@settings(deadline=None, max_examples=30)
@given(
    shear=st.integers(-6, 6),
    stretch=st.floats(0.3, 3.0),
    seed=st.integers(0, 2**31 - 1),
)
@example(shear=5, stretch=1.0, seed=0)
def test_regression_D013_cost_exact_for_unreduced_cells(
    shear: int, stretch: float, seed: int
) -> None:
    """Unreduced (strongly sheared) 2-D cells: the default scan is exact.

    With basis [[1, 0], [5, 1]] the old fixed 3^d scan got 36 % of entries
    wrong by up to 2.0 (D-013).
    """
    basis = np.array([[stretch, 0.0], [float(shear), 1.0]])
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(size=(9, 2)), rng.uniform(size=(9, 2))
    radius = max(required_images(basis)) + 3
    got = np.asarray(torus_sq_cost(jnp.asarray(a), jnp.asarray(b), jnp.asarray(basis)))
    np.testing.assert_allclose(
        got, _oracle_cost(a, b, basis, radius), rtol=1e-12, atol=1e-12
    )


def test_cost_rejects_too_few_images() -> None:
    """An explicit radius below required_images raises for a concrete basis."""
    basis = jnp.asarray([[1.0, 0.0], [5.0, 1.0]])
    a = jnp.zeros((3, 2))
    with pytest.raises(ValueError, match="too small"):
        torus_sq_cost(a, a, basis, images=1)


def test_cost_needs_explicit_images_under_jit() -> None:
    """images=None cannot size the scan from a traced basis."""
    a = jnp.zeros((3, 2))
    with pytest.raises(ValueError, match="concrete basis"):
        jax.jit(lambda bas: torus_sq_cost(a, a, bas))(jnp.eye(2))
    out = jax.jit(lambda bas: torus_sq_cost(a, a, bas, images=0))(jnp.eye(2))
    assert out.shape == (3, 3)


def test_required_images_orthogonal_is_zero() -> None:
    """Rounding alone is exact for diagonal cells."""
    assert required_images(np.diag([1.0, 2.0, 0.5])) == (0, 0, 0)
    assert required_images(np.array([[1.0, 0.0], [5.0, 1.0]]))[0] >= 5


def test_cost_periodic_example() -> None:
    """Points 0.05 and 0.95 of a unit 1-D cell are 0.1 apart (squared 0.01)."""
    a = jnp.asarray([[0.05]])
    b = jnp.asarray([[0.95]])
    got = float(torus_sq_cost(a, b, jnp.eye(1))[0, 0])
    assert got == pytest.approx(0.01, rel=1e-12)


# ============================================================
# SINKHORN
# ============================================================


def test_sinkhorn_marginals_converge() -> None:
    """Row and column sums of the converged plan are 1/N."""
    rng = np.random.default_rng(3)
    cost = jnp.asarray(rng.uniform(0.0, 1.0, (16, 16)))
    res = sinkhorn(cost, 0.05, max_iters=5000, tol=1e-12)
    plan = np.exp((np.asarray(res.f)[:, None] + np.asarray(res.g) - cost) / 0.05)
    np.testing.assert_allclose(plan.sum(axis=1), 1 / 16, atol=1e-11)
    np.testing.assert_allclose(plan.sum(axis=0), 1 / 16, atol=1e-11)
    assert float(res.marginal_error) < 1e-12 and int(res.n_iters) < 5000


def test_sinkhorn_rejects_bad_arguments() -> None:
    """Non-square cost and non-positive eps are refused."""
    with pytest.raises(AssertionError):
        sinkhorn(jnp.ones((2, 3)), 0.1)
    with pytest.raises(AssertionError):
        sinkhorn(jnp.ones((2, 2)), 0.0)


@settings(deadline=None, max_examples=30)
@given(
    perm_seed=st.integers(0, 2**31 - 1),
    shift=st.floats(-0.2, 0.2, allow_nan=False),
)
def test_relabelled_pattern_has_zero_distance(perm_seed: int, shift: float) -> None:
    """b = permuted, rigidly shifted copy of a: matching distance = N^{1/2}|shift|."""
    a = _jittered_lattice(5, 0.3, seed=1)
    perm = np.random.default_rng(perm_seed).permutation(a.shape[0])
    b = np.mod(a[perm] + np.array([shift, 0.0]) / 5.0, 1.0)
    cost = torus_sq_cost(jnp.asarray(a), jnp.asarray(b), jnp.eye(2) * 5.0)
    res = sinkhorn_annealed(cost, EPS_SCHEDULE, max_iters=3000)
    sigma, is_perm = round_to_matching(cost, res.g)
    assert bool(is_perm)
    np.testing.assert_array_equal(perm[np.asarray(sigma)], np.arange(a.shape[0]))
    w = float(matching_distance(cost, sigma, 1.0))
    assert w == pytest.approx(5.0 * abs(shift), abs=1e-12)


@settings(deadline=None, max_examples=25)
@given(seed=st.integers(0, 2**31 - 1), noise=st.floats(0.05, 0.6))
def test_rounded_matching_is_optimal_and_bounded(seed: int, noise: float) -> None:
    """Rounded Sinkhorn = exact W2 <= identity distance; entropic estimate close."""
    a = _jittered_lattice(5, 0.4, seed=seed)
    rng = np.random.default_rng(seed + 1)
    b = np.mod(a + rng.uniform(-noise, noise, a.shape) / 5.0, 1.0)
    basis = np.eye(2) * 5.0
    cost = torus_sq_cost(jnp.asarray(a), jnp.asarray(b), jnp.asarray(basis))
    res = sinkhorn_annealed(cost, EPS_SCHEDULE, max_iters=5000, tol=1e-10)
    sigma, is_perm = round_to_matching(cost, res.g)
    exact = _exact_w2(np.asarray(cost), 1.0)
    identity = float(np.sqrt(np.trace(np.asarray(cost))))
    assert exact <= identity + 1e-12
    if bool(is_perm):
        w = float(matching_distance(cost, sigma, 1.0))
        assert w >= exact - 1e-12
        assert w == pytest.approx(exact, rel=1e-3)
    ent = float(entropic_distance(cost, res, EPS_SCHEDULE[-1], 1.0))
    assert ent == pytest.approx(exact, rel=0.05, abs=0.05)


def test_round_to_matching_detects_collision() -> None:
    """Two rows sharing one argmax column are reported as a non-bijection."""
    cost = jnp.asarray([[0.0, 5.0], [0.0, 5.0]])
    sigma, is_perm = round_to_matching(cost, jnp.zeros(2))
    assert not bool(is_perm)
    assert np.asarray(sigma).tolist() == [0, 0]


def test_matching_distance_example() -> None:
    """Identity on a diagonal cost returns rho^{1/2} sqrt(trace)."""
    cost = jnp.diag(jnp.asarray([1.0, 4.0, 4.0]))
    w = matching_distance(cost, jnp.arange(3), 4.0)
    assert float(w) == pytest.approx(np.sqrt(4.0 * 9.0), rel=1e-14)


# ============================================================
# REPAIR
# ============================================================


@settings(deadline=None, max_examples=40)
@given(
    seed=st.integers(0, 2**31 - 1),
    n=st.sampled_from((1, 2, 5, 16)),
)
def test_repair_returns_bijection(seed: int, n: int) -> None:
    """Any map (collisions, ties) is repaired into a permutation."""
    rng = np.random.default_rng(seed)
    cost = jnp.asarray(rng.integers(0, 3, (n, n)).astype(float))
    g = jnp.asarray(rng.integers(0, 2, n).astype(float))
    sigma = jnp.asarray(rng.integers(0, n, n))
    out = np.asarray(repair_matching(cost, g, sigma))
    assert sorted(out.tolist()) == list(range(n))


@settings(deadline=None, max_examples=20)
@given(seed=st.integers(0, 2**31 - 1))
def test_repair_keeps_bijection_and_bounds_exact(seed: int) -> None:
    """A permutation is returned unchanged; a repaired map costs >= the optimum."""
    rng = np.random.default_rng(seed)
    cost = jnp.asarray(rng.uniform(0.0, 1.0, (7, 7)))
    perm = jnp.asarray(rng.permutation(7))
    np.testing.assert_array_equal(repair_matching(cost, jnp.zeros(7), perm), perm)
    sigma, _ = round_to_matching(cost, jnp.zeros(7))
    fixed = repair_matching(cost, jnp.zeros(7), sigma)
    w = float(matching_distance(cost, fixed, 1.0))
    assert w >= _exact_w2(np.asarray(cost), 1.0) - 1e-12


def test_repair_collision_example() -> None:
    """Tied rows on one column: the lower row index keeps it."""
    cost = jnp.asarray([[0.0, 5.0], [0.0, 5.0]])
    out = repair_matching(cost, jnp.zeros(2), jnp.asarray([0, 0]))
    assert np.asarray(out).tolist() == [0, 1]


def test_stop_on_matching_stops_early_with_bijection() -> None:
    """The matching stop returns a bijective rounding in no more iterations."""
    a = _jittered_lattice(6, 0.4, seed=11)
    rng = np.random.default_rng(12)
    b = np.mod(a + rng.uniform(-0.3, 0.3, a.shape) / 6.0, 1.0)
    cost = torus_sq_cost(jnp.asarray(a), jnp.asarray(b), jnp.eye(2) * 6.0)
    full = sinkhorn(cost, 0.01, max_iters=5000, tol=1e-12)
    fast = sinkhorn(cost, 0.01, max_iters=5000, tol=1e-12, stop_on_matching=True)
    sigma, is_perm = round_to_matching(cost, fast.g)
    assert bool(is_perm) and int(fast.n_iters) <= int(full.n_iters)
    w = float(matching_distance(cost, sigma, 1.0))
    assert w == pytest.approx(_exact_w2(np.asarray(cost), 1.0), rel=1e-12)
