#!/usr/bin/env python3

"""
Property tests: phase-table fast path vs the oracle and the trig baseline.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ----- project -----
import oracle
from strategies import chunk_sizes, fractional_configs, integer_kvectors
from tolerances import GRAD_SUM_C, GRAD_SUM_FLOOR, RTOL_KERNEL_E, RTOL_KERNEL_F
from xla_cco.energy import StealthyEnergy, auto_chunk_size

# ============================================================
# HELPERS
# ============================================================

FAST = settings(deadline=None)


@st.composite
def systems(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Draw (s, n, w, V) with matching dimension; M is padded to a grid of 8."""
    s = draw(fractional_configs(n_grid=(1, 2, 7, 16)))
    n = draw(integer_kvectors(dim=s.shape[1], max_m=16))
    w = np.random.default_rng(draw(st.integers(0, 2**31))).uniform(0.5, 2.0, len(n))
    total = -(-n.shape[0] // 8) * 8
    n = np.vstack([n, np.zeros((total - n.shape[0], n.shape[1]), dtype=n.dtype)])
    w = np.concatenate([w, np.zeros(total - w.shape[0])])
    return s, n, w, draw(st.floats(0.5, 50.0, allow_nan=False))


def _pair(
    n: np.ndarray, w: np.ndarray, vol: float, chunk: int | None
) -> tuple[StealthyEnergy, StealthyEnergy]:
    """Return (table path, trig baseline) energies over the same k-set."""
    fast = StealthyEnergy(
        n_vectors=jnp.asarray(n, dtype=jnp.float64),
        weights=jnp.asarray(w),
        volume=jnp.asarray(vol),
        chunk_size=chunk,
        n_max=int(np.abs(n).max()),
    )
    return fast, dataclasses.replace(fast, n_max=None)


def _grad_atol(
    n: np.ndarray, w: np.ndarray, vol: float, n_part: int, ref: np.ndarray
) -> float:
    """Absolute gradient tolerance: relative to |grad|, floored by round-off.

    Each dPhi/ds_j sums terms up to (4 pi n_max / V) sum(w) N before they
    cancel (exactly, for coincident particles), so two summation orders differ
    by about eps times that bound however small the gradient is (D-011, D-017).
    """
    term = 4 * np.pi * float(np.abs(n).max()) * float(np.sum(w)) * n_part / vol
    roundoff = GRAD_SUM_C * float(np.finfo(np.float64).eps) * term + GRAD_SUM_FLOOR
    return max(1e-12 * max(1.0, float(np.abs(ref).max())), roundoff)


def _check_tables(case: tuple, chunk: int | None) -> None:
    """Phase-table energy and gradient equal the oracle and the trig baseline."""
    s, n, w, vol = case
    fast, base = _pair(n, w, vol, chunk)
    sj = jnp.asarray(s)
    val, grad = fast.value_and_grad(sj)
    bval, bgrad = base.value_and_grad(sj)
    ref = oracle.energy(s, n, w, vol)
    np.testing.assert_allclose(float(val), ref, rtol=RTOL_KERNEL_E, atol=1e-12)
    np.testing.assert_allclose(float(val), float(bval), rtol=RTOL_KERNEL_E, atol=1e-12)
    atol = _grad_atol(n, w, vol, s.shape[0], np.asarray(bgrad))
    np.testing.assert_allclose(
        np.asarray(grad), np.asarray(bgrad), rtol=RTOL_KERNEL_F, atol=atol
    )


# ============================================================
# TESTS
# ============================================================


@pytest.mark.high
@pytest.mark.slow
@FAST
@given(systems(), chunk_sizes())
def test_tables_match_oracle_and_baseline(case: tuple, chunk: int | None) -> None:
    _check_tables(case, chunk)


@pytest.mark.medium
def test_regression_D017_coincident_particles_gradient_tolerance() -> None:
    """16 coincident particles: grad = 0 exactly; paths differ by ~1e-12 round-off.

    CI (derandomized profile) found this case; the old atol of 1e-12 * max|grad|
    ignored the pre-cancellation size of the terms.
    """
    s = np.full((16, 2), 0.8125)
    n = np.array([[1, 1]] * 10 + [[0, 0]] * 6)
    w = np.array(
        [
            1.45544253,
            0.90468007,
            0.56146029,
            0.52479145,
            1.71990536,
            1.86913337,
            1.40995366,
            1.59424484,
            1.31543749,
            1.90260864,
        ]
        + [0.0] * 6
    )
    _check_tables((s, n, w, 1.0), 1)


@pytest.mark.high
@pytest.mark.slow
@FAST
@given(systems(), chunk_sizes())
def test_tables_gradient_matches_autodiff(case: tuple, chunk: int | None) -> None:
    s, n, w, vol = case
    fast, _ = _pair(n, w, vol, chunk)
    _, grad = fast.value_and_grad(jnp.asarray(s))
    rgrad = jax.grad(fast)(jnp.asarray(s))
    atol = _grad_atol(n, w, vol, s.shape[0], np.asarray(rgrad))
    np.testing.assert_allclose(
        np.asarray(grad), np.asarray(rgrad), rtol=RTOL_KERNEL_F, atol=atol
    )


@pytest.mark.medium
@given(
    st.integers(1, 20_000),
    st.integers(1, 20_000),
    st.integers(1, 2**34),
    st.integers(1, 512),
)
def test_auto_chunk_size_bounds(n: int, m: int, budget: int, per_pair: int) -> None:
    size = auto_chunk_size(n, m, budget, per_pair)
    if size is None:
        assert max(1, budget // (n * per_pair)) >= m
    else:
        assert 1 <= size < m
        assert size == 1 or size * n * per_pair <= budget
