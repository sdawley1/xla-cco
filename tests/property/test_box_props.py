#!/usr/bin/env python3

"""
Property tests of the periodic Box and wrap_fractional.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# ----- project -----
from strategies import boxes, fractional_configs
from xla_cco.box import Box, wrap_fractional

# ============================================================
# TESTS
# ============================================================


@pytest.mark.critical
@settings(deadline=None)
@given(boxes())
def test_reciprocal_duality(basis: np.ndarray) -> None:
    recip = np.asarray(Box(jnp.asarray(basis)).reciprocal())
    d = basis.shape[0]
    np.testing.assert_allclose(basis @ recip.T, 2.0 * np.pi * np.eye(d), atol=1e-12)


@pytest.mark.high
@settings(deadline=None)
@given(boxes(), st.data())
def test_frac_cart_round_trip(basis: np.ndarray, data: st.DataObject) -> None:
    box = Box(jnp.asarray(basis))
    s = data.draw(fractional_configs(dim=basis.shape[0]))
    back = np.asarray(box.to_fractional(box.to_cartesian(jnp.asarray(s))))
    np.testing.assert_allclose(back, s, atol=1e-12)


@pytest.mark.high
@settings(deadline=None)
@given(
    hnp.arrays(
        np.float64,
        (4, 3),
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    )
)
@example(
    np.array([[1.0 - 2.0**-53, -0.0, -1e-300], [5.0, 0.0, -1.0], [-2.5, 0.5, 1e-17]])
)
def test_wrap_idempotent_in_unit_interval(s: np.ndarray) -> None:
    once = wrap_fractional(jnp.asarray(s))
    out = np.asarray(once)
    assert np.all((out >= 0.0) & (out < 1.0))
    np.testing.assert_array_equal(np.asarray(wrap_fractional(once)), out)
    # wrapping moves a coordinate by an integer only (to within roundoff)
    shift = s - out
    np.testing.assert_allclose(shift, np.round(shift), atol=1e-12)


@pytest.mark.critical
@settings(deadline=None)
@given(
    boxes(),
    hnp.arrays(np.float64, (3, 3), elements=st.floats(-2, 2, allow_nan=False)),
    hnp.arrays(np.float64, 3, elements=st.floats(-2, 2, allow_nan=False)),
    st.data(),
)
def test_cartesian_forces_chain_rule(
    basis: np.ndarray, m: np.ndarray, b: np.ndarray, data: st.DataObject
) -> None:
    d = basis.shape[0]
    amat = m[:d, :d] @ m[:d, :d].T  # symmetric positive semi-definite
    bvec = b[:d]
    s = data.draw(fractional_configs(dim=d))
    box = Box(jnp.asarray(basis))

    def f(sf: jax.Array) -> jax.Array:
        r = sf @ box.basis
        return jnp.sum(0.5 * jnp.einsum("nd,de,ne->n", r, amat, r) + r @ bvec)

    grad_s = jax.grad(f)(jnp.asarray(s))
    r_np = s @ basis
    expected = -(r_np @ amat + bvec)  # -grad_r f, analytic (A symmetric)
    got = np.asarray(box.cartesian_forces(grad_s))
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)
