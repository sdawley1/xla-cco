#!/usr/bin/env python3

"""
Example tests of the periodic Box and wrap_fractional.

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

# ----- project -----
from xla_cco.box import Box, cubic_box, wrap_fractional

# ============================================================
# TESTS
# ============================================================


def test_cubic_box_side_and_volume() -> None:
    box = cubic_box(100, 3)
    side = 100.0 ** (1.0 / 3.0)
    np.testing.assert_allclose(np.asarray(box.basis), side * np.eye(3), rtol=1e-15)
    assert box.dim == 3
    np.testing.assert_allclose(float(box.volume), 100.0, rtol=1e-13)
    assert box.basis.dtype == jnp.float64


def test_cubic_box_1d_is_length_n() -> None:
    np.testing.assert_allclose(np.asarray(cubic_box(7, 1).basis), [[7.0]], rtol=1e-15)


def test_known_sheared_box_reciprocal_and_maps() -> None:
    basis = np.array([[2.0, 0.0], [1.0, 1.0]])
    box = Box(jnp.asarray(basis))
    expected = 2.0 * np.pi * np.array([[0.5, -0.5], [0.0, 1.0]])
    np.testing.assert_allclose(np.asarray(box.reciprocal()), expected, rtol=1e-14)
    s = jnp.array([[0.5, 0.25]])
    np.testing.assert_allclose(np.asarray(box.to_cartesian(s)), [[1.25, 0.25]])
    np.testing.assert_allclose(
        np.asarray(box.to_fractional(jnp.array([[1.25, 0.25]]))), [[0.5, 0.25]]
    )
    np.testing.assert_allclose(float(box.volume), 2.0)


def test_box_methods_work_under_jit_and_vmap() -> None:
    box = cubic_box(8, 2)
    s = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    jitted = jax.jit(lambda b, x: b.to_fractional(b.to_cartesian(x)))
    np.testing.assert_allclose(np.asarray(jitted(box, s)), np.asarray(s), atol=1e-15)
    batched = jax.vmap(box.to_cartesian)(jnp.stack([s, s + 0.1]))
    assert batched.shape == (2, 2, 2)
    assert jax.jit(lambda b: b.reciprocal())(box).shape == (2, 2)


def test_wrap_fractional_pinned_values() -> None:
    below_one = 1.0 - 2.0**-53
    out = np.asarray(
        wrap_fractional(jnp.array([[below_one, -0.0, -1e-300, 5.0, -0.25, 1.0]]))
    )
    assert np.all((out >= 0.0) & (out < 1.0))
    np.testing.assert_array_equal(out[0, [0, 1, 3, 5]], [below_one, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[0, 4], 0.75)
    assert out[0, 2] == below_one  # s - floor(s) rounds to 1.0; clamped below 1
