#!/usr/bin/env python3

"""
Tests for the closed-form StealthyEnergy.hessian against autodiff.

Oracle: ``AbstractEnergy.hessian`` (``jax.hessian`` of ``__call__``), which
differentiates the plain-JAX energy and shares no code with the closed form.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ----- project -----
from tests.strategies import fractional_configs, integer_kvectors
from xla_cco.box import cubic_box
from xla_cco.energy import AbstractEnergy, StealthyEnergy, make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors

# ============================================================
# TESTS
# ============================================================


@settings(deadline=None, max_examples=40)
@given(data=st.data())
def test_closed_form_hessian_matches_autodiff(data: st.DataObject) -> None:
    """Random integer k-sets, weights and positions in d = 1, 2, 3."""
    n_vec = data.draw(integer_kvectors(max_m=12, max_abs=3))
    d = n_vec.shape[1]
    s = data.draw(fractional_configs(n_grid=(1, 2, 5), dim=d))
    w = data.draw(
        st.lists(st.floats(0.0, 2.0), min_size=n_vec.shape[0], max_size=n_vec.shape[0])
    )
    energy = StealthyEnergy(
        n_vectors=jnp.asarray(n_vec, jnp.float64),
        weights=jnp.asarray(w, jnp.float64),
        volume=jnp.asarray(data.draw(st.floats(0.5, 20.0))),
    )
    got = np.asarray(energy.hessian(jnp.asarray(s)))
    ref = np.asarray(AbstractEnergy.hessian(energy, jnp.asarray(s)))
    scale = max(1.0, float(np.max(np.abs(ref))))
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-11 * scale)


def test_hessian_symmetric_with_padding_and_tables() -> None:
    """Factory energy (padded k-set, phase-table path): symmetric, = autodiff."""
    box = cubic_box(12, 2)
    basis = np.asarray(box.basis)
    kset = enumerate_kvectors(basis, 0.0, 2 * np.pi / basis[0, 0] * 2.5)
    energy = make_stealthy_energy(box, kset)
    s = jnp.asarray(np.random.default_rng(3).uniform(size=(12, 2)))
    h = np.asarray(energy.hessian(s))
    assert h.shape == (24, 24)
    np.testing.assert_allclose(h, h.T, atol=1e-12)
    ref = np.asarray(AbstractEnergy.hessian(energy, s))
    np.testing.assert_allclose(h, ref, rtol=1e-10, atol=1e-10)


def test_hessian_rejects_float32() -> None:
    """float32 positions raise TypeError, like the energy itself."""
    energy = StealthyEnergy(
        n_vectors=jnp.ones((1, 1)), weights=jnp.ones(1), volume=jnp.asarray(1.0)
    )
    with pytest.raises(TypeError):
        energy.hessian(jnp.zeros((2, 1), jnp.float32))
