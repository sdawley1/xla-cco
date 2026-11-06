#!/usr/bin/env python3

"""
Example tests of the phase-table fast path and the auto chunk-size helper.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses

# ----- numerics -----
import jax.numpy as jnp
import numpy as np
import pytest

# ----- project -----
import oracle
from systems import small_system, square_lattice
from tolerances import C_E, RTOL_KERNEL_E, RTOL_KERNEL_F
from xla_cco.box import cubic_box
from xla_cco.energy import (
    StealthyEnergy,
    _phase_block,
    _power_table,
    auto_chunk_size,
    make_stealthy_energy,
)
from xla_cco.kvectors import enumerate_kvectors

# ============================================================
# TESTS
# ============================================================


def test_factory_sets_n_max_and_baseline_has_none() -> None:
    _, kset, en = small_system(12, 2, 4.0)
    assert en.n_max == int(np.abs(kset.n).max())
    assert StealthyEnergy(en.n_vectors, en.weights, en.volume).n_max is None


@pytest.mark.parametrize("dim,n,k_out", [(1, 9, 5.0), (2, 12, 4.0), (3, 10, 3.5)])
@pytest.mark.parametrize("chunk", [None, 1, 7, 64])
def test_tables_match_oracle_and_trig_baseline(
    dim: int, n: int, k_out: float, chunk: int | None
) -> None:
    box = cubic_box(n, dim)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k_out)
    en = make_stealthy_energy(box, kset, chunk_size=chunk)
    base = dataclasses.replace(en, n_max=None)
    s = np.random.default_rng(3).uniform(size=(n, dim))
    val, grad = en.value_and_grad(jnp.asarray(s))
    bval, bgrad = base.value_and_grad(jnp.asarray(s))
    ref = oracle.energy(s, np.asarray(en.n_vectors), np.asarray(en.weights), n)
    np.testing.assert_allclose(float(val), ref, rtol=RTOL_KERNEL_E)
    np.testing.assert_allclose(float(val), float(bval), rtol=RTOL_KERNEL_E)
    scale = max(1.0, float(np.abs(bgrad).max()))
    np.testing.assert_allclose(
        np.asarray(grad), np.asarray(bgrad), rtol=RTOL_KERNEL_F, atol=1e-11 * scale
    )


def test_tables_near_ordered_configuration() -> None:
    """A lattice (E ~ 0, cancellation-dominated) matches the trig path."""
    s = square_lattice(8) + 1e-9 * np.random.default_rng(0).normal(size=(64, 2))
    box = cubic_box(64, 2)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, 2.0 * np.pi * 7.5 / 8.0)
    en = make_stealthy_energy(box, kset)
    base = dataclasses.replace(en, n_max=None)
    v, g = en.value_and_grad(jnp.asarray(s))
    bv, bg = base.value_and_grad(jnp.asarray(s))
    m_real = int(kset.n.shape[0])
    floor = C_E * np.finfo(np.float64).eps * np.sqrt(64 * m_real * float(bv))
    assert abs(float(v) - float(bv)) <= floor + RTOL_KERNEL_E * float(bv)
    np.testing.assert_allclose(np.asarray(g), np.asarray(bg), atol=1e-12)


def test_power_table_equals_exponential() -> None:
    s = jnp.asarray(np.random.default_rng(1).uniform(size=(6, 3)))
    table = np.asarray(_power_table(s, 5))
    assert table.shape == (6, 6, 3)
    for p in range(6):
        np.testing.assert_allclose(
            table[p], np.exp(2j * np.pi * p * np.asarray(s)), atol=1e-14
        )


def test_phase_block_handles_negative_indices() -> None:
    s = jnp.asarray(np.random.default_rng(2).uniform(size=(5, 2)))
    n = jnp.asarray([[2.0, -3.0], [-1.0, 0.0], [0.0, 0.0]])
    ph = np.asarray(_phase_block(_power_table(s, 3), n))
    want = np.exp(2j * np.pi * (np.asarray(n) @ np.asarray(s).T))
    np.testing.assert_allclose(ph, want, atol=1e-14)


def test_underbounded_n_max_is_nan_not_wrong() -> None:
    _, _, en = small_system(12, 2, 4.0)
    assert en.n_max is not None and en.n_max >= 2
    bad = dataclasses.replace(en, n_max=en.n_max - 1)
    s = jnp.asarray(np.random.default_rng(5).uniform(size=(12, 2)))
    val, grad = bad.value_and_grad(s)
    assert np.isnan(float(val)) and np.all(np.isnan(np.asarray(grad)))


def test_auto_chunk_size_examples() -> None:
    assert auto_chunk_size(100, 128, budget_bytes=1 << 30) is None
    assert auto_chunk_size(10_000, 9_024, budget_bytes=1 << 30) == 1118
    assert auto_chunk_size(10_000, 9_024, budget_bytes=1, bytes_per_pair=1) == 1
    with pytest.raises(AssertionError):
        auto_chunk_size(0, 10)
