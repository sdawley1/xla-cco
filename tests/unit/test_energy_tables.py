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
import jax
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


def _tables(en: StealthyEnergy) -> StealthyEnergy:
    """Force the table path: n_max = max|n| of the energy's own k rows."""
    return dataclasses.replace(en, n_max=int(np.abs(np.asarray(en.n_vectors)).max()))


@pytest.mark.high
def test_factory_always_selects_trig_and_tables_stay_reachable() -> None:
    """R32: the factory never picks tables, at any M; direct construction can."""
    _, small_k, small = small_system(12, 2, 4.0)
    assert small.n_max is None
    box = cubic_box(50, 2)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, 12.0)
    assert kset.n.shape[0] >= 75  # a large k-set (the former table regime)
    big = make_stealthy_energy(box, kset)
    assert big.n_max is None
    forced = _tables(big)
    assert forced.n_max == int(np.abs(kset.n).max())


@pytest.mark.high
@pytest.mark.parametrize("dim,n,k_out", [(1, 9, 5.0), (2, 12, 4.0), (3, 10, 3.5)])
@pytest.mark.parametrize("chunk", [None, 1, 7, 64])
def test_tables_match_oracle_and_trig_baseline(
    dim: int, n: int, k_out: float, chunk: int | None
) -> None:
    box = cubic_box(n, dim)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k_out)
    en = _tables(make_stealthy_energy(box, kset, chunk_size=chunk))
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


@pytest.mark.high
def test_tables_near_ordered_configuration() -> None:
    """A lattice (E ~ 0, cancellation-dominated) matches the trig path."""
    s = square_lattice(8) + 1e-9 * np.random.default_rng(0).normal(size=(64, 2))
    box = cubic_box(64, 2)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, 2.0 * np.pi * 7.5 / 8.0)
    en = _tables(make_stealthy_energy(box, kset))
    base = dataclasses.replace(en, n_max=None)
    v, g = en.value_and_grad(jnp.asarray(s))
    bv, bg = base.value_and_grad(jnp.asarray(s))
    m_real = int(kset.n.shape[0])
    floor = C_E * np.finfo(np.float64).eps * np.sqrt(64 * m_real * float(bv))
    assert abs(float(v) - float(bv)) <= floor + RTOL_KERNEL_E * float(bv)
    np.testing.assert_allclose(np.asarray(g), np.asarray(bg), atol=1e-12)


@pytest.mark.high
def test_power_table_equals_exponential() -> None:
    s = jnp.asarray(np.random.default_rng(1).uniform(size=(6, 3)))
    table = np.asarray(_power_table(s, 5))
    assert table.shape == (6, 6, 3)
    for p in range(6):
        np.testing.assert_allclose(
            table[p], np.exp(2j * np.pi * p * np.asarray(s)), atol=1e-14
        )


@pytest.mark.high
def test_phase_block_handles_negative_indices() -> None:
    s = jnp.asarray(np.random.default_rng(2).uniform(size=(5, 2)))
    n = jnp.asarray([[2.0, -3.0], [-1.0, 0.0], [0.0, 0.0]])
    ph = np.asarray(_phase_block(_power_table(s, 3), n))
    want = np.exp(2j * np.pi * (np.asarray(n) @ np.asarray(s).T))
    np.testing.assert_allclose(ph, want, atol=1e-14)


@pytest.mark.high
def test_regression_D010_undersized_n_max_raises_at_construction() -> None:
    en = _tables(small_system(12, 2, 4.0)[2])
    assert en.n_max is not None and en.n_max >= 2
    with pytest.raises(ValueError, match="n_max"):
        dataclasses.replace(en, n_max=en.n_max - 1)
    with pytest.raises(ValueError, match="non-negative"):
        dataclasses.replace(en, n_max=-1)
    ok = dataclasses.replace(en, n_max=en.n_max + 3)  # a larger bound is fine
    assert ok.n_max == en.n_max + 3


@pytest.mark.medium
def test_non_integer_n_vectors_raise_at_construction() -> None:
    n = jnp.asarray([[1.0, 0.0], [0.5, 1.0]])
    w, v = jnp.ones(2), jnp.asarray(4.0)
    with pytest.raises(ValueError, match="integer"):
        StealthyEnergy(n, w, v, None, 1)
    with pytest.raises(ValueError, match="integer"):
        StealthyEnergy(n, w, v)  # the trig path is checked too


@pytest.mark.medium
def test_traced_n_vectors_skip_checks_and_nan_guards_the_table() -> None:
    """Under jit ``n_vectors`` is a tracer: no check runs, NaN fill guards."""
    en = _tables(small_system(12, 2, 4.0)[2])
    assert en.n_max is not None and en.n_max >= 2
    s = jnp.asarray(np.random.default_rng(5).uniform(size=(12, 2)))
    n_max = en.n_max - 1

    @jax.jit
    def run(n: jax.Array) -> jax.Array:
        bad = StealthyEnergy(n, en.weights, en.volume, None, n_max)
        return bad.value_and_grad(s)[0]

    assert np.isnan(float(run(en.n_vectors)))


@pytest.mark.medium
def test_auto_chunk_size_examples() -> None:
    assert auto_chunk_size(100, 128, budget_bytes=1 << 30) is None
    assert auto_chunk_size(10_000, 9_024, budget_bytes=1 << 30) == 1118
    assert auto_chunk_size(10_000, 9_024, budget_bytes=1, bytes_per_pair=1) == 1
    with pytest.raises(AssertionError):
        auto_chunk_size(0, 10)
