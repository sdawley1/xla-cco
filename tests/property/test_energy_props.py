#!/usr/bin/env python3

"""
Property tests of the stealthy energy against oracles and symmetries.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

# ----- project -----
import oracle
from strategies import boxes, chunk_sizes, fractional_configs, integer_kvectors
from tolerances import ATOL_FD, RTOL_KERNEL_E, RTOL_KERNEL_F
from xla_cco.box import Box, cubic_box
from xla_cco.energy import StealthyEnergy, collective_density, make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors

# ============================================================
# HELPERS
# ============================================================

FAST = settings(deadline=None)


def _weights(seed: int, m: int) -> np.ndarray:
    """Positive weights in [0.5, 2) from an explicit seed."""
    return np.random.default_rng(seed).uniform(0.5, 2.0, size=m)


def _energy(
    n: np.ndarray, w: np.ndarray, vol: float, chunk: int | None = None
) -> StealthyEnergy:
    # pad M to a multiple of 8 (zero rows, zero weights) so jit shapes come
    # from a small fixed grid; padding is exactly inert (tested separately)
    total = -(-n.shape[0] // 8) * 8
    n = np.vstack([n, np.zeros((total - n.shape[0], n.shape[1]), dtype=n.dtype)])
    w = np.concatenate([w, np.zeros(total - w.shape[0])])
    return StealthyEnergy(
        n_vectors=jnp.asarray(n, dtype=jnp.float64),
        weights=jnp.asarray(w),
        volume=jnp.asarray(vol),
        chunk_size=chunk,
    )


@st.composite
def systems(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Draw (s, n, w, V) with matching dimension."""
    s = draw(fractional_configs(n_grid=(1, 2, 7, 16)))
    n = draw(integer_kvectors(dim=s.shape[1], max_m=16))
    w = _weights(draw(st.integers(0, 2**31)), n.shape[0])
    vol = draw(st.floats(0.5, 50.0, allow_nan=False))
    return s, n, w, vol


# ============================================================
# TESTS
# ============================================================


@FAST
@given(systems())
def test_energy_matches_oracle(case: tuple) -> None:
    s, n, w, vol = case
    ref = oracle.energy(s, n, w, vol)
    got = float(_energy(n, w, vol)(jnp.asarray(s)))
    np.testing.assert_allclose(got, ref, rtol=RTOL_KERNEL_E, atol=1e-12)


@FAST
@given(systems(), chunk_sizes())
def test_value_and_grad_matches_autodiff(case: tuple, chunk: int | None) -> None:
    s, n, w, vol = case
    en = _energy(n, w, vol, chunk)
    val, grad = en.value_and_grad(jnp.asarray(s))
    rval, rgrad = jax.value_and_grad(en)(jnp.asarray(s))
    np.testing.assert_allclose(float(val), float(rval), rtol=1e-12, atol=1e-12)
    scale = max(1.0, float(np.abs(rgrad).max()))
    np.testing.assert_allclose(
        np.asarray(grad), np.asarray(rgrad), rtol=1e-12, atol=1e-12 * scale
    )


@FAST
@given(systems())
def test_gradient_matches_central_difference(case: tuple) -> None:
    s, n, w, vol = case
    en = _energy(n, w, vol)
    _, grad = en.value_and_grad(jnp.asarray(s))
    h = 1e-6
    fd = np.zeros_like(s)
    for i in range(s.shape[0]):
        for c in range(s.shape[1]):
            sp, sm = s.copy(), s.copy()
            sp[i, c] += h
            sm[i, c] -= h
            fd[i, c] = (oracle.energy(sp, n, w, vol) - oracle.energy(sm, n, w, vol)) / (
                2 * h
            )
    scale = max(1.0, float(np.abs(fd).max()))
    np.testing.assert_allclose(np.asarray(grad), fd, atol=ATOL_FD * scale)


@FAST
@given(boxes(), st.integers(0, 2**31), st.sampled_from((2, 7, 16)))
def test_forces_match_cpp_closed_form(
    basis: np.ndarray, seed: int, n_part: int
) -> None:
    d = basis.shape[0]
    rng = np.random.default_rng(seed)
    s = rng.uniform(size=(n_part, d))
    n = rng.integers(-3, 4, size=(10, d))
    n[~np.any(n != 0, axis=1)] = 1
    w = _weights(seed, 10)
    vol = float(abs(np.linalg.det(basis)))
    box = Box(jnp.asarray(basis))
    _, grad = _energy(n, w, vol).value_and_grad(jnp.asarray(s))
    ref = oracle.cartesian_forces(s, n, w, basis, vol)
    got = np.asarray(box.cartesian_forces(grad))
    np.testing.assert_allclose(
        got, ref, rtol=RTOL_KERNEL_F, atol=RTOL_KERNEL_F * max(1.0, np.abs(ref).max())
    )


@FAST
@given(systems(), st.integers(0, 2**31))
def test_translation_invariant(case: tuple, seed: int) -> None:
    s, n, w, vol = case
    shift = np.random.default_rng(seed).uniform(-1, 1, size=s.shape[1])
    en = _energy(n, w, vol)
    np.testing.assert_allclose(
        float(en(jnp.asarray(s + shift))),
        float(en(jnp.asarray(s))),
        rtol=1e-10,
        atol=1e-10,
    )


@FAST
@given(systems(), st.integers(0, 2**31))
def test_permutation_invariant(case: tuple, seed: int) -> None:
    s, n, w, vol = case
    perm = np.random.default_rng(seed).permutation(s.shape[0])
    en = _energy(n, w, vol)
    np.testing.assert_allclose(
        float(en(jnp.asarray(s[perm]))), float(en(jnp.asarray(s))), rtol=1e-12
    )


@FAST
@given(systems(), st.integers(0, 2**31))
def test_periodic_image_invariant(case: tuple, seed: int) -> None:
    s, n, w, vol = case
    shifts = np.random.default_rng(seed).integers(-3, 4, size=s.shape)
    en = _energy(n, w, vol)
    np.testing.assert_allclose(
        float(en(jnp.asarray(s + shifts))),
        float(en(jnp.asarray(s))),
        rtol=1e-10,
        atol=1e-10,
    )


@FAST
@given(systems())
def test_gradient_sums_to_zero(case: tuple) -> None:
    s, n, w, vol = case
    _, grad = _energy(n, w, vol).value_and_grad(jnp.asarray(s))
    scale = max(1.0, float(np.abs(grad).max()) * s.shape[0])
    np.testing.assert_allclose(np.asarray(grad).sum(axis=0), 0.0, atol=1e-11 * scale)


@FAST
@given(systems())
def test_nonnegative(case: tuple) -> None:
    s, n, w, vol = case
    assert float(_energy(n, w, vol)(jnp.asarray(s))) >= 0.0


@FAST
@given(fractional_configs(n_grid=(1, 2, 7, 16)), st.floats(0.5, 50.0))
def test_single_vector_energy_bounds(s: np.ndarray, vol: float) -> None:
    n = np.ones((1, s.shape[1]))
    val = float(_energy(n, np.ones(1), vol)(jnp.asarray(s)))
    assert 0.0 <= val <= s.shape[0] ** 2 / vol * (1 + 1e-12)


@FAST
@given(systems(), st.integers(0, 20))
def test_window_additivity(case: tuple, cut: int) -> None:
    s, n, w, vol = case
    cut = min(cut, n.shape[0])
    whole = float(_energy(n, w, vol)(jnp.asarray(s)))
    parts = 0.0
    for sl in (slice(0, cut), slice(cut, None)):
        if n[sl].shape[0]:
            parts += float(_energy(n[sl], w[sl], vol)(jnp.asarray(s)))
    np.testing.assert_allclose(whole, parts, rtol=1e-12, atol=1e-12)


@FAST
@given(systems(), st.sampled_from((1, 64, 128)))
def test_padding_rows_inert(case: tuple, pad_to: int) -> None:
    s, n, w, vol = case
    m = n.shape[0]
    total = -(-m // pad_to) * pad_to
    n_pad = np.vstack([n, np.zeros((total - m, n.shape[1]), dtype=n.dtype)])
    w_pad = np.concatenate([w, np.zeros(total - m)])
    sj = jnp.asarray(s)
    v0, g0 = _energy(n, w, vol).value_and_grad(sj)
    v1, g1 = _energy(n_pad, w_pad, vol).value_and_grad(sj)
    np.testing.assert_allclose(float(v1), float(v0), rtol=1e-12, atol=1e-12)
    scale = max(1.0, float(np.abs(g0).max()))
    np.testing.assert_allclose(
        np.asarray(g1), np.asarray(g0), rtol=1e-11, atol=1e-11 * scale
    )


@FAST
@given(systems(), st.sampled_from((1, 3, 5, 8, 13)))
def test_chunked_equals_unchunked(case: tuple, chunk: int) -> None:
    s, n, w, vol = case
    sj = jnp.asarray(s)
    v0, g0 = _energy(n, w, vol, None).value_and_grad(sj)
    v1, g1 = _energy(n, w, vol, chunk).value_and_grad(sj)
    np.testing.assert_allclose(float(v1), float(v0), rtol=1e-12, atol=1e-12)
    scale = max(1.0, float(np.abs(g0).max()))
    np.testing.assert_allclose(
        np.asarray(g1), np.asarray(g0), rtol=1e-11, atol=1e-11 * scale
    )
    re0, im0 = collective_density(sj, jnp.asarray(n, dtype=jnp.float64), None)
    re1, im1 = collective_density(sj, jnp.asarray(n, dtype=jnp.float64), chunk)
    np.testing.assert_allclose(np.asarray(re1), np.asarray(re0), atol=1e-12)
    np.testing.assert_allclose(np.asarray(im1), np.asarray(im0), atol=1e-12)


@FAST
@given(systems())
def test_collective_density_matches_oracle(case: tuple) -> None:
    s, n, _, _ = case
    re, im = collective_density(jnp.asarray(s), jnp.asarray(n, dtype=jnp.float64), 4)
    sk = oracle.structure_factor(s, n) * s.shape[0]
    np.testing.assert_allclose(
        np.asarray(re) ** 2 + np.asarray(im) ** 2, sk, rtol=1e-11, atol=1e-11
    )


@settings(deadline=None)
@given(
    st.integers(0, 2**31),
    st.sampled_from((1, 64)),
    st.sampled_from((7, 13)),
)
def test_factory_matches_oracle_large_m(seed: int, pad_to: int, chunk: int) -> None:
    """Factory output equals the oracle for M > 64 (kills weight truncation)."""
    n_part = 12
    box = cubic_box(n_part, 2)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, 12.0)
    assert kset.n.shape[0] > 64
    kset = dataclasses.replace(kset, weights=_weights(seed, kset.n.shape[0]))
    s = np.random.default_rng(seed).uniform(size=(n_part, 2))
    en = make_stealthy_energy(box, kset, chunk_size=chunk, pad_to=pad_to)
    assert en.n_vectors.shape[0] % chunk != 0 or pad_to == 1
    vol = float(box.volume)
    val, grad = en.value_and_grad(jnp.asarray(s))
    ref = oracle.energy(s, kset.n, kset.weights, vol)
    np.testing.assert_allclose(float(val), ref, rtol=RTOL_KERNEL_E)
    ref_f = oracle.cartesian_forces(s, kset.n, kset.weights, np.asarray(box.basis), vol)
    got_f = np.asarray(box.cartesian_forces(grad))
    np.testing.assert_allclose(
        got_f, ref_f, rtol=RTOL_KERNEL_F, atol=RTOL_KERNEL_F * np.abs(ref_f).max()
    )
