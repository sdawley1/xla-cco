#!/usr/bin/env python3

"""
Example tests of the stealthy energy, collective density and factory.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any, Iterator

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ----- project -----
import oracle
from tolerances import ATOL_FD, RTOL_KERNEL_E, RTOL_KERNEL_F
from xla_cco.box import Box, cubic_box
from xla_cco.energy import (
    AbstractEnergy,
    StealthyEnergy,
    collective_density,
    make_stealthy_energy,
)
from xla_cco.kvectors import enumerate_kvectors

# ============================================================
# HELPERS
# ============================================================


def _small_case(
    seed: int = 0, n: int = 12, d: int = 2
) -> tuple[np.ndarray, Box, Any, StealthyEnergy]:
    """Return (s, box, kset, energy) for a random small system."""
    s = np.random.default_rng(seed).uniform(size=(n, d))
    box = cubic_box(n, d)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, 4.0)
    return s, box, kset, make_stealthy_energy(box, kset, chunk_size=None)


def _walk_jaxpr_sizes(jaxpr: Any) -> Iterator[int]:
    """Yield the element count of every outvar in a jaxpr and its sub-jaxprs."""
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            yield int(np.prod(var.aval.shape, dtype=np.int64))
        for val in eqn.params.values():
            subs = val if isinstance(val, (list, tuple)) else [val]
            for sub in subs:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from _walk_jaxpr_sizes(inner)


# ============================================================
# TESTS
# ============================================================


@pytest.mark.critical
def test_energy_matches_oracle_example() -> None:
    s, box, kset, en = _small_case()
    ref = oracle.energy(s, kset.n, kset.weights, float(box.volume))
    np.testing.assert_allclose(float(en(jnp.asarray(s))), ref, rtol=RTOL_KERNEL_E)


@pytest.mark.critical
def test_forces_match_cpp_closed_form() -> None:
    s, box, kset, en = _small_case(seed=3)
    _, grad = en.value_and_grad(jnp.asarray(s))
    ref = oracle.cartesian_forces(
        s, kset.n, kset.weights, np.asarray(box.basis), float(box.volume)
    )
    got = np.asarray(box.cartesian_forces(grad))
    np.testing.assert_allclose(
        got, ref, rtol=RTOL_KERNEL_F, atol=RTOL_KERNEL_F * np.abs(ref).max()
    )


@pytest.mark.critical
def test_gradient_matches_central_difference() -> None:
    s, _, _, en = _small_case(seed=5, n=6)
    _, grad = en.value_and_grad(jnp.asarray(s))
    h = 1e-6
    fd = np.zeros_like(s)
    for i in range(s.shape[0]):
        for c in range(s.shape[1]):
            sp, sm = s.copy(), s.copy()
            sp[i, c] += h
            sm[i, c] -= h
            fd[i, c] = (float(en(jnp.asarray(sp))) - float(en(jnp.asarray(sm)))) / (
                2 * h
            )
    scale = max(1.0, float(np.abs(fd).max()))
    np.testing.assert_allclose(np.asarray(grad), fd, atol=ATOL_FD * scale)


@pytest.mark.high
def test_single_particle_energy() -> None:
    box = cubic_box(1, 2)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, 9.0)
    en = make_stealthy_energy(box, kset)
    val = float(en(jnp.asarray([[0.37, 0.81]])))
    np.testing.assert_allclose(val, kset.weights.sum() / 1.0, rtol=RTOL_KERNEL_E)


@pytest.mark.critical
@pytest.mark.parametrize("dim,m", [(1, 6), (2, 4), (3, 3)])
def test_lattice_ground_state(dim: int, m: int) -> None:
    n = m**dim
    grid = np.stack(np.meshgrid(*[np.arange(m)] * dim, indexing="ij"), -1)
    s = grid.reshape(-1, dim).astype(np.float64) / m
    box = cubic_box(n, dim)
    k_out = 0.999 * 2.0 * np.pi * m / float(np.asarray(box.basis)[0, 0])
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k_out)
    assert kset.n.shape[0] > 0
    en = make_stealthy_energy(box, kset)
    assert float(en(jnp.asarray(s))) <= 1e-26


@pytest.mark.high
def test_padding_rows_inert_example() -> None:
    s, box, kset, _ = _small_case(seed=2)
    ref = make_stealthy_energy(box, kset, pad_to=1)
    padded = make_stealthy_energy(box, kset, pad_to=128)
    assert padded.n_vectors.shape[0] % 128 == 0
    sj = jnp.asarray(s)
    np.testing.assert_allclose(float(padded(sj)), float(ref(sj)), rtol=1e-14)
    np.testing.assert_allclose(
        np.asarray(padded.value_and_grad(sj)[1]),
        np.asarray(ref.value_and_grad(sj)[1]),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.medium
def test_factory_pads_and_types() -> None:
    _, box, kset, _ = _small_case()
    en = make_stealthy_energy(box, kset, chunk_size=7, pad_to=64)
    m = kset.n.shape[0]
    assert en.n_vectors.shape == (-(-m // 64) * 64, 2)
    assert en.n_vectors.dtype == jnp.float64
    assert float(en.weights[m:].sum()) == 0.0
    assert en.chunk_size == 7


@pytest.mark.medium
def test_factory_rejects_empty_kset() -> None:
    box = cubic_box(4, 2)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, 1e-3)
    assert kset.n.shape[0] == 0
    with pytest.raises(ValueError, match="empty"):
        make_stealthy_energy(box, kset)


@pytest.mark.high
@pytest.mark.parametrize("chunk", [1, 5, 16, 40, 200, 1000])
def test_chunked_equals_unchunked_example(chunk: int) -> None:
    s, box, kset, _ = _small_case(seed=7, n=20)
    sj = jnp.asarray(s)
    ref = make_stealthy_energy(box, kset, chunk_size=None)
    ch = make_stealthy_energy(box, kset, chunk_size=chunk)
    v0, g0 = ref.value_and_grad(sj)
    v1, g1 = ch.value_and_grad(sj)
    np.testing.assert_allclose(float(v1), float(v0), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(g1), np.asarray(g0), rtol=1e-10, atol=1e-10)
    re0, im0 = collective_density(sj, ref.n_vectors, None)
    re1, im1 = collective_density(sj, ref.n_vectors, chunk)
    np.testing.assert_allclose(np.asarray(re1), np.asarray(re0), atol=1e-12)
    np.testing.assert_allclose(np.asarray(im1), np.asarray(im0), atol=1e-12)


@pytest.mark.high
def test_value_and_grad_chunked_no_full_matrix() -> None:
    n, chunk = 50, 16
    box = cubic_box(n, 2)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, 12.0)
    en = make_stealthy_energy(box, kset, chunk_size=chunk, pad_to=64)
    m = en.n_vectors.shape[0]
    assert m >= 200 > chunk
    s = jnp.asarray(np.random.default_rng(1).uniform(size=(n, 2)))
    jaxpr = jax.make_jaxpr(lambda x: eqx.filter_jit(en.value_and_grad)(x))(s)
    sizes = list(_walk_jaxpr_sizes(jaxpr.jaxpr))
    assert max(sizes) < n * m
    assert max(sizes) <= max(n * chunk, m * 2, n * 2) * 2  # O(N*chunk)
    unchunked = make_stealthy_energy(box, kset, chunk_size=None)
    big = jax.make_jaxpr(lambda x: unchunked.value_and_grad(x))(s)
    assert max(_walk_jaxpr_sizes(big.jaxpr)) >= n * m  # the check can fail


@pytest.mark.high
def test_vmap_jit_hessian_compatible() -> None:
    s, _, _, en = _small_case(seed=11, n=5)
    batch = jnp.stack([jnp.asarray(s), jnp.asarray(s) + 0.1])
    got = eqx.filter_jit(eqx.filter_vmap(lambda x: en.value_and_grad(x)[0]))(batch)
    ref = np.array([float(en(batch[i])) for i in range(2)])
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-12)
    hess = np.asarray(jax.hessian(en)(jnp.asarray(s)))
    flat = hess.reshape(s.size, s.size)
    np.testing.assert_allclose(flat, flat.T, atol=1e-9)
    # Hessian is the Jacobian of the analytic gradient (checks the fast path).
    jac = jax.jacfwd(lambda x: en.value_and_grad(x)[1])(jnp.asarray(s))
    np.testing.assert_allclose(
        np.asarray(jac).reshape(s.size, s.size), flat, rtol=1e-9, atol=1e-9
    )


@pytest.mark.medium
def test_energy_is_pytree_with_static_chunk() -> None:
    _, _, _, en = _small_case()
    leaves = jax.tree_util.tree_leaves(en)
    assert len(leaves) == 3


@pytest.mark.medium
@pytest.mark.parametrize("which", ["call", "vag", "density", "factory"])
def test_rejects_float32(which: str) -> None:
    s, box, kset, en = _small_case(n=4)
    s32 = jnp.asarray(s, dtype=jnp.float32)
    with pytest.raises(TypeError, match="float32"):
        if which == "call":
            en(s32)
        elif which == "vag":
            en.value_and_grad(s32)
        elif which == "density":
            collective_density(s32, en.n_vectors)
        else:
            make_stealthy_energy(Box(jnp.asarray(box.basis, dtype=jnp.float32)), kset)


@pytest.mark.medium
def test_abstract_energy_default_value_and_grad() -> None:
    class Quad(AbstractEnergy):
        def __call__(self, s: Any) -> Any:
            return jnp.sum(s**2)

    val, grad = Quad().value_and_grad(jnp.array([[1.0, 2.0]]))
    assert float(val) == 5.0
    np.testing.assert_allclose(np.asarray(grad), [[2.0, 4.0]])
    with pytest.raises(TypeError):
        AbstractEnergy()  # type: ignore[abstract]


@pytest.mark.critical
@pytest.mark.parametrize("pad_to", [1, 64])
def test_factory_matches_oracle_large_m(pad_to: int) -> None:
    """Factory Phi and forces equal the oracle for M > pad_to (non-divisor chunk)."""
    n_part = 12
    box = cubic_box(n_part, 2)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, 12.0)
    assert kset.n.shape[0] > 64
    s = np.random.default_rng(4).uniform(size=(n_part, 2))
    en = make_stealthy_energy(box, kset, chunk_size=13, pad_to=pad_to)
    val, grad = en.value_and_grad(jnp.asarray(s))
    vol = float(box.volume)
    ref = oracle.energy(s, kset.n, kset.weights, vol)
    np.testing.assert_allclose(float(val), ref, rtol=RTOL_KERNEL_E)
    ref_f = oracle.cartesian_forces(s, kset.n, kset.weights, np.asarray(box.basis), vol)
    got_f = np.asarray(box.cartesian_forces(grad))
    np.testing.assert_allclose(
        got_f, ref_f, rtol=RTOL_KERNEL_F, atol=RTOL_KERNEL_F * np.abs(ref_f).max()
    )
