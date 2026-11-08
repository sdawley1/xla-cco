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
import dataclasses
from typing import Any, Iterator

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ----- project -----
import oracle
from strategies import chunk_sizes, fractional_configs, integer_kvectors
from tolerances import ATOL_FD, RTOL_KERNEL_E, RTOL_KERNEL_F, TABLE_RHO_C
from xla_cco.box import Box, cubic_box
from xla_cco.energy import (
    AbstractEnergy,
    StealthyEnergy,
    _phase_block,
    _power_table,
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


# ============================================================
# VALUE, GRADIENT AND DENSITY IN ONE PASS (MD)
# ============================================================


def _vgd_case(
    dim: int, n: int, chunk: Any, tables: bool
) -> tuple[jnp.ndarray, StealthyEnergy]:
    """Padded energy (pad_to 64: zero rows) and positions for value_grad_density."""
    box = cubic_box(n, dim)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, {1: 9.0, 2: 4.5, 3: 3.2}[dim])
    en = make_stealthy_energy(box, kset, chunk_size=chunk, pad_to=64)
    if tables:
        en = dataclasses.replace(en, n_max=int(np.abs(kset.n).max()))
    assert en.n_vectors.shape[0] > kset.n.shape[0]  # padding rows present
    s = np.random.default_rng(dim + n).uniform(-1.0, 2.0, size=(n, dim))
    return jnp.asarray(s), en


def _assert_bitwise(got: Any, ref: Any) -> None:
    """Exact equality of every array (no tolerance)."""
    assert len(got) == len(ref)
    for a, b in zip(got, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), strict=True)


def _table_rho(
    s: jnp.ndarray, en: StealthyEnergy, jit: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-table rho of every k row, blocked as the energy's chunks.

    The table sums are not chunk-invariant bitwise (measured: 1 ulp moves
    between chunkings, and between eager and jitted unchunked sums), so the
    reference sums the same k blocks (zero-padded to whole chunks, padding
    dropped) with a scan of one block per step, jitted when the energy is.
    """
    if jit:
        out = jax.jit(lambda x: _table_rho_impl(x, en))(s)
    else:
        out = _table_rho_impl(s, en)
    return np.asarray(out.real), np.asarray(out.imag)


def _table_rho_impl(s: jnp.ndarray, en: StealthyEnergy) -> jnp.ndarray:
    """Complex blocked table rho (body of ``_table_rho``)."""
    assert en.n_max is not None
    table = _power_table(s, en.n_max)
    m = en.n_vectors.shape[0]
    size = m if en.chunk_size is None else min(en.chunk_size, m)
    n_chunks = -(-m // size)
    if n_chunks == 1:
        return jnp.sum(_phase_block(table, en.n_vectors), axis=1)
    pad = jnp.zeros((n_chunks * size - m, en.n_vectors.shape[1]))
    blocks = jnp.concatenate([en.n_vectors, pad]).reshape(n_chunks, size, -1)
    _, out = jax.lax.scan(
        lambda _, n_c: (None, jnp.sum(_phase_block(table, n_c), axis=1)), None, blocks
    )
    return out.reshape(-1)[:m]


@pytest.mark.critical
@pytest.mark.parametrize("dim,n", [(1, 9), (2, 12), (3, 10)])
@pytest.mark.parametrize("chunk", [None, 1, 5, 7, 16, 1000])
@pytest.mark.parametrize("jit", [False, True])
def test_value_grad_density_bitwise_trig(
    dim: int, n: int, chunk: Any, jit: bool
) -> None:
    s, en = _vgd_case(dim, n, chunk, tables=False)
    vgd = eqx.filter_jit(en.value_grad_density) if jit else en.value_grad_density
    vag = eqx.filter_jit(en.value_and_grad) if jit else en.value_and_grad
    cd = eqx.filter_jit(collective_density) if jit else collective_density
    val, grad, re, im = vgd(s)
    _assert_bitwise((val, grad), vag(s))
    _assert_bitwise((re, im), cd(s, en.n_vectors, chunk))
    pad = np.asarray(en.weights) == 0.0
    assert np.all(np.asarray(re)[pad] == n) and np.all(np.asarray(im)[pad] == 0.0)


@pytest.mark.critical
@pytest.mark.parametrize("dim,n", [(1, 9), (2, 12), (3, 10)])
@pytest.mark.parametrize("chunk", [None, 1, 5, 7, 16, 1000])
@pytest.mark.parametrize("jit", [False, True])
def test_value_grad_density_bitwise_tables(
    dim: int, n: int, chunk: Any, jit: bool
) -> None:
    """Table path: Phi/grad bitwise to value_and_grad; rho is the table rho the
    gradient used (bitwise) and equals collective_density to table rounding."""
    s, en = _vgd_case(dim, n, chunk, tables=True)
    vgd = eqx.filter_jit(en.value_grad_density) if jit else en.value_grad_density
    vag = eqx.filter_jit(en.value_and_grad) if jit else en.value_and_grad
    val, grad, re, im = vgd(s)
    _assert_bitwise((val, grad), vag(s))
    assert en.n_max is not None
    atol = TABLE_RHO_C * np.finfo(float).eps * n * (1 + 2 * np.pi) * en.n_max * dim
    tab = _table_rho(s, en, jit)
    if jit and (chunk is None or chunk >= en.n_vectors.shape[0]):
        # one jitted block: XLA fuses the rho sum with the gradient's use of the
        # phases, so its bits are not reproducible standalone (1 ulp measured)
        np.testing.assert_allclose(np.asarray(re), tab[0], rtol=0, atol=atol)
        np.testing.assert_allclose(np.asarray(im), tab[1], rtol=0, atol=atol)
    else:
        _assert_bitwise((re, im), tab)
    ref_re, ref_im = collective_density(s, en.n_vectors, chunk)
    np.testing.assert_allclose(np.asarray(re), np.asarray(ref_re), rtol=0, atol=atol)
    np.testing.assert_allclose(np.asarray(im), np.asarray(ref_im), rtol=0, atol=atol)


@pytest.mark.high
@settings(deadline=None, max_examples=40)
@given(data=st.data(), chunk=chunk_sizes(), tables=st.booleans())
def test_value_grad_density_property_bitwise(
    data: Any, chunk: Any, tables: bool
) -> None:
    s = data.draw(fractional_configs())
    d = s.shape[1]
    n_vec = data.draw(integer_kvectors(dim=d)).astype(np.float64)
    total = -(-n_vec.shape[0] // 8) * 8  # zero padding rows on a small shape grid
    n_pad = np.vstack([n_vec, np.zeros((total - n_vec.shape[0], d))])
    w = np.concatenate([np.ones(n_vec.shape[0]), np.zeros(total - n_vec.shape[0])])
    en = StealthyEnergy(
        n_vectors=jnp.asarray(n_pad), weights=jnp.asarray(w),
        volume=jnp.asarray(float(s.shape[0])), chunk_size=chunk,
        n_max=int(np.abs(n_vec).max()) if tables else None,
    )  # fmt: skip
    sj = jnp.asarray(s)
    val, grad, re, im = en.value_grad_density(sj)
    _assert_bitwise((val, grad), en.value_and_grad(sj))
    ref = _table_rho(sj, en) if tables else collective_density(sj, en.n_vectors, chunk)
    _assert_bitwise((re, im), ref)


def _count_primitive(jaxpr: Any, name: str) -> int:
    """Number of equations with primitive ``name`` in a jaxpr and its sub-jaxprs."""
    count = 0
    for eqn in jaxpr.eqns:
        count += int(eqn.primitive.name == name)
        for val in eqn.params.values():
            for sub in val if isinstance(val, (list, tuple)) else [val]:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    count += _count_primitive(inner, name)
    return count


@pytest.mark.high
def test_value_grad_density_one_chunked_density_pass() -> None:
    """Chunked trig path: rho is computed once (one cos in the density map, one in
    the gradient scan) and no (n, m) array is ever formed."""
    n, chunk = 50, 16
    box = cubic_box(n, 2)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, 12.0)
    en = make_stealthy_energy(box, kset, chunk_size=chunk, pad_to=64)
    m = en.n_vectors.shape[0]
    s = jnp.asarray(np.random.default_rng(1).uniform(size=(n, 2)))
    jaxpr = jax.make_jaxpr(en.value_grad_density)(s).jaxpr
    assert max(_walk_jaxpr_sizes(jaxpr)) < n * m
    assert _count_primitive(jaxpr, "cos") == 2
    assert _count_primitive(jaxpr, "sin") == 2


@pytest.mark.medium
def test_value_grad_density_shapes_and_float32() -> None:
    s, en = _vgd_case(2, 12, 7, tables=False)
    val, grad, re, im = en.value_grad_density(s)
    m = en.n_vectors.shape[0]
    assert val.shape == () and grad.shape == s.shape
    assert re.shape == (m,) and im.shape == (m,) and re.dtype == jnp.float64
    with pytest.raises(TypeError):
        en.value_grad_density(s.astype(jnp.float32))
