#!/usr/bin/env python3

"""
Property tests of the MD integrators and thermal sampling.

Shapes come from fixed grids so jit compiles a handful of times only. The
properties are exact (metamorphic or oracle) rather than statistical:
reversibility, temperature scaling of the Gaussian draws, the count-mode
accumulator replay, the cold BAOAB splitting and harmonic-sample geometry.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from typing import Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from jax import Array
from jaxtyping import Float

# ----- project -----
from tolerances import (
    ATOL_MD_CART_ORACLE,
    ATOL_MD_REVERSE,
    MD_COM_ROUNDOFF_C,
    RTOL_MD_QUADFORM,
)
from xla_cco.box import Box
from xla_cco.dynamics.integrators import (
    ForceFn,
    ForceOut,
    andersen_reset,
    baoab_step,
    verlet_step,
)
from xla_cco.dynamics.thermal import (
    harmonic_sample,
    kinetic_energy,
    kinetic_temperature,
    maxwell_boltzmann,
)
from xla_cco.energy import AbstractEnergy, make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors

# ============================================================
# STRATEGIES AND HELPERS
# ============================================================

EPS = float(np.finfo(np.float64).eps)
SHAPES = st.sampled_from([(1, 1), (4, 2), (9, 3), (16, 2), (33, 1)])
SEEDS = st.integers(0, 2**31 - 1)
TEMPS = st.floats(1e-4, 50.0, allow_nan=False, allow_infinity=False)
# XLA CPU flushes subnormals to zero, so rates and accumulators stay normal.
RATES = st.one_of(st.just(0.0), st.floats(1e-12, 1.0))
ACCS = st.one_of(st.just(0.0), st.floats(1e-12, 1.0, exclude_max=True))


def _aux(r: Array) -> Tuple[Array, Array]:
    """Dummy (re, im) of one padding row."""
    z = jnp.zeros((1,), dtype=r.dtype)
    return z, z


def _free(r: Array) -> ForceOut:
    """Zero force."""
    re, im = _aux(r)
    return jnp.zeros(()), jnp.zeros_like(r), re, im


def _stealthy_force(n: int, d: int) -> ForceFn:
    """Cartesian force of a cubic-cell stealthy energy (density 1)."""
    basis = np.eye(d) * float(n) ** (1.0 / d)
    box = Box(basis=jnp.asarray(basis))
    energy = make_stealthy_energy(box, enumerate_kvectors(basis, 0.0, 4.0))
    inv = jnp.linalg.inv(box.basis)

    def force(r: Array) -> ForceOut:
        phi, g = energy.value_and_grad(r @ inv)
        re, im = _aux(r)
        return phi, box.cartesian_forces(g), re, im

    return force


_FORCES = {
    shape: _stealthy_force(*shape) for shape in [(4, 2), (9, 3), (16, 2), (33, 1)]
}


def _verlet_steps(
    force: ForceFn, r: Array, v: Array, dt: float, n: int
) -> Tuple[Array, Array]:
    """n Verlet steps (lax.fori_loop, bounded)."""

    def body(_: int, c: Tuple[Array, Array, Array]) -> Tuple[Array, Array, Array]:
        rr, vv, ff = c
        rr, vv, out = verlet_step(rr, vv, ff, force, dt)
        return rr, vv, out[1]

    rf, vf, _ = jax.lax.fori_loop(0, n, body, (r, v, force(r)[1]))
    return rf, vf


_VERLET = jax.jit(_verlet_steps, static_argnums=(0, 3, 4))


class _FracQuadratic(AbstractEnergy):
    """Phi(s) = 1/2 sum_j (s_j - a_j) C (s_j - a_j)^T."""

    a: Float[Array, "n d"]
    c: Float[Array, "d d"]

    def __call__(self, s: Float[Array, "n d"]) -> Float[Array, ""]:
        u = s - self.a
        return 0.5 * jnp.sum((u @ self.c) * u)


# ============================================================
# PROPERTIES
# ============================================================


@pytest.mark.critical
@settings(deadline=None, max_examples=25)
@given(shape=st.sampled_from([(4, 2), (9, 3), (16, 2), (33, 1)]), seed=SEEDS,
       dt=st.sampled_from([0.01, 0.05]))  # fmt: skip
@example(shape=(16, 2), seed=0, dt=0.05)
def test_verlet_reversible_random_states(
    shape: Tuple[int, int], seed: int, dt: float
) -> None:
    """Forward 40 steps, flip v, 40 steps: back to (r0, -v0) within ATOL_MD_REVERSE."""
    n, d = shape
    rng = np.random.default_rng(seed)
    side = float(n) ** (1.0 / d)
    r0 = jnp.asarray(rng.uniform(size=(n, d)) * side)
    v0 = jnp.asarray(rng.normal(size=(n, d)))
    force = _FORCES[shape]
    r1, v1 = _VERLET(force, r0, v0, dt, 40)
    r2, v2 = _VERLET(force, r1, -v1, dt, 40)
    np.testing.assert_allclose(
        np.asarray(r2), np.asarray(r0), rtol=0, atol=ATOL_MD_REVERSE
    )
    np.testing.assert_allclose(
        np.asarray(-v2), np.asarray(v0), rtol=0, atol=ATOL_MD_REVERSE
    )


@pytest.mark.critical
@settings(deadline=None)
@given(shape=SHAPES, seed=SEEDS, temp=TEMPS)
@example(shape=(16, 2), seed=0, temp=1.0)
def test_maxwell_boltzmann_scaling_and_com(
    shape: Tuple[int, int], seed: int, temp: float
) -> None:
    """v(T) = sqrt(T) v(1) for the same key; remove_com gives sum v = 0 (roundoff)."""
    n, d = shape
    key = jax.random.PRNGKey(seed)
    raw1 = np.asarray(maxwell_boltzmann(key, n, d, jnp.asarray(1.0), False))
    raw_t = np.asarray(maxwell_boltzmann(key, n, d, jnp.asarray(temp), False))
    np.testing.assert_allclose(raw_t, math.sqrt(temp) * raw1, rtol=4 * EPS, atol=0)
    v = np.asarray(maxwell_boltzmann(key, n, d, jnp.asarray(temp), True))
    bound = MD_COM_ROUNDOFF_C * n * EPS * max(np.abs(raw_t).max(), 1e-300)
    assert np.abs(v.sum(axis=0)).max() <= bound
    # COM removal is a shift by the mean of the same draw
    np.testing.assert_allclose(
        v, raw_t - raw_t.mean(axis=0), rtol=0, atol=4 * EPS * np.abs(raw_t).max()
    )


@pytest.mark.high
@settings(deadline=None)
@given(shape=SHAPES, seed=SEEDS, dof=st.integers(1, 200))
def test_kinetic_temperature_is_twice_ke_over_dof(
    shape: Tuple[int, int], seed: int, dof: int
) -> None:
    """KE = sum v^2 / 2 and T_kin = 2 KE / dof."""
    v = np.random.default_rng(seed).normal(size=shape)
    ke = float(kinetic_energy(jnp.asarray(v)))
    assert ke == pytest.approx(0.5 * float(np.sum(v * v)), rel=1e-14)
    assert float(kinetic_temperature(jnp.asarray(v), dof)) == pytest.approx(
        2.0 * ke / dof, rel=1e-14
    )


@pytest.mark.critical
@settings(deadline=None)
@given(shape=SHAPES, seed=SEEDS, temp=TEMPS, acc=ACCS, p=RATES,
       mode=st.sampled_from(["count", "bernoulli"]))  # fmt: skip
@example(shape=(4, 2), seed=0, temp=4.0, acc=0.5, p=0.5, mode="count")
def test_andersen_reset_row_scales_with_sqrt_t(
    shape: Tuple[int, int], seed: int, temp: float, acc: float, p: float, mode: str
) -> None:
    """At most one row changes; it equals sqrt(T) times the T = 1 row (same key)."""
    n, d = shape
    key = jax.random.PRNGKey(seed)
    v = jnp.asarray(np.random.default_rng(seed).normal(size=(n, d)) + 100.0)
    a = jnp.asarray(acc)
    k1, v1, acc1, n1 = andersen_reset(key, v, a, jnp.asarray(1.0), p, mode)
    kt, vt, acct, nt = andersen_reset(key, v, a, jnp.asarray(temp), p, mode)
    assert int(n1) == int(nt) and int(n1) in (0, 1)
    np.testing.assert_array_equal(np.asarray(k1), np.asarray(kt))
    assert float(acc1) == float(acct)
    rows = np.flatnonzero(np.any(np.asarray(vt) != np.asarray(v), axis=1))
    assert rows.size == int(nt)
    if mode == "count":
        expected = acc + p >= 1.0
        assert int(nt) == int(expected)
        assert float(acct) == (acc + p - 1.0 if expected else acc + p)
    for j in rows:
        np.testing.assert_allclose(
            np.asarray(vt[j]), math.sqrt(temp) * np.asarray(v1[j]), rtol=4 * EPS
        )


@pytest.mark.critical
@settings(deadline=None)
@given(p=RATES, steps=st.sampled_from([1, 17, 250, 1000]))
@example(p=0.01, steps=1000)
def test_andersen_count_replays_float_accumulator(p: float, steps: int) -> None:
    """Count mode resets exactly when the replayed accumulator reaches 1."""

    @jax.jit
    def run(key: Array) -> Tuple[Array, Array]:
        def body(
            _: int, c: Tuple[Array, Array, Array, Array]
        ) -> Tuple[Array, Array, Array, Array]:
            k, v, acc, tot = c
            k, v, acc, nr = andersen_reset(k, v, acc, jnp.asarray(1.0), p, "count")
            return k, v, acc, tot + nr

        init = (key, jnp.zeros((3, 2)), jnp.zeros(()), jnp.zeros((), jnp.int32))
        out = jax.lax.fori_loop(0, steps, body, init)
        return out[2], out[3]

    acc, total = run(jax.random.PRNGKey(1))
    a, expected = 0.0, 0
    for _ in range(steps):
        a += p
        if a >= 1.0:
            a -= 1.0
            expected += 1
    assert int(total) == expected and float(acc) == a
    assert abs(expected - math.floor(p * steps)) <= 1


@pytest.mark.critical
@settings(deadline=None, max_examples=40)
@given(seed=SEEDS, dt=st.sampled_from([0.05, 0.2]), gamma=st.floats(0.0, 5.0))
@example(seed=0, dt=0.2, gamma=1.0)
def test_baoab_cold_equals_splitting(seed: int, dt: float, gamma: float) -> None:
    """T = 0: one BAOAB step equals B, A, (v *= exp(-g dt)), A, force, B exactly."""
    rng = np.random.default_rng(seed)
    w2 = rng.uniform(0.1, 3.0, size=(4, 2))
    r0, v0 = rng.normal(size=(4, 2)), rng.normal(size=(4, 2))

    def force(r: Array) -> ForceOut:
        re, im = _aux(r)
        return 0.5 * jnp.sum(w2 * r * r), -w2 * r, re, im

    key, f0 = jax.random.PRNGKey(seed), jnp.asarray(-w2 * r0)
    r0j, v0j = jnp.asarray(r0), jnp.asarray(v0)
    _, r1, v1, out = baoab_step(key, r0j, v0j, f0, force, dt, gamma, jnp.zeros(()))
    v = v0 - 0.5 * dt * w2 * r0
    r = r0 + 0.5 * dt * v
    v = math.exp(-gamma * dt) * v
    r = r + 0.5 * dt * v
    v = v - 0.5 * dt * w2 * r
    np.testing.assert_allclose(np.asarray(r1), r, rtol=0, atol=ATOL_MD_CART_ORACLE)
    np.testing.assert_allclose(np.asarray(v1), v, rtol=0, atol=ATOL_MD_CART_ORACLE)
    np.testing.assert_allclose(
        np.asarray(out[1]), -w2 * r, rtol=0, atol=ATOL_MD_CART_ORACLE
    )


@pytest.mark.high
@settings(deadline=None)
@given(
    shape=SHAPES,
    seed=SEEDS,
    dt=st.floats(1e-3, 0.5),
    gamma=st.floats(0.0, 5.0),
    temp=TEMPS,
)
def test_baoab_free_particle_drift_identity(
    shape: Tuple[int, int], seed: int, dt: float, gamma: float, temp: float
) -> None:
    """Zero force: r' = r + dt (v + v') / 2 for any noise draw."""
    rng = np.random.default_rng(seed)
    r0, v0 = jnp.asarray(rng.normal(size=shape)), jnp.asarray(rng.normal(size=shape))
    _, r1, v1, _ = baoab_step(
        jax.random.PRNGKey(seed),
        r0,
        v0,
        jnp.zeros(shape),
        _free,
        dt,
        gamma,
        jnp.asarray(temp),
    )
    scale = 1.0 + float(jnp.max(jnp.abs(r0))) + dt * float(jnp.max(jnp.abs(v1)))
    np.testing.assert_allclose(
        np.asarray(r1),
        np.asarray(r0 + 0.5 * dt * (v0 + v1)),
        rtol=0,
        atol=8 * EPS * scale,
    )


@pytest.mark.high
@settings(deadline=None, max_examples=30)
@given(seed=SEEDS, temp=st.floats(1e-6, 2.0), shear=st.floats(-1.0, 1.0),
       lx=st.floats(0.5, 40.0), ly=st.floats(0.5, 40.0))  # fmt: skip
@example(seed=0, temp=1.0, shear=0.0, lx=32.0, ly=16.0 * math.sqrt(3.0))
def test_harmonic_sample_independent_particles(
    seed: int, temp: float, shear: float, lx: float, ly: float
) -> None:
    """Independent fractional quadratics: dr K dr^T / T equals |xi|^2 exactly.

    K = inv(B) C inv(B)^T per particle (6 nonzero modes, no zero modes); xi
    are the generator's standard normals (harmonic_sample draws them once).
    """
    basis = np.array([[lx, 0.0], [shear * lx, ly]])
    inv = np.linalg.inv(basis)
    c = np.array([[3.0, 1.0], [1.0, 2.0]])
    a = np.random.default_rng(seed).uniform(size=(3, 2))
    energy = _FracQuadratic(a=jnp.asarray(a), c=jnp.asarray(c))
    out = harmonic_sample(np.random.default_rng(seed), energy, a, basis, temp)
    disp = out - a @ basis
    k_r = inv @ c @ inv.T
    quad = float(np.sum((disp @ k_r) * disp)) / temp
    xi = np.random.default_rng(seed).standard_normal(6)  # same stream, 6 modes
    assert quad == pytest.approx(float(xi @ xi), rel=RTOL_MD_QUADFORM)
    assert np.abs(disp).max() > 0.0 or temp == 0.0
