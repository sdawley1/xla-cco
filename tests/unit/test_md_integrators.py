#!/usr/bin/env python3

"""
Tests for xla_cco.dynamics.integrators and xla_cco.dynamics.thermal.

Oracles: the exact discrete solution of velocity Verlet on a harmonic
oscillator, the continuous solution (global order), the exact shadow
invariant (energy bound), a NumPy Cartesian Verlet/BAOAB loop, autodiff and
finite-difference Cartesian Hessians, and the exact canonical statistics of
Gaussian velocities, Andersen resets and BAOAB on harmonic potentials
(Leimkuhler & Matthews 2013). Statistical tests use seeded draws, Bonferroni
critical values at MD_FAMILY_ALPHA and blocked errors for correlated series.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from typing import Callable, List, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jaxtyping import Float
from scipy import stats

# ----- project -----
import oracle
from tolerances import (
    ATOL_FD,
    ATOL_MD_CART_ORACLE,
    ATOL_MD_COM,
    ATOL_MD_ONE_STEP,
    ATOL_MD_REVERSE,
    ATOL_MD_VERLET_EXACT,
    ATOL_VMAP_S,
    MD_BIAS_DETECT,
    MD_BLOCK_SE_RTOL,
    MD_BLOCK_Z,
    MD_DRIFT_IDENTITY_C,
    MD_FAMILY_ALPHA,
    MD_KS_PVALUE,
    MD_ORDER_RATIO_TOL,
    MD_SQRT_T_ULPS,
    MD_VERLET_ENERGY_FACTOR,
    RTOL_MD_HESSIAN,
    RTOL_MD_ZERO_MODE,
)
from xla_cco.box import Box
from xla_cco.dynamics import thermal
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
from xla_cco.energy import AbstractEnergy, StealthyEnergy, make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors

# ============================================================
# CONSTANTS
# ============================================================

EPS = float(np.finfo(np.float64).eps)
RECT_BASIS = np.diag([32.0, 16.0 * math.sqrt(3.0)])  # the URL rectangular cell
SHEAR_BASIS = np.array([[4.0, 0.0], [1.3, 3.5]])  # sheared, |det| = 14

# ============================================================
# HELPERS
# ============================================================


def _z_crit(m: int) -> float:
    """Two-sided Bonferroni critical z for m comparisons at MD_FAMILY_ALPHA."""
    assert m >= 1, "need at least one comparison"
    return float(stats.norm.isf(MD_FAMILY_ALPHA / (2.0 * m)))


def _aux(r: Array) -> Tuple[Array, Array]:
    """Dummy (re, im) collective densities of one padding row."""
    z = jnp.zeros((1,), dtype=r.dtype)
    return z, z


def _harmonic(omega2: np.ndarray) -> ForceFn:
    """Independent Cartesian oscillators, F = -omega^2 r (componentwise)."""
    w2 = jnp.asarray(omega2, dtype=jnp.float64)

    def force(r: Array) -> ForceOut:
        re, im = _aux(r)
        return 0.5 * jnp.sum(w2 * r * r), -w2 * r, re, im

    return force


def _free(r: Array) -> ForceOut:
    """Zero force (free particles)."""
    re, im = _aux(r)
    return jnp.zeros(()), jnp.zeros_like(r), re, im


def _stealthy(basis: np.ndarray, k_outer: float) -> StealthyEnergy:
    """Stealthy energy of the (0, k_outer) window in the cell ``basis``."""
    kset = enumerate_kvectors(basis, 0.0, k_outer)
    return make_stealthy_energy(Box(basis=jnp.asarray(basis)), kset)


def _cartesian_force(energy: AbstractEnergy, basis: np.ndarray) -> ForceFn:
    """Force of an energy defined on fractional s = r inv(B) (Box convention)."""
    box = Box(basis=jnp.asarray(basis))
    inv = jnp.linalg.inv(box.basis)

    def force(r: Array) -> ForceOut:
        phi, g = energy.value_and_grad(r @ inv)
        re, im = _aux(r)
        return phi, box.cartesian_forces(g), re, im

    return force


def _verlet_run(
    force_fn: ForceFn, r: Array, v: Array, dt: float, n_steps: int
) -> Tuple[Array, Array]:
    """n_steps of verlet_step under lax.scan (jitted); returns (r, v)."""

    @jax.jit
    def run(r: Array, v: Array) -> Tuple[Array, Array]:
        def body(c: Tuple[Array, Array, Array], _: None) -> Tuple[Tuple, None]:
            rr, vv, ff = c
            rr, vv, out = verlet_step(rr, vv, ff, force_fn, dt)
            return (rr, vv, out[1]), None

        (rf, vf, _), _ = jax.lax.scan(
            body, (r, v, force_fn(r)[1]), None, length=n_steps
        )
        return rf, vf

    return run(jnp.asarray(r), jnp.asarray(v))


def _discrete_oscillator(
    x0: np.ndarray, v0: np.ndarray, omega: np.ndarray, h: float, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Verlet iterate n: x_n = x0 cos(n th) + b sin(n th), cos th = 1 - (wh)^2/2.

    v_n = (x_{n+1} - x_{n-1}) / (2 h) for velocity Verlet, b = h v0 / sin(th);
    th = 2 arcsin(w h / 2) since 1 - cos th = 2 sin^2(th / 2).
    """
    theta = 2.0 * np.arcsin(0.5 * omega * h)  # well-conditioned arccos(1 - (wh)^2/2)
    b = h * v0 / np.sin(theta)
    x = x0 * np.cos(n * theta) + b * np.sin(n * theta)
    v = np.sin(theta) / h * (-x0 * np.sin(n * theta) + b * np.cos(n * theta))
    return x, v


def _np_verlet(
    force: Callable[[np.ndarray], np.ndarray], r: np.ndarray, v: np.ndarray,
    dt: float, n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Plain NumPy velocity Verlet (the C++ Evolve loop)."""
    f = force(r)
    for _ in range(n_steps):
        v = v + 0.5 * dt * f
        r = r + dt * v
        f = force(r)
        v = v + 0.5 * dt * f
    return r, v


def _np_baoab_cold(
    force: Callable[[np.ndarray], np.ndarray], r: np.ndarray, v: np.ndarray,
    dt: float, gamma: float, n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """NumPy BAOAB at T = 0 (the O step is v -> exp(-gamma dt) v)."""
    c = math.exp(-gamma * dt)
    f = force(r)
    for _ in range(n_steps):
        v = v + 0.5 * dt * f
        r = r + 0.5 * dt * v
        v = c * v
        r = r + 0.5 * dt * v
        f = force(r)
        v = v + 0.5 * dt * f
    return r, v


def _andersen_scan(
    key: Array, v: Array, n_steps: int, temperature: float, p: float, mode: str
) -> Tuple[Array, Array, Array, Array]:
    """Free-particle Verlet + Andersen for n_steps; returns (v, acc, n, changed).

    ``changed`` (n_steps, n) flags the particles whose velocity changed.
    """
    t = jnp.asarray(temperature)

    @jax.jit
    def run(key: Array, v: Array) -> Tuple[Array, Array, Array, Array]:
        def body(c: Tuple, _: None) -> Tuple[Tuple, Tuple[Array, Array]]:
            k, vv, acc = c
            k, v_new, acc, n_reset = andersen_reset(k, vv, acc, t, p, mode)
            changed = jnp.any(v_new != vv, axis=1)
            return (k, v_new, acc), (n_reset, changed)

        init = (key, v, jnp.zeros(()))
        (_, vf, acc), (counts, changed) = jax.lax.scan(body, init, None, length=n_steps)
        return vf, acc, jnp.sum(counts), changed

    return run(key, v)


def _cartesian_hessian_autodiff(
    energy: AbstractEnergy, s: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """Reference H_r = jax.hessian of r -> Phi(r inv(B)) at r = s B (row-major)."""
    n, d = s.shape
    inv = jnp.linalg.inv(jnp.asarray(basis))

    def phi_r(x: Array) -> Array:
        return energy(x.reshape(n, d) @ inv)

    r = jnp.asarray(s @ basis).reshape(-1)
    return np.asarray(jax.hessian(phi_r)(r))


def _whitened_covariance(
    samples: np.ndarray, hess: np.ndarray, temperature: float, zero_rel: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project displacement samples (S, nd) on the eigenmodes of ``hess``.

    Returns (cov of q_i sqrt(lambda_i / T) over nonzero modes, zero-mode
    projections, lambda of the nonzero modes).
    """
    lam, vec = np.linalg.eigh(hess)
    keep = lam > zero_rel * lam.max()
    q = samples @ vec
    w = q[:, keep] * np.sqrt(lam[keep] / temperature)
    cov = (w.T @ w) / samples.shape[0]
    return cov, q[:, ~keep], lam[keep]


def _assert_whitened_identity(cov: np.ndarray, n_samples: int) -> None:
    """Entries of a sample covariance of iid N(0, 1) vectors vs the identity.

    Diagonal: mean of chi^2_1, sd sqrt(2 / S); off-diagonal sd sqrt(1 / S).
    """
    k = cov.shape[0]
    zc = _z_crit(k * (k + 1) // 2 + 1)
    sd = np.where(
        np.eye(k, dtype=bool), math.sqrt(2.0 / n_samples), 1.0 / math.sqrt(n_samples)
    )
    z = np.abs(cov - np.eye(k)) / sd
    assert z.max() < zc, f"whitened covariance off identity: max z {z.max():.2f}"
    # aggregate: trace/k is chi^2_{k S} / (k S), sd sqrt(2 / (k S))
    z_tr = abs(np.trace(cov) / k - 1.0) / math.sqrt(2.0 / (k * n_samples))
    assert z_tr < zc, f"mean whitened variance off 1: z {z_tr:.2f}"


def _assert_joint_gaussian(x: np.ndarray, var: float) -> None:
    """Rows of x (m, d) ~ N(0, var I_d) jointly: |x|^2 / var ~ chi^2_d (KS) and
    zero cross-covariances (z-tests, sd var / sqrt(m), Bonferroni over pairs).

    Catches one N(0, 1) draw broadcast over the d components (then
    |x|^2 / var ~ d chi^2_1 and corr = 1).
    """
    m, d = x.shape
    assert d >= 2, "joint test needs d >= 2"
    p = stats.kstest(np.sum(x * x, axis=1) / var, stats.chi2(d).cdf).pvalue
    assert p > MD_KS_PVALUE, ("|v|^2 / T not chi^2_d", p)
    pairs = [(a, b) for a in range(d) for b in range(a + 1, d)]
    zc = _z_crit(len(pairs))
    for a, b in pairs:
        z = abs(float(np.mean(x[:, a] * x[:, b]))) / (var / math.sqrt(m))
        assert z < zc, (a, b, z)


class _RingQuadratic(AbstractEnergy):
    """Phi(s) = 1/2 sum_j (u_j - u_{j+1}) C (u_j - u_{j+1})^T, u = s - a (ring).

    Translation invariant: d zero modes. H_s = kron(L_ring, C) (row-major).
    """

    a: Float[Array, "n d"]
    c: Float[Array, "d d"]

    def __call__(self, s: Float[Array, "n d"]) -> Float[Array, ""]:
        u = s - self.a
        du = u - jnp.roll(u, -1, axis=0)
        return 0.5 * jnp.sum((du @ self.c) * du)

    def hessian(self, s: Float[Array, "n d"]) -> Float[Array, "nd nd"]:
        """Closed form kron(L_ring, C) (row-major), cheap for repeated draws."""
        lap = jnp.asarray(_ring_laplacian(s.shape[0]))
        return jnp.kron(lap, self.c)


class _FracQuadratic(AbstractEnergy):
    """Phi(s) = 1/2 sum_j (s_j - a_j) C (s_j - a_j)^T (independent particles)."""

    a: Float[Array, "n d"]
    c: Float[Array, "d d"]

    def __call__(self, s: Float[Array, "n d"]) -> Float[Array, ""]:
        u = s - self.a
        return 0.5 * jnp.sum((u @ self.c) * u)


def _ring_laplacian(n: int) -> np.ndarray:
    """Graph Laplacian of the n-cycle."""
    lap = (
        2.0 * np.eye(n) - np.roll(np.eye(n), 1, axis=1) - np.roll(np.eye(n), -1, axis=1)
    )
    return lap


# ============================================================
# ORACLE SELF-TEST
# ============================================================


@pytest.mark.medium
@pytest.mark.parametrize("phi", [0.0, 0.9])
def test_blocked_standard_error_on_ar1(phi: float) -> None:
    """AR(1) x_t = phi x_{t-1} + e_t: SE of the mean -> sd_x sqrt((1+phi)/(1-phi)/n)."""
    n = 2**18
    e = np.random.default_rng(50).normal(size=n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1.0 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    sd_x = 1.0 / math.sqrt(1.0 - phi**2)
    exact = sd_x * math.sqrt((1.0 + phi) / (1.0 - phi) / n)
    se = oracle.blocked_standard_error(x)
    assert abs(se / exact - 1.0) <= MD_BLOCK_SE_RTOL, (se, exact)
    naive = float(np.std(x)) / math.sqrt(n)
    assert se >= naive * (1.0 - MD_BLOCK_SE_RTOL)


# ============================================================
# VELOCITY VERLET
# ============================================================


@pytest.mark.critical
def test_verlet_matches_exact_discrete_oscillator() -> None:
    """Anisotropic oscillators follow the exact discrete Verlet solution.

    cos(w~ h) = 1 - (w h)^2 / 2; catches a flipped force sign and a dropped
    half kick (both change the discrete frequency or the velocity phase).
    """
    rng = np.random.default_rng(10)
    omega = rng.uniform(0.2, 2.0, size=(3, 2))
    x0, v0 = rng.normal(size=(3, 2)), rng.normal(size=(3, 2))
    h, n = 0.15, 600
    r, v = _verlet_run(_harmonic(omega**2), x0, v0, h, n)
    x_ex, v_ex = _discrete_oscillator(x0, v0, omega, h, n)
    np.testing.assert_allclose(np.asarray(r), x_ex, rtol=0, atol=ATOL_MD_VERLET_EXACT)
    np.testing.assert_allclose(np.asarray(v), v_ex, rtol=0, atol=ATOL_MD_VERLET_EXACT)


@pytest.mark.critical
def test_verlet_single_step_matches_definition() -> None:
    """One step: v += h/2 F; r += h v; F = F(r); v += h/2 F (with re/im passthrough)."""
    force = _harmonic(np.array([[1.7, 0.4]]))
    r0, v0 = jnp.asarray([[0.3, -1.1]]), jnp.asarray([[0.5, 0.25]])
    f0 = force(r0)[1]
    r1, v1, out = verlet_step(r0, v0, f0, force, 0.1)
    w2 = np.array([[1.7, 0.4]])
    vh = np.asarray(v0) - 0.05 * w2 * np.asarray(r0)
    r_ex = np.asarray(r0) + 0.1 * vh
    v_ex = vh - 0.05 * w2 * r_ex
    np.testing.assert_allclose(np.asarray(r1), r_ex, rtol=0, atol=ATOL_MD_ONE_STEP)
    np.testing.assert_allclose(np.asarray(v1), v_ex, rtol=0, atol=ATOL_MD_ONE_STEP)
    np.testing.assert_allclose(
        np.asarray(out[1]), -w2 * r_ex, rtol=0, atol=ATOL_MD_ONE_STEP
    )
    assert len(out) == 4 and out[2].shape == (1,) and out[3].shape == (1,)


@pytest.mark.critical
def test_verlet_global_order_two() -> None:
    """Halving dt divides the error vs the continuous solution by 4."""
    omega, t_end = 1.0, 2.0
    x0, v0 = np.array([[0.8]]), np.array([[-0.3]])
    x_ex = x0 * math.cos(omega * t_end) + v0 / omega * math.sin(omega * t_end)
    v_ex = -x0 * omega * math.sin(omega * t_end) + v0 * math.cos(omega * t_end)
    errs = []
    for h in (0.02, 0.01, 0.005):
        r, v = _verlet_run(
            _harmonic(np.array([[omega**2]])), x0, v0, h, round(t_end / h)
        )
        errs.append(
            math.hypot(float(r[0, 0] - x_ex[0, 0]), float(v[0, 0] - v_ex[0, 0]))
        )
    for coarse, fine in zip(errs[:-1], errs[1:], strict=True):
        assert abs(coarse / fine - 4.0) <= MD_ORDER_RATIO_TOL, errs


@pytest.mark.critical
def test_verlet_time_reversible_on_stealthy_force() -> None:
    """n steps, flip v, n steps returns to the start (nonlinear force, sheared cell)."""
    basis = SHEAR_BASIS * math.sqrt(16.0 / abs(np.linalg.det(SHEAR_BASIS)))
    energy = _stealthy(basis, 2.6)
    force = _cartesian_force(energy, basis)
    rng = np.random.default_rng(11)
    r0 = rng.uniform(size=(16, 2)) @ basis
    v0 = rng.normal(size=(16, 2))
    r1, v1 = _verlet_run(force, r0, v0, 0.05, 150)
    assert float(jnp.max(jnp.abs(r1 - r0))) > 0.1  # the particles moved
    r2, v2 = _verlet_run(force, r1, -v1, 0.05, 150)
    np.testing.assert_allclose(np.asarray(r2), r0, rtol=0, atol=ATOL_MD_REVERSE)
    np.testing.assert_allclose(np.asarray(-v2), v0, rtol=0, atol=ATOL_MD_REVERSE)


@pytest.mark.critical
def test_verlet_energy_bounded_over_1e6_steps() -> None:
    """max |E - <E>| / <E> <= 1.1 (w h)^2 / 8 over 1e6 steps (no drift).

    Exact invariant K = v^2 + (1 - (wh)^2/4) w^2 x^2 gives the bound
    (wh)^2 / (8 - (wh)^2) (MD_VERLET_ENERGY_FACTOR).
    """
    omega, h, n_steps = 1.3, 0.1, 1_000_000
    force = _harmonic(np.array([[omega**2]]))

    @jax.jit
    def run(r: Array, v: Array) -> Array:
        def body(c: Tuple, _: None) -> Tuple[Tuple, None]:
            rr, vv, ff, e_sum, e_min, e_max = c
            rr, vv, out = verlet_step(rr, vv, ff, force, h)
            e = out[0] + kinetic_energy(vv)
            return (
                rr,
                vv,
                out[1],
                e_sum + e,
                jnp.minimum(e_min, e),
                jnp.maximum(e_max, e),
            ), None

        e0 = force(r)[0] + kinetic_energy(v)
        init = (r, v, force(r)[1], jnp.zeros(()), e0, e0)
        c, _ = jax.lax.scan(body, init, None, length=n_steps)
        return jnp.stack([c[3] / n_steps, c[4], c[5]])

    mean, e_min, e_max = np.asarray(run(jnp.asarray([[0.7]]), jnp.asarray([[0.4]])))
    dev = max(e_max - mean, mean - e_min) / mean
    bound = MD_VERLET_ENERGY_FACTOR * (omega * h) ** 2 / 8.0
    assert dev <= bound, (dev, bound)
    assert dev >= 0.5 * (omega * h) ** 2 / 8.0  # the oscillation is really there


@pytest.mark.critical
@pytest.mark.parametrize("basis", [RECT_BASIS, SHEAR_BASIS], ids=["rect", "shear"])
def test_verlet_fractional_quadratic_matches_cartesian_oracle(
    basis: np.ndarray,
) -> None:
    """Anisotropic quadratic on s = r inv(B): Verlet equals a NumPy Cartesian loop.

    Phi = 1/2 sum_j (s_j - a_j) C (s_j - a_j)^T is, in Cartesian coordinates,
    1/2 sum_j (r_j - a_j B) K (r_j - a_j B)^T with K = inv(B) C inv(B)^T; a
    fractional integration with an identity metric would see C instead of K.
    """
    rng = np.random.default_rng(12)
    inv = np.linalg.inv(basis)
    c = np.array([[900.0, 250.0], [250.0, 400.0]])
    a = rng.uniform(size=(5, 2))
    energy = _FracQuadratic(a=jnp.asarray(a), c=jnp.asarray(c))
    k_r = inv @ c @ inv.T
    anchor = a @ basis
    r0 = anchor + rng.normal(size=(5, 2))
    v0 = rng.normal(size=(5, 2))
    r, v = _verlet_run(_cartesian_force(energy, basis), r0, v0, 0.1, 500)
    r_np, v_np = _np_verlet(lambda x: -(x - anchor) @ k_r, r0, v0, 0.1, 500)
    np.testing.assert_allclose(np.asarray(r), r_np, rtol=0, atol=ATOL_MD_CART_ORACLE)
    np.testing.assert_allclose(np.asarray(v), v_np, rtol=0, atol=ATOL_MD_CART_ORACLE)


# ============================================================
# MAXWELL-BOLTZMANN AND KINETIC ENERGY
# ============================================================


def _mb_batch(n_keys: int, n: int, d: int, temperature: float, com: bool) -> np.ndarray:
    """maxwell_boltzmann over n_keys seeds (vmapped), shape (n_keys, n, d)."""
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(n_keys))
    t = jnp.asarray(temperature)
    out = jax.vmap(lambda k: maxwell_boltzmann(k, n, d, t, com))(keys)
    return np.asarray(out)


@pytest.mark.critical
def test_maxwell_boltzmann_zero_total_momentum() -> None:
    """remove_com: sum_j v_j = 0 to 1e-14 per component; without it, not zero."""
    v = _mb_batch(2000, 16, 3, 1.0, True)
    assert np.abs(v.sum(axis=1)).max() <= ATOL_MD_COM
    raw = _mb_batch(200, 16, 3, 1.0, False)
    assert np.abs(raw.sum(axis=1)).max() > 1.0  # sd of the sum is 4


@pytest.mark.critical
@pytest.mark.parametrize("com", [True, False])
def test_maxwell_boltzmann_kinetic_energy_z(com: bool) -> None:
    """Mean KE = d(n - 1) T / 2 (COM removed) or d n T / 2; z-test, chi^2 variance."""
    n, d, temp, k = 16, 3, 2.5, 8000
    v = _mb_batch(k, n, d, temp, com)
    ke = 0.5 * np.sum(v * v, axis=(1, 2))
    dof = d * (n - 1) if com else d * n
    se = temp * math.sqrt(dof / 2.0) / math.sqrt(k)
    z = abs(ke.mean() - 0.5 * dof * temp) / se
    assert z < _z_crit(2), z
    # power: a 5% bias in T would give z >= 6
    assert MD_BIAS_DETECT * 0.5 * dof * temp / se >= 6.0


@pytest.mark.critical
def test_maxwell_boltzmann_components_ks() -> None:
    """Components ~ N(0, T (n - 1) / n) after COM removal (KS), jointly independent."""
    n, d, temp = 16, 3, 0.7
    v = _mb_batch(3000, n, d, temp, True)
    scale = math.sqrt(temp * (n - 1) / n)
    for a in range(d):
        p = stats.kstest(v[:, :, a].ravel() / scale, "norm").pvalue
        assert p > MD_KS_PVALUE, (a, p)
    raw = _mb_batch(3000, n, d, temp, False)
    for a in range(d):
        p = stats.kstest(raw[:, :, a].ravel() / math.sqrt(temp), "norm").pvalue
        assert p > MD_KS_PVALUE, (a, p)
    # joint law of each particle's d components (independent axes)
    _assert_joint_gaussian(raw.reshape(-1, d), temp)
    _assert_joint_gaussian(v.reshape(-1, d), scale**2)


@pytest.mark.high
def test_kinetic_energy_and_temperature() -> None:
    """KE = sum v^2 / 2; T_kin = 2 KE / dof."""
    v = jnp.asarray([[1.0, 2.0], [-3.0, 0.5]])
    # exact: every partial sum and the halving are representable in binary
    assert float(kinetic_energy(v)) == 0.5 * (1 + 4 + 9 + 0.25)
    assert float(kinetic_temperature(v, 2)) == 14.25 / 2
    assert kinetic_energy(v).shape == () and kinetic_temperature(v, 3).shape == ()


@pytest.mark.medium
def test_thermal_input_contracts() -> None:
    """Non-positive dof, bad shapes and negative temperatures are refused."""
    with pytest.raises(ValueError):
        kinetic_temperature(jnp.ones((2, 2)), 0)
    with pytest.raises(ValueError):
        maxwell_boltzmann(jax.random.PRNGKey(0), 0, 2, jnp.asarray(1.0), True)
    with pytest.raises(TypeError):
        kinetic_energy(jnp.ones((2, 2), dtype=jnp.float32))
    energy = _FracQuadratic(a=jnp.zeros((2, 2)), c=jnp.eye(2))
    with pytest.raises(ValueError):
        harmonic_sample(
            np.random.default_rng(0), energy, np.zeros((2, 2)), np.eye(2), -1.0
        )
    with pytest.raises(ValueError):
        harmonic_sample(
            np.random.default_rng(0), energy, np.zeros((2, 2)), np.eye(3), 1.0
        )


# ============================================================
# ANDERSEN
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize(
    ("acc", "p", "n_exp", "acc_exp"),
    [
        (0.75, 0.25, 1, 0.0),
        (0.5, 0.25, 0, 0.75),
        (0.0, 1.0, 1, 0.0),
        (0.0, 0.0, 0, 0.0),
    ],
)
def test_andersen_count_accumulator(
    acc: float, p: float, n_exp: int, acc_exp: float
) -> None:
    """acc += p; reset once when acc >= 1, then acc -= 1 (C++ count rule)."""
    v = jnp.asarray(np.random.default_rng(13).normal(size=(5, 2)))
    key = jax.random.PRNGKey(3)
    k2, v2, acc2, nr = andersen_reset(
        key, v, jnp.asarray(acc), jnp.asarray(1.0), p, "count"
    )
    assert int(nr) == n_exp and float(acc2) == acc_exp
    rows = np.flatnonzero(np.any(np.asarray(v2) != np.asarray(v), axis=1))
    assert rows.size == n_exp
    assert k2.dtype == jnp.uint32 and k2.shape == (2,)
    assert not np.array_equal(np.asarray(k2), np.asarray(key))


@pytest.mark.critical
@pytest.mark.parametrize(
    ("p", "n_steps"), [(0.125, 1001), (0.01, 1000), (1.0, 37), (0.3, 997)]
)
def test_andersen_count_exact_number_and_one_particle(p: float, n_steps: int) -> None:
    """Exactly floor(p n) resets from acc = 0; each reset redraws ONE full particle."""
    v0 = jnp.asarray(np.random.default_rng(14).normal(size=(6, 3)))
    _, acc, total, changed = _andersen_scan(
        jax.random.PRNGKey(5), v0, n_steps, 1.0, p, "count"
    )
    expected, a = 0, 0.0  # replay of the float accumulator (same operations)
    for _ in range(n_steps):
        a += p
        if a >= 1.0:
            a -= 1.0
            expected += 1
    assert expected == math.floor(p * n_steps)
    assert int(total) == expected and float(acc) == a
    per_step = np.asarray(changed).sum(axis=1)
    assert per_step.max() <= 1 and int(per_step.sum()) == expected


@pytest.mark.critical
def test_andersen_reset_redraws_all_components_of_one_particle() -> None:
    """The chosen particle's whole velocity vector is redrawn; others are untouched."""
    v = jnp.asarray(np.random.default_rng(15).normal(size=(7, 3)))
    _, v2, _, nr = andersen_reset(
        jax.random.PRNGKey(9), v, jnp.asarray(0.0), jnp.asarray(1.0), 1.0, "count"
    )
    diff = np.asarray(v2) != np.asarray(v)
    assert int(nr) == 1 and diff.any(axis=1).sum() == 1
    assert diff[diff.any(axis=1)].all()


@pytest.mark.critical
def test_andersen_particle_choice_uniform() -> None:
    """Reset particle index is uniform over n (chi-square)."""
    n, steps = 8, 16000
    v0 = jnp.zeros((n, 2)) + 1e6  # a redraw always changes the row
    _, _, _, changed = _andersen_scan(
        jax.random.PRNGKey(6), v0, steps, 1.0, 1.0, "count"
    )
    counts = np.asarray(changed).sum(axis=0)
    assert counts.sum() <= steps
    assert stats.chisquare(counts).pvalue > MD_FAMILY_ALPHA


@pytest.mark.critical
def test_andersen_bernoulli_binomial_count() -> None:
    """Bernoulli mode: resets ~ Binomial(n_steps, p) (z-test); acc unchanged."""
    p, steps = 0.3, 100_000
    v0 = jnp.zeros((4, 2)) + 1e6
    _, acc, total, changed = _andersen_scan(
        jax.random.PRNGKey(7), v0, steps, 1.0, p, "bernoulli"
    )
    sd = math.sqrt(steps * p * (1 - p))
    z = abs(int(total) - steps * p) / sd
    assert z < _z_crit(1), z
    assert MD_BIAS_DETECT * steps * p / sd >= 6.0
    assert float(acc) == 0.0
    per_step = np.asarray(changed).sum(axis=1)
    assert per_step.max() == 1 and int(per_step.sum()) == int(total)


@pytest.mark.critical
@pytest.mark.parametrize(
    ("mode", "p", "steps"), [("count", 1.0, 640), ("bernoulli", 0.5, 1280)]
)
def test_andersen_free_particles_reach_temperature(
    mode: str, p: float, steps: int
) -> None:
    """Zero force, v0 = 0: after ~20 resets per particle, v ~ N(0, T) (var z, KS).

    T = 2.5 != 1, so a reset with sigma = T (variance 6.25) is detected.
    """
    lanes, n, d, temp = 1024, 32, 2, 2.5
    force = _free
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(lanes) + 100)
    t = jnp.asarray(temp)

    def lane(key: Array) -> Array:
        def body(c: Tuple, _: None) -> Tuple[Tuple, None]:
            k, r, v, f, acc = c
            r, v, out = verlet_step(r, v, f, force, 0.01)
            k, v, acc, _ = andersen_reset(k, v, acc, t, p, mode)
            return (k, r, v, out[1], acc), None

        z = jnp.zeros((n, d))
        c, _ = jax.lax.scan(body, (key, z, z, z, jnp.zeros(())), None, length=steps)
        return c[2]

    v = np.asarray(jax.jit(jax.vmap(lane))(keys))
    x = v.reshape(-1, d)
    zc = _z_crit(2 * d)
    for a in range(d):
        m = x.shape[0]
        z_var = abs(np.mean(x[:, a] ** 2) - temp) / (temp * math.sqrt(2.0 / m))
        z_mean = abs(np.mean(x[:, a])) / math.sqrt(temp / m)
        assert z_var < zc and z_mean < zc, (a, z_var, z_mean)
        assert MD_BIAS_DETECT / math.sqrt(2.0 / m) >= 6.0
        p_ks = stats.kstest(x[:, a] / math.sqrt(temp), "norm").pvalue
        assert p_ks > MD_KS_PVALUE, (a, p_ks)
    _assert_joint_gaussian(x, temp)


@pytest.mark.critical
@pytest.mark.parametrize("mode", ["count", "bernoulli"])
def test_andersen_key_streams_are_independent(mode: str) -> None:
    """key', k_pick, k_draw, k_coin = split(key, 4): the index comes from k_pick and
    the redrawn row from k_draw, so the row is independent of the choice.

    A reused subkey (e.g. the row drawn from k_pick) changes the row.
    """
    n, d, temp = 7, 3, 2.0
    v = jnp.full((n, d), 1e6)
    for seed in range(20):
        key = jax.random.PRNGKey(seed)
        k_next, k_pick, k_draw, _ = jax.random.split(key, 4)
        k2, v2, _, nr = andersen_reset(
            key, v, jnp.asarray(0.0), jnp.asarray(temp), 1.0, mode
        )
        assert int(nr) == 1
        np.testing.assert_array_equal(np.asarray(k2), np.asarray(k_next))
        pick = int(jax.random.randint(k_pick, (), 0, n))
        changed = np.flatnonzero(np.any(np.asarray(v2) != np.asarray(v), axis=1))
        assert changed.tolist() == [pick]
        row = math.sqrt(temp) * np.asarray(jax.random.normal(k_draw, (d,)))
        np.testing.assert_allclose(
            np.asarray(v2[pick]), row, rtol=MD_SQRT_T_ULPS * EPS, atol=0
        )


@pytest.mark.critical
def test_baoab_noise_key_stream() -> None:
    """key', k_xi = split(key); xi = normal(k_xi, (n, d)) (free particle, v = 0)."""
    n, d, dt, gamma, temp = 5, 3, 0.2, 1.3, 0.6
    key = jax.random.PRNGKey(8)
    k_next, k_xi = jax.random.split(key)
    z = jnp.zeros((n, d))
    k2, _, v1, _ = baoab_step(key, z, z, z, _free, dt, gamma, jnp.asarray(temp))
    np.testing.assert_array_equal(np.asarray(k2), np.asarray(k_next))
    amp = math.sqrt(-math.expm1(-2.0 * gamma * dt) * temp)
    xi = np.asarray(jax.random.normal(k_xi, (n, d)))
    np.testing.assert_allclose(np.asarray(v1), amp * xi, rtol=MD_SQRT_T_ULPS * EPS)


@pytest.mark.medium
def test_andersen_rejects_bad_mode_and_rate() -> None:
    """Unknown mode, p outside [0, 1] are refused."""
    v, key, acc, t = (
        jnp.zeros((3, 2)),
        jax.random.PRNGKey(0),
        jnp.zeros(()),
        jnp.ones(()),
    )
    with pytest.raises(ValueError):
        andersen_reset(key, v, acc, t, 0.1, "poisson")
    with pytest.raises(ValueError):
        andersen_reset(key, v, acc, t, 1.5, "count")
    with pytest.raises(ValueError):
        andersen_reset(key, v, acc, t, -0.1, "bernoulli")


# ============================================================
# BAOAB
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize(("omega", "dt"), [(1.0, 1.0), (2.0, 0.5)])
def test_baoab_harmonic_configurational_exact(omega: float, dt: float) -> None:
    """BAOAB at w dt = 1: <x^2> = T / w^2 exactly (Leimkuhler-Matthews), blocked z."""
    n, steps, burn, temp, gamma = 256, 20_000, 1_000, 0.7, 1.0
    force = _harmonic(np.full((n, 1), omega**2))
    t = jnp.asarray(temp)

    @jax.jit
    def run(key: Array) -> Array:
        def body(c: Tuple, _: None) -> Tuple[Tuple, Array]:
            k, r, v, f = c
            k, r, v, out = baoab_step(k, r, v, f, force, dt, gamma, t)
            return (k, r, v, out[1]), jnp.mean(r * r)

        z = jnp.zeros((n, 1))
        _, x2 = jax.lax.scan(body, (key, z, z, z), None, length=steps)
        return x2

    x2 = np.asarray(run(jax.random.PRNGKey(21)))[burn:]
    se = oracle.blocked_standard_error(x2)
    z = abs(x2.mean() - temp / omega**2) / se
    assert z < MD_BLOCK_Z, (x2.mean(), temp / omega**2, se)
    assert MD_BIAS_DETECT * temp / omega**2 / se >= 6.0


@pytest.mark.critical
@pytest.mark.parametrize("basis", [RECT_BASIS, SHEAR_BASIS], ids=["rect", "shear"])
def test_baoab_cold_matches_numpy_oracle(basis: np.ndarray) -> None:
    """T = 0 BAOAB is deterministic: the NumPy splitting with c = exp(-g dt)."""
    rng = np.random.default_rng(22)
    inv = np.linalg.inv(basis)
    c = np.array([[900.0, 250.0], [250.0, 400.0]])
    a = rng.uniform(size=(4, 2))
    energy = _FracQuadratic(a=jnp.asarray(a), c=jnp.asarray(c))
    k_r, anchor = inv @ c @ inv.T, a @ basis
    r0, v0 = anchor + rng.normal(size=(4, 2)), rng.normal(size=(4, 2))
    force = _cartesian_force(energy, basis)
    dt, gamma = 0.1, 0.8

    @jax.jit
    def run(r: Array, v: Array) -> Tuple[Array, Array]:
        def body(cc: Tuple, _: None) -> Tuple[Tuple, None]:
            k, rr, vv, ff = cc
            k, rr, vv, out = baoab_step(k, rr, vv, ff, force, dt, gamma, jnp.zeros(()))
            return (k, rr, vv, out[1]), None

        cc, _ = jax.lax.scan(
            body, (jax.random.PRNGKey(0), r, v, force(r)[1]), None, length=300
        )
        return cc[1], cc[2]

    r, v = run(jnp.asarray(r0), jnp.asarray(v0))
    r_np, v_np = _np_baoab_cold(lambda x: -(x - anchor) @ k_r, r0, v0, dt, gamma, 300)
    np.testing.assert_allclose(np.asarray(r), r_np, rtol=0, atol=ATOL_MD_CART_ORACLE)
    np.testing.assert_allclose(np.asarray(v), v_np, rtol=0, atol=ATOL_MD_CART_ORACLE)


@pytest.mark.critical
def test_baoab_free_particle_single_step_statistics() -> None:
    """Free particles: r' = r + dt/2 (v + v'), v' = c v + sqrt((1 - c^2) T) xi."""
    n, d, dt, gamma, temp = 20_000, 2, 0.3, 1.7, 1.9
    c = math.exp(-gamma * dt)
    rng = np.random.default_rng(23)
    r0, v0 = jnp.asarray(rng.normal(size=(n, d))), jnp.asarray(rng.normal(size=(n, d)))
    k2, r1, v1, out = baoab_step(
        jax.random.PRNGKey(4),
        r0,
        v0,
        jnp.zeros((n, d)),
        _free,
        dt,
        gamma,
        jnp.asarray(temp),
    )
    np.testing.assert_allclose(
        np.asarray(r1),
        np.asarray(r0 + 0.5 * dt * (v0 + v1)),
        rtol=0,
        atol=MD_DRIFT_IDENTITY_C * EPS * (1.0 + float(jnp.max(jnp.abs(r1)))),
    )
    xi = (np.asarray(v1) - c * np.asarray(v0)) / math.sqrt((1 - c * c) * temp)
    m = xi.size
    z_var = abs(np.mean(xi**2) - 1.0) / math.sqrt(2.0 / m)
    z_mean = abs(np.mean(xi)) * math.sqrt(m)
    assert z_var < _z_crit(2) and z_mean < _z_crit(2), (z_var, z_mean)
    assert stats.kstest(xi.ravel(), "norm").pvalue > MD_KS_PVALUE
    _assert_joint_gaussian(xi, 1.0)
    assert k2.dtype == jnp.uint32 and out[1].shape == (n, d)


# ============================================================
# DETERMINISM, VMAP, JIT
# ============================================================


@pytest.mark.high
def test_same_key_bitwise_equal() -> None:
    """Same key and inputs -> bitwise identical outputs; another key differs."""
    key, t = jax.random.PRNGKey(31), jnp.asarray(0.8)
    a = maxwell_boltzmann(key, 9, 2, t, True)
    np.testing.assert_array_equal(
        np.asarray(a), np.asarray(maxwell_boltzmann(key, 9, 2, t, True))
    )
    assert not np.array_equal(
        np.asarray(a),
        np.asarray(maxwell_boltzmann(jax.random.PRNGKey(32), 9, 2, t, True)),
    )
    force = _harmonic(np.ones((9, 2)))
    f = force(a)[1]
    o1 = baoab_step(key, a, a, f, force, 0.1, 1.0, t)
    o2 = baoab_step(key, a, a, f, force, 0.1, 1.0, t)
    for x, y in zip(
        jax.tree_util.tree_leaves(o1), jax.tree_util.tree_leaves(o2), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    b1 = andersen_reset(key, a, jnp.asarray(0.5), t, 0.7, "count")
    b2 = andersen_reset(key, a, jnp.asarray(0.5), t, 0.7, "count")
    for x, y in zip(b1, b2, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.high
def test_vmap_over_lanes_equals_loop() -> None:
    """vmap of verlet/baoab/andersen over lanes equals per-lane calls."""
    lanes, n, d = 4, 6, 2
    rng = np.random.default_rng(33)
    r = jnp.asarray(rng.normal(size=(lanes, n, d)))
    v = jnp.asarray(rng.normal(size=(lanes, n, d)))
    temps = jnp.asarray([0.5, 1.0, 1.5, 2.0])
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(lanes))
    force = _cartesian_force(
        _stealthy(np.eye(2) * math.sqrt(n), 3.0), np.eye(2) * math.sqrt(n)
    )
    f = jax.vmap(lambda x: force(x)[1])(r)

    def one(k: Array, rr: Array, vv: Array, ff: Array, t: Array) -> List[Array]:
        r1, v1, o1 = verlet_step(rr, vv, ff, force, 0.05)
        k2, r2, v2, o2 = baoab_step(k, rr, vv, ff, force, 0.05, 2.0, t)
        k3, v3, a3, n3 = andersen_reset(k, vv, jnp.asarray(0.9), t, 0.3, "count")
        return [r1, v1, o1[1], k2, r2, v2, o2[1], k3, v3, a3, n3]

    batched = jax.vmap(one)(keys, r, v, f, temps)
    for i in range(lanes):
        single = one(keys[i], r[i], v[i], f[i], temps[i])
        for xb, xs in zip(batched, single, strict=True):
            np.testing.assert_allclose(
                np.asarray(xb[i]), np.asarray(xs), rtol=0, atol=ATOL_VMAP_S
            )


@pytest.mark.high
def test_jit_compiles_once_per_shape() -> None:
    """Tracing happens once per shape: new values reuse the compiled step."""
    traces = [0]

    def force(r: Array) -> ForceOut:
        traces[0] += 1  # runs at trace time only
        re, im = _aux(r)
        return jnp.sum(r * r), -r, re, im

    step = jax.jit(lambda k, r, v, f, t: baoab_step(k, r, v, f, force, 0.1, 1.0, t))
    for i in range(3):
        x = jnp.full((5, 2), float(i))
        step(jax.random.PRNGKey(i), x, x, x, jnp.asarray(1.0 + i))
    assert traces[0] == 1
    step(
        jax.random.PRNGKey(0),
        jnp.zeros((6, 2)),
        jnp.zeros((6, 2)),
        jnp.zeros((6, 2)),
        jnp.asarray(1.0),
    )
    assert traces[0] == 2


# ============================================================
# HARMONIC SAMPLING
# ============================================================


@pytest.mark.critical
def test_cartesian_hessian_matches_autodiff_and_finite_differences() -> None:
    """H_r from energy.hessian + inv(B) transform vs autodiff and FD in r (sheared)."""
    basis = SHEAR_BASIS * math.sqrt(12.0 / abs(np.linalg.det(SHEAR_BASIS)))
    energy = _stealthy(basis, 2.4)
    s = np.random.default_rng(41).uniform(size=(12, 2))
    h_r = thermal._cartesian_hessian(energy, s, basis)
    ref = _cartesian_hessian_autodiff(energy, s, basis)
    scale = np.abs(ref).max()
    np.testing.assert_allclose(h_r, ref, rtol=0, atol=RTOL_MD_HESSIAN * scale)
    # central differences of the Cartesian gradient
    inv = jnp.linalg.inv(jnp.asarray(basis))
    grad_r = jax.jit(jax.grad(lambda x: energy(x.reshape(12, 2) @ inv)))
    r0, step = (s @ basis).reshape(-1), 1e-5
    fd = np.empty_like(ref)
    for i in range(r0.size):
        e = np.zeros_like(r0)
        e[i] = step
        fd[:, i] = (
            np.asarray(grad_r(jnp.asarray(r0 + e)))
            - np.asarray(grad_r(jnp.asarray(r0 - e)))
        ) / (2 * step)
    np.testing.assert_allclose(h_r, fd, rtol=0, atol=ATOL_FD * max(1.0, scale))


@pytest.mark.critical
@pytest.mark.parametrize("basis", [RECT_BASIS, SHEAR_BASIS], ids=["rect", "shear"])
def test_harmonic_sample_quadratic_covariance(basis: np.ndarray) -> None:
    """Ring quadratic on fractional s: Cov(dr) = T pinv(kron(L, inv(B) C inv(B)^T)).

    Whitened covariance is the identity on the 2(n-1) nonzero modes and the
    d translation modes carry no displacement.
    """
    n, temp, samples = 5, 0.37, 3000
    rng = np.random.default_rng(42)
    inv = np.linalg.inv(basis)
    c = np.array([[900.0, 250.0], [250.0, 400.0]])
    a = rng.uniform(size=(n, 2))
    energy = _RingQuadratic(a=jnp.asarray(a), c=jnp.asarray(c))
    h_r = np.kron(_ring_laplacian(n), inv @ c @ inv.T)
    # the closed forms agree with autodiff (oracle self-check)
    auto_s = np.asarray(AbstractEnergy.hessian(energy, jnp.asarray(a)))
    np.testing.assert_allclose(
        np.asarray(energy.hessian(jnp.asarray(a))),
        auto_s,
        rtol=0,
        atol=RTOL_MD_HESSIAN * np.abs(auto_s).max(),
    )
    auto_r = _cartesian_hessian_autodiff(energy, a, basis)
    np.testing.assert_allclose(
        h_r, auto_r, rtol=0, atol=RTOL_MD_HESSIAN * np.abs(h_r).max()
    )
    draw = np.random.default_rng(43)
    disp = np.stack(
        [
            harmonic_sample(draw, energy, a, basis, temp) - a @ basis
            for _ in range(samples)
        ]
    )
    cov, zero, lam = _whitened_covariance(disp.reshape(samples, -1), h_r, temp, 1e-9)
    assert lam.size == 2 * (n - 1) and zero.shape[1] == 2
    _assert_whitened_identity(cov, samples)
    assert np.abs(zero).max() <= RTOL_MD_ZERO_MODE * np.abs(disp).max()


@pytest.mark.critical
def test_harmonic_sample_stealthy_lattice_covariance() -> None:
    """Stealthy ground state (fractional square lattice) in a sheared cell.

    Reference Hessian: jax.hessian of r -> Phi(r inv(B)) (independent of the
    inv(B) transform under test).
    """
    side, temp, samples = 4, 1e-3, 2000
    basis = SHEAR_BASIS * math.sqrt(16.0 / abs(np.linalg.det(SHEAR_BASIS)))
    energy = _stealthy(basis, 2.2)
    g = np.arange(side) / side
    s_gs = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2) + 0.1
    assert float(energy(jnp.asarray(s_gs))) < 1e-25  # a ground state
    h_ref = _cartesian_hessian_autodiff(energy, s_gs, basis)
    draw = np.random.default_rng(44)
    disp = np.stack(
        [
            harmonic_sample(draw, energy, s_gs, basis, temp) - s_gs @ basis
            for _ in range(samples)
        ]
    )
    cov, zero, lam = _whitened_covariance(disp.reshape(samples, -1), h_ref, temp, 1e-9)
    assert 0 < lam.size < s_gs.size and zero.shape[1] == s_gs.size - lam.size
    _assert_whitened_identity(cov, samples)
    assert np.abs(zero).max() <= RTOL_MD_ZERO_MODE * np.abs(disp).max()


@pytest.mark.high
def test_harmonic_sample_zero_temperature_is_ground_state() -> None:
    """T = 0 returns s_gs @ B exactly (unwrapped, no displacement)."""
    energy = _RingQuadratic(a=jnp.zeros((3, 2)), c=jnp.eye(2))
    s = np.array([[0.1, 1.2], [-0.3, 0.5], [0.9, 0.9]])
    out = harmonic_sample(np.random.default_rng(0), energy, s, RECT_BASIS, 0.0)
    np.testing.assert_array_equal(out, s @ RECT_BASIS)
    assert out.shape == (3, 2) and out.dtype == np.float64
