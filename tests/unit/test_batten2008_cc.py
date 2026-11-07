#!/usr/bin/env python3

"""
Tests for the collective-coordinate analyses of the 2008 flows
(experiments/scripts/batten2008_{modes,hessian,floor}.py).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import sys
from fractions import Fraction
from pathlib import Path

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ----- project -----
from xla_cco.energy import collective_density
from xla_cco.minimize import relax
from xla_cco.minimize.lbfgs import LBFGSSettings

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
from batten2008_floor import FiberEnergy, grid_starts, plane_offsets  # noqa: E402
from batten2008_flows import build_energy  # noqa: E402
from batten2008_hessian import (  # noqa: E402
    density_wave_stiffness,
    sample_frames,
    section_axes,
    tenfold_distance,
    wavelength_power,
)
from batten2008_modes import (  # noqa: E402
    mode_indices,
    phase_stats,
    relax_times,
    structure_factors,
)

# ============================================================
# HELPERS
# ============================================================

N, CHI = 12, Fraction(1, 3)  # M = 4 modes; side L = 12


def fiber(seed: int) -> FiberEnergy:
    """FiberEnergy for N = 12, chi = 1/3, on a random orthonormal plane."""
    energy, _, _ = build_energy(N, CHI)
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(N, 2)))
    return FiberEnergy(base=energy, q=jnp.asarray(q.T[:, :, None]))


def random_s(seed: int) -> jnp.ndarray:
    """Random fractional configuration (N, 1)."""
    return jnp.asarray(np.random.default_rng(seed).uniform(size=(N, 1)))


# ============================================================
# TESTS: FIBER ENERGY (valley floor)
# ============================================================


@pytest.mark.high
@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=20, deadline=None)
def test_fiber_value_is_phi_and_gradient_is_off_plane(seed: int) -> None:
    f = fiber(seed)
    s = random_s(seed + 1)
    value, grad = f.value_and_grad(s)
    base_value, base_grad = f.base.value_and_grad(s)
    q = np.asarray(f.q)[:, :, 0]
    g = np.asarray(grad)[:, 0]
    assert float(value) == pytest.approx(float(base_value), rel=1e-14)
    assert np.allclose(q @ g, 0.0, atol=1e-12 * np.linalg.norm(np.asarray(base_grad)))
    # the removed part is exactly the in-plane part of the true gradient
    gb = np.asarray(base_grad)[:, 0]
    assert np.allclose(g, gb - q.T @ (q @ gb), rtol=1e-12, atol=1e-14)


@pytest.mark.high
@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=20, deadline=None)
def test_fiber_autodiff_matches_analytic_projection(seed: int) -> None:
    f = fiber(seed)
    s = random_s(seed + 2)
    auto_value, auto_grad = jax.value_and_grad(f.__call__)(s)
    value, grad = f.value_and_grad(s)
    assert float(auto_value) == pytest.approx(float(value), rel=1e-13)
    assert np.allclose(np.asarray(auto_grad), np.asarray(grad), rtol=1e-10, atol=1e-12)


@pytest.mark.high
def test_relax_on_fiber_keeps_in_plane_coordinates_and_lowers_phi() -> None:
    f = fiber(7)
    s0 = jnp.stack([random_s(s) for s in (11, 12, 13)])
    res = relax(f, s0, LBFGSSettings(energy_tol=1e-20, max_evals_per_pass=400))
    d = np.asarray(res.s)[:, :, 0] - np.asarray(s0)[:, :, 0]
    d -= np.round(d)  # result is wrapped into [0, 1)
    q = np.asarray(f.q)[:, :, 0]
    assert np.max(np.abs(d @ q.T)) < 1e-10
    phi0 = np.asarray(jax.vmap(f.base)(s0))
    assert np.all(np.asarray(res.energy) <= phi0)


# ============================================================
# TESTS: GRID STARTS
# ============================================================


@pytest.mark.medium
def test_plane_offsets_reassemble_the_frames() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(9, N))
    e, _ = np.linalg.qr(rng.normal(size=(N, 2)))
    axes = (x.mean(axis=0), e[:, 0], e[:, 1])
    ab, off = plane_offsets(x, axes)
    back = axes[0] + ab[:, :1] * axes[1] + ab[:, 1:] * axes[2] + off
    assert np.allclose(back, x, atol=1e-12)
    assert np.allclose(off @ e, 0.0, atol=1e-12)


@pytest.mark.medium
def test_grid_point_on_a_frame_starts_at_that_frame() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, N))
    e, _ = np.linalg.qr(rng.normal(size=(N, 2)))
    axes = (np.zeros(N), e[:, 0], e[:, 1])
    ab, off = plane_offsets(x, axes)
    ra = np.sort(np.append(np.linspace(-5, 5, 4), ab[2, 0]))
    rb = np.sort(np.append(np.linspace(-5, 5, 4), ab[2, 1]))
    starts = grid_starts(ra, rb, ab, off, axes)
    i, j = (
        int(np.flatnonzero(rb == ab[2, 1])[0]),
        int(np.flatnonzero(ra == ab[2, 0])[0]),
    )
    assert np.allclose(starts[i * ra.shape[0] + j], x[2], atol=1e-12)


# ============================================================
# TESTS: MODES
# ============================================================


@pytest.mark.high
@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=20, deadline=None)
def test_mode_structure_factors_sum_to_phi(seed: int) -> None:
    energy, _, m = build_energy(N, CHI)
    nv = jnp.asarray(np.arange(1.0, m + 1)[:, None])
    rho_fn = jax.jit(jax.vmap(lambda s: collective_density(s, nv)))
    s = np.random.default_rng(seed).uniform(size=(3, N, 1))
    re, im = structure_factors(s, m, rho_fn)
    phi = np.asarray(jax.vmap(energy)(jnp.asarray(s)))
    assert np.allclose((re**2 + im**2).sum(axis=1), phi, rtol=1e-12, atol=1e-14)


@pytest.mark.medium
def test_relax_times_last_crossing_example() -> None:
    u = np.array([0.0, 0.25, 0.5, 1.0])
    sk = np.array([[1.0, 1e-3], [1e-3, 1e-3], [1.0, 1e-3], [1e-9, 1e-9]])
    out = relax_times(u, sk, (1e-2, 1e-6))
    # mode 0 is last above 1e-2 at u = 0.5; mode 1 never above 1e-2
    assert out[:, 0].tolist() == [0.5, 0.0]
    assert out[:, 1].tolist() == [0.5, 0.5]


@pytest.mark.medium
@given(m=st.integers(1, 700))
def test_mode_indices_in_range_and_include_ends(m: int) -> None:
    picks = mode_indices(m, (0.0, 0.5, 1.0))
    assert picks[0] == 1 and picks[-1] == m
    assert np.all(np.diff(picks) > 0) and np.all((picks >= 1) & (picks <= m))


# ============================================================
# TESTS: HESSIAN SECTIONS
# ============================================================


@pytest.mark.medium
def test_sample_frames_first_at_or_after() -> None:
    u = np.array([0.0, 1e-6, 1e-3, 0.2, 1.0])
    assert sample_frames(u, (0.0, 1e-5, 0.1, 1.0)).tolist() == [0, 2, 3, 4]


@pytest.mark.medium
def test_section_axes_orthonormal_and_downhill() -> None:
    rng = np.random.default_rng(5)
    grad = rng.normal(size=N)
    vecs, _ = np.linalg.qr(rng.normal(size=(N, N)))
    tau, v = section_axes(grad, vecs)
    assert tau @ grad < 0.0
    assert np.allclose([tau @ tau, v @ v, tau @ v], [1.0, 1.0, 0.0], atol=1e-12)


@pytest.mark.medium
def test_tenfold_distance_example() -> None:
    x = np.arange(5.0)[:, None] * np.ones((1, 2))
    e = np.array([100.0, 30.0, 9.0, 1.0, 0.5])
    assert tenfold_distance(x, e, 0) == pytest.approx(2.0 * np.sqrt(2.0))
    assert tenfold_distance(x, e, 2) == pytest.approx(2.0 * np.sqrt(2.0))  # 0.5 <= 0.9
    assert tenfold_distance(x, e, 3) == pytest.approx(np.sqrt(2.0))  # last frame


@pytest.mark.high
def test_density_wave_stiffness_is_k_squared_at_the_lattice() -> None:
    energy, basis, m = build_energy(N, CHI)  # N = 12, M = 4 < N - M: no aliasing
    side = float(basis[0, 0])
    x = np.arange(N, dtype=float) + 0.3  # translated integer lattice: a ground state
    h = np.asarray(energy.hessian(jnp.asarray(x[:, None] / side))) / side**2
    stiff = density_wave_stiffness(x, h, m)
    k2 = (2.0 * np.pi * np.arange(1, m + 1) / N) ** 2
    assert np.allclose(stiff, k2[:, None], rtol=1e-10)


@pytest.mark.medium
def test_wavelength_power_of_a_single_cosine() -> None:
    n, mode = 16, 3
    x = np.arange(n, dtype=float)
    v = np.cos(2.0 * np.pi * mode * x / n)
    power = wavelength_power(x, (v / np.linalg.norm(v))[:, None])
    assert int(np.argmax(power)) + 1 == mode
    assert power[mode - 1] == pytest.approx(n / 2.0)
    assert np.allclose(np.delete(power[: n // 2 - 1], mode - 1), 0.0, atol=1e-20)
    assert np.all(np.isnan(wavelength_power(x, np.zeros((n, 0)))))


@pytest.mark.medium
def test_phase_stats_radial_decay_and_one_turn() -> None:
    t = np.linspace(0.0, 1.0, 401)
    radial = 1e-2 * np.exp(-20.0 * t) * np.exp(1j * 0.7)  # fixed phase
    turning = 1e-2 * np.exp(1j * 2.0 * np.pi * t)  # one full turn, constant S
    rho = np.stack([radial, turning], axis=1)
    out = phase_stats(rho.real, rho.imag)
    assert out["turns"][0] == pytest.approx(0.0, abs=1e-12)
    assert out["turns"][1] == pytest.approx(1.0, rel=1e-9)
    assert out["s_lock"][0] == pytest.approx(1e-4)  # locked from the start
    assert out["unresolved"].tolist() == [0.0, 0.0]
