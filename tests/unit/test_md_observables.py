#!/usr/bin/env python3

"""
Tests of the MD observables: wave densities, COM correction and winding.

References are independent: ``collective_density`` in fractional
coordinates, ``oracle.cumulative_winding`` (NumPy, np.angle),
closed-form rotations and the Rice formula for a circular complex Gaussian
process, <|theta'|> = sigma_rho' / sigma_rho.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from typing import Any, Optional, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# ----- project -----
from oracle import PHASE_FLOOR, cumulative_winding
from strategies import boxes, chunk_sizes, fractional_configs, integer_kvectors
from tolerances import (
    ATOL_MD_COM_MEAN,
    ATOL_MD_COM_RHO,
    ATOL_MD_COM_WIND,
    ATOL_MD_WAVE_PER_PARTICLE,
    ATOL_MD_WIND_ONE_STEP,
    ATOL_MD_WINDING,
    MD_RICE_Z,
    RTOL_MD_WINDING,
)
from xla_cco.dynamics.observables import (
    center_of_mass,
    com_correct,
    wave_densities,
    winding_update,
)
from xla_cco.energy import collective_density

# ============================================================
# CONSTANTS
# ============================================================

FAST = settings(deadline=None, max_examples=60)
# test floor of the COM property test: keeps every counted step well conditioned
# (|rho| >= 1e-2), see ATOL_MD_COM_WIND
COM_TEST_FLOOR = 1e-4

# ============================================================
# HELPERS
# ============================================================


def _to_pairs(z: np.ndarray) -> jnp.ndarray:
    """Complex (..., w) -> float64 (..., w, 2) = (Re, Im)."""
    return jnp.asarray(np.stack([z.real, z.imag], axis=-1))


def _run_winding(
    rho: np.ndarray, n_part: int, floor: float, active: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan winding_update over a complex path (F, w); per-frame wind_abs/net.

    Returns (wind_abs (F, w), wind_net (F, w), n_valid (w,), rho_prev (w, 2)),
    frame 0 holding zeros, as cumulative_winding.
    """
    path = _to_pairs(rho)
    w = rho.shape[1]
    act = jnp.ones(w, dtype=bool) if active is None else jnp.asarray(active)

    def body(carry: Tuple[Any, ...], new: Any) -> Tuple[Tuple[Any, ...], Any]:
        prev, wabs, wnet, nval, _ = carry
        prev, wabs, wnet, nval = winding_update(
            prev, new, wabs, wnet, nval, act, n_part=n_part, phase_floor=floor
        )
        return (prev, wabs, wnet, nval, act), (wabs, wnet)

    def run(p: Any) -> Any:
        zero = jnp.zeros(w)
        init = (p[0], zero, zero, jnp.zeros(w, dtype=jnp.int32), act)
        carry, (wabs, wnet) = jax.lax.scan(body, init, p[1:])
        return carry, wabs, wnet

    carry, wabs, wnet = jax.jit(run)(path)
    head = np.zeros((1, w))
    return (
        np.concatenate([head, np.asarray(wabs)]),
        np.concatenate([head, np.asarray(wnet)]),
        np.asarray(carry[3]),
        np.asarray(carry[0]),
    )


def _update(
    old: np.ndarray,
    new: np.ndarray,
    active: np.ndarray,
    n_part: int,
    floor: float = PHASE_FLOOR,
) -> Tuple[np.ndarray, ...]:
    """One winding_update from zero accumulators; NumPy outputs."""
    w = old.shape[0]
    acc = jnp.full(w, 0.25)  # nonzero, so "unchanged" is distinguishable
    out = winding_update(
        _to_pairs(old), _to_pairs(new), acc, -acc, jnp.full(w, 3, dtype=jnp.int32),
        jnp.asarray(active), n_part, floor,
    )  # fmt: skip
    return tuple(np.asarray(x) for x in out)


def _signed_oracle(rho: np.ndarray, n: int, floor: float) -> Tuple[np.ndarray, ...]:
    """NumPy net winding and valid-step count, independent of the code under test."""
    sk = np.abs(rho) ** 2 / n
    dth = np.angle(rho[1:] * np.conj(rho[:-1]))
    ok = (sk[1:] > floor) & (sk[:-1] > floor)
    return np.sum(np.where(ok, dth, 0.0), axis=0), np.sum(ok, axis=0)


def _synthetic_path(seed: int, frames: int, n_part: int) -> np.ndarray:
    """Complex rho (F, 8): random walks, half-turns and sub-floor passages."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=1.2, size=(frames, 5))  # large angles, both signs
    amp = np.exp(rng.normal(size=(frames, 5)))
    rho = np.empty((frames, 8), dtype=complex)
    rho[:, :5] = amp * np.exp(1j * np.cumsum(steps, axis=0))
    rho[::7, 1] *= 1e-9  # |rho| > floor but S = |rho|^2 / N < floor
    rho[::11, 2] = 0.0
    rho[:, 5] = np.where(np.arange(frames) % 2 == 0, 3.0, -3.0)  # exact half-turns
    rho[:, 6] = 3e-8 * np.exp(1j * 0.3 * np.arange(frames))  # S = 9e-16 / N
    rho[:, 7] = 2.0 * np.exp(-1j * np.pi * np.arange(frames) / 2.0)
    assert n_part > 9, "column 6 must sit below the floor"
    return rho


# ============================================================
# WAVE DENSITIES AND CENTRE OF MASS
# ============================================================


def _box_cases() -> Tuple[np.ndarray, ...]:
    """A rectangular (32 x 16 sqrt 3, scaled) and a sheared 2D basis."""
    rect = np.diag([4.0, 2.0 * math.sqrt(3.0)])
    shear = np.array([[2.0, 0.0], [0.7, 1.5]])
    return rect, shear


@pytest.mark.critical
@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("chunk", [None, 1, 3, 7, 13, 100])
def test_wave_densities_match_fractional_density(which: int, chunk: Any) -> None:
    basis = _box_cases()[which]
    rng = np.random.default_rng(4 + which)
    s = rng.uniform(-2.0, 3.0, size=(11, 2))  # unwrapped fractional positions
    n_vec = np.vstack([rng.integers(-4, 5, size=(12, 2)), np.zeros((1, 2))])
    kvec = n_vec @ (2.0 * np.pi * np.linalg.inv(basis).T)
    r = s @ basis
    out = np.asarray(wave_densities(jnp.asarray(r), jnp.asarray(kvec), chunk))
    s_ref = jnp.asarray(r @ np.linalg.inv(basis))
    re, im = collective_density(s_ref, jnp.asarray(n_vec.astype(np.float64)))
    assert out.shape == (13, 2) and out.dtype == np.float64
    atol = ATOL_MD_WAVE_PER_PARTICLE * s.shape[0]
    np.testing.assert_allclose(out[:, 0], np.asarray(re), rtol=0, atol=atol)
    np.testing.assert_allclose(out[:, 1], np.asarray(im), rtol=0, atol=atol)
    assert out[-1, 0] == 11.0 and out[-1, 1] == 0.0  # k = 0 row: rho = N


@pytest.mark.high
@FAST
@given(
    basis=boxes(),
    data=st.data(),
    chunk=chunk_sizes(),
)
def test_wave_densities_property_matches_fractional(
    basis: np.ndarray, data: Any, chunk: Any
) -> None:
    d = basis.shape[0]
    s = data.draw(fractional_configs(dim=d))
    shift = data.draw(
        hnp.arrays(np.int64, s.shape, elements=st.integers(-3, 3)), label="unwrap"
    )
    n_vec = data.draw(integer_kvectors(dim=d)).astype(np.float64)
    r = (s + shift) @ basis
    kvec = n_vec @ (2.0 * np.pi * np.linalg.inv(basis).T)
    out = np.asarray(wave_densities(jnp.asarray(r), jnp.asarray(kvec), chunk))
    re, im = collective_density(
        jnp.asarray(r @ np.linalg.inv(basis)), jnp.asarray(n_vec)
    )
    atol = ATOL_MD_WAVE_PER_PARTICLE * s.shape[0]
    np.testing.assert_allclose(out[:, 0], np.asarray(re), rtol=0, atol=atol)
    np.testing.assert_allclose(out[:, 1], np.asarray(im), rtol=0, atol=atol)


@pytest.mark.medium
def test_wave_densities_rejects_float32() -> None:
    r = jnp.zeros((3, 2), dtype=jnp.float32)
    with pytest.raises(TypeError):
        wave_densities(r, jnp.ones((2, 2)))


@pytest.mark.high
def test_center_of_mass_and_com_correct_example() -> None:
    rng = np.random.default_rng(8)
    r = rng.normal(size=(9, 3)) * 5.0
    kvec = rng.normal(size=(4, 3))
    r_cm = np.asarray(center_of_mass(jnp.asarray(r)))
    np.testing.assert_allclose(r_cm, r.mean(axis=0), rtol=0, atol=ATOL_MD_COM_MEAN)
    rho = np.exp(1j * (r @ kvec.T)).sum(axis=0)
    out = np.asarray(com_correct(_to_pairs(rho), jnp.asarray(kvec), jnp.asarray(r_cm)))
    ref = rho * np.exp(-1j * (kvec @ r_cm))  # analytic: rho_k exp(-i k . R_cm)
    np.testing.assert_allclose(
        out[:, 0] + 1j * out[:, 1], ref, rtol=0, atol=ATOL_MD_COM_RHO
    )


@pytest.mark.critical
def test_com_corrected_single_particle_never_winds() -> None:
    """N = 1: rho exp(-i k . R_cm) = 1 exactly in exact arithmetic, at any r."""
    rng = np.random.default_rng(2)
    traj = np.cumsum(rng.normal(scale=0.4, size=(60, 1, 2)), axis=0)
    kvec = jnp.asarray(rng.normal(scale=2.0, size=(5, 2)))
    rho = np.empty((60, 5), dtype=complex)
    for t in range(60):
        r = jnp.asarray(traj[t])
        out = np.asarray(com_correct(wave_densities(r, kvec), kvec, center_of_mass(r)))
        rho[t] = out[:, 0] + 1j * out[:, 1]
    np.testing.assert_allclose(rho, 1.0 + 0.0j, rtol=0, atol=ATOL_MD_COM_RHO)
    wabs, wnet, n_valid, _ = _run_winding(rho, n_part=1, floor=PHASE_FLOOR)
    assert np.all(n_valid == 59)
    np.testing.assert_allclose(wabs[-1], 0.0, atol=ATOL_MD_WINDING)
    np.testing.assert_allclose(wnet[-1], 0.0, atol=ATOL_MD_WINDING)


def _corrected_path(traj: np.ndarray, kvec: np.ndarray) -> np.ndarray:
    """COM-corrected complex rho (F, w) of a Cartesian trajectory (F, n, d)."""
    k = jnp.asarray(kvec)
    out = np.empty((traj.shape[0], kvec.shape[0]), dtype=complex)
    for t in range(traj.shape[0]):
        r = jnp.asarray(traj[t])
        pair = np.asarray(com_correct(wave_densities(r, k), k, center_of_mass(r)))
        out[t] = pair[:, 0] + 1j * pair[:, 1]
    return out


@pytest.mark.critical
@FAST
@given(data=st.data(), basis=boxes(dim=2))
@example(data=None, basis=np.eye(2))
def test_com_winding_invariant_under_rigid_translations(
    data: Any, basis: np.ndarray
) -> None:
    if data is None:  # pinned example: fixed seed, N = 5, unit cell
        rng = np.random.default_rng(0)
        s0 = rng.uniform(size=(5, 2))
        moves = rng.normal(scale=0.1, size=(8, 5, 2))
        trans = rng.uniform(-8.0, 8.0, size=(8, 2))
        n_vec = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -2.0]])
    else:
        s0 = data.draw(fractional_configs(n_grid=(2, 7, 16, 40), dim=2))
        n = s0.shape[0]
        small = st.floats(-0.2, 0.2, allow_nan=False)
        moves = data.draw(hnp.arrays(np.float64, (8, n, 2), elements=small))
        big = st.floats(-8.0, 8.0, allow_nan=False)
        trans = data.draw(hnp.arrays(np.float64, (8, 2), elements=big))
        n_vec = data.draw(integer_kvectors(dim=2, max_m=6)).astype(np.float64)
    kvec = n_vec @ (2.0 * np.pi * np.linalg.inv(basis).T)
    traj = (s0[None] + np.cumsum(moves, axis=0)) @ basis
    rho_a = _corrected_path(traj, kvec)
    rho_b = _corrected_path(traj + trans[:, None, :], kvec)
    wa, na, va, _ = _run_winding(rho_a, s0.shape[0], COM_TEST_FLOOR)
    wb, nb, vb, _ = _run_winding(rho_b, s0.shape[0], COM_TEST_FLOOR)
    np.testing.assert_array_equal(va, vb)
    np.testing.assert_allclose(wa[-1], wb[-1], rtol=0, atol=ATOL_MD_COM_WIND)
    # net winding modulo 2 pi: an exact half-turn may round to +pi or -pi
    gap = np.angle(np.exp(1j * (na[-1] - nb[-1])))
    np.testing.assert_allclose(gap, 0.0, rtol=0, atol=ATOL_MD_COM_WIND)


# ============================================================
# WINDING
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_winding_matches_cumulative_winding(seed: int) -> None:
    n_part = 40
    rho = _synthetic_path(seed, frames=400, n_part=n_part)
    wabs, wnet, n_valid, _ = _run_winding(rho, n_part, PHASE_FLOOR)
    ref = cumulative_winding(rho, n_part) * 2.0 * np.pi  # turns -> radians
    tol = dict(rtol=RTOL_MD_WINDING, atol=ATOL_MD_WINDING)
    np.testing.assert_allclose(wabs, ref, **tol)
    net_ref, ok_ref = _signed_oracle(rho, n_part, PHASE_FLOOR)
    np.testing.assert_allclose(wnet[-1], net_ref, **tol)
    np.testing.assert_array_equal(n_valid, ok_ref)
    assert n_valid[6] == 0 and wabs[-1, 6] == 0.0  # sub-floor wave never counts
    assert np.all(wabs[-1, :5] > np.abs(wnet[-1, :5]))  # back-and-forth walks


@pytest.mark.critical
def test_winding_half_turn_is_atan2_of_product() -> None:
    """(1, 0) -> (-1, 0): Im(rho_new conj rho_old) = +0, so Delta = +pi."""
    rho = np.array([[1.0 + 0.0j], [-1.0 + 0.0j]])
    wabs, wnet, n_valid, _ = _run_winding(rho, n_part=1, floor=PHASE_FLOOR)
    assert wnet[-1, 0] == np.angle(rho[1, 0] * np.conj(rho[0, 0])) == np.pi
    assert wabs[-1, 0] == np.pi and n_valid[0] == 1


@pytest.mark.critical
@pytest.mark.parametrize("omega", [0.3, -0.3, 2.9, -2.9])
def test_winding_of_exact_rotation(omega: float) -> None:
    dt, frames, amp = 1.0, 301, 5.0
    t = dt * np.arange(frames)
    rho = amp * np.exp(1j * omega * t)[:, None]
    wabs, wnet, n_valid, prev = _run_winding(rho, n_part=10, floor=PHASE_FLOOR)
    tol = dict(rtol=RTOL_MD_WINDING, atol=ATOL_MD_WINDING)
    np.testing.assert_allclose(wnet[:, 0], omega * t, **tol)
    np.testing.assert_allclose(wabs[:, 0], abs(omega) * t, **tol)
    assert n_valid[0] == frames - 1
    np.testing.assert_array_equal(prev[0], [rho[-1, 0].real, rho[-1, 0].imag])


def _gaussian_paths(seed: int, length: int, dt: float, waves: int) -> Tuple[Any, ...]:
    """Stationary circular complex Gaussian paths (L, w) and sigma_rho'/sigma_rho.

    rho(t) = sum_k c_k exp(i omega_k t), c_k ~ CN(0, P_k), P symmetric in omega
    (real autocorrelation: rho and rho' are uncorrelated at equal times), so
    sigma_rho^2 = sum P_k and sigma_rho'^2 = sum P_k omega_k^2 exactly.
    """
    rng = np.random.default_rng(seed)
    omega = 2.0 * np.pi * np.fft.fftfreq(length, d=dt)
    power = np.exp(-0.5 * omega**2)  # correlation time ~ 1 >> dt
    cn = rng.normal(size=(2, waves, length)) * np.sqrt(power / 2.0)
    rho = length * np.fft.ifft(cn[0] + 1j * cn[1], axis=1)
    ratio = math.sqrt(float(np.sum(power * omega**2)) / float(np.sum(power)))
    assert np.all(np.isfinite(rho)) and ratio > 0.0, "bad synthetic process"
    return rho.T, ratio


@pytest.mark.high
def test_rice_formula_mean_phase_speed() -> None:
    dt, length, waves = 0.02, 1 << 15, 128
    rho, ratio = _gaussian_paths(seed=11, length=length, dt=dt, waves=waves)
    wabs, wnet, n_valid, _ = _run_winding(rho, n_part=1, floor=PHASE_FLOOR)
    assert np.all(n_valid == length - 1)
    speed = wabs[-1] / ((length - 1) * dt)  # <|theta'|> per realization
    z = (speed.mean() - ratio) / (speed.std(ddof=1) / math.sqrt(waves))
    assert abs(z) < MD_RICE_Z, f"z = {z:.2f}, mean {speed.mean():.4f} vs {ratio:.4f}"
    # no net drift: the mean net winding is zero within the same z criterion
    z_net = np.mean(wnet[-1]) / (np.std(wnet[-1], ddof=1) / math.sqrt(waves))
    assert abs(z_net) < MD_RICE_Z, f"net winding z = {z_net:.2f}"


@pytest.mark.critical
def test_winding_floor_uses_s_not_modulus() -> None:
    """|rho| = 3e-8 > floor but S = 9e-16 / 100 < floor: the step is excluded."""
    n_part, small = 100, 3e-8
    old = np.array([small, 1.0, 1.0, small, 1.0])
    new = np.array([1j, small * 1j, 1j, small * 1j, 1j])
    prev, wabs, wnet, n_valid = _update(old, new, np.ones(5, bool), n_part)
    np.testing.assert_array_equal(n_valid, [3, 3, 4, 3, 4])
    np.testing.assert_array_equal(wabs[[0, 1, 3]], 0.25)
    np.testing.assert_array_equal(wnet[[0, 1, 3]], -0.25)
    np.testing.assert_allclose(
        wabs[[2, 4]], 0.25 + np.pi / 2, rtol=0, atol=ATOL_MD_WIND_ONE_STEP
    )
    np.testing.assert_allclose(
        wnet[[2, 4]], -0.25 + np.pi / 2, rtol=0, atol=ATOL_MD_WIND_ONE_STEP
    )
    np.testing.assert_array_equal(prev, np.stack([new.real, new.imag], axis=-1))


@pytest.mark.critical
def test_winding_floor_counts_above_threshold() -> None:
    """S just above the floor at both ends (N = 1, |rho| = 2e-8) counts."""
    old, new = np.array([2e-8 + 0j]), np.array([-2e-8j])
    _, wabs, wnet, n_valid = _update(old, new, np.ones(1, bool), n_part=1)
    assert n_valid[0] == 4
    np.testing.assert_allclose(
        [wabs[0], wnet[0]],
        [0.25 + np.pi / 2, -0.25 - np.pi / 2],
        rtol=0,
        atol=ATOL_MD_WIND_ONE_STEP,
    )


@pytest.mark.critical
def test_winding_inactive_waves_keep_state() -> None:
    rng = np.random.default_rng(5)
    old = rng.normal(size=6) + 1j * rng.normal(size=6)
    new = rng.normal(size=6) + 1j * rng.normal(size=6)
    active = np.array([True, False, True, False, False, True])
    prev, wabs, wnet, n_valid = _update(old, new, active, n_part=4)
    pick = np.where(active[:, None], np.stack([new.real, new.imag], -1), 0.0)
    keep = np.where(~active[:, None], np.stack([old.real, old.imag], -1), 0.0)
    np.testing.assert_array_equal(prev, pick + keep)
    np.testing.assert_array_equal(wabs[~active], 0.25)
    np.testing.assert_array_equal(wnet[~active], -0.25)
    np.testing.assert_array_equal(n_valid, np.where(active, 4, 3))
    dth = np.angle(new * np.conj(old))[active]
    np.testing.assert_allclose(
        wnet[active], -0.25 + dth, rtol=0, atol=ATOL_MD_WIND_ONE_STEP
    )


@pytest.mark.medium
def test_winding_update_shapes_and_dtypes() -> None:
    w = 4
    zero = jnp.zeros(w)
    out = winding_update(
        jnp.ones((w, 2)), jnp.ones((w, 2)), zero, zero,
        jnp.zeros(w, dtype=jnp.int32), jnp.ones(w, dtype=bool), 3, 1e-16,
    )  # fmt: skip
    assert [x.shape for x in out] == [(w, 2), (w,), (w,), (w,)]
    assert out[3].dtype == jnp.int32 and out[1].dtype == jnp.float64
