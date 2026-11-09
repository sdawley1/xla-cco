#!/usr/bin/env python3

"""
MD around a stealthy ground state: linear dynamics, spectra, winding, equipartition.

System: N = 32, d = 2, K = 3.2 (M = 12, chi = 12/62 ~ 0.19), a ground state
from gradient flow (Phi < 1e-20). References are independent of the driver:
the Cartesian Hessian from ``jax.hessian`` of a direct Phi(r), its exact
discrete velocity-Verlet propagation mode by mode (cos(w~ dt) = 1 - w^2 dt^2
/ 2) mapped through the density Jacobian J (drho = J dr), the eigenvalues of
K = (2/V) J J^T, and ``batten2008_winding.cumulative_winding`` (NumPy).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

# ----- project -----
from systems import stealthy_ground_state
from tolerances import (
    ATOL_MD_WAVE_PER_PARTICLE,
    ATOL_MD_WINDING,
    MD_FAMILY_ALPHA,
    MD_SPECTRUM_BINS,
    MD_SPECTRUM_PEAK_FRAC,
    RTOL_MD_GS_EIG,
    RTOL_MD_LINEAR,
    RTOL_MD_WINDING,
)
from xla_cco.box import Box
from xla_cco.dynamics import run
from xla_cco.dynamics.state import MDSettings, MDState
from xla_cco.dynamics.thermal import harmonic_sample
from xla_cco.energy import make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors
from xla_cco.minimize.result import STATUS_RUNNING

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
from batten2008_winding import cumulative_winding  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

N, D, K = 32, 2, 3.2
ZERO_REL = 1e-9  # Hessian eigenvalues below ZERO_REL * max are zero modes

# ============================================================
# FIXTURES AND HELPERS
# ============================================================


@pytest.fixture(scope="module")
def system() -> Dict[str, Any]:
    """Ground state, energy, waves (|k| < 2K) and the exact linear model."""
    s_gs, basis, phi = stealthy_ground_state(N, D, K, 0)
    kset = enumerate_kvectors(basis, 0.0, K)
    energy = make_stealthy_energy(Box(basis=jnp.asarray(basis)), kset)
    waves = run.make_waves(basis, K, 2.0 * K)
    r0 = s_gs @ basis
    kvec = np.asarray(waves.kvec[: waves.n_constrained])
    vol = abs(float(np.linalg.det(basis)))

    def phi_cart(r: jax.Array) -> jax.Array:
        theta = r @ jnp.asarray(kvec).T
        re, im = jnp.sum(jnp.cos(theta), 0), jnp.sum(jnp.sin(theta), 0)
        return jnp.sum(re * re + im * im) / vol

    hess = np.asarray(jax.hessian(phi_cart)(jnp.asarray(r0))).reshape(N * D, N * D)
    lam, vec = np.linalg.eigh(0.5 * (hess + hess.T))
    phase = np.exp(1j * (r0 @ kvec.T))  # (N, M)
    jac = (1j * phase.T[:, :, None] * kvec[:, None, :]).reshape(len(kvec), N * D)
    return dict(s_gs=s_gs, basis=basis, energy=energy, waves=waves, r0=r0, lam=lam,
                vec=vec, jac=jac, kvec=kvec, vol=vol, m=len(kvec), phi=phi)  # fmt: skip


def _lane(
    sys_: Dict[str, Any], temperature: float, seed: int, st: MDSettings,
    full_mb: bool = False,
) -> Tuple[MDState, np.ndarray, np.ndarray]:  # fmt: skip
    """Harmonic positions; velocities in the nonzero-mode subspace (linear
    tests) or, with ``full_mb``, full Maxwell-Boltzmann with the COM removed.

    Returns the batched one-lane state and the initial (displacement,
    velocity) as flat (N d,) vectors.
    """
    rng = np.random.default_rng(seed)
    lam, vec = sys_["lam"], sys_["vec"]
    nz = lam > ZERO_REL * lam[-1]
    assert int(nz.sum()) == 2 * sys_["m"], "2M nonzero modes at a ground state"
    r = harmonic_sample(rng, sys_["energy"], sys_["s_gs"], sys_["basis"], temperature)
    v = vec[:, nz] @ (math.sqrt(temperature) * rng.standard_normal(int(nz.sum())))
    if full_mb:  # flat modes must not start cold (resets would feed them)
        v = math.sqrt(temperature) * rng.standard_normal((N, D))
        v = (v - v.mean(axis=0)).reshape(-1)
    lane = run.init_state(
        sys_["energy"], sys_["waves"], jnp.asarray(sys_["basis"]), jnp.asarray(r),
        jnp.asarray(v.reshape(N, D)), jnp.asarray(temperature),
        jax.random.PRNGKey(seed), st)  # fmt: skip
    batch: MDState = jax.tree_util.tree_map(lambda x: x[None], lane)
    return batch, (r - sys_["r0"]).reshape(-1), v


def _verlet_modes(
    lam: np.ndarray, c0: np.ndarray, b0: np.ndarray, t: np.ndarray, dt: float
) -> np.ndarray:
    """Exact velocity-Verlet mode coordinates c(t) (len(t), modes).

    For lam > 0: c = c0 cos(w~ t) + dt b0 / sin(w~ dt) sin(w~ t) with
    cos(w~ dt) = 1 - lam dt^2 / 2 (the discrete map's exact solution);
    zero modes drift: c = c0 + b0 t.
    """
    pos = lam > ZERO_REL * lam.max()
    w = np.where(pos, np.arccos(1.0 - 0.5 * np.clip(lam, 0, None) * dt * dt) / dt, 0.0)
    sin_dt = np.where(pos, np.sin(w * dt), 1.0)
    osc = c0 * np.cos(np.outer(t, w)) + dt * b0 / sin_dt * np.sin(np.outer(t, w))
    return np.asarray(np.where(pos, osc, c0 + np.outer(t, b0)))


def _complex(rho: jax.Array) -> np.ndarray:
    """(..., 2) (Re, Im) -> complex (...)."""
    a = np.asarray(rho)
    return np.asarray(a[..., 0] + 1j * a[..., 1])


# ============================================================
# LINEAR DYNAMICS AND SPECTRA
# ============================================================


@pytest.mark.critical
def test_linearized_dynamics_match_hessian_modes(system: Dict[str, Any]) -> None:
    """rho_k(t) of the constrained waves = rho(r0) + J dr(t), dr from the modes."""
    dt, n_steps, every, temp = 0.01, 5000, 50, 1e-10
    st = MDSettings(dt=dt, scheme="verlet", chunk_steps=n_steps, record_rho_every=every,
                    max_rho_records=n_steps // every)  # fmt: skip
    lanes, dr0, v0 = _lane(system, temp, 11, st)
    out = run.simulate(system["energy"], system["waves"], jnp.asarray(system["basis"]),
                       st, lanes, n_steps)  # fmt: skip
    assert int(out.state.status[0]) == STATUS_RUNNING and out.trace is not None
    t = dt * every * np.arange(1, n_steps // every + 1)
    vec = system["vec"]
    c = _verlet_modes(system["lam"], vec.T @ dr0, vec.T @ v0, t, dt)
    rho_r0 = np.exp(1j * (system["r0"] @ system["kvec"].T)).sum(0)
    pred = rho_r0 + (c @ vec.T) @ system["jac"].T  # (T, M)
    got = _complex(out.trace.rho[0, :, : system["m"]])
    scale = math.sqrt(float(np.mean(np.sum(np.abs(pred) ** 2, axis=1))))
    err = float(np.max(np.sqrt(np.sum(np.abs(got - pred) ** 2, axis=1)))) / scale
    assert t[-1] == pytest.approx(50.0) and scale > 1e3 * abs(rho_r0).max()
    assert err <= RTOL_MD_LINEAR, err


@pytest.mark.high
def test_power_spectrum_peaks_on_sqrt_eig_k(system: Dict[str, Any]) -> None:
    """FFT peaks of rho_k(t) lie on sqrt(eig K), K = (2/V) J J^T (real J)."""
    dt, n_steps, every, temp = 0.02, 20_000, 4, 1e-8
    st = MDSettings(dt=dt, scheme="verlet", chunk_steps=n_steps, record_rho_every=every,
                    max_rho_records=n_steps // every)  # fmt: skip
    lanes, dr0, v0 = _lane(system, temp, 5, st)
    out = run.simulate(system["energy"], system["waves"], jnp.asarray(system["basis"]),
                       st, lanes, n_steps)  # fmt: skip
    assert out.trace is not None and int(out.trace.count[0]) == n_steps // every
    rho = _complex(out.trace.rho[0, :, : system["m"]])
    rho = (rho - rho.mean(0)) * np.hanning(rho.shape[0])[:, None]
    power = np.sum(np.abs(np.fft.fft(rho, axis=0)) ** 2, axis=1)
    omega = 2.0 * math.pi * np.fft.fftfreq(rho.shape[0], d=dt * every)
    half = rho.shape[0] // 2
    folded = power[:half] + np.concatenate([[0.0], power[-1:-half:-1]])
    jr = np.concatenate([system["jac"].real, system["jac"].imag])  # (2M, N d)
    eig_k = np.linalg.eigvalsh(2.0 / system["vol"] * jr @ jr.T)
    w_modes = np.sqrt(eig_k)
    lam = system["lam"][system["lam"] > ZERO_REL * system["lam"][-1]]
    np.testing.assert_allclose(np.sort(eig_k), np.sort(lam), rtol=RTOL_MD_GS_EIG)
    bin_w = 2.0 * math.pi / (dt * n_steps)
    peaks = [i for i in range(1, half - 1) if folded[i] >= folded[i - 1]
             and folded[i] >= folded[i + 1]
             and folded[i] > MD_SPECTRUM_PEAK_FRAC * folded.max()]  # fmt: skip
    assert len(peaks) >= 3
    for i in peaks:
        assert np.min(np.abs(w_modes - omega[i])) <= MD_SPECTRUM_BINS * bin_w, i
    # converse: every resolved mode (no other mode within 3 bins, else the two
    # merge) carrying >= 5 % of the largest modal power has a peak nearby
    nz = system["lam"] > ZERO_REL * system["lam"][-1]
    amp = np.sum(np.abs(system["jac"] @ system["vec"][:, nz]) ** 2, axis=0)
    a0, b0 = system["vec"][:, nz].T @ dr0, system["vec"][:, nz].T @ v0
    weight = amp * (a0**2 + b0**2 / system["lam"][nz])
    w_nz = np.sqrt(system["lam"][nz])
    gap = np.array([np.min(np.abs(np.delete(w_nz, i) - w)) for i, w in enumerate(w_nz)])
    check = (weight > 0.05 * weight.max()) & (gap > 3.0 * bin_w)
    assert int(check.sum()) >= 8, "most modes are resolved and excited"
    for w_i in w_nz[check]:
        near = min(abs(omega[i] - w_i) for i in peaks)
        assert near <= MD_SPECTRUM_BINS * bin_w, w_i


# ============================================================
# WINDING
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("scheme,every", [("verlet", 1), ("andersen", 2), ("baoab", 3)])
def test_winding_matches_offline_cumulative_winding(
    system: Dict[str, Any], scheme: str, every: int
) -> None:
    """wind_abs / wind_net / n_valid equal an offline recomputation from the trace."""
    n_steps = 240
    st = MDSettings(dt=0.02, scheme=scheme, andersen_p=0.2, andersen_mode="bernoulli",
                    gamma=0.5, chunk_steps=100, wind_every=every, record_rho_every=1,
                    max_rho_records=n_steps, record_steps=(7, 100),
                    drift_tol=math.inf)  # fmt: skip
    lanes, _, _ = _lane(system, 1e-4, 3, st)
    rho0 = _complex(lanes.rho_prev[0])
    out = run.simulate(system["energy"], system["waves"], jnp.asarray(system["basis"]),
                       st, lanes, n_steps)  # fmt: skip
    assert out.trace is not None
    frames = np.concatenate([rho0[None], _complex(out.trace.rho[0])])
    nc = system["waves"].n_constrained
    cad = np.arange(0, n_steps + 1, every)
    for cols, idx in [(slice(0, nc), np.arange(n_steps + 1)), (slice(nc, None), cad)]:
        path = frames[idx][:, cols]
        w_ref = cumulative_winding(path, N)[-1] * 2.0 * math.pi
        net = np.angle(path[1:] * np.conj(path[:-1])).sum(0)
        got_abs = np.asarray(out.state.wind_abs[0, cols])
        np.testing.assert_allclose(got_abs, w_ref, rtol=RTOL_MD_WINDING,
                                   atol=ATOL_MD_WINDING)  # fmt: skip
        got_net = np.asarray(out.state.wind_net[0, cols])
        np.testing.assert_allclose(got_net, net, rtol=RTOL_MD_WINDING,
                                   atol=ATOL_MD_WINDING)  # fmt: skip
        assert np.all(np.asarray(out.state.n_valid[0, cols]) == len(idx) - 1)
    assert float(out.state.wind_abs[0].min()) > 0.0
    _check_snapshot_rho(system, out.state, scheme)


def _check_snapshot_rho(sys_: Dict[str, Any], state: MDState, scheme: str) -> None:
    """Snapshot rho = direct sum at the snapshot positions (COM rule of scheme)."""
    kv = np.asarray(sys_["waves"].kvec)
    snap = state.snap
    assert np.asarray(snap.step[0]).tolist() == [7, 100]
    for j in range(2):
        r = np.asarray(snap.r[0, j])
        rho = np.exp(1j * (r @ kv.T)).sum(0)
        if scheme != "verlet":
            rho = rho * np.exp(-1j * (kv @ r.mean(0)))
        got = _complex(snap.rho[0, j])
        every = 1 if scheme == "verlet" else (2 if scheme == "andersen" else 3)
        fresh = np.ones(kv.shape[0], bool)
        fresh[sys_["waves"].n_constrained :] = int(snap.step[0, j]) % every == 0
        np.testing.assert_allclose(got[fresh], rho[fresh],
                                   atol=ATOL_MD_WAVE_PER_PARTICLE * N)  # fmt: skip


# ============================================================
# EQUIPARTITION (smoke)
# ============================================================


@pytest.mark.high
def test_equipartition_smoke_andersen(system: Dict[str, Any]) -> None:
    """<Phi> / (M T) = 1 over 32 Andersen lanes at T = 1e-8 (t-test on lanes).

    Each lane starts canonical (harmonic positions, full MB velocities with
    the COM removed); a nonzero-mode-only start leaves the flat modes cold
    and the resets then drain the stiff modes (-7 %, AUDIT-T5). The lane
    means are independent, so (mean - 1) / SE is Student t with n - 1 dof;
    the critical value is t_{n-1}(1 - MD_FAMILY_ALPHA / 2) = 3.63 at n = 32.
    """
    temp, n_lanes, n_steps = 1e-8, 32, 4000
    st = MDSettings(dt=0.02, scheme="andersen", andersen_p=0.01, chunk_steps=n_steps)
    lanes = [_lane(system, temp, 100 + i, st, full_mb=True)[0] for i in range(n_lanes)]
    batch: MDState = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x), *lanes)
    out = run.simulate(system["energy"], system["waves"], jnp.asarray(system["basis"]),
                       st, batch, n_steps).state  # fmt: skip
    ratio = np.asarray(out.sum_phi) / np.asarray(out.n_sum) / (system["m"] * temp)
    se = float(np.std(ratio, ddof=1) / math.sqrt(n_lanes))
    t_stat = (float(ratio.mean()) - 1.0) / se
    t_crit = float(stats.t.ppf(1.0 - MD_FAMILY_ALPHA / 2.0, df=n_lanes - 1))
    assert np.all(np.asarray(out.n_sum) == n_steps) and se < 0.1
    assert abs(t_stat) < t_crit, (ratio.mean(), se, t_crit)
    # power: the same data reject a 30 % temperature error
    assert abs((float(ratio.mean()) / 1.3 - 1.0) / se) > t_crit
