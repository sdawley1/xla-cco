#!/usr/bin/env python3

"""
Statistical validation of the MD library against exact harmonic results (V1).

System: N = 64, d = 2, K = 3.2 (M = 24 constrained waves, chi = 24/126 ~ 0.19),
a ground state from gradient flow (Phi < 1e-20), density 1 (V = N = 64).
Near the ground state Phi = |u|^2 / V with u = (Re rho_q, Im rho_q) over the
M constrained waves, i.e. 2M stiff harmonic coordinates (Hessian
H = (2/V) J^T J). In the canonical ensemble each u_i has <u_i^2> = V T / 2,
so

- <Phi> = M T and Phi / T ~ Gamma(M, 1) (half a chi^2 with 2M dof);
- <S(k)> = <|rho_k|^2> / N = T V / N (= T at density 1) for every constrained k;
- <KE> = d (N - 1) T / 2 with the COM momentum removed (NVE), d N T / 2 with
  it (Andersen redraws whole velocity rows, so it samples the COM too).

Protocol, per lane (512 independent lanes per temperature, both temperatures
in one vmapped batch): harmonic_sample positions plus full Maxwell-Boltzmann
velocities (the COM momentum kept: it is part of the Andersen ensemble, and
a COM-free start would leave the COM degree of freedom cold for ~N/(2p)
steps, biasing the Andersen <KE> by up to 1/N; the NVE segment removes it in
begin_segment) -> count-mode Andersen (p = 0.01) equilibration -> three
branches from the same equilibrated state: NVE (velocity Verlet, COM removed)
and Andersen production at dt = 0.02, and Andersen production at dt = 0.01.

Errors are lane-level: lanes are independent, so the lane means give a
Student t statistic with n - 1 dof (no within-lane correlation model is
needed; in NVE the near-integrable harmonic dynamics keep each lane's mode
energies, which within-lane blocking would miss). Each test applies a
Bonferroni correction over its own m comparisons at family rate
MD_FAMILY_ALPHA.

Velocity Verlet samples the shadow-canonical measure of the harmonic
problem (the map conserves v^2/2 + (1 - lambda dt^2 / 4) lambda x^2 / 2 per
mode, and Andersen redraws preserve it), so <Phi> / (M T) = 1 + (1/2M)
sum_i lambda_i dt^2 / (4 - lambda_i dt^2) ~ 1 + 5e-4 at dt = 0.02; the tests
assert that this predicted bias is negligible against the statistical error.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
import math
import time
from typing import Any, Dict, List, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

# ----- project -----
from systems import stealthy_ground_state
from tolerances import (
    MD_BIAS_DETECT,
    MD_CONTROL_T_FACTOR,
    MD_FAMILY_ALPHA,
    MD_KS_PVALUE,
    MD_MIN_NEFF,
    MD_POWER_Z,
    MD_SHADOW_BIAS_FRAC,
)
from xla_cco.box import Box
from xla_cco.dynamics import run
from xla_cco.dynamics.state import MDSettings, MDState
from xla_cco.dynamics.thermal import harmonic_sample
from xla_cco.energy import make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors
from xla_cco.minimize.result import STATUS_RUNNING

# ============================================================
# CONSTANTS
# ============================================================

N, D, K, GS_SEED = 64, 2, 3.2, 0
TEMPS: Tuple[float, ...] = (1e-8, 1e-6)
N_LANES = 512  # per temperature
LANE_SEED = 2026
DT, DT_HALF, P_ANDERSEN = 0.02, 0.01, 0.01
N_EQUIL, N_NVE, N_AND = 4000, 5000, 10_000
CHUNK = 2500
BRANCHES: Tuple[str, ...] = ("nve", "andersen")  # the dt = 0.02 production runs

# ============================================================
# STATISTICS
# ============================================================


def _lane_t(x: np.ndarray, mu: float) -> Tuple[float, float, float]:
    """(mean, standard error, t) of independent lane values against ``mu``."""
    assert x.ndim == 1 and x.size >= 2, "need >= 2 independent lanes"
    assert np.all(np.isfinite(x)), "non-finite lane value"
    mean = float(x.mean())
    se = float(x.std(ddof=1)) / math.sqrt(x.size)
    assert se > 0.0, "zero spread across lanes"
    return mean, se, (mean - mu) / se


def _t_crit(n: int, m: int) -> float:
    """Two-sided Bonferroni critical |t| for m comparisons of n-lane means."""
    assert n >= 2 and m >= 1, "need lanes and comparisons"
    out = float(stats.t.ppf(1.0 - MD_FAMILY_ALPHA / (2.0 * m), df=n - 1))
    assert math.isfinite(out) and out > 0.0, "bad critical value"
    return out


def _report(rows: List[Tuple[str, float, float, float]], t_crit: float, m: int) -> None:
    """Print t_crit (family of m comparisons) and one 'label mean +- se (t)'
    line per listed comparison (pytest -s)."""
    assert rows and t_crit > 0.0, "nothing to report"
    assert m >= len(rows), "the family holds every listed comparison"
    print(f"\n  t_crit = {t_crit:.3f} (m = {m})", flush=True)
    for label, mean, se, t in rows:
        print(f"  {label:<28s} {mean:.5f} +- {se:.5f}  t = {t:+.2f}", flush=True)


# ============================================================
# FIXTURES AND HELPERS
# ============================================================


@pytest.fixture(scope="module")
def system() -> Dict[str, Any]:
    """Ground state, energy, constrained waves and the stiff spectrum eig K."""
    s_gs, basis, phi = stealthy_ground_state(N, D, K, GS_SEED)
    energy = make_stealthy_energy(
        Box(basis=jnp.asarray(basis)), enumerate_kvectors(basis, 0.0, K)
    )
    waves = run.make_waves(basis, K, K)  # constrained waves only
    kvec = np.asarray(waves.kvec)
    vol = abs(float(np.linalg.det(basis)))
    phase = np.exp(1j * ((s_gs @ basis) @ kvec.T))  # (N, M)
    jac = (1j * phase.T[:, :, None] * kvec[:, None, :]).reshape(len(kvec), N * D)
    jr = np.concatenate([jac.real, jac.imag])  # (2M, N d): d(Re, Im rho)/dr
    lam = np.linalg.eigvalsh(2.0 / vol * jr @ jr.T)  # nonzero Hessian modes
    m = waves.n_constrained
    assert phi < 1e-20 and m == 24 and vol == pytest.approx(N), (phi, m, vol)
    assert lam.min() > 0.1, "2M stiff modes at a ground state"
    return dict(s_gs=s_gs, basis=basis, energy=energy, waves=waves, m=m,
                vol=vol, lam=lam)  # fmt: skip


def _initial_lanes(sys_: Dict[str, Any], st: MDSettings) -> MDState:
    """Canonical starts: harmonic positions, full MB velocities (COM kept)."""
    rng = np.random.default_rng(LANE_SEED)
    temps = np.repeat(np.asarray(TEMPS), N_LANES)
    r = np.stack([harmonic_sample(rng, sys_["energy"], sys_["s_gs"],
                                  sys_["basis"], float(t)) for t in temps])  # fmt: skip
    v = np.sqrt(temps)[:, None, None] * rng.standard_normal((temps.size, N, D))
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(temps.size))
    basis = jnp.asarray(sys_["basis"])

    def one(r0: jax.Array, v0: jax.Array, t: jax.Array, k: jax.Array) -> MDState:
        return run.init_state(sys_["energy"], sys_["waves"], basis, r0, v0, t, k, st)

    lanes: MDState = jax.vmap(one)(jnp.asarray(r), jnp.asarray(v),
                                   jnp.asarray(temps), keys)  # fmt: skip
    assert lanes.r.shape == (temps.size, N, D), "one lane per draw"
    assert np.all(np.isfinite(np.asarray(lanes.phi))), "finite start"
    return lanes


def _production(
    sys_: Dict[str, Any], st: MDSettings, state: MDState, n_steps: int
) -> MDState:
    """begin_segment (sums and winding reset) then n_steps under ``st``."""
    start = jax.vmap(lambda s: run.begin_segment(s, st, True, True))(state)
    out = run.simulate(sys_["energy"], sys_["waves"], jnp.asarray(sys_["basis"]),
                       st, start, n_steps).state  # fmt: skip
    assert np.all(np.asarray(out.status) == STATUS_RUNNING), "a lane aborted"
    assert np.all(np.asarray(out.n_sum) == n_steps), "one sample per step"
    return out


def _summary(sys_: Dict[str, Any], out: MDState, dof: int) -> Dict[str, np.ndarray]:
    """Per-lane normalized means (Phi/(MT), Phi^2/(MT)^2, KE/(dof T/2), S/T)
    and the final Phi / T."""
    t = np.asarray(out.temperature)
    n_sum = np.asarray(out.n_sum).astype(np.float64)
    mt = sys_["m"] * t
    s_ref = t * sys_["vol"] / N  # <S(k)> = T V / N
    res = dict(
        temperature=t,
        phi=np.asarray(out.sum_phi) / n_sum / mt,
        phi2=np.asarray(out.sum_phi2) / n_sum / (mt * mt),
        ke=np.asarray(out.sum_kinetic) / n_sum / (0.5 * dof * t),
        s=np.asarray(out.sum_s)[:, : sys_["m"]] / n_sum[:, None] / s_ref[:, None],
        phi_end=np.asarray(out.phi) / t,
    )
    assert res["s"].shape == (t.size, sys_["m"]), "constrained S rows"
    assert all(np.all(np.isfinite(x)) for x in res.values()), "finite summaries"
    return res


@pytest.fixture(scope="module")
def runs(system: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """Equilibrate all lanes, then the NVE / Andersen / Andersen-dt/2 branches."""
    t0 = time.monotonic()
    st_and = MDSettings(dt=DT, scheme="andersen", andersen_p=P_ANDERSEN,
                        andersen_mode="count", remove_com=False,
                        chunk_steps=CHUNK)  # fmt: skip
    st_nve = MDSettings(dt=DT, scheme="verlet", remove_com=True, chunk_steps=CHUNK)
    st_half = dataclasses.replace(st_and, dt=DT_HALF)
    lanes = _initial_lanes(system, st_and)
    eq = run.simulate(system["energy"], system["waves"], jnp.asarray(system["basis"]),
                      st_and, lanes, N_EQUIL).state  # fmt: skip
    out = {
        "nve": _summary(system, _production(system, st_nve, eq, N_NVE), D * (N - 1)),
        "andersen": _summary(system, _production(system, st_and, eq, N_AND), D * N),
        "andersen_half": _summary(
            system, _production(system, st_half, eq, N_AND), D * N
        ),
    }
    print(f"\n  V1 runs: {time.monotonic() - t0:.1f} s", flush=True)
    assert set(out) == {"nve", "andersen", "andersen_half"}, "three branches"
    return out


def _at(res: Dict[str, np.ndarray], temp: float) -> np.ndarray:
    """Boolean lane mask of temperature ``temp``."""
    mask = np.asarray(res["temperature"] == temp)
    assert int(mask.sum()) == N_LANES, "N_LANES lanes per temperature"
    return mask


def _shadow_bias(sys_: Dict[str, Any], dt: float) -> float:
    """Predicted Verlet bias of <Phi> / (M T): (1/2M) sum lam dt^2/(4 - lam dt^2)."""
    x = sys_["lam"] * dt * dt
    assert np.all(x < 4.0), "Verlet stable for every mode"
    return float(np.sum(x / (4.0 - x)) / (2 * sys_["m"]))


# ============================================================
# EQUIPARTITION
# ============================================================


@pytest.mark.slow
@pytest.mark.high
def test_energy_means_match_harmonic_values(
    system: Dict[str, Any], runs: Dict[str, Dict[str, np.ndarray]]
) -> None:
    """<Phi>/(M T) = 1 and <KE>/(dof T / 2) = 1 at both T, NVE and Andersen."""
    m = len(TEMPS) * len(BRANCHES) * 2
    t_crit = _t_crit(N_LANES, m)
    rows: List[Tuple[str, float, float, float]] = []
    for temp in TEMPS:
        for br in BRANCHES:
            mask = _at(runs[br], temp)
            for name in ("phi", "ke"):
                rows.append((f"{name} {br} T={temp:g}",
                             *_lane_t(runs[br][name][mask], 1.0)))  # fmt: skip
            se_phi = rows[-2][2]
            assert _shadow_bias(system, DT) < MD_SHADOW_BIAS_FRAC * se_phi
    _report(rows, t_crit, m)
    assert len(rows) == m
    for label, mean, se, t in rows:
        assert abs(t) < t_crit, (label, mean, se, t, t_crit)


@pytest.mark.slow
@pytest.mark.high
def test_structure_factor_each_constrained_k(
    system: Dict[str, Any], runs: Dict[str, Dict[str, np.ndarray]]
) -> None:
    """<S(k)> / (T V / N) = 1 for every constrained k, both T and branches."""
    mk = system["m"]
    m = len(TEMPS) * len(BRANCHES) * mk
    t_crit = _t_crit(N_LANES, m)
    worst: List[Tuple[str, float, float, float]] = []
    for temp in TEMPS:
        for br in BRANCHES:
            s = runs[br]["s"][_at(runs[br], temp)]
            res = [_lane_t(s[:, j], 1.0) for j in range(mk)]
            j = int(np.argmax([abs(r[2]) for r in res]))
            worst.append((f"S max|t| {br} T={temp:g} k#{j}", *res[j]))
            means = np.array([r[0] for r in res])
            print(f"  S/T {br} T={temp:g}: range [{means.min():.4f}, "
                  f"{means.max():.4f}]", flush=True)  # fmt: skip
    _report(worst, t_crit, m)
    assert len(worst) == len(TEMPS) * len(BRANCHES)
    for label, mean, se, t in worst:
        assert abs(t) < t_crit, (label, mean, se, t, t_crit)


@pytest.mark.slow
@pytest.mark.high
def test_phi_distribution_is_gamma(
    system: Dict[str, Any], runs: Dict[str, Dict[str, np.ndarray]]
) -> None:
    """KS: final Phi / T ~ Gamma(M, 1), pooled over branches per T; a 1.1 T
    report is rejected.

    Independence: one sample per lane, the NVE value from the first half of
    the lanes and the Andersen value from the second half (the branches share
    their start, so a lane contributes to one of them only).
    """
    half = N_LANES // 2
    gamma = stats.gamma(a=system["m"]).cdf
    for temp in TEMPS:
        x_nve = runs["nve"]["phi_end"][_at(runs["nve"], temp)][:half]
        x_and = runs["andersen"]["phi_end"][_at(runs["andersen"], temp)][half:]
        x = np.concatenate([x_nve, x_and])
        p = float(stats.kstest(x, gamma).pvalue)
        p_ctl = float(stats.kstest(x / MD_CONTROL_T_FACTOR, gamma).pvalue)
        print(f"  KS T={temp:g}: n = {x.size}, p = {p:.3g}, "
              f"p(1.1 T) = {p_ctl:.3g}", flush=True)  # fmt: skip
        assert x.size == N_LANES
        assert p > MD_KS_PVALUE, (temp, p)
        assert p_ctl < MD_KS_PVALUE, (temp, p_ctl)


# ============================================================
# POWER AND CONTROL
# ============================================================


@pytest.mark.slow
@pytest.mark.high
def test_power_and_temperature_control(
    runs: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """The <Phi>/(M T) test detects a 5 % bias with z >= 6 (n_eff >= 600),
    and the same data reported at 1.1 T fail it (both outcomes asserted).

    n_eff = Var(Phi / M T) / SE^2, with the per-sample variance measured from
    the running sums of Phi and Phi^2 (theory: 1/M).
    """
    m = len(TEMPS) * len(BRANCHES) * 2  # the family of the equipartition test
    t_crit = _t_crit(N_LANES, m)
    for temp in TEMPS:
        for br in BRANCHES:
            mask = _at(runs[br], temp)
            phi = runs[br]["phi"][mask]
            var1 = float(runs[br]["phi2"][mask].mean() - phi.mean() ** 2)
            mean, se, t = _lane_t(phi, 1.0)
            n_eff, z5 = var1 / se**2, MD_BIAS_DETECT / se
            _, _, t_ctl = _lane_t(phi / MD_CONTROL_T_FACTOR, 1.0)
            print(f"  power {br} T={temp:g}: var1 = {var1:.4f}, n_eff = "
                  f"{n_eff:.0f}, z(5%) = {z5:.2f}, t = {t:+.2f}, "
                  f"t(1.1 T) = {t_ctl:+.2f}", flush=True)  # fmt: skip
            assert z5 >= MD_POWER_Z and n_eff >= MD_MIN_NEFF, (br, temp, z5, n_eff)
            assert abs(t) < t_crit and abs(t_ctl) > t_crit, (br, temp, t, t_ctl)


# ============================================================
# TIME-STEP INDEPENDENCE
# ============================================================


@pytest.mark.slow
@pytest.mark.high
def test_configurational_averages_independent_of_dt(
    system: Dict[str, Any], runs: Dict[str, Dict[str, np.ndarray]]
) -> None:
    """Andersen <Phi> and <S(k)> agree at dt = 0.02 and 0.01 (paired lanes).

    Both runs start from the same equilibrated lane, so the per-lane
    differences are independent across lanes (paired t). The predicted
    Verlet bias difference (shadow measure) must be negligible against the
    error, i.e. the test checks that nothing beyond O(dt^2) depends on dt.
    """
    mk = system["m"]
    m = len(TEMPS) * (1 + mk)
    t_crit = _t_crit(N_LANES, m)
    gap = _shadow_bias(system, DT) - _shadow_bias(system, DT_HALF)
    rows: List[Tuple[str, float, float, float]] = []
    for temp in TEMPS:
        a, h = runs["andersen"], runs["andersen_half"]
        mask = _at(a, temp)
        rows.append((f"dPhi(dt) T={temp:g}",
                     *_lane_t(a["phi"][mask] - h["phi"][mask], 0.0)))  # fmt: skip
        assert gap < MD_SHADOW_BIAS_FRAC * rows[-1][2], (gap, rows[-1])
        diff_s = a["s"][mask] - h["s"][mask]
        rows.extend((f"dS(dt) T={temp:g} k#{j}", *_lane_t(diff_s[:, j], 0.0))
                    for j in range(mk))  # fmt: skip
    print(f"  predicted shadow gap {gap:.2e}", flush=True)
    _report(rows[:1] + rows[1 + mk : 2 + mk], t_crit, m)
    print(f"  dt family max|t| = {max(abs(r[3]) for r in rows):.2f}", flush=True)
    assert len(rows) == m
    for label, mean, se, t in rows:
        assert abs(t) < t_crit, (label, mean, se, t, t_crit)
