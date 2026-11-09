#!/usr/bin/env python3

"""
Tests for the MD winding analysis, statistics and synthetic generator
(experiments/scripts/batten2008_md_{analysis,stats,synthetic}.py).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy import stats as sps

# ----- project -----
from tests.tolerances import (
    ATOL_MD_ALPHA_EXACT,
    ATOL_MD_ALPHA_OU,
    ATOL_MD_ALPHA_RW,
    ATOL_MD_SYN_ALPHA,
    MD_H5_NULL_CELLS,
    MD_H5_NULL_PASS,
    MD_H5_NULL_SET_MIN,
    MD_H5_NULL_SETS,
    RTOL_MD_D_RW,
    RTOL_MD_RATE_EXACT,
    RTOL_MD_SYN_EQUIPARTITION,
    RTOL_MD_SYN_RATE,
    RTOL_MD_WINDOW_EXACT,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
import batten2008_md_analysis as ana  # noqa: E402
import batten2008_md_stats as mds  # noqa: E402
import batten2008_md_synthetic as syn  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

TWO_PI = 2.0 * np.pi
CLI_DT = 0.01  # time step of the cli_files fixtures
finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False,
                   allow_infinity=False, allow_subnormal=False)  # fmt: skip
samples = hnp.arrays(np.float64, st.integers(1, 12), elements=finite)

# ============================================================
# HELPERS
# ============================================================


def _cli_windings(
    omega_abs: np.ndarray, omega_net: np.ndarray, net_msd: Any
) -> Tuple[List[Any], np.ndarray, np.ndarray, np.ndarray]:
    """(phases, record steps, W_abs, W_net) of cli_files (radians, dt = 0.01)."""
    lanes, waves = omega_abs.shape
    phases, rec = syn._protocol(syn.default_proto(100_000, 2, 120))
    segs = ana.segments({"phases": [dict(p.__dict__) for p in phases]})
    w_abs = np.zeros((lanes, rec.size, waves))
    w_net = np.zeros((lanes, rec.size, waves))
    for s0, n in segs:
        j = np.flatnonzero((rec > s0) & (rec <= s0 + n))
        t = (rec[j] - s0) * CLI_DT
        w_abs[:, j] = omega_abs[:, None, :] * t[None, :, None]
        if net_msd is None:
            w_net[:, j] = omega_net[:, None, :] * t[None, :, None]
        else:
            sign = np.where(np.arange(waves) % 2 == 0, 1.0, -1.0)
            w_net[:, j] = TWO_PI * np.sqrt(net_msd(t))[None, :, None] * sign
    assert len(segs) == 2 and w_abs.shape == w_net.shape, "two segments"
    return phases, rec, w_abs, w_net


def _cli_trace(
    phases: List[Any], s_mean: np.ndarray, every: int
) -> Dict[str, np.ndarray]:
    """Uniform rho trace every ``every`` steps over ALL phases (A4 fixture).

    |rho| = sqrt(3 s_mean) inside the production segments and 1e3 outside,
    so a selection that leaks non-production samples changes <1/|rho|>.
    """
    lanes, waves = s_mean.shape
    total = sum(int(p.n_steps) for p in phases)
    steps = np.arange(every, total + 1, every)
    segs = ana.segments({"phases": [dict(p.__dict__) for p in phases]})
    prod = np.any([(steps > a) & (steps <= a + b) for a, b in segs], axis=0)
    mod = np.where(prod[None, :, None], np.sqrt(3.0 * s_mean)[:, None, :], 1e3)
    rho = np.zeros((lanes, steps.size, waves, 2))
    rho[..., 1] = mod  # phase pi/2: |rho| = mod
    assert prod.any() and not prod.all(), "samples inside and outside production"
    return {"trace_rho": rho, "trace_time": np.tile(steps * CLI_DT, (lanes, 1)),
            "trace_count": np.full(lanes, steps.size)}  # fmt: skip


def cli_files(
    path: Path, omega_abs: np.ndarray, omega_net: np.ndarray, temps: List[float],
    n_gs: int, s_mean: np.ndarray, net_msd: Any = None, trace_every: int = 0,
) -> Path:  # fmt: skip
    """Write a minimal CLI-schema md_chi<tag>.npz / .json with exact windings.

    Lane l, wave w: W_abs = omega_abs[l, w] t and W_net = omega_net[l, w] t
    (radians, t since the segment began), or, if ``net_msd`` (a function of
    t in turns^2) is given, W_net = +-2 pi sqrt(net_msd(t)) with alternating
    signs. Waves: 2 constrained (|k| = 0.5, 0.9) then 2 unconstrained (1.4,
    1.8) with K = 1, N = 3; sums give <S_k> = s_mean[l, w] and <Phi> = M T;
    snapshot rho has the constant modulus sqrt(N s_mean[l, w]).
    """
    lanes, waves = omega_abs.shape
    assert lanes == n_gs * len(temps) and waves == 4, "4 waves per lane"
    phases, rec, w_abs, w_net = _cli_windings(omega_abs, omega_net, net_msd)
    lane_t = np.tile(np.asarray(temps), n_gs)
    n_sum = np.full((2, lanes), 1000)
    rho = np.zeros((lanes, rec.size, waves, 2))
    rho[..., 0] = np.sqrt(3.0 * s_mean)[:, None, :]
    arrays = {
        "snap_step": np.tile(rec, (lanes, 1)), "snap_time": np.tile(rec * CLI_DT,
                                                                    (lanes, 1)),
        "snap_wind_abs": w_abs, "snap_wind_net": w_net, "snap_rho": rho,
        "snap_r": np.zeros((lanes, rec.size, 3, 2)),
        "snap_phi": np.ones((lanes, rec.size)),
        "snap_kinetic": np.ones((lanes, rec.size)),
        "snap_count": np.full(lanes, rec.size),
        "cycle_sum_s": n_sum[..., None] * s_mean[None], "cycle_n_sum": n_sum,
        "cycle_sum_phi": n_sum * 2 * lane_t[None],
        "cycle_sum_kinetic": n_sum * 2 * lane_t,
        "status": np.zeros(lanes, np.int64),
        "lane_index": np.repeat(np.arange(n_gs), len(temps)),
        "lane_temperature": lane_t, "temperatures": np.asarray(temps),
        "indices": np.arange(n_gs),
        "waves_kvec": np.array([[0.5, 0.0], [0.0, 0.9], [1.4, 0.0], [0.0, 1.8]]),
        "waves_constrained": np.array([True, True, False, False]),
        "record_steps": rec, "basis": np.sqrt(3.0) * np.eye(2),
    }  # fmt: skip
    if trace_every > 0:
        arrays.update(_cli_trace(phases, s_mean, trace_every))
    meta = {"phases": [dict(p.__dict__) for p in phases], "settings": {"dt": CLI_DT},
            "M": 2, "K": 1.0, "N": 3, "d": 2, "chi": 1 / 3}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta))
    return path


def ou_msd(t: np.ndarray) -> np.ndarray:
    """MSD of an Ornstein-Uhlenbeck-velocity walk, D = 1, tau = 1 (turns^2)."""
    return 2.0 * (t - 1.0 + np.exp(-t))


def tiny_synthetic(tmp: Path, inject: Dict[str, float], common: bool) -> Path:
    """Synthetic five-start set (chi = 1/10), analysed; returns its root."""
    cfg = {"n": 64, "n_gs": 6, "temps": [1e-6, 1e-5, 1e-4], "dt": 0.025, "seed": 3,
           "common": common}  # fmt: skip
    paths = syn.generate(tmp, list(syn.STARTS), ["1/10"], cfg,
                         syn.default_proto(16_000, 2, 120), inject)  # fmt: skip
    for p in paths:
        res = ana.analyze_file(p, None)
        np.savez(p.parent / ana.output_name(p), **res)
    return tmp


@pytest.fixture(scope="module")
def null_set(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Null-consistent synthetic set: identical draws for every start."""
    return tiny_synthetic(tmp_path_factory.mktemp("null"), {}, True)


@pytest.fixture(scope="module")
def injected_set(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same set with the hip2d constrained rates 15 % faster."""
    return tiny_synthetic(tmp_path_factory.mktemp("inject"), {"hip2d": 1.15}, True)


# ============================================================
# WINDING, RATES AND RICE PREDICTIONS
# ============================================================


@pytest.mark.critical
def test_rates_of_exact_rotations_in_turns(tmp_path: Path) -> None:
    """W = omega t (radians) -> unsigned |omega| / 2 pi, net omega / 2 pi turns/time.

    Wave 1 swings back and forth (net 0, unsigned 2 rad/time), so swapping
    the signed and unsigned windings, or dropping the 2 pi, is caught.
    """
    temps, n_gs = [1e-6, 1e-4], 2
    om_net = (
        np.tile(np.array([-0.7, 0.0, 0.3, 1.9]), (4, 1)) * (1 + np.arange(4))[:, None]
    )
    om_abs = np.abs(om_net)
    om_abs[:, 1] = 2.0
    s_mean = np.tile(np.array([1e-6, 1e-6, 0.5, 0.8]), (4, 1))
    path = cli_files(tmp_path / "x" / "N3" / "md_chi1-3.npz", om_abs, om_net,
                     temps, n_gs, s_mean)  # fmt: skip
    arrays, meta = ana.load_md(path)
    q = ana.lane_quantities(arrays, meta)
    np.testing.assert_allclose(q["rate_abs"], om_abs / TWO_PI, rtol=RTOL_MD_RATE_EXACT)
    np.testing.assert_allclose(q["rate_net"], om_net / TWO_PI, rtol=RTOL_MD_RATE_EXACT,
                               atol=1e-15)  # fmt: skip
    np.testing.assert_allclose(q["abs_w1"], q["abs_w2"], rtol=RTOL_MD_WINDOW_EXACT)
    np.testing.assert_allclose(
        q["abs_w2"][:, 0], om_abs / TWO_PI, rtol=RTOL_MD_WINDOW_EXACT
    )


@pytest.mark.critical
def test_ground_state_major_grid(tmp_path: Path) -> None:
    """Lane l = g n_T + t: analyze_file puts lane (g, T) at grid [g, t]."""
    temps, n_gs = [1e-6, 1e-5, 1e-4], 3
    lanes = n_gs * len(temps)
    code = 1.0 + np.arange(lanes)  # unique per lane: g n_T + t + 1
    om = np.tile(code[:, None], (1, 4))
    s_mean = np.repeat(np.tile(np.asarray(temps), n_gs)[:, None], 4, axis=1)
    path = cli_files(tmp_path / "x" / "N3" / "md_chi1-3.npz", om, om, temps, n_gs,
                     s_mean)  # fmt: skip
    res = ana.analyze_file(path, None)
    for g in range(n_gs):
        for t in range(len(temps)):
            np.testing.assert_allclose(res["rate_abs"][g, t], (g * 3 + t + 1) / TWO_PI,
                                       rtol=RTOL_MD_RATE_EXACT)  # fmt: skip
    np.testing.assert_allclose(res["s_over_t_bin"][..., :20][np.isfinite(
        res["s_over_t_bin"][..., :20])], 1.0, rtol=1e-12)  # fmt: skip
    np.testing.assert_allclose(res["phi_ratio"], 1.0, rtol=1e-12)
    assert res["temps"].tolist() == temps


@pytest.mark.medium
def test_lane_layout_rejects_temperature_major(tmp_path: Path) -> None:
    """Lanes labelled temperature major are a ValueError, not a silent mix-up."""
    om = np.ones((4, 4))
    path = cli_files(tmp_path / "x" / "N3" / "md_chi1-3.npz", om, om, [1e-6, 1e-4], 2,
                     np.full((4, 4), 1e-6))  # fmt: skip
    arrays, _ = ana.load_md(path)
    assert ana.lane_layout(arrays) == (2, 2)
    arrays["lane_temperature"] = np.repeat(arrays["temperatures"], 2)
    with pytest.raises(ValueError, match="state major"):
        ana.lane_layout(arrays)


@pytest.mark.high
def test_windowed_rates_of_quadratic_winding() -> None:
    """W = c t^2 with snapshots at t/4, t/2, t: w1 = 0.75 c t, w2 = 1.5 c t."""
    t = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    w = 0.3 * t[:, None] ** 2
    r = ana.segment_rates(t, w, w, 8.0)
    np.testing.assert_allclose(
        r["abs_w1"], [0.75 * 0.3 * 8.0], rtol=RTOL_MD_WINDOW_EXACT
    )
    np.testing.assert_allclose(
        r["abs_w2"], [1.5 * 0.3 * 8.0], rtol=RTOL_MD_WINDOW_EXACT
    )
    np.testing.assert_allclose(r["net_end"], [0.3 * 8.0], rtol=RTOL_MD_WINDOW_EXACT)


@pytest.mark.high
@given(a=st.floats(-50, 50), b=st.floats(-50, 50),
       t=hnp.arrays(np.float64, st.integers(2, 60),
                    elements=st.floats(1e-3, 1e4), unique=True))  # fmt: skip
def test_ls_slope_exact_for_linear_property(a: float, b: float, t: np.ndarray) -> None:
    """ls_slope recovers a of y = a t + b for any distinct times."""
    t = np.sort(t)
    if np.ptp(t) < 1e-2 * t.max():
        return  # nearly coincident times: the slope is ill conditioned
    y = (a * t + b)[:, None]
    got = ana.ls_slope(t, y)[0]
    assert abs(got - a) <= 1e-8 * (abs(a) + abs(b) / np.ptp(t) + 1.0)


@pytest.mark.critical
def test_rice_prediction_formulas() -> None:
    """|k| sqrt(T / S) / 2 pi (constrained) and |k| sqrt(T / (pi S)) / 2 pi."""
    kmag = np.array([1.0, 2.0, 3.0])
    con = np.array([True, True, False])
    s = np.array([1e-4, 4e-4, 0.5])
    got = ana.rice_prediction(kmag, np.array(1e-4), s, con)
    want = np.array([1.0, 2.0 * 0.5, 3.0 * math.sqrt(1e-4 / (math.pi * 0.5))]) / TWO_PI
    np.testing.assert_allclose(got, want, rtol=1e-14)
    assert np.isnan(ana.rice_prediction(kmag, np.array(1.0), np.zeros(3), con)).all()


@pytest.mark.critical
def test_phase_floor_excludes_steps() -> None:
    """A frame with S below the floor removes both of its steps from the winding."""
    rho = np.array([[1.0], [1e-5 * np.exp(2.0j)], [np.exp(0.5j)]])
    w_abs, w_net = ana.cumulative_turns(rho, 1, 1e-8)
    np.testing.assert_allclose(w_abs[:, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(w_net[:, 0], 0.0, atol=0.0)
    w_abs, w_net = ana.cumulative_turns(rho, 1, 1e-12)  # S = 1e-10 now resolved
    np.testing.assert_allclose(w_abs[-1, 0], (2.0 + 1.5) / TWO_PI, rtol=1e-14)
    np.testing.assert_allclose(w_net[-1, 0], 0.5 / TWO_PI, rtol=1e-14)


@pytest.mark.high
@given(re=hnp.arrays(np.float64, 8, elements=st.floats(-10, 10)),
       im=hnp.arrays(np.float64, 8, elements=st.floats(-10, 10)),
       floor=st.floats(0.0, 50.0))  # fmt: skip
def test_winding_step_property(re: np.ndarray, im: np.ndarray, floor: float) -> None:
    """|Delta| <= pi, zero wherever S <= floor at either end, net = angle otherwise."""
    rho = re + 1j * im
    dth, ok = ana.winding_step(rho[1:], rho[:-1], 2, floor)
    s = np.abs(rho) ** 2 / 2
    assert np.all(np.abs(dth) <= np.pi + 1e-15)
    assert np.array_equal(ok, (s[1:] > floor) & (s[:-1] > floor))
    assert np.all(dth[~ok] == 0.0)
    ang = np.angle(rho[1:] * np.conj(rho[:-1]))
    np.testing.assert_allclose(dth[ok], ang[ok], atol=1e-15)


# ============================================================
# BINS AND MSD
# ============================================================


@pytest.mark.high
def test_bin_ids_hand_examples() -> None:
    """Bins are (a, b]: 0.05 -> 0, just above -> 1; (1, 1.05] -> 20; 1.999 -> 39."""
    kmag = np.array([0.05, 0.0500001, 0.999, 1.05, 1.0500001, 1.999])
    con = np.array([True, True, True, False, False, False])
    assert ana.bin_ids(kmag, 1.0, con).tolist() == [0, 1, 19, 20, 21, 39]
    vals = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    means = ana.bin_mean(vals, ana.bin_ids(kmag, 1.0, con))
    assert means[0, 0] == 1.0 and means[0, 39] == 6.0
    assert np.isnan(means[0, 2]) and np.sum(np.isfinite(means)) == 6


@pytest.mark.medium
@given(k=hnp.arrays(np.float64, st.integers(1, 30), elements=st.floats(1e-3, 1.999)))
def test_bin_ids_monotone_property(k: np.ndarray) -> None:
    """Within a class, bins are non-decreasing in |k| and contain |k| / K."""
    con = k < 1.0
    ids = ana.bin_ids(k, 1.0, con)
    order = np.argsort(k)
    for cls in (True, False):
        sel = order[con[order] == cls]
        assert np.all(np.diff(ids[sel]) >= 0)
    lo = np.where(con, ids * 0.05, 1.0 + (ids - 20) * 0.05)
    assert np.all((k > lo - 1e-12) & (k <= lo + 0.05 + 1e-12))


@pytest.mark.critical
def test_alpha_window_excludes_ballistic_lags() -> None:
    """OU-velocity MSD: alpha ~ 1 over t >= 10 / rate, ~1.5 over all lags."""
    t = np.geomspace(1e-2, 1e4, 120)
    msd = ou_msd(t)
    t_min = ana.window_start(np.full(12, 1.0))  # median inverse rate 1 -> 10
    assert t_min == pytest.approx(10.0)
    alpha, d_theta = ana.fit_msd(t, msd, t_min)
    assert abs(alpha - 1.0) < ATOL_MD_ALPHA_OU
    assert d_theta == pytest.approx(1.0, rel=ATOL_MD_ALPHA_OU)
    alpha_all, _ = ana.fit_msd(t, msd, 0.0)
    assert alpha_all > 1.0 + 5 * ATOL_MD_ALPHA_OU


@pytest.mark.critical
def test_alpha_through_lane_quantities(tmp_path: Path) -> None:
    """lane_quantities fits alpha in its data-chosen window (not over all lags)."""
    om = np.full((2, 4), 2.0 * np.pi * 1.0)  # unsigned 1 turn / time: t_min = 10
    path = cli_files(
        tmp_path / "x" / "N3" / "md_chi1-3.npz",
        om,
        om,
        [1e-6],
        2,
        np.full((2, 4), 1e-6),
        net_msd=ou_msd,
    )
    arrays, meta = ana.load_md(path)
    q = ana.lane_quantities(arrays, meta)
    np.testing.assert_allclose(q["t_min"], 10.0, rtol=1e-12)
    assert np.all(np.abs(q["alpha"] - 1.0) < ATOL_MD_ALPHA_OU)
    np.testing.assert_allclose(q["msd_c"], q["msd_u"], rtol=1e-12)


@pytest.mark.high
def test_msd_exponent_of_random_walks() -> None:
    """Pooled MSD of 4000 Gaussian walks: alpha = 1 and D = D_true."""
    rng = np.random.default_rng(11)
    t = np.unique(np.round(np.geomspace(1, 1000, 60)))
    steps = rng.standard_normal((4000, 1000)) * np.sqrt(2 * 0.3)  # D = 0.3, dt = 1
    walk = np.cumsum(steps, axis=1)[:, t.astype(int) - 1]
    msd = np.mean(walk**2, axis=0)
    alpha, d_theta = ana.fit_msd(t, msd, 1.0)
    assert abs(alpha - 1.0) < ATOL_MD_ALPHA_RW
    assert d_theta == pytest.approx(0.3, rel=RTOL_MD_D_RW)


@pytest.mark.high
@given(a=st.floats(0.2, 2.0), c=st.floats(1e-6, 1e3))
def test_fit_msd_power_law_property(a: float, c: float) -> None:
    """fit_msd returns the exponent of an exact power law; NaN without points."""
    t = np.geomspace(1e-2, 1e3, 50)
    alpha, _ = ana.fit_msd(t, c * t**a, 1.0)
    assert abs(alpha - a) < ATOL_MD_ALPHA_EXACT
    assert np.isnan(ana.fit_msd(t, c * t**a, 2e3)[0])


@pytest.mark.high
def test_particle_msd_removes_com() -> None:
    """Rigid drift gives 0; ballistic u t (zero mean u) gives mean|u|^2 t^2."""
    rng = np.random.default_rng(2)
    t = np.array([0.0, 1.0, 3.0])
    u = rng.standard_normal((50, 2))
    u -= u.mean(axis=0)
    r0 = rng.uniform(0, 5, (50, 2))
    r = r0[None] + t[:, None, None] * (u[None] + np.array([3.0, -1.0]))
    np.testing.assert_allclose(ana.particle_msd(r), np.mean(np.sum(u**2, 1)) * t**2,
                               rtol=1e-12, atol=1e-12)  # fmt: skip
    rigid = r0[None] + t[:, None, None] * np.array([3.0, -1.0])
    np.testing.assert_allclose(ana.particle_msd(rigid), 0.0, atol=1e-12)


@pytest.mark.high
def test_collapse_metric() -> None:
    """MSD = c T t^2 collapses on t sqrt(T) (metric 0); MSD = T t does not."""
    lag = np.geomspace(1e-2, 1e3, 80)
    temps = np.array([1e-6, 1e-5, 1e-4])
    good = 2.0 * temps[:, None] * lag[None] ** 2
    assert ana.collapse_metric(lag, good, temps) < 1e-9
    bad = temps[:, None] * lag[None]
    assert ana.collapse_metric(lag, bad, temps) > 0.2
    assert np.isnan(ana.collapse_metric(lag[:2], good[:, :2], temps * [1, 1e6, 1e12]))


@pytest.mark.medium
def test_spectrum_moments_and_power_spectrum() -> None:
    """Moments of a known spectrum; the power spectrum of e^{i w t} peaks at w."""
    mom = ana.spectrum_moments(np.array([1.0, 2.0, 3.0, 6.0]))
    np.testing.assert_allclose(mom[[0, 2]], [3.0, 1.0 / 3.0], rtol=1e-14)
    np.testing.assert_allclose(mom[4], 4 / (1 + 0.5 + 1 / 3 + 1 / 6) / 3.0, rtol=1e-14)
    t = np.arange(512) * 0.1
    rho = np.exp(1j * 2.0 * t)[:, None] * np.array([1.0, 3.0])
    omega, power = ana.power_spectrum(rho, 0.1)
    assert abs(omega[np.argmax(power)] - 2.0) < TWO_PI / (512 * 0.1)
    assert power.sum() == pytest.approx(1.0)


# ============================================================
# STATISTICS
# ============================================================


@pytest.mark.critical
def test_kruskal_eta2_hand_example() -> None:
    """Three separated groups of three: H = 7.2, eta^2_H = (H - k + 1)/(n - k) = 0.867.

    The plain epsilon^2 = H (n+1)/(n^2-1) = 0.9 (pre-Amendment A1) is rejected.
    """
    groups = [np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), np.array([7.0, 8, 9])]
    h, p, eta2 = mds.kruskal_eta2(groups)
    assert h == pytest.approx(7.2, rel=1e-12)
    assert eta2 == pytest.approx((7.2 - 2.0) / 6.0, rel=1e-12)
    assert eta2 != pytest.approx(7.2 * 10 / 80, rel=1e-3)
    assert p == pytest.approx(sps.chi2.sf(7.2, 2), rel=1e-12)
    assert mds.kruskal_eta2([np.ones(3), np.ones(4)]) == (0.0, 1.0, 0.0)


@pytest.mark.high
@given(a=samples, b=samples, c=samples)
def test_kruskal_eta2_property(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    """eta^2_H = (H - 2)/(n - 3) with scipy's H, in [-2/(n-3), 1]."""
    n = a.size + b.size + c.size
    if n < 4:
        return
    h, _, eta2 = mds.kruskal_eta2([a, b, c])
    pooled = np.concatenate([a, b, c])
    if np.all(pooled == pooled[0]):
        assert eta2 == 0.0
        return
    assert h == pytest.approx(sps.kruskal(a, b, c).statistic, rel=1e-12)
    assert eta2 == pytest.approx((h - 2.0) / (n - 3.0), rel=1e-12, abs=1e-12)
    assert -2.0 / (n - 3) - 1e-12 <= eta2 <= 1.0 + 1e-12


def null_h5_set(seed: int, effect: float = 0.0) -> List[Dict[str, Any]]:
    """Analysis dicts of 5 starts x 2 chi x 3 harmonic T, 10 iid ground states each.

    nu ~ 1 + 0.003 N(0, 1) per ground state (hip2d scaled by 1 + effect).
    """
    rng = np.random.default_rng(seed)
    out = []
    for tag, chi in (("1-10", 0.1), ("2-5", 0.4)):
        for s in syn.STARTS:
            nu = 1.0 + 0.003 * rng.standard_normal((10, 3))
            nu *= 1.0 + (effect if s == "hip2d" else 0.0)
            out.append({"system": s, "tag": tag, "chi": np.asarray(chi),
                        "temps": np.array([1e-6, 1e-5, 1e-4]), "nu_c": nu,
                        "d_theta": rng.uniform(0.1, 0.2, (10, 3))})  # fmt: skip
    return out


@pytest.mark.critical
def test_regression_D047_h5_null_set_pass_rate() -> None:
    """D-047 / A2: >= 18 of 20 start-independent sets pass H5 part 1 (mean eta^2_H).

    Under the old per-cell rule (all 6 cells eta^2_H < 0.06) only ~0.37 of
    the sets pass, under the plain epsilon^2 ~0.006.
    """
    passed = 0
    for seed in range(MD_H5_NULL_SETS):
        parts, report = mds.h5_parts(null_h5_set(seed), True)
        assert parts[0]["part"] == "mean KW eta2_H"
        assert len(report["eta2_H_per_cell"]) == 6
        passed += parts[0]["outcome"] == mds.PASS
    assert passed >= MD_H5_NULL_SET_MIN


@pytest.mark.critical
def test_a2_mean_eta2_part_and_per_cell_descriptive() -> None:
    """A2: one cell above 0.06 does not fail H5 if the mean is below; an effect does."""
    parts, report = mds.h5_parts(null_h5_set(4), True)
    cells = [p["value"] for p in report["eta2_H_per_cell"]]
    assert parts[0]["value"] == pytest.approx(np.mean(cells), rel=1e-12)
    assert max(cells) > mds.H5_ETA2 > np.mean(cells)  # seed 4: one cell misses
    assert parts[0]["outcome"] == mds.PASS
    assert [p["part"] for p in parts[1:]] == ["max median diff"] * 6
    assert mds.combine([p["outcome"] for p in parts]) == mds.PASS
    parts, _ = mds.h5_parts(null_h5_set(4, effect=0.15), True)
    assert parts[0]["value"] > mds.H5_ETA2 and parts[0]["outcome"] == mds.FAIL


@pytest.mark.high
def test_h5_cell_null_pass_rate() -> None:
    """Per cell, eta^2_H < 0.06 holds at P(chi2_4 < 6.7) = 0.85 under the null."""
    passed = 0
    for seed in range(MD_H5_NULL_CELLS):
        rng = np.random.default_rng(seed)
        groups = {s: 1.0 + 0.003 * rng.standard_normal(10) for s in "abcde"}
        parts = mds.h5_cell(groups, "null")
        passed += mds.combine([p["outcome"] for p in parts]) == mds.PASS
    lo, hi = MD_H5_NULL_PASS
    assert lo <= passed / MD_H5_NULL_CELLS <= hi


@pytest.mark.high
@example(values=np.arange(10.0), k=5)
@given(
    values=hnp.arrays(np.float64, st.integers(2, 12), elements=finite),
    k=st.integers(2, 6),
)
def test_h5_identical_starts_pass_property(values: np.ndarray, k: int) -> None:
    """Starts holding the same ground-state values never miss H5 (eta^2_H <= 0)."""
    if np.any(values == 0.0) or np.ptp(values) == 0.0 or np.median(values) <= 0:
        return  # the median difference is relative to a positive pooled median
    groups = {f"s{i}": np.roll(values, i) for i in range(k)}
    parts = mds.h5_cell(groups, "same")
    assert parts[0]["value"] <= 1e-12
    assert {p["outcome"] for p in parts} == {mds.PASS}


@pytest.mark.high
@given(a=samples, b=samples)
def test_cliffs_delta_property(a: np.ndarray, b: np.ndarray) -> None:
    """Cliff's delta equals the brute-force pair count, is antisymmetric, |d| <= 1."""
    d = mds.cliffs_delta(a, b)
    brute = sum(int(x > y) - int(x < y) for x in a for y in b) / (a.size * b.size)
    assert d == pytest.approx(brute, abs=1e-15)
    assert d == pytest.approx(-mds.cliffs_delta(b, a), abs=1e-15)
    assert -1.0 <= d <= 1.0


@pytest.mark.critical
def test_holm_hand_example_and_report() -> None:
    """Holm: [.01, .04, .03, .005] -> [.03, .06, .06, .02]; the report uses it."""
    np.testing.assert_allclose(mds.holm([0.01, 0.04, 0.03, 0.005]),
                               [0.03, 0.06, 0.06, 0.02], rtol=1e-14)  # fmt: skip
    rng = np.random.default_rng(5)
    groups = {s: rng.normal(i, 0.3, 10) for i, s in enumerate(("a", "b", "c"))}
    rep = mds.d_theta_report(groups)
    raw = [r["p"] for r in rep["pairs"]]
    np.testing.assert_allclose([r["p_holm"] for r in rep["pairs"]], mds.holm(raw))
    assert max(r["p_holm"] / r["p"] for r in rep["pairs"]) == pytest.approx(3.0)


@pytest.mark.high
@given(p=hnp.arrays(np.float64, st.integers(1, 20), elements=st.floats(0.0, 1.0)))
def test_holm_property(p: np.ndarray) -> None:
    """Adjusted p in [p, 1], order preserving, and min(1, m min p) for the smallest."""
    adj = mds.holm(p)
    assert np.all(adj >= p - 1e-15) and np.all(adj <= 1.0)
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(adj[order]) >= -1e-15)
    assert adj[order[0]] == pytest.approx(min(1.0, p.size * p[order[0]]))


@pytest.mark.critical
def test_classify_thresholds() -> None:
    """PASS if met; a miss by >= 10 % FAILs; a smaller (or NaN) one is INCONCLUSIVE."""
    assert mds.classify(True, 0.5) == mds.PASS
    assert mds.classify(False, 0.0999) == mds.INCONCLUSIVE
    assert mds.classify(False, 0.10) == mds.FAIL
    assert mds.classify(False, float("nan")) == mds.INCONCLUSIVE
    assert mds.combine([mds.PASS, mds.INCONCLUSIVE]) == mds.INCONCLUSIVE
    assert mds.combine([mds.PASS, mds.INCONCLUSIVE, mds.FAIL]) == mds.FAIL
    assert mds.combine([]) == mds.NOT_EVALUATED


@pytest.mark.critical
@pytest.mark.parametrize(
    ("med", "outcome"),
    [(1.01, "PASS"), (1.03, "INCONCLUSIVE"), (0.89, "FAIL"), (1.12, "FAIL")],
)
def test_band_part_outcomes(med: float, outcome: str) -> None:
    """H1 Phi band [0.98, 1.02]: the median over ground states decides."""
    x = med + np.array([-0.001, 0.0, 0.001])
    assert mds.band_part("phi", "c", x, mds.H1_PHI, 1.0)["outcome"] == outcome


@pytest.mark.critical
def test_h5_cell_constructed() -> None:
    """Identical starts PASS; +15 % on one start FAILs; +5 % is INCONCLUSIVE."""
    base = 1.0 + 0.002 * np.arange(10)
    names = ("a", "b", "c", "d", "e")
    same = {s: base.copy() for s in names}
    assert {p["outcome"] for p in mds.h5_cell(same, "x")} == {mds.PASS}
    for factor, outcome in ((1.15, mds.FAIL), (1.05, mds.INCONCLUSIVE)):
        groups = {**same, "e": base * factor}
        parts = mds.h5_cell(groups, "x")
        assert mds.combine([p["outcome"] for p in parts]) == outcome
        h, _, eta2 = mds.kruskal_eta2([groups[s] for s in names])
        assert parts[0]["value"] == pytest.approx((h - 4.0) / 45.0, rel=1e-12) == eta2


# ============================================================
# SCHEMA AND GENERATOR
# ============================================================


@pytest.mark.medium
def test_schema_of_synthetic_files(tmp_path: Path) -> None:
    """The generator writes the CLI keys; load_md / segments / output_name read them."""
    cfg = {"n": 16, "n_gs": 2, "temps": [1e-6, 1e-4], "dt": 0.05, "seed": 0,
           "common": False}  # fmt: skip
    (path,) = syn.generate(tmp_path, ["flows2d"], ["1/5"], cfg,
                           syn.default_proto(64, 2, 20))  # fmt: skip
    arrays, meta = ana.load_md(path)
    assert ana.lane_layout(arrays) == (2, 2)
    segs = ana.segments(meta)
    assert [n for _, n in segs] == [64, 64] and segs[1][0] > segs[0][0] + 64
    lanes, waves = 4, arrays["waves_kvec"].shape[0]
    assert arrays["snap_wind_abs"].shape == (lanes, 20, waves)
    assert arrays["cycle_sum_s"].shape == (2, lanes, waves)
    assert ana.output_name(path) == "md_analysis_chi1-5.npz"
    assert ana.output_name(Path("md_langevin_g0.03_chi2-5.npz")) == (
        "md_analysis_langevin_g0.03_chi2-5.npz"
    )
    del arrays["cycle_sum_s"]
    np.savez(tmp_path / "bad.npz", **arrays)
    (tmp_path / "bad.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="cycle_sum_s"):
        ana.load_md(tmp_path / "bad.npz")


@pytest.mark.medium
def test_oscillator_coeffs_stationary() -> None:
    """The exact map keeps diag(1, w^2) stationary: E S E^T + C C^T = S."""
    omega = np.array([0.3, 1.0, 4.0])
    e, c = syn.oscillator_coeffs(omega, 0.05)
    for i, w in enumerate(omega):
        sig = np.diag([1.0, w**2])
        np.testing.assert_allclose(e[i] @ sig @ e[i].T + c[i] @ c[i].T, sig,
                                   rtol=1e-10, atol=1e-12)  # fmt: skip
    assert np.all(np.abs(np.linalg.eigvals(e[0])) < 1.0)


# ============================================================
# END TO END
# ============================================================


@pytest.mark.slow
@pytest.mark.critical
def test_synthetic_recovery(null_set: Path) -> None:
    """Analysis recovers the model: Rice ratios, S/T = 1, Phi/(MT) = 1, alpha = 1."""
    main, _, _ = mds.collect([null_set])
    assert len(main) == 5
    z = main[0]
    con = z["constrained"]
    ratio = z["ratio"]
    assert np.nanmean(ratio[..., con]) == pytest.approx(1.0, rel=RTOL_MD_SYN_RATE)
    assert np.nanmean(ratio[..., ~con]) == pytest.approx(1.0, rel=RTOL_MD_SYN_RATE)
    s_t = z["s_mean"][..., con] / z["temps"][None, :, None]
    assert np.mean(s_t) == pytest.approx(1.0, rel=RTOL_MD_SYN_EQUIPARTITION)
    assert np.mean(z["phi_ratio"]) == pytest.approx(1.0, rel=RTOL_MD_SYN_EQUIPARTITION)
    assert abs(np.median(z["alpha"]) - 1.0) < ATOL_MD_SYN_ALPHA
    assert np.mean(z["nu_c"]) == pytest.approx(1.0, rel=RTOL_MD_SYN_RATE)
    assert np.all(z["net_rms_u"][:, 1:] > z["net_rms_u"][:, :-1])  # grows with T


@pytest.mark.slow
@pytest.mark.critical
def test_synthetic_verdicts_null_and_injected(
    null_set: Path, injected_set: Path
) -> None:
    """Null set: every evaluable hypothesis PASSes; hip2d +15 %: H5 and H2 FAIL."""
    main, lang, _ = mds.collect([null_set])
    res = mds.evaluate(main, lang)
    verdict = {h: res["hypotheses"][h]["verdict"] for h in mds.HYPOTHESES}
    for h in ("H1", "H2", "H3", "H4", "H5", "H6", "H8"):
        assert verdict[h] == mds.PASS, (h, res["hypotheses"][h]["worst"][:1])
    assert verdict["H7"] == mds.NOT_EVALUATED  # no Langevin files in this set
    main, lang, _ = mds.collect([injected_set])
    res = mds.evaluate(main, lang)
    assert res["hypotheses"]["H5"]["verdict"] == mds.FAIL
    assert res["hypotheses"]["H2"]["verdict"] == mds.FAIL
    worst = res["hypotheses"]["H5"]["worst"][0]
    assert worst["dev"] == pytest.approx(0.15, abs=0.03)
    assert worst["part"] == "mean KW eta2_H" and worst["value"] > mds.H5_ETA2
    pairs = [p["pair"] for p in res["hypotheses"]["H5"]["parts"] if "pair" in p]
    assert pairs and all("hip2d" in pair for pair in pairs)


@pytest.mark.slow
@pytest.mark.medium
def test_clean_json_and_markdown(null_set: Path) -> None:
    """verdicts.json is strict JSON (NaN -> null); the table lists every hypothesis."""
    out = mds.clean_json({"a": np.float64("nan"), "b": [np.int64(3), np.inf], "c": 1.5})
    assert out == {"a": None, "b": [3, None], "c": 1.5}
    main, lang, _ = mds.collect([null_set])
    res = mds.evaluate(main, lang)
    json.dumps(mds.clean_json(res), allow_nan=False)
    table = mds.markdown(res)
    assert all(f"| {h} |" in table for h in mds.HYPOTHESES)
    assert res["exploratory"]["label"] == mds.EXPL_LABEL  # post hoc, separate
    assert table.index("Exploratory (post hoc") > table.index("| H8 |")


@pytest.mark.slow
@pytest.mark.critical
def test_exploratory_rows_never_enter_registered_parts(null_set: Path) -> None:
    """H1 / H2 verdict parts are the registered per-bin parts only (AUDIT-E5 F1).

    The post hoc pooled groups must never be appended to, or replace, the
    registered parts, so H1 holds exactly h1_parts and H2 exactly h2_parts
    (mutants M6a and M6c of the audit).
    """
    main, lang, _ = mds.collect([null_set])
    res = mds.evaluate(main, lang)
    patterns = {"H1": r"^(phi|S bin \d+)$", "H2": r"^(Rice bin|T ratio bin) \d+$"}
    registered = {"H1": mds.h1_parts(main, True), "H2": mds.h2_parts(main, True)}
    for h, rx in patterns.items():
        parts = res["hypotheses"][h]["parts"]
        bad = [p["part"] for p in parts if not re.match(rx, p["part"])]
        assert not bad, (h, bad[:3])
        assert [(p["part"], p["cell"]) for p in parts] == [
            (p["part"], p["cell"]) for p in registered[h]
        ], h
    assert res["exploratory"]["rows"], "the null set has pooled rows to leak"


@pytest.mark.slow
@pytest.mark.critical
@pytest.mark.parametrize("w_min", [1, 50])
def test_verdicts_independent_of_exploratory_w_min(
    null_set: Path, monkeypatch: pytest.MonkeyPatch, w_min: int
) -> None:
    """Changing the post hoc W_min changes the pooled rows, never a verdict (F1)."""
    main, lang, _ = mds.collect([null_set])
    base = mds.evaluate(main, lang)
    monkeypatch.setattr(mds, "EXPL_W_MIN", {"S/T": w_min, "Rice": w_min})
    other = mds.evaluate(main, lang)
    assert len(other["exploratory"]["rows"]) != len(base["exploratory"]["rows"])
    for h in mds.HYPOTHESES:
        a, b = base["hypotheses"][h], other["hypotheses"][h]
        assert (a["verdict"], a["counts"]) == (b["verdict"], b["counts"]), h


def _cell_data(net: np.ndarray, alpha: np.ndarray, d_theta: np.ndarray, tag: str,
               chi: float) -> Dict[str, Any]:  # fmt: skip
    """Minimal analysis dict (one start, one harmonic T) for h4 / h8 parts."""
    g = net.size
    return {"system": "flows2d", "tag": tag, "chi": np.asarray(chi),
            "temps": np.array([1e-6]), "net_mean_c": net[:, None],
            "abs_mean_c": np.ones((g, 1)), "alpha": alpha[:, None],
            "d_theta": d_theta[:, None]}  # fmt: skip


@pytest.mark.critical
def test_a1_significance_parts_fail_on_any_miss() -> None:
    """A1.2: H4 |z| >= 3 and H8 Mann-Whitney p >= 0.01 FAIL (never INCONCLUSIVE)."""
    base = np.linspace(-1e-3, 1e-3, 10)
    ok = _cell_data(base, np.ones(10), np.ones(10), "1-10", 0.1)
    assert mds.h4_parts([ok], True)[0]["outcome"] == mds.PASS
    biased = _cell_data(base + 0.02, np.ones(10), np.ones(10), "1-10", 0.1)
    part = mds.h4_parts([biased], True)[0]
    assert abs(part["z"]) > 3 and part["dev"] < 0.10  # a small but significant miss
    assert part["outcome"] == mds.FAIL
    d = np.linspace(1.0, 2.0, 10)
    a = _cell_data(base, np.ones(10), d, "1-10", 0.1)
    same = _cell_data(base, np.ones(10), d * 1.15, "2-5", 0.4)  # 15 %, p > 0.01
    far = _cell_data(base, np.ones(10), d + 10.0, "2-5", 0.4)
    assert mds.h8_parts([a, far], [])[0]["outcome"] == mds.PASS
    miss = mds.h8_parts([a, same], [])[0]
    assert miss["p"] >= mds.H8_P and miss["outcome"] == mds.FAIL


@pytest.mark.critical
def test_a1_tolerance_parts_inconclusive_whatever_z() -> None:
    """A1.2: an interval / tolerance part missing by < 10 % is INCONCLUSIVE at any z."""
    tight = 1.04 + 1e-6 * np.arange(10)  # |z| huge, 4 % off (H1 Phi band 2 %)
    loose = 1.04 + 0.2 * np.linspace(-1, 1, 10)  # |z| < 3, 4 % off
    for x in (tight, loose):
        assert mds.band_part("phi", "c", x, mds.H1_PHI, 1.0)["outcome"] == (
            mds.INCONCLUSIVE
        )
    part = mds._ratio_part("BAOAB / NVE", "c", tight, mds.H7_TOL)
    assert abs(part["z"]) > 3 and part["outcome"] == mds.INCONCLUSIVE
    assert mds.band_part("phi", "c", tight + 0.08, mds.H1_PHI, 1.0)["outcome"] == (
        mds.FAIL
    )


@pytest.mark.high
def test_a1_h6_collapse_and_beta_use_harmonic_t() -> None:
    """A1.4: an anharmonic T = 1e-3 curve does not enter the H6 collapse and beta."""
    lag = np.geomspace(1e-2, 1e3, 40)
    temps = np.array([1e-6, 1e-5, 1e-4, 1e-3])
    msd = 2.0 * temps[:, None] * lag[None] ** 2
    msd[3] *= 5.0  # T = 1e-3 off the collapse
    rms = np.sqrt(temps) * np.array([1.0, 1.0, 1.0, 30.0])  # and off beta = 1/2
    z = {"system": "x", "tag": "1-10", "temps": temps, "part_lag": lag,
         "part_msd": np.repeat(msd[None], 3, axis=0),
         "net_rms_u": np.repeat(rms[None], 3, axis=0),
         "ratio_bin": np.full((3, 4, 40), np.nan)}  # fmt: skip
    parts = {p["part"]: p for p in mds.h6_parts([z], True)}
    assert parts["beta"]["value"] == pytest.approx(0.5, abs=1e-9)
    assert parts["collapse t sqrt(T)"]["value"] < 1e-9
    desc = {p["part"]: p for p in mds.h6_parts([z], False)}
    assert desc["collapse t sqrt(T)"]["value"] > 0.2 and desc["beta"]["value"] > 0.6


# ============================================================
# AMENDMENT A3 (H6b) AND AUDIT MUTANTS
# ============================================================


@pytest.mark.critical
def test_conditional_rice_formula() -> None:
    """|k| sqrt(N T / pi) <1/|rho|> / 2 pi; the fixed-rho Rice at constant |rho|."""
    rho = np.zeros((2, 3, 2, 2))
    rho[..., 0] = np.array([1.0, 2.0, 4.0])[None, :, None]  # |rho| = 1, 2, 4
    inv = ana.mean_inverse_modulus(rho)
    np.testing.assert_allclose(inv, (1.0 + 0.5 + 0.25) / 3.0, rtol=1e-15)
    kmag, n, temp = np.array([1.0, 2.0]), 9, np.array([[1e-4]])
    got = ana.conditional_rice(kmag, n, temp, inv)
    want = kmag * math.sqrt(n * 1e-4 / math.pi) * (1.75 / 3.0) / TWO_PI
    np.testing.assert_allclose(got[0], want, rtol=1e-14)
    s = np.full(2, 0.7)
    fixed = ana.conditional_rice(kmag, n, temp, 1.0 / np.sqrt(n * s)[None])
    np.testing.assert_allclose(
        fixed[0], ana.rice_prediction(kmag, temp, s, np.zeros(2, bool))[0], rtol=1e-14
    )
    zero = np.zeros((1, 2, 1, 2))
    assert np.isnan(ana.mean_inverse_modulus(zero)).all()


@pytest.mark.critical
def test_h6b_through_analysis_and_verdicts(tmp_path: Path) -> None:
    """Constant |rho|: rice_cond = unconstrained Rice; H6b bins in [0.9, 1.1]."""
    om = np.tile(np.array([1.0, 1.0, 0.02, 0.03]), (2, 1))
    s_mean = np.tile(np.array([1e-6, 1e-6, 0.5, 0.8]), (2, 1))
    path = cli_files(tmp_path / "x" / "N3" / "md_chi1-3.npz", om, om, [1e-6], 2,
                     s_mean, trace_every=100)  # fmt: skip
    res = ana.analyze_file(path, None)
    unc = ~res["constrained"]
    np.testing.assert_allclose(res["rice_cond"][..., unc], res["rice"][..., unc],
                               rtol=1e-12)  # fmt: skip
    z = {"system": "x", "tag": "1-3", "temps": np.array([1e-6, 1e-5]),
         "ratio_cond_bin": np.full((3, 2, 40), np.nan)}  # fmt: skip
    for value, outcome in ((1.08, mds.PASS), (1.12, mds.FAIL), (0.95, mds.PASS)):
        z["ratio_cond_bin"][:, :, 25] = value
        parts = mds.h6b_parts([z], True)
        assert [p["outcome"] for p in parts] == [outcome, outcome]
    assert "H6b" in mds.HYPOTHESES and mds.h6b_parts([z], False) == []


@pytest.mark.critical
def test_h8_threshold_is_p_001() -> None:
    """M13: Mann-Whitney p = 0.017 (between 0.01 and 0.05) is a miss, i.e. FAIL."""
    base = np.linspace(-1e-3, 1e-3, 10)
    d = np.arange(10.0)
    a = _cell_data(base, np.ones(10), 1.0 + 0.1 * d, "1-10", 0.1)
    b = _cell_data(base, np.ones(10), 1.4 + 0.1 * d, "2-5", 0.4)
    part = mds.h8_parts([a, b], [])[0]
    assert 0.01 < part["p"] < 0.05
    assert part["outcome"] == mds.FAIL


@pytest.mark.critical
def test_h3_needs_z_below_3() -> None:
    """M17: a tight 2 % window mismatch (within 3 % but |z| >> 3) is not a PASS."""
    w1 = np.ones((10, 1, 2))
    w2 = (1.02 + 1e-5 * np.arange(10))[:, None, None] * np.ones((10, 1, 2))
    z = {"system": "x", "tag": "1-10", "temps": np.array([1e-6]), "w1_c": w1,
         "w2_c": w2}  # fmt: skip
    parts = mds.h3_parts([z], True)
    assert all(abs(p["z"]) > 3 and p["dev"] < 0.03 for p in parts)
    assert {p["outcome"] for p in parts} == {mds.INCONCLUSIVE}


@pytest.mark.critical
def test_nu_is_rate_over_k_not_over_rice(tmp_path: Path) -> None:
    """M19 (A1.3): nu = mean over constrained waves of rate / (|k| / 2 pi), S != T."""
    om = np.tile(np.array([0.4, 1.3, 0.02, 0.03]), (2, 1))
    s_mean = np.tile(np.array([4e-6, 9e-6, 0.5, 0.8]), (2, 1))  # S = 4 T, 9 T
    path = cli_files(tmp_path / "x" / "N3" / "md_chi1-3.npz", om, om, [1e-6], 2, s_mean)
    res = ana.analyze_file(path, None)
    want = np.mean(np.array([0.4, 1.3]) / np.array([0.5, 0.9]))
    np.testing.assert_allclose(res["nu_c"], want, rtol=1e-11)
    rice_ratio = np.mean(np.array([0.4 / 0.5 * 2.0, 1.3 / 0.9 * 3.0]))
    assert abs(float(res["nu_c"][0, 0]) - rice_ratio) > 0.5


@pytest.mark.critical
def test_h2_t_ratio_uses_1e6_over_1e4() -> None:
    """M20: three T with distinct rates; the H2 ratio is rate(1e-6) / rate(1e-4)."""
    rates = np.array([1.0, 1.5, 1.02])  # at T = 1e-6, 1e-5, 1e-4
    z = {"system": "x", "tag": "1-10", "temps": np.array([1e-6, 1e-5, 1e-4]),
         "ratio_bin": np.full((4, 3, 40), np.nan),
         "rate_abs_bin": np.full((4, 3, 40), np.nan)}  # fmt: skip
    z["rate_abs_bin"][:, :, 7] = rates
    parts = [p for p in mds.h2_parts([z], True) if p["part"].startswith("T ratio")]
    assert len(parts) == 1
    assert parts[0]["value"] == pytest.approx(1.0 / 1.02, rel=1e-12)
    assert parts[0]["outcome"] == mds.PASS


@pytest.mark.critical
def test_h5_median_difference_is_relative() -> None:
    """M21: nu ~ 5, one start +2 %: relative difference 0.02 PASSes (absolute 0.1)."""
    base = 5.0 + 0.001 * np.arange(10)
    groups = {s: base.copy() for s in "abcd"}
    groups["e"] = base * 1.02
    part = mds.h5_cell(groups, "x")[1]
    assert part["value"] == pytest.approx(0.02, rel=1e-3)
    assert part["outcome"] == mds.PASS


@pytest.mark.critical
def test_synthetic_phase_increment_is_independent() -> None:
    """The generator's own winding step matches the analysis and honours the floor."""
    rng = np.random.default_rng(9)
    new = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    old = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    floor = 0.5  # S = |rho|^2 / 2 > 0.5 for about 37 % of the samples
    got, ok = syn.phase_increment(new, old, 2, floor)
    ref, ref_ok = ana.winding_step(new, old, 2, floor)
    assert np.array_equal(ok, ref_ok) and 0 < ok.sum() < ok.size
    np.testing.assert_allclose(got, ref, atol=1e-15)
    assert syn.phase_increment is not ana.winding_step
    assert "winding_step" not in vars(syn)


@pytest.mark.critical
@pytest.mark.parametrize(
    ("every", "evaluated"), [(100, True), (400, False), (0, False)]
)
def test_a4_trace_only_inside_production(
    tmp_path: Path, every: int, evaluated: bool
) -> None:
    """A4: <1/|rho|> from the dense trace inside production, >= 500 per segment.

    every = 100 gives 1000 samples per segment (evaluated; the 1e3-modulus
    samples outside production are excluded); 400 gives 250 (NaN); no trace
    gives NaN. Never filled from the snapshots.
    """
    om = np.tile(np.array([1.0, 1.0, 0.02, 0.03]), (2, 1))
    s_mean = np.tile(np.array([1e-6, 1e-6, 0.5, 0.8]), (2, 1))
    path = cli_files(tmp_path / "x" / "N3" / "md_chi1-3.npz", om, om, [1e-6], 2,
                     s_mean, trace_every=every)  # fmt: skip
    res = ana.analyze_file(path, None)
    want = 1.0 / np.sqrt(3.0 * s_mean)
    if evaluated:
        np.testing.assert_allclose(res["inv_rho"][:, 0], want, rtol=1e-12)
        assert np.all(res["trace_min"] == 100_000 // every)
        assert len(mds.h6b_parts([res], True)) > 0
    else:
        assert np.isnan(res["inv_rho"]).all() and np.isnan(res["ratio_cond_bin"]).all()
        assert mds.h6b_parts([res], True) == []
        assert mds.combine([p["outcome"] for p in mds.h6b_parts([res], True)]) == (
            mds.NOT_EVALUATED
        )


# ============================================================
# D-088: TRACE SPECTRA RESTRICTED TO THE PRODUCTION SEGMENTS
# ============================================================

SPEC_EVERY = 100  # trace cadence of the D-088 fixture (1000 rows per segment)
SPEC_ROWS = 100_000 // SPEC_EVERY


def _tone_trace(
    arrays: Dict[str, np.ndarray], meta: Dict[str, Any], m_in: int, amp_out: float,
    m_out: int, m_last: int = -1,
) -> Dict[str, np.ndarray]:  # fmt: skip
    """Copy of ``arrays`` whose rho trace is a pure tone per region.

    Inside the production segments rho = exp(i w_in t) with w_in = 2 pi m_in /
    (SPEC_ROWS dt_trace), on the segment's FFT grid; outside (equilibration,
    Andersen) rho = amp_out exp(i w_out t). ``m_last >= 0`` replaces m_in in
    the last production segment.
    """
    out = dict(arrays)
    t = arrays["trace_time"]
    steps = np.rint(t / CLI_DT).astype(np.int64)
    prod = np.any([(steps > a) & (steps <= a + b) for a, b in ana.segments(meta)],
                  axis=0)  # fmt: skip
    span = SPEC_ROWS * SPEC_EVERY * CLI_DT
    a_last, n_last = ana.segments(meta)[-1]
    last = (steps > a_last) & (steps <= a_last + n_last)
    m_prod = np.where(last & (m_last >= 0), m_last, m_in)
    phase = np.where(prod, TWO_PI * m_prod / span * t, TWO_PI * m_out / span * t)
    amp = np.where(prod, 1.0, amp_out)
    rho = np.zeros(arrays["trace_rho"].shape)
    rho[..., 0] = (amp * np.cos(phase))[..., None]
    rho[..., 1] = (amp * np.sin(phase))[..., None]
    out["trace_rho"] = rho
    assert prod.any() and not prod.all(), "rows inside and outside production"
    return out


def _spectra_fixture(tmp: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Arrays and meta of a cli_files output with a trace every SPEC_EVERY steps."""
    om = np.tile(np.array([1.0, 1.0, 0.02, 0.03]), (2, 1))
    s_mean = np.tile(np.array([1e-6, 1e-6, 0.5, 0.8]), (2, 1))
    path = cli_files(tmp / "x" / "N3" / "md_spectra_chi1-3.npz", om, om, [1e-6], 2,
                     s_mean, trace_every=SPEC_EVERY)  # fmt: skip
    return ana.load_md(path)


def _segment_periodogram(rho: np.ndarray) -> np.ndarray:
    """Reference: normalized |FFT(rho - mean)|^2 (Q,) of one complex series."""
    x = rho - rho.mean()
    p = np.abs(np.fft.fft(x)) ** 2
    return p / p.sum()


@pytest.mark.critical
def test_regression_D088_trace_spectrum_production_only(tmp_path: Path) -> None:
    """D-088: the rho power spectrum uses only the production (NVE) trace rows.

    Repro: a trace across all phases whose non-production rows carry a loud
    tone at w_out (amplitude 1e3) and whose production rows carry w_in. The
    fixed spectrum is the segment-averaged periodogram of the production
    rows alone (a delta at w_in); the defect transformed the whole uniform
    trace, so w_out dominated. Edge: a trace without production rows (count
    truncated before the first segment) gives no spectrum.
    """
    arrays, meta = _spectra_fixture(tmp_path)
    tone = _tone_trace(arrays, meta, 50, 1e3, 200)
    got = ana._trace_spectra(tone, meta, 2, 1)
    assert int(got["spec_rows"]) == SPEC_ROWS and int(got["spec_segments"]) == 2
    omega, power = got["spec_omega"], got["spec_power"][0, 0]
    w_in = TWO_PI * 50 / (SPEC_ROWS * SPEC_EVERY * CLI_DT)
    peak = int(np.argmax(power))
    assert omega[peak] == pytest.approx(w_in, rel=1e-12)
    assert power[peak] == pytest.approx(1.0, abs=1e-12)  # all power at w_in
    steps = np.rint(tone["trace_time"][0] / CLI_DT).astype(np.int64)
    ref = np.zeros(SPEC_ROWS)
    for a, b in ana.segments(meta):
        rows = np.flatnonzero((steps > a) & (steps <= a + b))
        tr = tone["trace_rho"][0, rows, 0]
        series = tr[:, 0] + 1j * tr[:, 1]
        ref += _segment_periodogram(series) / 2.0
    order = np.argsort(TWO_PI * np.fft.fftfreq(SPEC_ROWS, d=SPEC_EVERY * CLI_DT))
    np.testing.assert_allclose(power, ref[order], atol=1e-12)
    two = ana._trace_spectra(_tone_trace(arrays, meta, 50, 1e3, 200, 80), meta, 2, 1)
    p2 = two["spec_power"][0, 0]  # segments averaged: half the power at each tone
    for m in (50, 80):
        w = TWO_PI * m / (SPEC_ROWS * SPEC_EVERY * CLI_DT)
        assert p2[np.argmin(np.abs(omega - w))] == pytest.approx(0.5, abs=1e-12)
    short = dict(tone, trace_count=np.full(2, 5))  # rows end before production
    assert ana._trace_spectra(short, meta, 2, 1) == {}


@pytest.mark.high
@given(m_in=st.integers(1, SPEC_ROWS // 2 - 1), amp_out=st.floats(0.0, 1e4),
       m_out=st.integers(0, SPEC_ROWS - 1))  # fmt: skip
@example(m_in=50, amp_out=1e3, m_out=200)  # the D-088 repro
def test_regression_D088_spectrum_ignores_non_production_property(
    m_in: int, amp_out: float, m_out: int
) -> None:
    """D-088: the spectrum is unchanged by any trace values outside production."""
    arrays, meta = _D088_FIXTURE
    base = ana._trace_spectra(_tone_trace(arrays, meta, m_in, 0.0, 0), meta, 2, 1)
    got = ana._trace_spectra(_tone_trace(arrays, meta, m_in, amp_out, m_out), meta,
                             2, 1)  # fmt: skip
    np.testing.assert_array_equal(got["spec_power"], base["spec_power"])
    w_in = TWO_PI * m_in / (SPEC_ROWS * SPEC_EVERY * CLI_DT)
    assert got["spec_omega"][np.argmax(got["spec_power"][1, 0])] == pytest.approx(
        w_in, rel=1e-9)  # fmt: skip


def _d088_fixture() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Module-level fixture of the property test (no function-scoped tmp_path)."""
    with tempfile.TemporaryDirectory() as tmp:
        arrays, meta = _spectra_fixture(Path(tmp))
    assert "trace_rho" in arrays, "a traced fixture"
    return arrays, meta


_D088_FIXTURE = _d088_fixture()


# ============================================================
# D-089: SHARED --out-dir
# ============================================================


@pytest.mark.critical
def test_regression_D089_out_dir_keeps_starts_apart(tmp_path: Path) -> None:
    """D-089: two starts at one chi get distinct analysis files under one --out-dir."""
    out = tmp_path / "results"
    a = tmp_path / "flows2d" / "N1024" / "md_chi1-10.npz"
    b = tmp_path / "gau2d" / "N1000" / "md_chi1-10.npz"
    c = tmp_path / "gau2d" / "N1000" / "md_langevin_g0.03_chi1-10.npz"
    old = {(out / ana.output_name(p)) for p in (a, b)}  # pre-fix destination
    assert len(old) == 1  # the starts collided
    dests = [ana.output_path(p, out) for p in (a, b, c)]
    assert dests[0] == out / "flows2d" / "N1024" / "md_analysis_chi1-10.npz"
    assert dests[1] == out / "gau2d" / "N1000" / "md_analysis_chi1-10.npz"
    assert dests[2].name == "md_analysis_langevin_g0.03_chi1-10.npz"
    assert ana.output_path(a, None) == a.parent / "md_analysis_chi1-10.npz"


@pytest.mark.high
@given(
    systems=st.lists(
        st.text("abcdefghij0123456789", min_size=1, max_size=6),
        min_size=2, max_size=5, unique=True,
    ),
    tag=st.sampled_from(["1-10", "2-5"]),
)  # fmt: skip
@example(systems=["flows2d", "gau2d"], tag="1-10")
def test_regression_D089_out_dir_property(systems: List[str], tag: str) -> None:
    """D-089: distinct (system, N) inputs never share an analysis file."""
    root, out = Path("/data"), Path("/results")
    paths = [root / s / "N1000" / f"md_chi{tag}.npz" for s in systems]
    dests = [ana.output_path(p, out) for p in paths]
    assert len(set(dests)) == len(systems)
    assert all(d.parent.parent.name == s for d, s in zip(dests, systems, strict=True))


@pytest.mark.high
@given(
    rest=st.sampled_from(["", "_langevin_g0.03", "_spectra"]),
    tag=st.sampled_from(["1-10", "2-5", "7-20"]),
)
@example(rest="", tag="1-10")
def test_regression_D090_analysis_output_is_not_an_input(rest: str, tag: str) -> None:
    """D-090: an md_analysis*_chi<tag>.npz is rejected as an MD input by name.

    The README glob ``md_*chi*.npz`` matched a stray analysis file, which then
    failed in load_md with a missing-arrays error; the name check fails first,
    with a message naming the cause. Genuine MD outputs keep their names.
    """
    with pytest.raises(ValueError, match="analysis output"):
        ana.output_name(Path(f"/d/flows2d/N1024/md_analysis{rest}_chi{tag}.npz"))
    good = ana.output_name(Path(f"/d/flows2d/N1024/md{rest}_chi{tag}.npz"))
    assert good == f"md_analysis{rest}_chi{tag}.npz"
