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
import sys
from pathlib import Path
from typing import Any, Dict, List

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy import stats as sps

# ----- project -----
from tests.tolerances import (
    ATOL_MD_ALPHA_EXACT,
    ATOL_MD_ALPHA_OU,
    ATOL_MD_ALPHA_RW,
    ATOL_MD_SYN_ALPHA,
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
finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False,
                   allow_infinity=False, allow_subnormal=False)  # fmt: skip
samples = hnp.arrays(np.float64, st.integers(1, 12), elements=finite)

# ============================================================
# HELPERS
# ============================================================


def cli_files(
    path: Path, omega_abs: np.ndarray, omega_net: np.ndarray, temps: List[float],
    n_gs: int, s_mean: np.ndarray, net_msd: Any = None,
) -> Path:  # fmt: skip
    """Write a minimal CLI-schema md_chi<tag>.npz / .json with exact windings.

    Lane l, wave w: W_abs = omega_abs[l, w] t and W_net = omega_net[l, w] t
    (radians, t since the segment began), or, if ``net_msd`` (a function of
    t in turns^2) is given, W_net = +-2 pi sqrt(net_msd(t)) with alternating
    signs. Waves: 2 constrained (|k| = 0.5, 0.9) then 2 unconstrained (1.4,
    1.8) with K = 1; sums give <S_k> = s_mean[l, w] and <Phi> = M T.
    """
    lanes, waves = omega_abs.shape
    assert lanes == n_gs * len(temps) and waves == 4, "4 waves per lane"
    proto = syn.default_proto(100_000, 2, 120)
    phases, rec = syn._protocol(proto)
    dt = 0.01
    segs = ana.segments({"phases": [dict(p.__dict__) for p in phases]})
    w_abs = np.zeros((lanes, rec.size, waves))
    w_net = np.zeros((lanes, rec.size, waves))
    for s0, n in segs:
        j = np.flatnonzero((rec > s0) & (rec <= s0 + n))
        t = (rec[j] - s0) * dt
        w_abs[:, j] = omega_abs[:, None, :] * t[None, :, None]
        if net_msd is None:
            w_net[:, j] = omega_net[:, None, :] * t[None, :, None]
        else:
            sign = np.where(np.arange(waves) % 2 == 0, 1.0, -1.0)
            w_net[:, j] = TWO_PI * np.sqrt(net_msd(t))[None, :, None] * sign
    lane_t = np.tile(np.asarray(temps), n_gs)
    kvec = np.array([[0.5, 0.0], [0.0, 0.9], [1.4, 0.0], [0.0, 1.8]])
    n_sum = np.full((2, lanes), 1000)
    arrays = {
        "snap_step": np.tile(rec, (lanes, 1)),
        "snap_time": np.tile(rec * dt, (lanes, 1)),
        "snap_wind_abs": w_abs,
        "snap_wind_net": w_net,
        "snap_rho": np.zeros((lanes, rec.size, waves, 2)),
        "snap_r": np.zeros((lanes, rec.size, 3, 2)),
        "snap_phi": np.ones((lanes, rec.size)),
        "snap_kinetic": np.ones((lanes, rec.size)),
        "snap_count": np.full(lanes, rec.size),
        "cycle_sum_s": n_sum[..., None] * s_mean[None],
        "cycle_n_sum": n_sum,
        "cycle_sum_phi": n_sum * 2 * lane_t[None],
        "cycle_sum_kinetic": n_sum * 2 * lane_t,
        "status": np.zeros(lanes, np.int64),
        "lane_index": np.repeat(np.arange(n_gs), len(temps)),
        "lane_temperature": lane_t,
        "temperatures": np.asarray(temps),
        "indices": np.arange(n_gs),
        "waves_kvec": kvec,
        "waves_constrained": np.array([True, True, False, False]),
        "record_steps": rec,
        "basis": np.sqrt(3.0) * np.eye(2),
    }
    meta = {"phases": [dict(p.__dict__) for p in phases], "settings": {"dt": dt},
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
def test_kruskal_epsilon2_hand_example() -> None:
    """Three separated groups of three: H = 7.2, eps^2 = H (n+1)/(n^2-1) = 0.9."""
    groups = [np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), np.array([7.0, 8, 9])]
    h, p, eps2 = mds.kruskal_epsilon2(groups)
    assert h == pytest.approx(7.2, rel=1e-12)
    assert eps2 == pytest.approx(7.2 * 10 / 80, rel=1e-12)
    assert p == pytest.approx(sps.chi2.sf(7.2, 2), rel=1e-12)
    assert mds.kruskal_epsilon2([np.ones(3), np.ones(4)]) == (0.0, 1.0, 0.0)


@pytest.mark.high
@given(a=samples, b=samples, c=samples)
def test_kruskal_epsilon2_property(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    """eps^2 = H (n+1)/(n^2-1) with scipy's H, in [0, 1]."""
    n = a.size + b.size + c.size
    if n < 3:
        return
    h, _, eps2 = mds.kruskal_epsilon2([a, b, c])
    pooled = np.concatenate([a, b, c])
    if np.all(pooled == pooled[0]):
        assert eps2 == 0.0
        return
    assert h == pytest.approx(sps.kruskal(a, b, c).statistic, rel=1e-12)
    assert eps2 == pytest.approx(h * (n + 1) / (n * n - 1), rel=1e-12)
    assert -1e-12 <= eps2 <= 1.0 + 1e-12


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
        h, _, eps2 = mds.kruskal_epsilon2([groups[s] for s in names])
        assert parts[0]["value"] == pytest.approx(h * 51 / 2499, rel=1e-12) == eps2


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
    assert "hip2d" in worst.get("pair", []) or "hip2d" in worst.get("medians", {})


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
