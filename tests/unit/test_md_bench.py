#!/usr/bin/env python3

"""
Unit tests of the pure helpers of the V3 benchmark (``benchmarks/md_bench.py``).

References: the device winding rule (``observables.winding_update``, applied
step by step), closed forms for the ratio-of-means statistic, and the
production schedule of ``cli.md``.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple

# ----- numerics -----
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# ----- project -----
from tolerances import MD_BENCH_NULL_Z, MD_BENCH_TINY_RHO, RTOL_MD_BENCH_WIND
from xla_cco.dynamics.observables import winding_update

# ============================================================
# MODULE UNDER TEST
# ============================================================

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "md_bench.py"


def _load() -> ModuleType:
    """Import benchmarks/md_bench.py as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location("md_bench", SCRIPT)
    assert spec is not None and spec.loader is not None, "md_bench.py importable"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["md_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


MB = _load()

# ============================================================
# HELPERS
# ============================================================


def _device_winding(rho: np.ndarray, rho0: np.ndarray, floor: float) -> np.ndarray:
    """Unsigned winding (lanes, w) by winding_update applied every step."""
    lanes, n, w, _ = rho.shape
    out = np.zeros((lanes, w))
    for lane in range(lanes):
        prev = jnp.asarray(rho0[lane])
        wa, wn = jnp.zeros(w), jnp.zeros(w)
        nv = jnp.zeros(w, dtype=jnp.int64)
        active = jnp.ones(w, dtype=bool)
        for i in range(n):
            prev, wa, wn, nv = winding_update(
                prev, jnp.asarray(rho[lane, i]), wa, wn, nv, active, 4, floor
            )
        out[lane] = np.asarray(wa)
    return out


def _results(worst: Dict[int, float]) -> List[Dict[str, Any]]:
    """Synthetic wind_every trial results with the given max per-wave errors."""
    one = {str(e): {"max_wave_rel": [worst[e], 0.0]} for e in (2, 5, 10)}
    return [one, one]


RHO = hnp.arrays(
    np.float64,
    st.tuples(
        st.integers(1, 2), st.integers(1, 12), st.integers(1, 3), st.just(2)
    ),  # fmt: skip
    elements=st.floats(-3.0, 3.0, allow_nan=False),
)

# ============================================================
# TESTS
# ============================================================


@pytest.mark.high
@given(rho=RHO, floor=st.sampled_from([0.0, 0.05]))
@example(rho=np.full((1, 1, 1, 2), -1.04e-158), floor=0.0)  # D-072
def test_regression_D072_offline_winding_every_step_matches_device_rule(
    rho: np.ndarray, floor: float
) -> None:
    # components below MD_BENCH_TINY_RHO are set to 0: their S = |rho|^2 / N would
    # be subnormal, which NumPy keeps (> 0, counted) and XLA on CPU flushes to 0
    # (masked), so the two rules legitimately differ there (D-072)
    rho = np.where(np.abs(rho) < MD_BENCH_TINY_RHO, 0.0, rho)
    rho0 = rho[:, 0] * 0.5 + 0.1
    got = MB.offline_winding(rho, rho0, 1, floor, 4)
    ref = _device_winding(rho, rho0, floor)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=RTOL_MD_BENCH_WIND, atol=RTOL_MD_BENCH_WIND)


@pytest.mark.high
@given(polar=RHO, every=st.integers(1, 12))
def test_offline_winding_subsampled_never_exceeds_every_step(
    polar: np.ndarray, every: int
) -> None:
    # |wrap(sum delta)| <= sum |delta|: a sparser cadence can only undercount
    # (|rho| >= 0.1 everywhere, so the phase floor never masks a step)
    radius, angle = 0.1 + np.abs(polar[..., 0]), polar[..., 1]
    rho = np.stack([radius * np.cos(angle), radius * np.sin(angle)], -1)
    every = min(every, rho.shape[1])
    rho0 = np.broadcast_to(np.array([0.5, 0.0]), rho[:, 0].shape)
    full = MB.offline_winding(rho, rho0, 1, 0.0, 4)
    sub = MB.offline_winding(rho, rho0, every, 0.0, 4)
    assert np.all(sub <= full + 1e-12)
    assert np.all(sub >= 0.0)


@pytest.mark.medium
def test_offline_winding_example_quarter_turns() -> None:
    # rho on the unit circle at 0, 90, 180, 270, 360 deg: every-step winding 2 pi,
    # every second step sees two half turns (atan2 gives +pi each): 2 pi as well,
    # every fourth step sees no net turn: 0
    ang = np.deg2rad([90.0, 180.0, 270.0, 360.0])
    rho = np.stack([np.cos(ang), np.sin(ang)], -1)[None, :, None, :]
    rho0 = np.array([[[1.0, 0.0]]])
    w = [MB.offline_winding(rho, rho0, e, 0.0, 1)[0, 0] for e in (1, 2, 4)]
    assert w[0] == pytest.approx(2 * math.pi)
    assert w[1] == pytest.approx(2 * math.pi)
    assert w[2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.medium
@given(x=st.floats(1e-12, 1e12, allow_nan=False))
@example(x=1.0)
@example(x=2.0000000001)
@example(x=999999999999.998)  # D-071: log10 rounds up to 12
def test_regression_D071_round_up_125_property(x: float) -> None:
    y = MB.round_up_125(x)
    mant = y / 10.0 ** math.floor(math.log10(y) + 1e-12)
    assert y >= x * (1.0 - 1e-12)
    assert y <= 2.5 * x * (1.0 + 1e-12)
    assert min(abs(mant - m) for m in (1.0, 2.0, 5.0)) < 1e-9


@pytest.mark.medium
@given(kk=hnp.arrays(np.float64, st.integers(1, 50),
                     elements=st.floats(1e-6, 3.0)))  # fmt: skip
def test_bin_index_property(kk: np.ndarray) -> None:
    idx = MB.bin_index(kk)
    assert np.all((idx >= 0) & (idx < MB.N_BINS))
    order = np.argsort(kk)
    assert np.all(np.diff(idx[order]) >= 0)  # monotone in |k|
    inside = kk < 2.0
    assert np.all(idx[inside] == np.floor(kk[inside] / (2.0 / MB.N_BINS)).clip(0, 19))


@pytest.mark.high
@given(eps=st.floats(-0.05, 0.05).filter(lambda e: abs(abs(e) - 0.01) > 1e-6),
       seed=st.integers(0, 2**31 - 1))  # fmt: skip
def test_compare_pooled_uniform_change(eps: float, seed: int) -> None:
    # W_dt = (1 + eps) W_dt2 exactly: Delta_b = eps in every bin, SE_b = 0
    rng = np.random.default_rng(seed)
    kk = rng.uniform(0.05, 1.95, 60)
    wb = rng.uniform(1.0, 5.0, (8, 60))
    temps = np.repeat(np.asarray(MB.TEMPS), 2)
    out = MB.compare_pooled([((1.0 + eps) * wb, wb, kk, temps)])
    for t in MB.TEMPS:
        r = out[f"{t:g}"]
        d = np.asarray(r["delta"], dtype=float)
        assert np.allclose(d[np.isfinite(d)], eps, atol=1e-12)
        assert r["pass"] == (abs(eps) < MB.WIND_DT_REL)


@pytest.mark.critical
@given(seed=st.integers(0, 2**31 - 1))
@example(seed=0)
def test_regression_D070_ratio_of_means_unbiased_under_noise(seed: int) -> None:
    # matched pairs with no dt effect but CV 0.3 noise (decorrelated waves): the
    # ratio of means is unbiased within its delta-method SE; the mean of per-pair
    # ratios (the D-070 estimator) is biased by ~ +CV^2
    rng = np.random.default_rng(seed)
    n, s = 4000, math.sqrt(math.log(1.0 + 0.3**2))
    a = rng.lognormal(0.0, s, n)
    b = rng.lognormal(0.0, s, n)
    bins_a = [np.empty(0)] * MB.N_BINS
    bins_b = [np.empty(0)] * MB.N_BINS
    bins_a[3], bins_b[3] = a, b
    delta, se = MB._bin_stats(bins_a, bins_b)
    assert abs(delta[3]) < MD_BENCH_NULL_Z * se[3]
    assert np.mean(a / b) - 1.0 > MD_BENCH_NULL_Z * se[3]  # the old estimator


@pytest.mark.medium
def test_choose_wind_every_takes_largest_passing_prefix() -> None:
    assert MB.choose_wind_every(_results({2: 0.001, 5: 0.005, 10: 0.02})) == 5
    assert MB.choose_wind_every(_results({2: 0.02, 5: 0.001, 10: 0.001})) == 1
    assert MB.choose_wind_every(_results({2: 0.0, 5: 0.0, 10: 0.0099})) == 10


@pytest.mark.medium
@given(n_nve=st.integers(500, 5_000_000))
def test_production_records_cover_the_run(n_nve: int) -> None:
    steps, every, cap = MB.production_records(n_nve)
    total = 2 * MB.N_EQUIL + MB.CYCLES * (MB.N_ANDERSEN + n_nve)
    assert every == n_nve // MB.TRACE_SAMPLES and cap == total // every
    assert n_nve // every >= MB.TRACE_SAMPLES  # >= 500 samples per NVE segment
    assert len(steps) <= MB.N_SNAPSHOTS and list(steps) == sorted(set(steps))
    assert steps[-1] == total


@pytest.mark.medium
@given(r10=st.floats(10.0, 1e4), frac=st.floats(0.05, 1.0),
       lanes=st.integers(1, 80))  # fmt: skip
def test_lane_rate_interpolates_time_per_step(
    r10: float, frac: float, lanes: int
) -> None:
    r40 = r10 * frac
    assert MB.lane_rate(r10, r40, 10) == pytest.approx(r10)
    assert MB.lane_rate(r10, r40, 40) == pytest.approx(r40)
    rate = MB.lane_rate(r10, r40, lanes)
    assert rate > 0.0
    if lanes >= 10:
        assert rate <= r10 * (1.0 + 1e-12)


# ============================================================
# SELECTION (choose_dt, confirm_dt, compare_pooled)
# ============================================================


def _ladder(
    drift: Dict[float, float], std: Dict[float, float], wind_ok: Dict[float, bool]
) -> Dict[str, Any]:
    """Synthetic dt_ladder result: per-dt drift/std maxima and (c) verdicts."""
    res: Dict[str, Any] = {"drift": {}, "wind": {}}
    for dt in MB.LADDER:
        for name in MB.CALIB_SYSTEMS:
            res["drift"][f"{name}/{dt:g}"] = {
                "drift_rel": [0.1 * drift[dt], drift[dt]],
                "std_rel": [std[dt], 0.5 * std[dt]],
                "max_dev_dof": [1e-5, 2e-5],
            }
        res["wind"][f"{dt:g}"] = {
            f"{t:g}": {"pass": wind_ok[dt] or t < 1e-3, "max_abs_delta": 0.0,
                       "max_excess": 0.0, "all_wave_rel": 0.0}
            for t in MB.TEMPS
        }  # fmt: skip
    return res


OK_DRIFT = {dt: 1e-6 for dt in MB.LADDER}
OK_STD = {dt: 1e-5 for dt in MB.LADDER}
OK_WIND = {dt: True for dt in MB.LADDER}


@pytest.mark.high
def test_choose_dt_picks_largest_passing() -> None:
    dt, verdict = MB.choose_dt(_ladder(OK_DRIFT, OK_STD, OK_WIND))
    assert dt == max(MB.LADDER)
    assert all(v["pass"] for v in verdict.values())


@pytest.mark.high
@pytest.mark.parametrize("gate", ["drift", "std", "wind"])
def test_choose_dt_each_gate_vetoes(gate: str) -> None:
    drift, std, wind = dict(OK_DRIFT), dict(OK_STD), dict(OK_WIND)
    if gate == "drift":
        drift[0.04] = 2.0 * MB.DRIFT_REL
    elif gate == "std":
        std[0.04] = 2.0 * MB.STD_REL
    else:
        wind[0.04] = False
    dt, verdict = MB.choose_dt(_ladder(drift, std, wind))
    assert dt == 0.02
    assert not verdict["0.04"]["pass"] and verdict["0.02"]["pass"]


@pytest.mark.high
def test_choose_dt_none_when_no_dt_passes() -> None:
    dt, verdict = MB.choose_dt(_ladder(OK_DRIFT, OK_STD, dict.fromkeys(OK_WIND, False)))
    assert dt is None
    assert not any(v["pass"] for v in verdict.values())


@pytest.mark.high
def test_confirm_dt_skips_failing_full_length(monkeypatch: pytest.MonkeyPatch) -> None:
    # full-length drift fails at 0.04, passes at 0.02; 0.01 is never tried
    tried: List[float] = []

    def fake_trial(sysd: Any, st: Any, dt: float, n: int) -> Dict[str, Any]:
        tried.append(dt)
        bad = dt == 0.04
        return {"drift_rel": [2.0 * MB.DRIFT_REL if bad else 1e-6], "std_rel": [1e-5]}

    monkeypatch.setattr(MB, "drift_trial", fake_trial)
    monkeypatch.setattr(MB, "first_lanes", lambda st, n: st)
    systems = [{"system": n} for n in MB.CALIB_SYSTEMS]
    verdict = {f"{dt:g}": {"pass": True} for dt in MB.LADDER}
    dt, trials = MB.confirm_dt(systems, [None, None], verdict, 1000)
    assert dt == 0.02 and set(trials) == set(MB.CALIB_SYSTEMS)
    assert 0.01 not in tried
    verdict["0.02"]["pass"] = False  # ladder-failed dts are skipped
    dt, _ = MB.confirm_dt(systems, [None, None], verdict, 1000)
    assert dt == 0.01


def _bin_entries(delta: float, se: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """(a, b) of one bin with ratio of means 1 + delta and delta-method SE se:
    b = 1, a = 1 + delta + eps s_i with s_i = +-1 (mean 0), eps = se sqrt(n - 1)."""
    assert n % 2 == 0, "alternating signs need an even count"
    s = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return 1.0 + delta + se * math.sqrt(n - 1) * s, np.ones(n)


def _pooled(bins: Dict[int, Tuple[float, float, int]]) -> Dict[str, Any]:
    """compare_pooled on one lane per T with the given {bin: (delta, se, n)}."""
    ka, kb, kk = [], [], []
    for b, (delta, se, n) in bins.items():
        a_b, b_b = _bin_entries(delta, se, n)
        ka.append(a_b)
        kb.append(b_b)
        kk.append(np.full(n, (b + 0.5) * 2.0 / MB.N_BINS))
    wa = np.tile(np.concatenate(ka), (len(MB.TEMPS), 1))
    wb = np.tile(np.concatenate(kb), (len(MB.TEMPS), 1))
    temps = np.asarray(MB.TEMPS)
    return MB.compare_pooled([(wa, wb, np.concatenate(kk), temps)])["1e-06"]


@pytest.mark.high
def test_compare_pooled_bin_statistics_closed_form() -> None:
    r = _pooled({3: (0.02, 0.008, 50), 7: (-0.004, 0.002, 400)})
    assert r["delta"][3] == pytest.approx(0.02, abs=1e-12)
    assert r["se"][3] == pytest.approx(0.008, rel=1e-9)
    assert r["delta"][7] == pytest.approx(-0.004, abs=1e-12)
    assert r["se"][7] == pytest.approx(0.002, rel=1e-9)


@pytest.mark.high
def test_compare_pooled_noisy_bin_judged_with_its_se() -> None:
    # noisy bin (SE 0.8 % > NOISE_SE): |Delta| = 2 % passes since 2 - 2 * 0.8 < 1;
    # |Delta| = 3 % fails (3 - 1.6 > 1); a large quiet bin keeps the all-wave < 1 %
    quiet = (0.0, 0.0, 2000)
    ok = _pooled({3: (0.02, 0.008, 50), 12: quiet})
    assert ok["pass"] and ok["n_noise_dominated"] == 1
    assert ok["max_excess"] == pytest.approx(0.02 - 0.016, abs=1e-9)
    bad = _pooled({3: (0.03, 0.008, 50), 12: quiet})
    assert not bad["pass"]


@pytest.mark.high
def test_compare_pooled_resolved_bin_judged_on_point_estimate() -> None:
    # resolved bin (SE 0.2 % <= NOISE_SE): 1.2 % fails although 1.2 - 0.4 < 1
    quiet = (0.0, 0.0, 2000)
    bad = _pooled({3: (0.012, 0.002, 400), 12: quiet})
    assert not bad["pass"] and bad["n_noise_dominated"] == 0
    assert bad["max_excess"] == pytest.approx(0.012, abs=1e-9)
    assert _pooled({3: (0.008, 0.002, 400), 12: quiet})["pass"]


@pytest.mark.high
def test_compare_pooled_all_wave_check_alone_fails() -> None:
    # every bin noisy with Delta 1.5 %, SE 0.8 %: each bin passes (1.5 - 1.6 < 1)
    # but the summed winding changes by 1.5 % > 1 %
    r = _pooled({b: (0.015, 0.008, 50) for b in (2, 5, 9, 14)})
    assert r["max_excess"] < MB.WIND_DT_REL
    assert r["all_wave_rel"] == pytest.approx(0.015, rel=1e-9)
    assert not r["pass"]


# ============================================================
# COST PROJECTION
# ============================================================

INIT_S = 0.7


def _rows(rate_2k: float, rate_c: float) -> List[Dict[str, Any]]:
    """Synthetic throughput rows: every 2K configuration at rate_2k steps/s,
    constrained-only at rate_c (both chi, 10 and 40 lanes)."""
    rows: List[Dict[str, Any]] = []
    for tag in MB.CHI_TAGS:
        for lanes in (10, 40):
            for scheme, wes in (("verlet", (1, 2, 5, 10)), ("andersen", (1,)),
                                ("baoab", (1,))):  # fmt: skip
                for we in wes:
                    rows.append({"tag": tag, "lanes": lanes, "waves": "2K",
                                 "scheme": scheme, "wind_every": we,
                                 "k_chunk": None, "steps_per_s": rate_2k,
                                 "peak_bytes": 10**9, "W": 3000, "M": 800})  # fmt: skip
            rows.append({"tag": tag, "lanes": lanes, "waves": "constrained",
                         "scheme": "verlet", "wind_every": 1, "k_chunk": None,
                         "steps_per_s": rate_c, "peak_bytes": 10**8, "W": 800,
                         "M": 800})  # fmt: skip
    return rows


def _cal(dt: float, we: int) -> Dict[str, Any]:
    """Synthetic calibration result of one chi."""
    return {"dt": dt, "wind_every": we, "init_s_per_lane": [INIT_S, 0.5 * INIT_S]}


CALS = {"1-10": _cal(0.04, 5), "2-5": _cal(0.02, 2)}


def _e1_closed(rate: float, n_nve: int, lanes: int) -> float:
    """Hours of one E1 job: 2 equilibrations + cycles x (Andersen + NVE)."""
    steps = 2 * 200_000 + 2 * (100_000 + n_nve)
    return (steps / rate * 1.05 * 1.25 + lanes * INIT_S + 120.0) / 3600.0


@pytest.mark.high
@given(rate=st.floats(10.0, 1e5), n_nve=st.sampled_from([500_000, 1_000_000]),
       n_gs=st.sampled_from([6, 10]), n_t=st.sampled_from([3, 4]))  # fmt: skip
def test_e1_job_hours_closed_form(rate: float, n_nve: int, n_gs: int, n_t: int) -> None:
    out = MB.e1_job_hours(_rows(rate, 10 * rate), CALS["2-5"], "2-5", n_nve, n_gs, n_t)
    assert out["hours"] == pytest.approx(_e1_closed(rate, n_nve, n_gs * n_t), rel=1e-9)
    assert out["lanes"] == n_gs * n_t
    assert out["steps_per_lane_worst"] == 600_000 + 2 * n_nve


@pytest.mark.high
def test_e1b_and_e4_job_hours_closed_form() -> None:
    rows = _rows(500.0, 4000.0)
    e1b = MB.e1b_job_hours(rows, CALS["1-10"], "1-10")
    steps = 2 * 200_000 + 100_000 + 5000  # t = 200 at dt 0.04
    want = (steps / 4000.0 * 1.05 * 1.25 + 2 * INIT_S + 120.0) / 3600.0
    assert e1b["hours"] == pytest.approx(want, rel=1e-9)
    assert e1b["max_rho_records"] == steps // 5
    e4 = MB.e4_job_hours(rows, CALS["1-10"], "1-10", 1_000_000)
    want = ((400_000 + 2_000_000) / 500.0 * 1.05 * 1.25 + 10 * INIT_S + 120.0) / 3600
    assert e4["hours"] == pytest.approx(want, rel=1e-9)


@pytest.mark.high
@pytest.mark.parametrize(
    ("rate", "label"),
    [(1e4, "none"), (150.0, "n_nve=5e5"),
     (30.0, "n_nve=5e5, 6 ground states, drop T=1e-5")],
)  # fmt: skip
def test_production_plan_applies_reduction_order(rate: float, label: str) -> None:
    plan = MB.production_plan(_rows(rate, 1e6), CALS)
    assert plan["reduction_applied"] == label
    first = plan["tried"][0]
    n_jobs = len(MB.SYSTEMS) * len(MB.CHI_TAGS)
    assert first["E1_hours"] == pytest.approx(n_jobs * _e1_closed(rate, 10**6, 40))
    assert plan["production_within_budget"] == (plan["production_hours"] <= 60.0)
    assert [t["reduction"] for t in plan["tried"]][-1] == label
