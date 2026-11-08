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
from typing import Any, Dict, List

# ----- numerics -----
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# ----- project -----
from tolerances import MD_BENCH_NULL_Z, RTOL_MD_BENCH_WIND
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
def test_offline_winding_every_step_matches_device_rule(
    rho: np.ndarray, floor: float
) -> None:
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
