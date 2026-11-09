#!/usr/bin/env python3

"""
Tests for the post hoc diagnostics of the MD winding experiment
(experiments/scripts/batten2008_md_diagnostics.py): bin wave counts behind
the H1 / H2 outcomes, the thermostat pairing, the circular-Gaussian Rice
ratio of the unconstrained waves, the drift exponents, the energy and gate
summary and the spectral support.

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
from typing import Any, Dict, List

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
import batten2008_md_analysis as ana  # noqa: E402
import batten2008_md_diagnostics as dg  # noqa: E402

# ============================================================
# HELPERS
# ============================================================


def _layout(counts: List[int]) -> Dict[str, Any]:
    """kmag / constrained / bin_id with ``counts[b]`` constrained waves in bin b."""
    centres = 0.5 * (ana.BIN_EDGES_C[:-1] + ana.BIN_EDGES_C[1:])
    kc = np.repeat(centres[: len(counts)], counts)
    kmag = np.concatenate([kc, [1.3, 1.7]])
    con = np.concatenate([np.ones(kc.size, bool), np.zeros(2, bool)])
    return {"kmag": kmag, "constrained": con, "bin_id": ana.bin_ids(kmag, 1.0, con)}


def _cell(n_gs: int, temps: List[float], seed: int) -> Dict[str, Any]:
    """Minimal production analysis with alpha / d_theta / nu per (GS, T)."""
    rng = np.random.default_rng(seed)
    t = np.asarray(temps)
    z = {"system": "x", "tag": "1-10", "temps": t, "indices": np.arange(n_gs),
         "alpha": 1.0 + 0.01 * rng.standard_normal((n_gs, t.size)),
         "d_theta": rng.uniform(0.1, 1.0, (n_gs, t.size)),
         "nu_c": 1.0 + 0.001 * rng.standard_normal((n_gs, t.size))}  # fmt: skip
    return z


# ============================================================
# BIN WAVES
# ============================================================


@pytest.mark.critical
def test_bin_counts_and_outcome_waves() -> None:
    """Wave counts per bin, and the counts behind each outcome of the bin parts."""
    z = _layout([0, 2, 6, 20])
    counts = dg.bin_counts(z)
    assert counts.shape == (ana.N_BINS,) and counts[:4].tolist() == [0, 2, 6, 20]
    parts = [
        {"part": "S bin 1", "cell": "x chi=1-10 T=1e-06", "outcome": "FAIL"},
        {"part": "Rice bin 2", "cell": "x chi=1-10 T=1e-05", "outcome": "FAIL"},
        {"part": "S bin 3", "cell": "x chi=1-10 T=1e-06", "outcome": "PASS"},
        {"part": "phi", "cell": "x chi=1-10 T=1e-06", "outcome": "FAIL"},
        {"part": "T ratio bin 3", "cell": "x chi=1-10", "outcome": "PASS"},
    ]
    res = dg.bin_waves(parts, {("x", "1-10"): counts})
    assert res["FAIL"] == {"n_parts": 2, "median": 4.0, "min": 2, "max": 6}
    assert res["PASS"] == {"n_parts": 1, "median": 20.0, "min": 20, "max": 20}


# ============================================================
# THERMOSTAT, RATES, DRIFT
# ============================================================


@pytest.mark.high
def test_thermostat_pairs_common_ground_states() -> None:
    """BAOAB (GS 0-4) is compared with the same NVE ground states, at shared T."""
    main = _cell(10, [1e-6, 1e-5, 1e-4, 1e-3], 0)
    lang = dict(_cell(5, [1e-6, 1e-4], 1), gamma=np.asarray(0.03))
    main["alpha"][5:] = 99.0  # ground states absent from E4 must not enter
    rows = dg.thermostat([main], [lang])
    assert [r["T"] for r in rows] == [1e-6, 1e-4] and all(r["n_gs"] == 5 for r in rows)
    assert rows[1]["nve"]["alpha"] == float(np.median(main["alpha"][:5, 2]))
    assert rows[0]["baoab"]["d_theta"] == float(np.median(lang["d_theta"][:, 0]))
    assert dg.thermostat([], [lang]) == []


@pytest.mark.critical
def test_unconstrained_ratio_is_rate_over_circular_rice() -> None:
    """H6 rate / Rice = sqrt(pi) is a rate equal to |k| sqrt(T / S) / 2 pi: ratio 1."""
    ratio = np.full((3, 2, 2 * ana.N_BINS), math.sqrt(math.pi))
    ratio[..., ana.N_BINS + 5] = np.nan  # an empty unconstrained bin
    ratio[..., : ana.N_BINS] = 7.0  # constrained bins are ignored
    rows = dg.unconstrained({"ratio_bin": ratio, "temps": np.array([1e-6, 1e-4])})
    for r in rows:
        assert r["n_bins"] == ana.N_BINS - 1
        assert r["min"] == pytest.approx(1.0, abs=1e-15)
        assert r["max"] == pytest.approx(1.0, abs=1e-15)


@pytest.mark.high
@given(p=st.floats(-3.0, 3.0), c=st.floats(1e-6, 1e6), n=st.integers(2, 40))
@example(p=0.25, c=1.0, n=2)
def test_loglog_slope_power_law(p: float, c: float, n: int) -> None:
    """y = c x^p gives slope p; fewer than two positive points give NaN."""
    x = np.geomspace(1e-2, 1e3, n)
    assert dg.loglog_slope(x, c * x**p) == pytest.approx(p, abs=1e-9)
    y = np.full(n, -1.0)
    y[0] = 1.0
    assert math.isnan(dg.loglog_slope(x, y))


@pytest.mark.high
def test_drift_exponents_of_diffusive_drift() -> None:
    """MSD = c sqrt(T) t: t exponent 1 per T, end-MSD T exponent 1/2 (harmonic T)."""
    temps = np.array([1e-6, 1e-5, 1e-4, 1e-3])
    lag = np.geomspace(0.04, 4e4, 100)
    msd = 0.3 * np.sqrt(temps)[None, :, None] * lag[None, None, :]
    msd = np.repeat(msd, 3, axis=0)
    msd[:, 3, -1] *= 1e3  # T = 1e-3 is not harmonic: no effect on the T exponent
    res = dg.drift({"part_lag": lag, "part_msd": msd, "temps": temps})
    assert res["T_exponent_end"] == pytest.approx(0.5, abs=1e-12)
    assert res["t_exponent_last_decade"][:3] == pytest.approx([1.0] * 3, abs=1e-12)
    assert res["lag_end"] == lag[-1]


# ============================================================
# ENERGY AND SPECTRA
# ============================================================


@pytest.mark.medium
def test_energy_moments_and_gate_misses() -> None:
    """Phi / T mean / M and sd / sqrt(M) over harmonic T; |gate - 1| > 5 % misses."""
    m = 400
    rng = np.random.default_rng(5)
    phi = np.full((10, 4, 50), 1e9)  # T = 1e-3 column: excluded
    phi[:, :3] = rng.gamma(m, 1.0, (10, 3, 50))
    gate = np.ones((10, 4))
    gate[0, 0], gate[1, 2], gate[2, 3] = 1.06, 0.97, 0.90
    z = {"M": np.asarray(m), "temps": np.array([1e-6, 1e-5, 1e-4, 1e-3]),
         "phi_over_t": phi, "gate": gate}  # fmt: skip
    res = dg.energy(z)
    sample = phi[:, :3].ravel()
    assert res["mean_over_M"] == pytest.approx(sample.mean() / m, rel=1e-14)
    assert res["sd_over_sqrtM"] == pytest.approx(sample.std(ddof=1) / 20.0, rel=1e-12)
    assert res["gate_misses"] == 2 and res["gate_lanes"] == 40
    assert res["gate_max_dev"] == pytest.approx(0.10, abs=1e-15)


@pytest.mark.critical
def test_spectral_edge_box_spectrum() -> None:
    """Power on [0.5, 2] only: all of it inside 1.05 f_max = 2.1, edge at 2."""
    omega = np.linspace(-4.0, 4.0, 801)
    power = np.where((np.abs(omega) >= 0.5) & (np.abs(omega) <= 2.0), 1.0, 0.0)
    res = dg.spectral_edge(omega, power, 2.0)
    assert res["power_inside"] == pytest.approx(1.0, abs=1e-15)
    assert res["edge"] == pytest.approx(2.0, abs=1e-12)
    assert res["edge_over_f_max"] == pytest.approx(1.0, abs=1e-12)
    half = dg.spectral_edge(omega, power, 1.0)  # support beyond f_max
    assert 0.0 < half["power_inside"] < 1.0


@pytest.mark.low
def test_markdown_is_labelled_post_hoc() -> None:
    """The report says it is descriptive and not pre-registered; no verdict column."""
    diag = {"label": dg.LABEL, "thermostat": [], "cells": {}, "spectra": []}
    text = dg.markdown(diag)
    assert text.startswith("# MD winding: post hoc diagnostics")
    assert "not pre-registered" in text and "Verdict" not in text
