#!/usr/bin/env python3

"""
Tests for the spectrum statistics of the coupling figures
(experiments/scripts/batten2008_mp_plot.py, batten2008_mp_shoulder_plot.py)
on starts other than Poisson. The scripts draw with matplotlib, which the
project environment does not install (they are skipped there); run them
with the plottime interpreter:
  .../plottime/bin/python -m pytest --noconftest -o addopts="" \
      tests/unit/test_batten2008_coupling_plots.py

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import sys
from pathlib import Path

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
from batten2008_mp_plot import ks_eff, stats  # noqa: E402
from batten2008_mp_shoulder_plot import min_density  # noqa: E402

# ============================================================
# TESTS
# ============================================================


@pytest.mark.medium
@given(zeros=st.integers(1, 40), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=25, deadline=None)
@example(zeros=89, seed=0)  # D-018: HIP starts at chi = 0.4, 42-134 zero modes
def test_regression_D018_stats_with_zero_modes(zeros: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    ev = np.sort(rng.uniform(0.05, 2.0, size=(3, 200)), axis=1)
    ev[:, :zeros] = rng.normal(0.0, 5e-16, size=(3, zeros))  # Gram-matrix round-off
    ev.sort(axis=1)
    out = stats(ev, 0.8)
    assert np.isnan(out["inv"])  # <1/lambda> is undefined with zero modes
    assert out["var"] > 0.0 and 0.0 <= out["ks"] <= 1.0
    pos = stats(np.sort(rng.uniform(0.05, 2.0, size=(3, 200)), axis=1), 0.8)
    assert pos["inv"] > 0.0


@pytest.mark.medium
@given(spread=st.floats(1.05, 3.0), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=25, deadline=None)
@example(spread=1.2, seed=0)  # D-018: HIP starts at chi = 0.4 have Var lambda > 1
def test_regression_D018_ks_eff_undefined_for_variance_above_one(
    spread: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    ev = np.sort(np.abs(rng.normal(1.0, spread, size=(3, 300))), axis=1)
    assert float(np.mean(ev.var(axis=1))) >= 1.0 or np.isfinite(ks_eff(ev))
    if float(np.mean(ev.var(axis=1))) >= 1.0:
        assert np.isnan(ks_eff(ev))  # MP(c_eff) needs 0 < c_eff < 1
    small = np.sort(rng.uniform(0.5, 1.5, size=(3, 300)), axis=1)
    assert 0.0 <= ks_eff(small) <= 1.0


@pytest.mark.medium
@given(
    centre=st.floats(0.9, 1.1),
    width=st.floats(0.01, 0.15),
    seed=st.integers(0, 2**31 - 1),
)
@settings(max_examples=25, deadline=None)
@example(centre=1.0, width=0.09, seed=0)  # D-019: saturated RSA at chi = 0.1
def test_regression_D019_min_density_window_outside_support(
    centre: float, width: float, seed: int
) -> None:
    ev = np.random.default_rng(seed).uniform(centre - width, centre + width, (3, 400))
    assert np.isnan(min_density(ev, 0.3, 0.8))  # no support in the window
    wide = np.random.default_rng(seed).uniform(0.0, 2.0, (3, 400))
    assert min_density(wide, 0.3, 0.8) > 0.0
