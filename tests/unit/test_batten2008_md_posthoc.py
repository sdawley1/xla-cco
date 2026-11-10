#!/usr/bin/env python3

"""
Tests for the exploratory (post hoc, not pre-registered) pooled-bin analysis
of the MD winding statistics (experiments/scripts/batten2008_md_stats.py):
W_min from the per-wave scatter, the adjacent-bin merge, the pooled H1 / H2
values and their separation from the pre-registered verdicts.

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
from hypothesis.extra import numpy as hnp

# ----- project -----
from tests.tolerances import RTOL_MD_SCATTER_SD

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
import batten2008_md_analysis as ana  # noqa: E402
import batten2008_md_stats as mds  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

# Constrained waves per |k|/K bin of the pilot (flows2d, chi = 0.1, M = 210).
PILOT_COUNTS = np.array(
    [0, 2, 2, 6, 2, 6, 4, 12, 10, 6, 10, 12, 16, 8, 20, 14, 16, 20, 20, 24]
)

# ============================================================
# HELPERS
# ============================================================


def _fake_analysis(
    counts: np.ndarray, temps: List[float], n_gs: int, seed: int
) -> Dict[str, Any]:
    """Analysis dict with per-wave s_mean / ratio and the pre-registered bin means.

    ``counts[b]`` constrained waves in bin b (|k|/K at the bin centre), plus
    two unconstrained waves; per-wave values are 1 + noise (S scaled by T).
    """
    rng = np.random.default_rng(seed)
    centres = 0.5 * (ana.BIN_EDGES_C[:-1] + ana.BIN_EDGES_C[1:])
    kc = np.repeat(centres, counts)
    kmag = np.concatenate([kc, [1.4, 1.8]])
    con = np.concatenate([np.ones(kc.size, bool), np.zeros(2, bool)])
    ids = ana.bin_ids(kmag, 1.0, con)
    t = np.asarray(temps)
    shape = (n_gs, t.size, kmag.size)
    s_mean = (1.0 + 0.2 * rng.standard_normal(shape)) * t[None, :, None]
    ratio = 1.0 + 0.1 * rng.standard_normal(shape)
    z = {"system": "x", "tag": "1-10", "temps": t, "kmag": kmag, "constrained": con,
         "bin_id": ids, "s_mean": s_mean, "ratio": ratio,
         "s_over_t_bin": ana.bin_mean(s_mean / t[None, :, None], ids),
         "ratio_bin": ana.bin_mean(ratio, ids), "K": np.asarray(1.0),
         "phi_ratio": np.ones((n_gs, t.size)),
         "rate_abs_bin": np.full((n_gs, t.size, 40), np.nan)}  # fmt: skip
    assert np.array_equal(np.bincount(ids[con], minlength=20), counts), "layout"
    return z


def _rows(res: Dict[str, Any], hyp: str) -> List[Dict[str, Any]]:
    """Exploratory rows of one hypothesis."""
    out = [r for r in res["rows"] if r["hypothesis"] == hyp]
    assert out, f"no {hyp} rows"
    return out


# ============================================================
# W_MIN AND THE MERGE
# ============================================================


@pytest.mark.critical
def test_w_min_from_pilot_scatter() -> None:
    """W_min = ceil(sigma_w^2 / (n SE^2)): 15 for S/T, 4 for rate / Rice."""
    assert mds.w_min_from_scatter(0.358, 10, 0.03) == 15
    assert mds.w_min_from_scatter(0.166, 10, 0.03) == 4
    assert mds.w_min_from_scatter(0.03 * math.sqrt(10), 10, 0.03) == 1
    for stat, sigma in mds.EXPL_SIGMA_W.items():
        w = mds.EXPL_W_MIN[stat]
        assert w == mds.w_min_from_scatter(sigma, mds.EXPL_N_GS, mds.EXPL_SE_TARGET)
        assert sigma / math.sqrt(mds.EXPL_N_GS * w) <= mds.EXPL_SE_TARGET
        assert sigma / math.sqrt(mds.EXPL_N_GS * (w - 1)) > mds.EXPL_SE_TARGET
    assert mds.EXPL_W_MIN == {"S/T": 15, "Rice": 4}


@pytest.mark.critical
def test_pool_bins_pilot_and_edges() -> None:
    """Pilot layout merges from low |k| up; a short remainder joins the last group."""
    assert mds.pool_bins(PILOT_COUNTS, 15) == [
        (1, 5), (6, 7), (8, 9), (10, 11), (12, 12), (13, 14), (15, 16), (17, 17),
        (18, 18), (19, 19),
    ]  # fmt: skip
    assert mds.pool_bins(PILOT_COUNTS, 4)[:4] == [(1, 2), (3, 3), (4, 5), (6, 6)]
    assert mds.pool_bins(PILOT_COUNTS, 1) == [(b, b) for b in range(1, 20)]
    assert mds.pool_bins(np.array([5, 0, 0, 1]), 5) == [(0, 3)]  # remainder merged
    assert mds.pool_bins(np.array([0, 2, 0, 1, 0]), 5) == [(1, 3)]  # total < W_min
    assert mds.pool_bins(np.zeros(4, np.int64), 3) == []


@pytest.mark.high
@given(counts=hnp.arrays(np.int64, st.integers(1, 20), elements=st.integers(0, 30)),
       w_min=st.integers(1, 40))  # fmt: skip
@example(counts=PILOT_COUNTS, w_min=15)
def test_pool_bins_property(counts: np.ndarray, w_min: int) -> None:
    """Ordered, disjoint groups covering every occupied bin once; each >= W_min
    waves unless one group holds every wave and the total is below W_min."""
    groups = mds.pool_bins(counts, w_min)
    covered = [b for lo, hi in groups for b in range(lo, hi + 1)]
    assert covered == sorted(set(covered))  # ordered and disjoint
    occupied = set(np.flatnonzero(counts > 0).tolist())
    assert occupied <= set(covered)
    assert all(counts[lo] > 0 and counts[hi] > 0 for lo, hi in groups)
    sums = [int(counts[lo : hi + 1].sum()) for lo, hi in groups]
    if int(counts.sum()) >= w_min:
        assert all(s >= w_min for s in sums)
    else:
        assert len(groups) == (1 if counts.sum() > 0 else 0)
    if w_min == 1:
        assert groups == [(b, b) for b in sorted(occupied)]


# ============================================================
# POOLED VALUES
# ============================================================


@pytest.mark.critical
def test_w_min_1_reproduces_preregistered_bins() -> None:
    """With W_min = 1 every group is one bin: medians and z equal H1 / H2 parts."""
    z = _fake_analysis(PILOT_COUNTS, [1e-6, 1e-4, 1e-3], 10, 0)
    res = mds.exploratory_pooled([z], {"S/T": 1, "Rice": 1})
    for hyp, fn, prefix in (("H1", mds.h1_parts, "S bin "),
                            ("H2", mds.h2_parts, "Rice bin ")):  # fmt: skip
        pre = {(p["cell"], int(p["part"][len(prefix):])): p for p in fn([z], True)
               if p["part"].startswith(prefix)}  # fmt: skip
        rows = _rows(res, hyp)
        assert len(rows) == len(pre) == 19 * 2  # occupied bins x harmonic T
        for r in rows:
            p = pre[(r["cell"], r["bins"][0])]
            assert r["bins"][0] == r["bins"][1]
            assert r["median"] == pytest.approx(p["value"], rel=1e-14)
            assert r["z"] == pytest.approx(p["z"], rel=1e-12)


@pytest.mark.critical
def test_pooled_value_is_wave_weighted() -> None:
    """A merged group is the mean over its waves, not over its bins' means."""
    counts = np.zeros(20, np.int64)
    counts[[2, 3]] = (1, 3)
    z = _fake_analysis(counts, [1e-5], 4, 1)
    con = z["constrained"]
    vals = np.array([2.0, 1.0, 1.0, 1.0])  # bin 2: one wave at 2; bin 3: three at 1
    z["ratio"][..., np.flatnonzero(con)] = vals
    res = mds.exploratory_pooled([z], {"S/T": 4, "Rice": 4})
    (row,) = _rows(res, "H2")
    assert row["bins"] == [2, 3] and row["n_waves"] == 4
    assert row["median"] == pytest.approx(1.25, rel=1e-14)  # (2 + 3) / 4, not 1.5
    assert row["k_over_K"] == pytest.approx([0.10, 0.20], abs=1e-15)
    assert row["n"] == 4 and row["in_band"] is False and row["band"] == [0.95, 1.05]


@pytest.mark.high
def test_per_wave_scatter_recovers_sd() -> None:
    """sigma_w = rms over harmonic T and low-|k| waves of the sd over ground states."""
    z = _fake_analysis(PILOT_COUNTS, [1e-6, 1e-5, 1e-4, 1e-3], 400, 2)
    sig = mds.per_wave_scatter(z, 0.35)
    assert sig["S/T"] == pytest.approx(0.2, rel=RTOL_MD_SCATTER_SD)  # generator sd
    assert sig["Rice"] == pytest.approx(0.1, rel=RTOL_MD_SCATTER_SD)
    z["ratio"][:, 3] *= 100.0  # T = 1e-3 is not harmonic: ignored
    hi = np.flatnonzero(z["kmag"] > 0.35)
    z["ratio"][..., hi] *= 100.0  # waves above k_max: ignored
    assert mds.per_wave_scatter(z, 0.35)["Rice"] == pytest.approx(sig["Rice"])


@pytest.mark.critical
def test_exploratory_section_is_separate_from_verdicts() -> None:
    """The markdown verdict table is unchanged; a labelled post hoc section follows."""
    z = _fake_analysis(PILOT_COUNTS, [1e-6, 1e-4], 10, 3)
    parts = {h: mds.summarize([]) for h in mds.HYPOTHESES}
    base = {"hypotheses": {h: {"title": mds.TITLES[h], **parts[h]}
                           for h in mds.HYPOTHESES},
            "descriptive": {}, "notes": ["n"]}  # fmt: skip
    table = mds.markdown(base)
    full = dict(base, exploratory=mds.exploratory_pooled([z], mds.EXPL_W_MIN))
    text = mds.markdown(full)
    assert text.startswith(table)
    tail = text[len(table) :]
    assert "Exploratory (post hoc, not pre-registered)" in tail
    assert "PASS" not in tail and "FAIL" not in tail  # no verdict words
    res = full["exploratory"]
    assert res["label"] == mds.EXPL_LABEL and res["w_min"] == mds.EXPL_W_MIN
    merged = [r for r in res["rows"] if r["bins"][0] != r["bins"][1]]
    assert merged and all(f"{r['k_over_K'][0]:.2f}" in tail for r in merged)
