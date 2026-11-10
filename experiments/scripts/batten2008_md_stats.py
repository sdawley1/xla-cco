#!/usr/bin/env python3

"""
Pre-registered verdicts H1-H8 of the MD winding experiment (plan E2).

Reads every ``md_analysis_chi<tag>.npz`` (Andersen + NVE production) and
``md_analysis_langevin*_chi<tag>.npz`` (BAOAB, for H7) written by
batten2008_md_analysis.py and evaluates experiments/md_winding/HYPOTHESES.md
exactly as registered, with Amendments A1 and A2. The statistical unit is a ground
state; "within x %" refers to the median over ground states; z-tests use
ground-state-level standard errors (sd / sqrt(n)). Every criterion is split
into parts (one per cell, bin or segment); each part is

- PASS if it is met;
- interval and tolerance parts (A1.2): FAIL if the miss is >= 10 % relative
  to the prediction, else INCONCLUSIVE whatever its |z|;
- significance-test parts (A1.2: H4 mean net rate |z| < 3, H8 part 2
  Mann-Whitney p < 0.01): FAIL on any miss.

A hypothesis is FAIL if any part fails, else INCONCLUSIVE if any part is,
else PASS; NOT_EVALUATED when its inputs are absent. Harmonic T means
T <= 1e-4; T = 1e-3 parts are listed under "descriptive" with no verdict.
A separate ``exploratory`` section (post hoc, NOT pre-registered; never
part of a verdict) pools adjacent low-|k| bins of H1 <S>/T and H2 rate /
Rice to at least W_min waves (see exploratory_pooled).

Deviation measures: |median - 1| for the ratios (H1-H3, H6 rates, H7);
|mean net rate| / mean unsigned rate (H4); |alpha - 1| (H4);
|beta - 0.5| / 0.5 and the collapse spread itself (H6); the largest
pairwise difference of start medians relative to the pooled median (both
H5 parts). H5 uses the bias-corrected Kruskal-Wallis effect size
eta^2_H = (H - k + 1) / (n - k) (A1.1; null expectation 0), averaged over
the harmonic (chi, T) cells for H5 part 1 (A2), and H5 / H7 the
per-ground-state nu = mean over constrained waves of rate / (|k| / 2 pi)
(A1.3); H6 collapse and beta use the harmonic T only (A1.4).

Usage: python -I batten2008_md_stats.py --analysis DIR_OR_FILES...
       [--out-json experiments/md_winding/verdicts.json] [--out-md ...]

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ----- numerics -----
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_md_analysis import BIN_EDGES_C, N_BINS, collapse_metric  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
NOT_EVALUATED, DESCRIPTIVE = "NOT_EVALUATED", "DESCRIPTIVE"
HARMONIC_T_MAX = 1e-4
T_LOW, T_HIGH = 1e-6, 1e-4  # H2 rate ratio
FAIL_DEV = 0.10  # a miss by >= 10 % relative is a FAIL
Z_CRIT = 3.0
H1_PHI = (0.98, 1.02)
H1_S = (0.95, 1.05)
H2_BAND = (0.95, 1.05)
H3_TOL = 0.03
H4_ALPHA = (0.9, 1.1)
H5_ETA2 = 0.06  # A1.1: bias-corrected eta^2_H
H5_MEDDIFF = 0.03
H6_RATE = 0.10
H6_BETA = (0.4, 0.6)
H6_COLLAPSE = 0.20
H6B_BAND = (0.90, 1.10)  # A3: conditional Rice rate of the unconstrained waves
H7_TOL = 0.03
H8_P = 0.01
HYPOTHESES: Tuple[str, ...] = (
    "H1", "H2", "H3", "H4", "H5", "H6", "H6b", "H7", "H8",
)  # fmt: skip
TITLES = {
    "H1": "Equipartition at harmonic T",
    "H2": "Rice rate of the constrained waves",
    "H3": "Winding persists (Q1)",
    "H4": "Net winding is a random walk",
    "H5": "Start independence of the unsigned rate (Q2)",
    "H6": "Unconstrained waves",
    "H6b": "Conditional Rice rate of the unconstrained waves (A3)",
    "H7": "Thermostat independence",
    "H8": "chi dependence",
}

# ----- exploratory (post hoc, not pre-registered) -----
EXPL_LABEL = "exploratory (post hoc, not pre-registered)"
EXPL_SE_TARGET = 0.03  # wanted SE of a pooled-bin mean over ground states
EXPL_N_GS = 10  # ground states per cell (plan E1)
EXPL_K_MAX = 0.35  # |k|/K of the few-wave pilot bins (1-6) whose scatter sets W_min
# Per-wave sd across ground states (rms over harmonic T and the waves with
# |k|/K <= EXPL_K_MAX), measured on the pilot flows2d chi = 0.1 (job 17148922)
# by per_wave_scatter: 0.3576 for <S>/T and 0.1664 for rate / Rice.
EXPL_SIGMA_W: Dict[str, float] = {"S/T": 0.358, "Rice": 0.166}

Data = Dict[str, np.ndarray]
Part = Dict[str, Any]

# ============================================================
# STATISTICS
# ============================================================


def kruskal_eta2(groups: Sequence[np.ndarray]) -> Tuple[float, float, float]:
    """(H, p, eta^2_H) of the Kruskal-Wallis test; eta^2_H = (H - k + 1) / (n - k).

    Bias corrected (Amendment A1.1): its null expectation is 0 (it can be
    negative); NaN when n = k.
    """
    clean = [np.asarray(g, dtype=np.float64) for g in groups]
    clean = [g[np.isfinite(g)] for g in clean]
    assert len(clean) >= 2 and all(g.size >= 1 for g in clean), "need >= 2 groups"
    n = sum(g.size for g in clean)
    assert n >= 3, "need at least three observations"
    pooled = np.concatenate(clean)
    if np.all(pooled == pooled[0]):
        return 0.0, 1.0, 0.0  # all tied: no evidence of any difference
    h, p = stats.kruskal(*clean)
    k = len(clean)
    eta2 = (float(h) - k + 1.0) / (n - k) if n > k else float("nan")
    return float(h), float(p), eta2


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta = (#(a > b) - #(a < b)) / (n_a n_b) over all pairs."""
    assert np.ndim(a) <= 1 and np.ndim(b) <= 1, "one-dimensional samples"
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    assert a.size >= 1 and b.size >= 1, "need two non-empty samples"
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=np.float64)
    assert p.ndim == 1 and np.all((p >= 0) & (p <= 1)), "p-values in [0, 1]"
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adj = np.maximum.accumulate(adj)
    out = np.empty(m)
    out[order] = adj
    assert np.all(out >= p - 1e-15), "adjusted p >= raw p"
    return out


def mann_whitney(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(two-sided p, normal-approximation z of U) of the Mann-Whitney test."""
    assert np.ndim(a) <= 1 and np.ndim(b) <= 1, "one-dimensional samples"
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    assert a.size >= 1 and b.size >= 1, "need two non-empty samples"
    if np.all(np.concatenate([a, b]) == a[0]):
        return 1.0, 0.0
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    na, nb = a.size, b.size
    z = (float(res.statistic) - na * nb / 2.0) / math.sqrt(
        na * nb * (na + nb + 1) / 12.0
    )
    return float(res.pvalue), z


def location(x: np.ndarray, target: float) -> Dict[str, float]:
    """Median, mean, ground-state SE and z = (mean - target) / SE of finite x."""
    assert np.isfinite(target), "finite target"
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        nan = float("nan")
        return {"median": nan, "mean": nan, "se": nan, "z": nan, "n": 0}
    mean = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else float("nan")
    if se > 0.0:
        z = (mean - target) / se
    else:
        z = 0.0 if mean == target else math.copysign(math.inf, mean - target)
    out = {"median": float(np.median(v)), "mean": mean, "se": se, "z": z, "n": v.size}
    assert out["n"] >= 1, "non-empty sample"
    return out


# ============================================================
# VERDICT LOGIC
# ============================================================


def classify(met: bool, dev: float) -> str:
    """PASS if met; else FAIL if dev >= 10 %; else (or dev undefined) INCONCLUSIVE."""
    assert np.isnan(dev) or dev >= 0.0, "deviations are non-negative"
    assert isinstance(met, (bool, np.bool_)), "met must be a boolean"
    if met:
        return PASS
    if not np.isfinite(dev):
        return INCONCLUSIVE  # no estimate (e.g. too few MSD points): undecided
    return FAIL if dev >= FAIL_DEV else INCONCLUSIVE


def make_part(
    name: str, cell: str, value: float, z: float, dev: float, met: bool,
    outcome: Optional[str] = None, **extra: Any,
) -> Part:  # fmt: skip
    """One verdict part; ``outcome`` overrides classify(met, dev)."""
    assert name and cell, "parts need a name and a cell"
    out = {"part": name, "cell": cell, "value": float(value), "z": float(z),
           "dev": float(dev), "met": bool(met),
           "outcome": outcome or classify(bool(met), float(dev))}  # fmt: skip
    out.update(extra)
    assert out["outcome"] in (PASS, FAIL, INCONCLUSIVE), "known outcome"
    return out


def combine(outcomes: Sequence[str]) -> str:
    """FAIL if any part fails, else INCONCLUSIVE if any is, else PASS."""
    assert isinstance(outcomes, (list, tuple)), "a sequence of outcomes"
    assert all(o in (PASS, FAIL, INCONCLUSIVE) for o in outcomes), "known outcomes"
    if not outcomes:
        return NOT_EVALUATED
    if FAIL in outcomes:
        return FAIL
    return INCONCLUSIVE if INCONCLUSIVE in outcomes else PASS


def band_part(
    name: str, cell: str, x: np.ndarray, band: Tuple[float, float], target: float
) -> Part:
    """Median over ground states inside ``band``; dev = |median - target| / target."""
    assert np.ndim(x) == 1, "one value per ground state"
    assert band[0] < band[1] and target != 0.0, "need a band and a target"
    loc = location(x, target)
    med = loc["median"]
    met = bool(band[0] <= med <= band[1])
    dev = abs(med - target) / abs(target)
    return make_part(name, cell, med, loc["z"], dev, met, n=loc["n"])


# ============================================================
# HYPOTHESES
# ============================================================


def temp_indices(temps: np.ndarray, harmonic: bool) -> List[int]:
    """Indices of the harmonic temperatures (T <= 1e-4), or of the others."""
    assert temps.ndim == 1 and np.all(temps > 0), "positive temperatures"
    is_h = temps <= HARMONIC_T_MAX * (1.0 + 1e-9)
    out = [i for i in range(temps.size) if bool(is_h[i]) == harmonic]
    assert all(0 <= i < temps.size for i in out), "indices in range"
    return out


def label(z: Data, t: Optional[float] = None) -> str:
    """Cell label 'system chi=<tag> [T=<T>]'."""
    assert "system" in z and "tag" in z, "analysis data"
    out = f"{z['system']} chi={z['tag']}"
    assert out.strip(), "empty label"
    return out if t is None else f"{out} T={t:g}"


def find_t(temps: np.ndarray, value: float) -> Optional[int]:
    """Index of temperature ``value`` (relative 1e-9), or None."""
    assert np.isfinite(value) and value > 0, "positive temperature"
    hit = np.flatnonzero(np.isclose(temps, value, rtol=1e-9, atol=0.0))
    assert hit.size <= 1, "temperatures are distinct"
    return int(hit[0]) if hit.size else None


def h1_parts(main: List[Data], harmonic: bool) -> List[Part]:
    """<Phi>/(M T) in [0.98, 1.02]; <S_k>/T in [0.95, 1.05] per constrained bin."""
    assert all("s_over_t_bin" in z for z in main), "analysis data"
    out: List[Part] = []
    for z in main:
        for ti in temp_indices(z["temps"], harmonic):
            cell = label(z, z["temps"][ti])
            out.append(band_part("phi", cell, z["phi_ratio"][:, ti], H1_PHI, 1.0))
            for b in range(N_BINS):
                x = z["s_over_t_bin"][:, ti, b]
                if np.any(np.isfinite(x)):
                    out.append(band_part(f"S bin {b}", cell, x, H1_S, 1.0))
    assert all(p["part"] for p in out), "named parts"
    return out


def h2_parts(main: List[Data], harmonic: bool) -> List[Part]:
    """Rate / Rice in [0.95, 1.05] per bin; rate(1e-6) / rate(1e-4) per bin."""
    assert all("ratio_bin" in z for z in main), "analysis data"
    out: List[Part] = []
    for z in main:
        for ti in temp_indices(z["temps"], harmonic):
            cell = label(z, z["temps"][ti])
            for b in range(N_BINS):
                x = z["ratio_bin"][:, ti, b]
                if np.any(np.isfinite(x)):
                    out.append(band_part(f"Rice bin {b}", cell, x, H2_BAND, 1.0))
        lo, hi = find_t(z["temps"], T_LOW), find_t(z["temps"], T_HIGH)
        if harmonic and lo is not None and hi is not None:
            for b in range(N_BINS):
                x = z["rate_abs_bin"][:, lo, b] / z["rate_abs_bin"][:, hi, b]
                if np.any(np.isfinite(x)):
                    out.append(band_part(f"T ratio bin {b}", label(z), x, H2_BAND, 1.0))
    assert all(p["part"] for p in out), "named parts"
    return out


def h3_parts(main: List[Data], harmonic: bool) -> List[Part]:
    """Windowed rates [t/2, t] vs [t/4, t/2] per segment: within 3 %, |z| < 3."""
    assert all("w1_c" in z for z in main), "analysis data"
    out: List[Part] = []
    for z in main:
        for ti in temp_indices(z["temps"], harmonic):
            for c in range(z["w1_c"].shape[2]):
                x = z["w2_c"][:, ti, c] / z["w1_c"][:, ti, c]
                loc = location(x, 1.0)
                dev = abs(loc["median"] - 1.0)
                met = bool(dev < H3_TOL and abs(loc["z"]) < Z_CRIT)
                out.append(
                    make_part(
                        f"segment {c}",
                        label(z, z["temps"][ti]),
                        loc["median"],
                        loc["z"],
                        dev,
                        met,
                        n=loc["n"],
                    )
                )
    assert all(p["part"] for p in out), "named parts"
    return out


def _significance(z: float) -> str:
    """A1.2 significance-test part |z| < 3: PASS, else FAIL (NaN: INCONCLUSIVE)."""
    assert isinstance(z, float), "z must be a float"
    if not np.isfinite(z) and not np.isinf(z):
        return INCONCLUSIVE  # no estimate (fewer than two ground states)
    out = PASS if abs(z) < Z_CRIT else FAIL
    assert out in (PASS, FAIL), "no INCONCLUSIVE for a decided test"
    return out


def h4_parts(main: List[Data], harmonic: bool) -> List[Part]:
    """Mean net rate consistent with 0 (|z| < 3); MSD exponent in [0.9, 1.1]."""
    assert all("alpha" in z for z in main), "analysis data"
    out: List[Part] = []
    for z in main:
        for ti in temp_indices(z["temps"], harmonic):
            cell = label(z, z["temps"][ti])
            loc = location(z["net_mean_c"][:, ti], 0.0)
            scale = float(np.nanmean(z["abs_mean_c"][:, ti]))
            dev = abs(loc["mean"]) / scale
            out.append(
                make_part(
                    "net rate",
                    cell,
                    loc["mean"],
                    loc["z"],
                    dev,
                    bool(abs(loc["z"]) < Z_CRIT),
                    outcome=_significance(loc["z"]),
                    n=loc["n"],
                )
            )
            out.append(band_part("alpha", cell, z["alpha"][:, ti], H4_ALPHA, 1.0))
    assert all(p["part"] for p in out), "named parts"
    return out


def _by_chi_t(main: List[Data], harmonic: bool) -> Dict[Tuple[str, float], List]:
    """{(tag, T): [(system, data, t index)]} over the selected temperatures."""
    assert all("tag" in z for z in main), "analysis data"
    out: Dict[Tuple[str, float], List] = {}
    for z in main:
        for ti in temp_indices(z["temps"], harmonic):
            key = (str(z["tag"]), float(z["temps"][ti]))
            out.setdefault(key, []).append((str(z["system"]), z, ti))
    assert all(len(v) >= 1 for v in out.values()), "non-empty cells"
    return out


def h5_cell(groups: Dict[str, np.ndarray], cell: str) -> List[Part]:
    """Kruskal-Wallis eta^2_H < 0.06 and largest pairwise median difference < 3 %."""
    assert cell, "cell label"
    assert len(groups) >= 2, "need at least two starts"
    names = sorted(groups)
    h, p, eta2 = kruskal_eta2([groups[s] for s in names])
    meds = {s: float(np.nanmedian(groups[s])) for s in names}
    grand = float(np.nanmedian(np.concatenate([groups[s] for s in names])))
    hi, lo = max(names, key=meds.__getitem__), min(names, key=meds.__getitem__)
    maxdiff = (meds[hi] - meds[lo]) / grand
    _, z_pair = mann_whitney(groups[hi], groups[lo])
    z_kw = float(stats.norm.isf(max(p, 1e-300) / 2.0))
    return [
        make_part("KW eta2_H", cell, eta2, z_kw, maxdiff, bool(eta2 < H5_ETA2),
                  H=h, p=p, medians=meds),
        make_part("max median diff", cell, maxdiff, z_pair, maxdiff,
                  bool(maxdiff < H5_MEDDIFF), pair=[hi, lo]),
    ]  # fmt: skip


def d_theta_report(groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """D_theta by start: medians, Holm-corrected Mann-Whitney p and Cliff's delta."""
    names = sorted(groups)
    assert len(names) >= 2, "need at least two starts"
    pairs = list(itertools.combinations(names, 2))
    raw = [mann_whitney(groups[a], groups[b])[0] for a, b in pairs]
    adj = holm(raw)
    rows = [{"a": a, "b": b, "p": raw[i], "p_holm": float(adj[i]),
             "cliffs_delta": cliffs_delta(groups[a], groups[b])}
            for i, (a, b) in enumerate(pairs)]  # fmt: skip
    out = {"median": {s: float(np.nanmedian(groups[s])) for s in names},
           "pairs": rows}  # fmt: skip
    assert len(rows) == len(names) * (len(names) - 1) // 2, "all pairs"
    return out


def spearman_report(entries: List, key: str) -> Dict[str, Any]:
    """Spearman rho of ``key`` (per ground state) against the K-spectrum moments."""
    assert entries, "need data"
    if not all("k_moments" in z for _, z, _ in entries):
        return {"note": "K spectra absent (run the analysis with --flows)"}
    y = np.concatenate([z[key][:, ti] for _, z, ti in entries])
    mom = np.concatenate([z["k_moments"] for _, z, _ in entries])
    names = [str(n) for n in entries[0][1]["moment_names"]]
    out: Dict[str, Any] = {}
    for j, name in enumerate(names):
        ok = np.isfinite(y) & np.isfinite(mom[:, j])
        if ok.sum() >= 3 and np.ptp(mom[ok, j]) > 0 and np.ptp(y[ok]) > 0:
            rho, p = stats.spearmanr(y[ok], mom[ok, j])
            out[name] = {"rho": float(rho), "p": float(p), "n": int(ok.sum())}
    assert isinstance(out, dict), "report is a dict"
    return out


def h5_mean_eta2(cells: List[Part]) -> Part:
    """A2 H5 part 1: the mean of eta^2_H over the (chi, T) cells is < 0.06.

    ``cells`` are the per-cell "KW eta2_H" parts of h5_cell (descriptive
    under A2). Deviation (for a miss): the largest pairwise median
    difference of any cell (A1.2); z = mean / (sd / sqrt(cells)).
    """
    assert cells and all(p["part"] == "KW eta2_H" for p in cells), "eta^2 cells"
    eta = np.array([p["value"] for p in cells])
    mean = float(np.mean(eta))
    se = float(np.std(eta, ddof=1) / math.sqrt(eta.size)) if eta.size > 1 else 0.0
    z = mean / se if se > 0.0 else float("nan")
    dev = max(p["dev"] for p in cells)
    assert np.all(np.isfinite(eta)), "finite eta^2_H per cell"
    return make_part("mean KW eta2_H", f"{eta.size} cells", mean, z, dev,
                     bool(mean < H5_ETA2),
                     cells={p["cell"]: p["value"] for p in cells})  # fmt: skip


def h5_parts(main: List[Data], harmonic: bool) -> Tuple[List[Part], Dict[str, Any]]:
    """H5 parts (A2: mean eta^2_H over cells; median difference per cell) + report.

    The per-cell eta^2_H parts are descriptive (report["eta2_H_per_cell"]);
    the D_theta analysis is reported, not pass/fail.
    """
    assert all("nu_c" in z for z in main), "analysis data"
    parts: List[Part] = []
    eta_cells: List[Part] = []
    report: Dict[str, Any] = {}
    for (tag, t), entries in sorted(_by_chi_t(main, harmonic).items()):
        if len({s for s, _, _ in entries}) < 2:
            continue
        nu = {s: z["nu_c"][:, ti] for s, z, ti in entries}
        cell = f"chi={tag} T={t:g}"
        eta_part, diff_part = h5_cell(nu, cell)
        eta_cells.append(eta_part)
        parts.append(diff_part)
        dth = {s: z["d_theta"][:, ti] for s, z, ti in entries}
        report[cell] = {"d_theta": d_theta_report(dth),
                        "spearman_d_theta": spearman_report(entries, "d_theta"),
                        "spearman_nu": spearman_report(entries, "nu_c")}  # fmt: skip
    assert len(parts) == len(eta_cells), "one median-difference part per cell"
    if eta_cells:
        parts.insert(0, h5_mean_eta2(eta_cells))
        report["eta2_H_per_cell"] = eta_cells
    return parts, report


def _slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares slope of every row of y (G, n) against x (n,)."""
    assert y.shape[-1] == x.size >= 2, "need two points per row"
    xc = x - x.mean()
    out: np.ndarray = ((y - y.mean(axis=-1, keepdims=True)) @ xc) / float(xc @ xc)
    assert out.shape == y.shape[:-1], "one slope per row"
    return out


def h6_parts(main: List[Data], harmonic: bool) -> List[Part]:
    """Unconstrained Rice rate within 10 % per bin; beta in [0.4, 0.6]; collapse."""
    assert all("net_rms_u" in z for z in main), "analysis data"
    out: List[Part] = []
    for z in main:
        tis = temp_indices(z["temps"], harmonic)
        for ti in tis:
            for b in range(N_BINS, 2 * N_BINS):
                x = z["ratio_bin"][:, ti, b]
                if np.any(np.isfinite(x)):
                    out.append(
                        band_part(
                            f"unc Rice bin {b}",
                            label(z, z["temps"][ti]),
                            x,
                            (1 - H6_RATE, 1 + H6_RATE),
                            1.0,
                        )
                    )
        tis = tis if harmonic else list(range(z["temps"].size))  # descriptive: all T
        if len(tis) < 2:
            continue
        temps = z["temps"][tis]
        beta = _slopes(np.log(temps), np.log(z["net_rms_u"][:, tis]))
        out.append(band_part("beta", label(z), beta, H6_BETA, 0.5))
        msd = np.nanmedian(z["part_msd"][:, tis, :], axis=0)
        spread = collapse_metric(z["part_lag"], msd, temps)
        out.append(
            make_part(
                "collapse t sqrt(T)",
                label(z),
                spread,
                float("nan"),
                spread,
                bool(spread <= H6_COLLAPSE),
            )
        )
    assert all(p["part"] for p in out), "named parts"
    return out


def _paired(a: Data, ta: int, b: Data, tb: int, key: str) -> np.ndarray:
    """a[key] / b[key] over the ground states (indices) both contain."""
    common = np.intersect1d(a["indices"], b["indices"])
    ia = np.searchsorted(a["indices"], common)
    ib = np.searchsorted(b["indices"], common)
    assert np.array_equal(a["indices"][ia], b["indices"][ib]), "sorted indices"
    assert common.size >= 1, "no common ground states"
    return np.asarray(a[key][ia, ta] / b[key][ib, tb])


def _ratio_part(name: str, cell: str, x: np.ndarray, tol: float) -> Part:
    """Median ratio within ``tol`` of 1 and |z| < 3."""
    assert name and cell, "parts need labels"
    loc = location(x, 1.0)
    dev = abs(loc["median"] - 1.0)
    assert tol > 0.0, "tolerance must be positive"
    return make_part(
        name,
        cell,
        loc["median"],
        loc["z"],
        dev,
        bool(dev < tol and abs(loc["z"]) < Z_CRIT),
        n=loc["n"],
    )


def h6b_parts(main: List[Data], harmonic: bool) -> List[Part]:
    """A3 H6b: rate / conditional Rice in [0.90, 1.10] per unconstrained bin and T."""
    assert all("ratio_cond_bin" in z for z in main), "analysis data (A3)"
    out: List[Part] = []
    for z in main:
        for ti in temp_indices(z["temps"], harmonic):
            cell = label(z, z["temps"][ti])
            for b in range(N_BINS, 2 * N_BINS):
                x = z["ratio_cond_bin"][:, ti, b]
                if np.any(np.isfinite(x)):
                    out.append(band_part(f"cond Rice bin {b}", cell, x, H6B_BAND, 1.0))
    assert all(p["part"].startswith("cond Rice") for p in out), "named parts"
    return out


def h7_parts(main: List[Data], lang: List[Data]) -> List[Part]:
    """BAOAB vs Andersen+NVE normalized rate, and gamma vs gamma: within 3 %."""
    assert all("gamma" in z for z in lang), "Langevin analyses carry gamma"
    by_cell = {(str(z["system"]), str(z["tag"])): z for z in main}
    out: List[Part] = []
    groups: Dict[Tuple[str, str], List[Data]] = {}
    for z in lang:
        key = (str(z["system"]), str(z["tag"]))
        groups.setdefault(key, []).append(z)
        m = by_cell.get(key)
        for ti in [] if m is None else temp_indices(z["temps"], True):
            tm = find_t(m["temps"], float(z["temps"][ti]))
            if tm is not None:
                cell = f"{label(z, z['temps'][ti])} gamma={float(z['gamma']):g}"
                out.append(
                    _ratio_part(
                        "BAOAB / NVE", cell, _paired(z, ti, m, tm, "nu_c"), H7_TOL
                    )
                )
    for zs in groups.values():
        for a, b in itertools.combinations(
            sorted(zs, key=lambda q: float(q["gamma"])), 2
        ):
            for ti in temp_indices(a["temps"], True):
                tb = find_t(b["temps"], float(a["temps"][ti]))
                if tb is not None:
                    cell = (
                        f"{label(a, a['temps'][ti])} gamma {float(a['gamma']):g}"
                        f" / {float(b['gamma']):g}"
                    )
                    out.append(
                        _ratio_part(
                            "gamma", cell, _paired(a, ti, b, tb, "nu_c"), H7_TOL
                        )
                    )
    assert all(p["part"] for p in out), "named parts"
    return out


def h8_parts(main: List[Data], h2: List[Part]) -> List[Part]:
    """H2 passes at each chi; D_theta differs between chi (Mann-Whitney p < 0.01)."""
    assert all(p["part"] for p in h2), "H2 parts"
    out: List[Part] = []
    for tag in sorted({str(z["tag"]) for z in main}):
        verdict = combine(
            [p["outcome"] for p in h2 if f"chi={tag}" in p["cell"].split()]
        )
        if verdict != NOT_EVALUATED:
            out.append(
                make_part(
                    "H2 at chi",
                    f"chi={tag}",
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    verdict == PASS,
                    outcome=verdict,
                )
            )
    systems: Dict[str, List[Data]] = {}
    for z in main:
        systems.setdefault(str(z["system"]), []).append(z)
    for zs in systems.values():
        for a, b in itertools.combinations(
            sorted(zs, key=lambda q: float(q["chi"])), 2
        ):
            for ti in temp_indices(a["temps"], True):
                tb = find_t(b["temps"], float(a["temps"][ti]))
                if tb is None:
                    continue
                p, zval = mann_whitney(a["d_theta"][:, ti], b["d_theta"][:, tb])
                ratio = float(np.nanmedian(b["d_theta"][:, tb])
                              / np.nanmedian(a["d_theta"][:, ti]))  # fmt: skip
                met = bool(p < H8_P)
                outcome = PASS if met else FAIL  # A1.2: significance test
                cell = (
                    f"{a['system']} chi {a['tag']} vs {b['tag']} T={a['temps'][ti]:g}"
                )
                out.append(
                    make_part(
                        "D_theta differs",
                        cell,
                        ratio,
                        zval,
                        abs(ratio - 1.0),
                        met,
                        outcome=outcome,
                        p=p,
                    )
                )
    assert all(p["part"] for p in out), "named parts"
    return out


# ============================================================
# EXPLORATORY: POOLED LOW-|k| BINS (POST HOC, NOT PRE-REGISTERED)
# ============================================================


def w_min_from_scatter(sigma_w: float, n_gs: int, se_target: float) -> int:
    """Fewest waves W with sigma_w / sqrt(n_gs W) <= se_target (independent waves).

    The noise argument behind the exploratory pooling: a ground state's
    bin value is the mean over the W waves of the bin; if the per-wave
    values scatter with sd sigma_w across ground states and are independent
    within a ground state, the SE of the bin mean over n_gs ground states is
    sigma_w / sqrt(n_gs W). With the pilot's sigma_w (EXPL_SIGMA_W), n_gs =
    10 and SE <= 3 %: W_min = ceil(0.358^2 / (10 x 0.03^2)) = ceil(14.2) =
    15 for <S>/T and ceil(0.166^2 / 0.009) = ceil(3.06) = 4 for rate / Rice.
    Waves in one bin are positively correlated in the pilot (they share the
    soft modes), so the realised SE is larger; exploratory_pooled reports
    the measured SE, never this model value.
    """
    assert sigma_w >= 0.0 and n_gs >= 1, "non-negative scatter, ground states"
    assert se_target > 0.0, "positive SE target"
    out = max(1, math.ceil(sigma_w**2 / (n_gs * se_target**2) - 1e-12))
    assert sigma_w / math.sqrt(n_gs * out) <= se_target * (1.0 + 1e-9), "target met"
    return int(out)


EXPL_W_MIN: Dict[str, int] = {
    k: w_min_from_scatter(v, EXPL_N_GS, EXPL_SE_TARGET) for k, v in EXPL_SIGMA_W.items()
}


def pool_bins(counts: np.ndarray, w_min: int) -> List[Tuple[int, int]]:
    """Merge adjacent occupied bins from low |k| upward until each has >= w_min waves.

    Returns inclusive (first, last) occupied-bin ranges in order. A group
    closes once its wave count reaches w_min; a final group left below
    w_min joins the previous one (or is the only group if all waves together
    are fewer than w_min). Empty bins never start or end a group.
    """
    c = np.asarray(counts, dtype=np.int64)
    assert c.ndim == 1 and np.all(c >= 0), "non-negative counts per bin"
    assert w_min >= 1, "w_min >= 1"
    groups: List[Tuple[int, int]] = []
    lo, acc = -1, 0
    for b in np.flatnonzero(c > 0).tolist():
        lo = b if acc == 0 else lo
        acc += int(c[b])
        if acc >= w_min:
            groups.append((lo, b))
            acc = 0
    if acc > 0:  # remainder below w_min
        last = int(np.flatnonzero(c > 0)[-1])
        groups[-1:] = [(groups[-1][0] if groups else lo, last)]
    assert all(a <= b for a, b in groups), "ordered ranges"
    return groups


def per_wave_scatter(z: Data, k_max: float) -> Dict[str, float]:
    """sigma_w of <S>/T and rate / Rice: sqrt of the mean over harmonic T and the
    constrained waves with |k|/K <= k_max of the variance across ground states."""
    assert 0.0 < k_max <= 1.0, "k_max in (0, 1]"
    sel = z["constrained"].astype(bool) & (z["kmag"] / float(z["K"]) <= k_max)
    harm = temp_indices(z["temps"], True)
    assert sel.any() and harm, "waves and harmonic temperatures"
    temps = z["temps"][None, :, None]
    out = {}
    for stat, vals in (("S/T", z["s_mean"] / temps), ("Rice", z["ratio"])):
        v = vals[:, harm][..., sel]
        out[stat] = float(np.sqrt(np.nanmean(np.nanvar(v, axis=0, ddof=1))))
    return out


def _pooled_row(
    z: Data, ti: int, stat: str, values: np.ndarray, group: Tuple[int, int]
) -> Dict[str, Any]:
    """One pooled bin: per-ground-state mean over its waves, located against 1."""
    lo, hi = group
    ids, con = z["bin_id"], z["constrained"].astype(bool)
    cols = con & (ids >= lo) & (ids <= hi)
    assert cols.any() and 0 <= lo <= hi < N_BINS, "an occupied constrained group"
    loc = location(np.mean(values[:, ti][:, cols], axis=-1), 1.0)
    band = H1_S if stat == "S/T" else H2_BAND
    return {
        "hypothesis": "H1" if stat == "S/T" else "H2", "statistic": stat,
        "cell": label(z, float(z["temps"][ti])), "bins": [lo, hi],
        "k_over_K": [float(BIN_EDGES_C[lo]), float(BIN_EDGES_C[hi + 1])],
        "n_waves": int(cols.sum()), "median": loc["median"], "mean": loc["mean"],
        "se": loc["se"], "z": loc["z"], "n": loc["n"], "band": list(band),
        "in_band": bool(band[0] <= loc["median"] <= band[1]),
    }  # fmt: skip


def exploratory_pooled(main: List[Data], w_min: Dict[str, int]) -> Dict[str, Any]:
    """Post hoc, NOT pre-registered: H1 <S>/T and H2 rate / Rice on pooled bins.

    Per cell and harmonic T, adjacent constrained |k|/K bins are merged from
    low |k| upward until each holds >= w_min[stat] waves (pool_bins; W_min
    from w_min_from_scatter). Each pooled bin reports the median, mean,
    ground-state SE and z = (mean - 1) / SE of the per-ground-state mean
    over its waves, and whether the median lies in the pre-registered band
    (H1 [0.95, 1.05], H2 [0.95, 1.05]). Never enters a verdict.
    """
    assert set(w_min) == {"S/T", "Rice"}, "one W_min per statistic"
    assert all(w >= 1 for w in w_min.values()), "W_min >= 1"
    rows: List[Dict[str, Any]] = []
    for z in main:
        con = z["constrained"].astype(bool)
        counts = np.bincount(z["bin_id"][con], minlength=N_BINS)[:N_BINS]
        temps = z["temps"][None, :, None]
        for stat, vals in (("S/T", z["s_mean"] / temps), ("Rice", z["ratio"])):
            for ti in temp_indices(z["temps"], True):
                rows += [_pooled_row(z, ti, stat, vals, g)
                         for g in pool_bins(counts, w_min[stat])]  # fmt: skip
    return {
        "label": EXPL_LABEL, "w_min": dict(w_min), "se_target": EXPL_SE_TARGET,
        "sigma_w": dict(EXPL_SIGMA_W), "k_max": EXPL_K_MAX,
        "derivation": " ".join(str(w_min_from_scatter.__doc__).split()),
        "rows": rows,
        "in_band": {h: [sum(r["in_band"] for r in rows if r["hypothesis"] == h),
                        sum(r["hypothesis"] == h for r in rows)]
                    for h in ("H1", "H2")},
    }  # fmt: skip


def exploratory_markdown(expl: Dict[str, Any]) -> List[str]:
    """Markdown lines of the post hoc section: the merged (multi-bin) groups."""
    assert expl["label"] == EXPL_LABEL, "an exploratory_pooled result"
    rows = [
        "", "## Exploratory (post hoc, not pre-registered): pooled low-|k| bins", "",
        f"Low-|k| bins pooled to >= W_min waves (S/T: {expl['w_min']['S/T']}, "
        f"rate / Rice: {expl['w_min']['Rice']}); W_min = ceil(sigma_w^2 / (n "
        f"SE^2)) with the pilot per-wave sd sigma_w = {expl['sigma_w']} (|k|/K "
        f"<= {expl['k_max']}), n = {EXPL_N_GS}, SE = {expl['se_target']}. Not a "
        "verdict; the pre-registered table above is unchanged. Medians within "
        "the band (all pooled bins): "
        + ", ".join(f"{h} {a} / {b}" for h, (a, b) in expl["in_band"].items()),
        "", "Merged groups only (single bins equal the pre-registered parts):", "",
        "| Hyp | Cell | Bins (|k|/K) | Waves | Median | Mean ± SE | z | In band |",
        "|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for r in expl["rows"]:
        if r["bins"][0] == r["bins"][1]:
            continue
        rows.append(
            f"| {r['hypothesis']} {r['statistic']} | {r['cell']} | "
            f"{r['bins'][0]}-{r['bins'][1]} ({r['k_over_K'][0]:.2f}, "
            f"{r['k_over_K'][1]:.2f}] | {r['n_waves']} | {r['median']:.4f} | "
            f"{r['mean']:.4f} ± {r['se']:.4f} | {r['z']:.2f} | "
            f"{'yes' if r['in_band'] else 'no'} |"
        )
    assert len(rows) >= 8, "header lines"
    return rows


# ============================================================
# EVALUATION AND OUTPUT
# ============================================================

SEVERITY = {FAIL: 0, INCONCLUSIVE: 1, PASS: 2}


def summarize(parts: List[Part], n_worst: int = 5) -> Dict[str, Any]:
    """Verdict, outcome counts, the worst parts (by outcome, then dev), all parts."""
    assert n_worst >= 1, "show at least one part"
    verdict = combine([p["outcome"] for p in parts])
    worst = sorted(
        parts,
        key=lambda p: (
            SEVERITY[p["outcome"]],
            -(p["dev"] if np.isfinite(p["dev"]) else 0),
        ),
    )
    counts = {
        o: sum(p["outcome"] == o for p in parts) for o in (PASS, INCONCLUSIVE, FAIL)
    }
    assert sum(counts.values()) == len(parts), "every part counted"
    return {
        "verdict": verdict,
        "counts": counts,
        "worst": worst[:n_worst],
        "parts": parts,
    }


def evaluate(main: List[Data], lang: List[Data]) -> Dict[str, Any]:
    """All verdicts, the descriptive T = 1e-3 parts, the H5 report and notes."""
    assert all("nu_c" in z for z in main + lang), "analysis files expected"
    parts: Dict[str, List[Part]] = {
        "H1": h1_parts(main, True), "H2": h2_parts(main, True),
        "H3": h3_parts(main, True), "H4": h4_parts(main, True),
    }  # fmt: skip
    parts["H5"], report = h5_parts(main, True)
    parts["H6"] = h6_parts(main, True)
    parts["H6b"] = h6b_parts(main, True)
    parts["H7"] = h7_parts(main, lang)
    parts["H8"] = h8_parts(main, parts["H2"])
    desc_fns: Dict[str, Callable[..., Any]] = {
        "H1": h1_parts, "H2": h2_parts, "H3": h3_parts, "H4": h4_parts, "H6": h6_parts,
        "H6b": h6b_parts,
    }  # fmt: skip
    descriptive = {h: fn(main, False) for h, fn in desc_fns.items()}
    descriptive["H5"], desc_report = h5_parts(main, False)
    n_groups = sorted({len(v) for v in _by_chi_t(main, True).values()}) if main else []
    n_gs = sorted({int(z["nu_c"].shape[0]) for z in main})
    notes = [
        "H5 uses eta^2_H = (H-k+1)/(n-k) (Amendment A1.1; null mean 0; starts "
        f"per cell {n_groups}, ground states per start {n_gs}); with k = 5, "
        "n = 50 a start-independent cell misses eta^2_H < 0.06 with probability "
        "P(chi2_4 > 6.7) = 0.15, so (Amendment A2) H5 part 1 is the mean of "
        "eta^2_H over the harmonic (chi, T) cells; per-cell values are in "
        "reports.H5_harmonic.eta2_H_per_cell.",
        "A1.2: interval / tolerance parts missing by < 10 % are INCONCLUSIVE "
        "whatever |z|; significance-test parts (H4 net rate, H8 part 2) FAIL "
        "on any miss.",
        "Normalized unsigned rate nu = mean over constrained waves of "
        "rate / (|k| / 2 pi); H5, H7 use it per ground state.",
        "Interpretation (H4): t >> tau_corr is taken as t >= 10 x the median over "
        "constrained waves of 1 / (unsigned rate); alpha is the log-log slope there.",
        "Interpretation (H6): the net-winding rate of the unconstrained waves is "
        "the RMS over waves and segments of the end-point rate W_net(t_end) / "
        "t_end; beta is its log-log slope against the harmonic T per ground state.",
        "Interpretation (H8 part 2): Mann-Whitney per (start, harmonic T) cell, "
        "each at p < 0.01, with no multiplicity correction.",
        "H6b (A3, A4): <1/|rho_k|>_t is the mean over the uniform dense rho trace "
        "inside the production segments (>= 500 samples per segment); lanes "
        "without it are NOT_EVALUATED (never filled from the snapshots).",
    ]
    out = {"hypotheses": {h: {"title": TITLES[h], **summarize(parts[h])}
                          for h in HYPOTHESES},
           "descriptive": {h: summarize(v) for h, v in descriptive.items()},
           "reports": {"H5_harmonic": report, "H5_descriptive": desc_report},
           "notes": notes,
           "exploratory": exploratory_pooled(main, EXPL_W_MIN)}  # fmt: skip
    assert set(out["hypotheses"]) == set(HYPOTHESES), "every hypothesis"
    return out


def load_analysis(path: Path) -> Data:
    """An analysis npz as a dict (0-d strings as str)."""
    with np.load(path, allow_pickle=False) as z:
        out: Data = {k: (str(z[k]) if z[k].dtype.kind == "U" and z[k].ndim == 0
                         else np.asarray(z[k])) for k in z.files}  # fmt: skip
    assert "nu_c" in out and "temps" in out, f"{path} is not an analysis file"
    assert out["nu_c"].shape == (out["indices"].size, out["temps"].size), "grids"
    return out


def collect(paths: List[Path]) -> Tuple[List[Data], List[Data], List[str]]:
    """(main, langevin) analyses from files and directories (searched recursively)."""
    assert paths, "need input paths"
    files: List[Path] = []
    for p in paths:
        files += sorted(p.rglob("md_analysis*_chi*.npz")) if p.is_dir() else [p]
    main, lang, used = [], [], []
    for f in files:
        if f.name.startswith("md_analysis_chi"):
            main.append(load_analysis(f))
        elif f.name.startswith("md_analysis_langevin"):
            lang.append(load_analysis(f))
        else:
            continue
        used.append(str(f))
    assert len(used) == len(main) + len(lang), "every used file loaded"
    return main, lang, used


def _json_default(obj: Any) -> Any:
    """NumPy scalars and arrays -> Python values for json.dumps."""
    assert obj is not None, "json.dumps never passes None here"
    assert not isinstance(obj, (str, int, float, list, dict)), "native JSON types"
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def clean_json(obj: Any) -> Any:
    """JSON-safe copy: NumPy values -> Python, NaN and +-inf -> None (no recursion)."""
    text = json.dumps(obj, default=_json_default, allow_nan=True)
    out = json.loads(text, parse_constant=lambda _: None)
    assert "NaN" not in json.dumps(out, allow_nan=False), "non-finite values remain"
    assert type(out) is type(json.loads(text)), "same top-level type"
    return out


def markdown(result: Dict[str, Any]) -> str:
    """Verdict table (one row per hypothesis, with its worst part) and notes."""
    assert "hypotheses" in result, "an evaluate() result"
    rows = [
        "| ID | Hypothesis | Verdict | Parts (P / I / F) | Worst part |",
        "|---|---|---|---|---|",
    ]
    for h in HYPOTHESES:
        r = result["hypotheses"][h]
        c = r["counts"]
        w = r["worst"][0] if r["worst"] else None
        worst = "—" if w is None else (
            f"{w['part']} @ {w['cell']}: value {w['value']:.4g}, dev {w['dev']:.3g}, "
            f"z {w['z']:.3g} ({w['outcome']})")  # fmt: skip
        rows.append(
            f"| {h} | {r['title']} | **{r['verdict']}** | "
            f"{c[PASS]} / {c[INCONCLUSIVE]} / {c[FAIL]} | {worst} |"
        )
    rows += [
        "",
        "T = 1e-3, descriptive (no verdict), parts within / outside the criteria: "
        + ", ".join(
            f"{h} {v['counts'][PASS]} / {v['counts'][INCONCLUSIVE] + v['counts'][FAIL]}"
            for h, v in result["descriptive"].items()
        ),
        "",
    ]
    rows += [f"- {n}" for n in result["notes"]]
    if "exploratory" in result:
        rows += exploratory_markdown(result["exploratory"])
    assert len(rows) >= len(HYPOTHESES) + 2, "table rows"
    return "\n".join(rows) + "\n"


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])

    # ----- required -----
    parser.add_argument(
        "--analysis",
        type=Path,
        nargs="+",
        required=True,
        help="md_analysis*_chi*.npz files or directories",
    )

    # ----- optional -----
    root = Path(__file__).resolve().parents[1] / "md_winding"
    parser.add_argument("--out-json", type=Path, default=root / "verdicts.json")
    parser.add_argument(
        "--out-md", type=Path, default=None, help="default: --out-json with suffix .md"
    )
    args = parser.parse_args()

    # =========================================================
    # EVALUATE
    # =========================================================
    main_runs, lang_runs, inputs = collect(args.analysis)
    if not main_runs:
        sys.exit("no md_analysis_chi*.npz found")
    res = evaluate(main_runs, lang_runs)
    res["inputs"] = inputs
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(clean_json(res), indent=1) + "\n")
    md_path = args.out_md or args.out_json.with_suffix(".md")
    md_path.write_text(markdown(res))
    for h in HYPOTHESES:
        print(f"{h}: {res['hypotheses'][h]['verdict']}", flush=True)
    print(f"wrote {args.out_json} and {md_path}", flush=True)
