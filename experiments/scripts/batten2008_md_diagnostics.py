#!/usr/bin/env python3

"""
Post hoc diagnostics of the MD winding experiment (descriptive, NOT pre-registered).

Reads the analysis files of batten2008_md_analysis.py (searched recursively
under --results) and the verdicts.json of batten2008_md_stats.py, and writes
``diagnostics.{json,md}`` beside them. Nothing here enters a verdict; every
number is a summary of quantities the pre-registered pipeline already stores:

- ``bin_waves``: constrained waves per |k|/K bin behind the PASS,
  INCONCLUSIVE and FAIL parts of H1 (<S>/T) and H2 (rate / Rice);
- ``thermostat``: median MSD exponent alpha, D_theta and nu per (start,
  chi, T) for Andersen + NVE and for BAOAB at each gamma, on the ground
  states both contain (E4 uses ground states 0-4);
- ``unconstrained``: the H6 rate / Rice ratio divided by sqrt(pi), i.e. the
  unsigned rate of the unconstrained waves over |k| sqrt(T / S) / 2 pi, the
  Rice rate of a circular Gaussian rho (median over ground states, per bin);
- ``drift``: the log-log slope of the particle MSD over the last decade of
  lag, and the exponent of the end-of-segment MSD against T (harmonic T);
- ``energy``: mean / M and sd / sqrt(M) of the snapshot Phi / T at harmonic
  T (Gamma(M, 1): 1 and 1), and the equilibration gate misses (> 5 %);
- ``spectra``: the largest sqrt(eig K) of each E1b ground state, the
  frequency above which the rho_k power stays below 1e-3 of its peak, and
  the power fraction at omega <= 1.05 max sqrt(eig K).

Usage: python -I batten2008_md_diagnostics.py --results DIR

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_md_analysis import N_BINS  # noqa: E402
from batten2008_md_stats import (  # noqa: E402
    clean_json,
    collect,
    load_analysis,
    temp_indices,
)

# ============================================================
# CONSTANTS
# ============================================================

LABEL = "post hoc diagnostics (descriptive, not pre-registered)"
GATE_TOL = 0.05  # E1 equilibration gate (cli.md GATE_TOL)
EDGE_FRACTION = 1e-3  # spectral edge: power below this fraction of the peak
EDGE_MARGIN = 1.05  # power fraction counted up to this x max sqrt(eig K)
CELL_RE = re.compile(r"^(?P<system>\S+) chi=(?P<tag>\S+) T=(?P<t>\S+)$")
BIN_RE = re.compile(r"^(S|Rice) bin (?P<b>\d+)$")

Data = Dict[str, Any]

# ============================================================
# SUMMARIES
# ============================================================


def bin_counts(z: Data) -> np.ndarray:
    """Constrained waves per pre-registered |k|/K bin (N_BINS,)."""
    con = np.asarray(z["constrained"], dtype=bool)
    ids = np.asarray(z["bin_id"])[con]
    assert ids.size == int(con.sum()) and ids.size >= 1, "constrained waves"
    assert np.all((ids >= 0) & (ids < N_BINS)), "constrained bins 0..N_BINS-1"
    out: np.ndarray = np.bincount(ids, minlength=N_BINS)
    return out


def bin_waves(parts: List[Data], counts: Dict[Tuple[str, str], np.ndarray]) -> Data:
    """Waves per bin behind each outcome of the per-bin parts ('S bin', 'Rice bin').

    Parameters
    ----------
    parts : verdict parts of H1 or H2 (verdicts.json ``parts``).
    counts : {(system, tag): bin_counts} of the production analyses.

    Returns
    -------
    {outcome: {"n_parts", "median", "min", "max"}} of the bin wave counts.
    """
    assert all("part" in p and "cell" in p for p in parts), "verdict parts"
    groups: Dict[str, List[int]] = {}
    for p in parts:
        mb, mc = BIN_RE.match(p["part"]), CELL_RE.match(p["cell"])
        if mb is None or mc is None:
            continue
        waves = counts[(mc["system"], mc["tag"])][int(mb["b"])]
        groups.setdefault(p["outcome"], []).append(int(waves))
    out = {o: {"n_parts": len(v), "median": float(np.median(v)),
               "min": int(min(v)), "max": int(max(v))}
           for o, v in sorted(groups.items())}  # fmt: skip
    assert all(r["min"] >= 1 for r in out.values()), "bins with parts hold waves"
    return out


def _common(a: Data, b: Data) -> Tuple[np.ndarray, np.ndarray]:
    """Indices into a and b of the ground states both contain."""
    common = np.intersect1d(a["indices"], b["indices"])
    ia = np.searchsorted(a["indices"], common)
    ib = np.searchsorted(b["indices"], common)
    assert np.array_equal(a["indices"][ia], b["indices"][ib]), "sorted indices"
    assert common.size >= 1, "no common ground states"
    return ia, ib


def _medians(z: Data, rows: np.ndarray, ti: int) -> Dict[str, float]:
    """Median alpha, D_theta and nu over the ground states ``rows`` at T index ti."""
    assert rows.ndim == 1 and rows.size >= 1, "ground-state rows"
    out = {key: float(np.nanmedian(z[key][rows, ti]))
           for key in ("alpha", "d_theta", "nu_c")}  # fmt: skip
    assert len(out) == 3, "three statistics"
    return out


def thermostat(main: List[Data], lang: List[Data]) -> List[Data]:
    """alpha, D_theta and nu: Andersen + NVE against BAOAB per gamma (common GS)."""
    by_cell = {(str(z["system"]), str(z["tag"])): z for z in main}
    out: List[Data] = []
    for z in sorted(lang, key=lambda q: (str(q["system"]), str(q["tag"]),
                                         float(q["gamma"]))):  # fmt: skip
        m = by_cell.get((str(z["system"]), str(z["tag"])))
        if m is None:
            continue
        iz, im = _common(z, m)
        for ti in range(z["temps"].size):
            hit = np.flatnonzero(np.isclose(m["temps"], z["temps"][ti], rtol=1e-9))
            if hit.size:
                out.append({"system": str(z["system"]), "tag": str(z["tag"]),
                            "T": float(z["temps"][ti]), "gamma": float(z["gamma"]),
                            "n_gs": int(iz.size),
                            "nve": _medians(m, im, int(hit[0])),
                            "baoab": _medians(z, iz, ti)})  # fmt: skip
    assert all(r["n_gs"] >= 1 for r in out), "paired ground states"
    return out


def unconstrained(z: Data) -> List[Data]:
    """Per T: H6 rate / Rice over sqrt(pi) (median over GS) across the |k| > K bins."""
    assert z["ratio_bin"].shape[-1] == 2 * N_BINS, "40 bins"
    out: List[Data] = []
    for ti in range(z["temps"].size):
        rows = z["ratio_bin"][:, ti, N_BINS:] / math.sqrt(math.pi)
        ok = np.any(np.isfinite(rows), axis=0)
        med = np.nanmedian(rows[:, ok], axis=0)
        out.append({"T": float(z["temps"][ti]), "n_bins": int(ok.sum()),
                    "min": float(med.min()), "median": float(np.median(med)),
                    "max": float(med.max())})  # fmt: skip
    assert len(out) == z["temps"].size, "one row per T"
    return out


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of log y against log x over the positive finite pairs."""
    assert x.shape == y.shape and x.ndim == 1, "matching 1D arrays"
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 2:
        return float("nan")
    out = float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])
    assert np.isfinite(out), "finite slope"
    return out


def drift(z: Data) -> Data:
    """Late-lag particle-MSD exponent per T and the T exponent of the end MSD."""
    lag = np.asarray(z["part_lag"])
    msd = np.nanmedian(z["part_msd"], axis=0)  # (n_T, R)
    assert msd.shape == (z["temps"].size, lag.size), "MSD per T and lag"
    late = lag >= lag[-1] / 10.0
    harm = temp_indices(z["temps"], True)
    out = {"t_exponent_last_decade": [loglog_slope(lag[late], m[late]) for m in msd],
           "T_exponent_end": loglog_slope(z["temps"][harm], msd[harm, -1]),
           "msd_end": msd[:, -1].tolist(), "lag_end": float(lag[-1])}  # fmt: skip
    assert len(out["t_exponent_last_decade"]) == z["temps"].size, "one per T"
    return out


def energy(z: Data) -> Data:
    """Snapshot Phi / T against Gamma(M, 1) at harmonic T; gate misses (> 5 %)."""
    m = int(z["M"])
    harm = temp_indices(z["temps"], True)
    x = np.asarray(z["phi_over_t"])[:, harm].ravel()
    x = x[np.isfinite(x)]
    gate = np.asarray(z["gate"], dtype=np.float64)
    assert m >= 1 and x.size >= 2, "snapshots and M"
    out = {"M": m, "mean_over_M": float(x.mean() / m),
           "sd_over_sqrtM": float(x.std(ddof=1) / math.sqrt(m)),
           "gate_misses": int(np.sum(np.abs(gate - 1.0) > GATE_TOL)),
           "gate_lanes": int(gate.size),
           "gate_max_dev": float(np.nanmax(np.abs(gate - 1.0)))}  # fmt: skip
    assert out["gate_misses"] <= out["gate_lanes"], "misses within lanes"
    return out


def spectral_edge(omega: np.ndarray, power: np.ndarray, f_max: float) -> Data:
    """Edge (last omega with power >= EDGE_FRACTION x peak) and the power within f_max.

    ``omega`` sorted two-sided; only omega > 0 is used. ``f_max`` is the
    largest sqrt(eig K); the fraction counts omega <= EDGE_MARGIN f_max.
    """
    assert omega.shape == power.shape and f_max > 0.0, "spectrum and support"
    pos = omega > 0.0
    w, p = omega[pos], power[pos]
    assert w.size >= 2 and np.all(p >= 0.0), "positive frequencies"
    above = np.flatnonzero(p >= EDGE_FRACTION * p.max())
    inside = float(p[w <= EDGE_MARGIN * f_max].sum() / p.sum())
    return {"f_max": float(f_max), "edge": float(w[above[-1]]),
            "edge_over_f_max": float(w[above[-1]] / f_max),
            "power_inside": inside}  # fmt: skip


def spectra(spec: List[Data]) -> List[Data]:
    """spectral_edge of every E1b ground state with a K spectrum."""
    out: List[Data] = []
    for z in spec:
        if "spec_power" not in z or "k_eig" not in z:
            continue
        for g in range(z["indices"].size):
            f_max = float(np.sqrt(np.max(z["k_eig"][g])))
            row = spectral_edge(z["spec_omega"], z["spec_power"][g, 0], f_max)
            row.update(system=str(z["system"]), tag=str(z["tag"]),
                       gs=int(z["indices"][g]))  # fmt: skip
            out.append(row)
    assert all(0.0 <= r["power_inside"] <= 1.0 + 1e-12 for r in out), "fractions"
    return out


# ============================================================
# DRIVER AND OUTPUT
# ============================================================


def diagnose(results: Path, verdicts: Optional[Data]) -> Data:
    """Every diagnostic of the analyses under ``results``."""
    main, lang, _ = collect([results])
    assert main, f"no md_analysis_chi*.npz under {results}"
    spec = [load_analysis(f) for f in sorted(results.rglob("md_analysis_spectra*.npz"))]
    counts = {(str(z["system"]), str(z["tag"])): bin_counts(z) for z in main}
    out: Data = {"label": LABEL, "thermostat": thermostat(main, lang),
                 "spectra": spectra(spec), "cells": {}}  # fmt: skip
    for z in main:
        key = f"{z['system']} chi={z['tag']}"
        out["cells"][key] = {"bin_counts": counts[(str(z["system"]), str(z["tag"]))],
                             "unconstrained": unconstrained(z), "drift": drift(z),
                             "energy": energy(z)}  # fmt: skip
    if verdicts is not None:
        out["bin_waves"] = {h: bin_waves(verdicts["hypotheses"][h]["parts"], counts)
                            for h in ("H1", "H2")}  # fmt: skip
    assert len(out["cells"]) == len(main), "one entry per production cell"
    return out


def _thermo_md(rows: List[Data]) -> List[str]:
    """Markdown table of the thermostat comparison."""
    lines = ["| Start | chi | T | gamma | GS | alpha NVE / BAOAB | D_theta NVE / BAOAB"
             " | nu NVE / BAOAB |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        a, b = r["nve"], r["baoab"]
        lines.append(f"| {r['system']} | {r['tag']} | {r['T']:g} | {r['gamma']:g} | "
                     f"{r['n_gs']} | {a['alpha']:.3f} / {b['alpha']:.3f} | "
                     f"{a['d_theta']:.3f} / {b['d_theta']:.3f} | "
                     f"{a['nu_c']:.4f} / {b['nu_c']:.4f} |")  # fmt: skip
    assert len(lines) == len(rows) + 2, "header and rows"
    return lines


def _cells_md(cells: Dict[str, Data]) -> List[str]:
    """Markdown table of the per-cell drift, unconstrained-rate and energy values."""
    lines = ["| Cell | unc. rate / circular Rice (min / med / max over T) | MSD t-exp."
             " (last decade, per T) | MSD_end T-exp. | Phi/T mean/M, sd/sqrt(M) |"
             " gate misses |", "|---|---|---|---|---|---|"]  # fmt: skip
    for key, c in cells.items():
        u, d, e = c["unconstrained"], c["drift"], c["energy"]
        lo, hi = min(r["min"] for r in u), max(r["max"] for r in u)
        med = float(np.median([r["median"] for r in u]))
        texp = ", ".join(f"{v:.2f}" for v in d["t_exponent_last_decade"])
        lines.append(f"| {key} | {lo:.3f} / {med:.3f} / {hi:.3f} | {texp} | "
                     f"{d['T_exponent_end']:.3f} | {e['mean_over_M']:.4f}, "
                     f"{e['sd_over_sqrtM']:.3f} | {e['gate_misses']} / "
                     f"{e['gate_lanes']} |")  # fmt: skip
    assert len(lines) == len(cells) + 2, "header and rows"
    return lines


def markdown(diag: Data) -> str:
    """diagnostics.md: the thermostat, per-cell, bin-wave and spectral tables."""
    assert diag["label"] == LABEL, "a diagnose() result"
    lines = [f"# MD winding: {LABEL}", "", "## Thermostat (net winding)", ""]
    lines += _thermo_md(diag["thermostat"])
    lines += ["", "## Per cell", ""] + _cells_md(diag["cells"])
    if "bin_waves" in diag:
        lines += ["", "## Waves per bin behind the H1 / H2 per-bin outcomes", "",
                  "| Hyp | Outcome | Parts | Waves median | min | max |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        for h, rows in diag["bin_waves"].items():
            lines += [f"| {h} | {o} | {r['n_parts']} | {r['median']:g} | {r['min']}"
                      f" | {r['max']} |" for o, r in rows.items()]  # fmt: skip
    lines += [
        "",
        "## E1b spectra",
        "",
        "| Start | chi | GS | max sqrt(eig K) | edge | edge / max | power inside |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['system']} | {r['tag']} | {r['gs']} | {r['f_max']:.3f} | "
              f"{r['edge']:.3f} | {r['edge_over_f_max']:.3f} | "
              f"{r['power_inside']:.4f} |" for r in diag["spectra"]]  # fmt: skip
    assert len(lines) > 8, "tables written"
    return "\n".join(lines) + "\n"


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])

    # ----- required -----
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="directory of md_analysis*_chi*.npz and verdicts.json",
    )

    # ----- optional -----
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="default: <results>/verdicts.json if present",
    )
    args = parser.parse_args()

    # =========================================================
    # DIAGNOSE
    # =========================================================
    vpath = args.verdicts or args.results / "verdicts.json"
    ver = json.loads(vpath.read_text()) if vpath.is_file() else None
    res = diagnose(args.results, ver)
    (args.results / "diagnostics.json").write_text(
        json.dumps(clean_json(res), indent=1) + "\n")  # fmt: skip
    (args.results / "diagnostics.md").write_text(markdown(res))
    print(f"wrote {args.results}/diagnostics.{{json,md}}", flush=True)
    print("Done.", flush=True)
