#!/usr/bin/env python3

"""
INCONCLUSIVE remedy of the MD winding experiment: H3 / H7 on 2x-cycles reruns.

HYPOTHESES.md (decision table): "INCONCLUSIVE: ... One cell is then rerun at
2x length from the reserve budget, and the verdict is re-evaluated once";
NOTES.md section 1 reads 2x length as 2x the cycles. The reruns use the
production command with ``--cycles 4`` and the same cell seeds, into a
separate data root, so their first two cycles must reproduce production.

This script

1. checks that determinism (``prefix_agreement``): every ``cycle_*`` array
   of the first C_prod cycles, every ``snap_*`` array at the record steps
   both runs share, the dense rho trace at the trace times both share, and
   the set-up / equilibration-gate arrays must agree bitwise;
2. re-evaluates H3 and H7 with the registered part functions of
   batten2008_md_stats.py on all cycles of the rerun cells: H3 on the rerun
   Andersen + NVE cell; H7 BAOAB / NVE with the rerun BAOAB cells against
   the production NVE cell, and gamma / gamma between the rerun cells;
3. reports, per hypothesis, the production verdict, the rerun cell's parts
   against the same cell's production parts, and the verdict after the
   rerun cell's parts replace its production parts (``after_remedy``).

The registered production verdicts (verdicts.json) are not changed.

Usage: python -I batten2008_md_remedy.py --production DIR --rerun DIR
           --pair PROD.npz RERUN.npz [--pair ...] --out-json FILE

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_md_stats import (  # noqa: E402
    FAIL,
    INCONCLUSIVE,
    PASS,
    Data,
    Part,
    clean_json,
    collect,
    h3_parts,
    h7_parts,
    summarize,
)

# ============================================================
# CONSTANTS
# ============================================================

LABEL = "INCONCLUSIVE remedy (decision table: one cell rerun at 2x cycles)"
REMEDY_HYPOTHESES: Tuple[str, ...] = ("H3", "H7")
SETUP_KEYS: Tuple[str, ...] = (
    "temperatures", "indices", "lane_index", "lane_temperature", "basis",
    "waves_kvec", "waves_n_vectors", "waves_constrained",
    "equil_gate_first", "equil_gate", "equil_gate_pass", "equil_extended",
)  # fmt: skip
DIVERGENCE_LEVELS: Tuple[float, ...] = (1e-9, 1e-3, 1e-1)  # relative trace diff

# ============================================================
# DETERMINISM CHECK
# ============================================================


def _difference(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Bitwise equality and max |a - b| (NaN == NaN; NaN vs a number is inf)."""
    assert a.shape == b.shape, f"shapes differ: {a.shape} vs {b.shape}"
    assert a.dtype.kind == b.dtype.kind or {a.dtype.kind, b.dtype.kind} <= set("iuf")
    a2, b2 = np.atleast_1d(a), np.atleast_1d(b)
    same, worst = True, 0.0
    for i in range(a2.shape[0]):  # row by row: bounded memory for the rho trace
        x, y = a2[i], b2[i]
        nan_ok = x.dtype.kind in "fc"
        same = same and bool(np.array_equal(x, y, equal_nan=nan_ok))
        xf, yf = np.asarray(x, np.float64), np.asarray(y, np.float64)
        d = np.where(np.isnan(xf) & np.isnan(yf), 0.0, np.abs(xf - yf))
        d = np.where(np.isnan(d), np.inf, d)
        worst = max(worst, float(np.max(d)) if d.size else 0.0)
    return {"bitwise": same, "max_abs": worst, "n": int(a.size)}


def _common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of the values a and b share (both strictly increasing)."""
    assert a.ndim == b.ndim == 1, "1-d grids"
    _, ia, ib = np.intersect1d(a, b, assume_unique=True, return_indices=True)
    assert np.array_equal(a[ia], b[ib]), "matched grid values"
    return ia, ib


def prefix_agreement(
    prod: Dict[str, np.ndarray], rerun: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """Compare a production MD output with a longer rerun where they overlap.

    Parameters
    ----------
    prod, rerun
        Arrays of the two MD outputs (np.load of the npz). The rerun must
        have at least as many cycles (leading axis of ``cycle_*``).

    Returns
    -------
    dict
        ``keys``: per array {bitwise, max_abs, n}; ``bitwise`` / ``max_abs``
        over all of them; ``n_cycles`` compared; ``n_snapshots_common``.

    Raises
    ------
    ValueError
        If the rerun has fewer cycles than production.
    """
    n_c = int(prod["cycle_status"].shape[0])
    if rerun["cycle_status"].shape[0] < n_c:
        raise ValueError(f"rerun has fewer cycles than production ({n_c})")
    out: Dict[str, Dict[str, Any]] = {}
    for k in sorted(set(prod) & set(rerun)):
        if k.startswith("cycle_"):
            out[k] = _difference(prod[k], rerun[k][:n_c])
        elif k in SETUP_KEYS:
            out[k] = _difference(np.asarray(prod[k]), np.asarray(rerun[k]))
    ia, ib = _common(prod["record_steps"], rerun["record_steps"])
    n_rec = prod["record_steps"].size
    for k in sorted(set(prod) & set(rerun)):
        if k.startswith("snap_") and prod[k].ndim >= 2 and prod[k].shape[1] == n_rec:
            out[k] = _difference(prod[k][:, ia], rerun[k][:, ib])
    if "trace_time" in prod and "trace_time" in rerun:
        q_p, q_r = int(prod["trace_count"].min()), int(rerun["trace_count"].min())
        ta, tb = _common(prod["trace_time"][0, :q_p], rerun["trace_time"][0, :q_r])
        for k in ("trace_time", "trace_rho"):
            out[k] = _difference(prod[k][:, ta], rerun[k][:, tb])
    assert out and all(v["n"] >= 0 for v in out.values()), "compared arrays"
    return {
        "keys": out, "n_cycles": n_c, "n_snapshots_common": int(ia.size),
        "bitwise": all(v["bitwise"] for v in out.values()),
        "max_abs": max(v["max_abs"] for v in out.values()),
    }  # fmt: skip


def divergence_onset(
    prod: Dict[str, np.ndarray], rerun: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """When the rerun's dense rho trace leaves the production trace.

    Per shared trace row, rel = max over lanes of max |rho_rerun - rho_prod|
    / max |rho_prod| (lane by lane, to bound memory). Returns ``t`` and
    ``rel`` per row and ``t_reach``: the first time rel exceeds each of
    DIVERGENCE_LEVELS (None if never).
    """
    q_p, q_r = int(prod["trace_count"].min()), int(rerun["trace_count"].min())
    ta, tb = _common(prod["trace_time"][0, :q_p], rerun["trace_time"][0, :q_r])
    assert ta.size >= 1, "shared trace rows"
    rel = np.zeros(ta.size)
    for lane in range(prod["trace_rho"].shape[0]):
        a, b = prod["trace_rho"][lane][ta], rerun["trace_rho"][lane][tb]
        scale = np.max(np.abs(a), axis=tuple(range(1, a.ndim)))
        diff = np.max(np.abs(b - a), axis=tuple(range(1, a.ndim)))
        rel = np.maximum(rel, diff / np.where(scale > 0.0, scale, 1.0))
    t = prod["trace_time"][0, ta].astype(np.float64)
    reach = {f"{lv:g}": (float(t[np.argmax(rel > lv)]) if np.any(rel > lv) else None)
             for lv in DIVERGENCE_LEVELS}  # fmt: skip
    assert rel.shape == t.shape, "one value per shared row"
    return {"t": t, "rel": rel, "t_reach": reach}


def cycle_winding_ratio(
    prod: Dict[str, np.ndarray], rerun: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """Rerun / production unsigned constrained winding per (shared cycle, lane).

    A statistical check of the shared cycles when they are not bitwise equal:
    the summed ``cycle_wind_abs`` of the constrained waves, per lane and
    cycle, of the rerun over production (median, min, max, n).
    """
    con = np.asarray(prod["waves_constrained"], dtype=bool)
    assert np.array_equal(con, rerun["waves_constrained"]), "same waves"
    n_c = int(prod["cycle_wind_abs"].shape[0])
    w_p = prod["cycle_wind_abs"][..., con].sum(axis=-1)
    w_r = rerun["cycle_wind_abs"][:n_c][..., con].sum(axis=-1)
    ratio = (w_r / w_p).reshape(-1)
    assert ratio.size == w_p.size >= 1, "one ratio per cycle and lane"
    return {"median": float(np.median(ratio)), "min": float(ratio.min()),
            "max": float(ratio.max()), "n": int(ratio.size)}  # fmt: skip


# ============================================================
# RE-EVALUATION
# ============================================================


def _key(p: Part) -> Tuple[str, str]:
    """(part name, cell label): the identity of a verdict part."""
    assert "part" in p and "cell" in p, "a verdict part"
    return str(p["part"]), str(p["cell"])


def replace_parts(base: List[Part], new: List[Part]) -> List[Part]:
    """``base`` with each part of the same (part, cell) as one of ``new`` replaced.

    Parts of ``new`` with no counterpart in ``base`` (e.g. rerun segments 2
    and 3) are appended in their order.
    """
    by_key = {_key(p): p for p in new}
    assert len(by_key) == len(new), "rerun parts are unique"
    out = [by_key.pop(_key(p), p) for p in base]
    out += [p for p in new if _key(p) in by_key]
    assert len({_key(p) for p in out}) == len(out), "unique parts"
    return out


def remedy(
    prod_main: List[Data], prod_lang: List[Data],
    rerun_main: List[Data], rerun_lang: List[Data],
) -> Dict[str, Any]:  # fmt: skip
    """H3 / H7: production, the rerun cell against its production parts, after remedy.

    Parameters
    ----------
    prod_main, prod_lang
        Production analyses (Andersen + NVE; BAOAB).
    rerun_main, rerun_lang
        Analyses of the 2x-cycles reruns; H7 pairs the rerun BAOAB cells
        with the production NVE cell of the same (system, chi).

    Returns
    -------
    dict
        ``label`` and, per hypothesis, the summaries ``production``,
        ``production_cell``, ``rerun_cell``, ``after_remedy`` and the
        rerun ``cells``.
    """
    assert all("nu_c" in z for z in prod_main + rerun_main), "analysis files"
    prod = {"H3": h3_parts(prod_main, True), "H7": h7_parts(prod_main, prod_lang)}
    new = {"H3": h3_parts(rerun_main, True), "H7": h7_parts(prod_main, rerun_lang)}
    out: Dict[str, Any] = {}
    for h in REMEDY_HYPOTHESES:
        cells = sorted({p["cell"] for p in new[h]})
        old_cell = [p for p in prod[h] if p["cell"] in cells]
        missing = {_key(p) for p in old_cell} - {_key(p) for p in new[h]}
        assert not missing, f"{h}: rerun lacks production parts {sorted(missing)}"
        out[h] = {
            "cells": cells, "production": summarize(prod[h]),
            "production_cell": summarize(old_cell), "rerun_cell": summarize(new[h]),
            "after_remedy": summarize(replace_parts(prod[h], new[h])),
        }  # fmt: skip
    return {"label": LABEL, "hypotheses": out}


# ============================================================
# MARKDOWN
# ============================================================


def _counts(s: Dict[str, Any]) -> str:
    """'P / I / F' of a summary."""
    c = s["counts"]
    return f"{c[PASS]} / {c[INCONCLUSIVE]} / {c[FAIL]}"


def markdown(result: Dict[str, Any]) -> str:
    """Verdict table of the remedy, every rerun part, and the determinism check."""
    assert result["label"] == LABEL, "a remedy() result"
    rows = [f"# {LABEL}", "",
            "The registered production verdicts (verdicts.json) are unchanged.", "",
            "| H | production | production cell | rerun cell (all cycles) | "
            "after remedy |", "|---|---|---|---|---|"]  # fmt: skip
    for h, r in result["hypotheses"].items():
        cols = [
            f"**{r[s]['verdict']}** ({_counts(r[s])})"
            for s in ("production", "production_cell", "rerun_cell", "after_remedy")
        ]
        rows.append(f"| {h} | " + " | ".join(cols) + " |")
    for h, r in result["hypotheses"].items():
        old = {_key(p): p for p in r["production_cell"]["parts"]}
        rows += ["", f"## {h} rerun parts (production value, z where it exists)", "",
                 "| part | cell | production | rerun value | z | dev | outcome |",
                 "|---|---|---|---|---|---|---|"]  # fmt: skip
        for p in r["rerun_cell"]["parts"]:
            q = old.get(_key(p))
            prev = "—" if q is None else (
                f"{q['value']:.5f}, z {q['z']:.3g} ({q['outcome']})")  # fmt: skip
            rows.append(
                f"| {p['part']} | {p['cell']} | {prev} | {p['value']:.5f} | "
                f"{p['z']:.3g} | {p['dev']:.3g} | {p['outcome']} |"
            )
    for name, d in result.get("determinism", {}).items():
        bad = [k for k, v in d["keys"].items() if not v["bitwise"]]
        rows += ["", f"- determinism {name}: first {d['n_cycles']} cycles, "
                 f"{d['n_snapshots_common']} shared snapshots, {len(d['keys'])} "
                 f"arrays; bitwise {d['bitwise']}, max |diff| {d['max_abs']:.3g}"
                 + (f"; differ: {', '.join(bad)}" if bad else "")]  # fmt: skip
        if "onset" in d:
            o, w = d["onset"], d["winding_ratio"]
            reach = ", ".join(f"{lv} at t = " + ("never" if t is None else f"{t:.6g}")
                              for lv, t in o["t_reach"].items())  # fmt: skip
            rows += [f"  - rho trace rel. difference {o['rel_first']:.3g} at t = "
                     f"{o['t_first']:g} (first shared row); exceeds {reach}; "
                     f"{o['rel_last']:.3g} at t = {o['t_last']:g}",
                     f"  - unsigned constrained winding, rerun / production per "
                     f"(cycle, lane): median {w['median']:.5f}, range "
                     f"{w['min']:.5f}-{w['max']:.5f} (n = {w['n']})"]  # fmt: skip
    assert len(rows) >= 6, "table rows"
    return "\n".join(rows) + "\n"


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])

    # ----- required -----
    parser.add_argument("--production", type=Path, nargs="+", required=True,
                        help="production md_analysis files or directories")  # fmt: skip
    parser.add_argument("--rerun", type=Path, nargs="+", required=True,
                        help="rerun md_analysis files or directories")  # fmt: skip
    parser.add_argument("--out-json", type=Path, required=True)

    # ----- optional -----
    parser.add_argument(
        "--pair",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        metavar=("PROD", "RERUN"),
        help="raw md_*.npz outputs for the determinism check",
    )
    args = parser.parse_args()

    # =========================================================
    # DETERMINISM
    # =========================================================
    checks: Dict[str, Any] = {}
    for p_path, r_path in args.pair:
        with np.load(p_path) as zp, np.load(r_path) as zr:
            keys = [k for k in zp.files if k in zr.files]
            prod_arrays = {k: zp[k] for k in keys}
            rerun_arrays = {k: zr[k] for k in keys}
        chk = prefix_agreement(prod_arrays, rerun_arrays)
        onset = divergence_onset(prod_arrays, rerun_arrays)
        chk["onset"] = {"t_first": onset["t"][0], "rel_first": onset["rel"][0],
                        "t_last": onset["t"][-1], "rel_last": onset["rel"][-1],
                        "t_reach": onset["t_reach"]}  # fmt: skip
        chk["winding_ratio"] = cycle_winding_ratio(prod_arrays, rerun_arrays)
        checks[str(r_path)] = chk
        del prod_arrays, rerun_arrays
        print(f"{r_path}: bitwise {checks[str(r_path)]['bitwise']}, max |diff| "
              f"{checks[str(r_path)]['max_abs']:.3g}", flush=True)  # fmt: skip

    # =========================================================
    # RE-EVALUATE
    # =========================================================
    p_main, p_lang, p_used = collect(args.production)
    r_main, r_lang, r_used = collect(args.rerun)
    res = remedy(p_main, p_lang, r_main, r_lang)
    res["determinism"] = checks
    res["inputs"] = {"production": p_used, "rerun": r_used}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(clean_json(res), indent=1) + "\n")
    md_path = args.out_json.with_suffix(".md")
    md_path.write_text(markdown(res))
    for h, r in res["hypotheses"].items():
        print(f"{h}: rerun cell {r['rerun_cell']['verdict']}, after remedy "
              f"{r['after_remedy']['verdict']}", flush=True)  # fmt: skip
    print(f"wrote {args.out_json} and {md_path}", flush=True)
