#!/usr/bin/env python3

"""
Markdown rendering of the T12 ensemble-parity results (pure Python).

``results`` is the JSON-serializable dictionary built by
``tests/parity/test_ensemble_parity.py``; ``render`` returns the full report or,
with ``summary=True``, the compact version pasted into ``docs/parity.md``.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any, Dict, List

# ----- project -----
import ensemble_stats as es

# ============================================================
# HELPERS
# ============================================================

KS_LABELS = {
    "nn_min": "min NN distance",
    "nn_mean": "mean NN distance",
    "g2_peak": "first g2-peak height",
    "rms_disp": "RMS displacement",
}


def _fmt(x: float) -> str:
    return f"{x:.3g}"


def _ok(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def battery_dict(res: es.BatteryResult, k_shell: Any, n_tests: int) -> Dict[str, Any]:
    """Return a JSON-serializable view of a battery result."""
    worst = int(abs(res.sk_z).argmax())
    return {
        "n_a": res.n_a,
        "n_b": res.n_b,
        "frac_a": res.frac_a,
        "frac_b": res.frac_b,
        "prop": {"z": res.prop_z, "p": res.prop_p, "pass": res.prop_pass},
        "ks": {
            k: {"stat": res.ks_stat[k], "p": res.ks_p[k], "pass": res.ks_pass[k]}
            for k in res.ks_p
        },
        "sk": {
            "worst_abs_z": res.sk_worst,
            "worst_k": float(k_shell[worst]),
            "n_bins": int(res.sk_z.size),
            "pass": res.sk_pass,
        },
        "logphi_ks_p": res.logphi_ks_p,
        "p_threshold_prop": res.p_threshold_prop,
        "p_threshold_ks": res.p_threshold_ks,
        "zmax": res.zmax,
        "n_tests": n_tests,
        "passed": res.passed,
    }


def battery_table(b: Dict[str, Any]) -> List[str]:
    """Return the markdown table lines of one battery."""
    n = b["n_tests"]
    lines = [
        "| statistic | A | B | z or KS D | p raw | p corrected | verdict |",
        "|---|---|---|---|---|---|---|",
        f"| success fraction | {b['frac_a']:.4f} | {b['frac_b']:.4f} "
        f"| {b['prop']['z']:.3f} | {_fmt(b['prop']['p'])} "
        f"| {_fmt(min(1.0, b['prop']['p'] * n))} | {_ok(b['prop']['pass'])} |",
    ]
    for k, v in b["ks"].items():
        lines.append(
            f"| KS {KS_LABELS[k]} | | | {v['stat']:.4f} | {_fmt(v['p'])} "
            f"| {_fmt(min(1.0, v['p'] * n))} | {_ok(v['pass'])} |"
        )
    sk = b["sk"]
    lines.append(
        f"| S(k) worst of {sk['n_bins']} bins | | | {sk['worst_abs_z']:.3f} "
        f"(at |k|={sk['worst_k']:.3f}) | | | {_ok(sk['pass'])} (|z| < {b['zmax']}) |"
    )
    lines.append(
        f"| -log10 Phi KS (report only, R22) | | | | {_fmt(b['logphi_ks_p'])} | | n/a |"
    )
    return lines


# ============================================================
# RENDER
# ============================================================


def render(results: Dict[str, Any], summary: bool = False) -> str:
    """Return the report (full) or its compact docs summary (``summary``)."""
    meta = results["meta"]
    out = [
        f"# Ensemble parity vs C++ L-BFGS ({meta['date']})",
        "",
        f"- git hash: `{meta['git']}`; SLURM job: {meta['job_id']}; "
        f"device: {meta['device']}; JAX {meta['jax_version']}",
        f"- configs per task: {meta['limit']} (tasks 0000-0002 = "
        f"{3 * meta['limit']} per chi); success: Phi < {meta['tol']:.8e}",
        f"- family-wise alpha = {meta['alpha']}, Bonferroni over {meta['n_tests']} "
        f"p-tests (5 chi x 5 tests x {{JAX vs C++, C++ false-alarm control}}): "
        f"p-test threshold {_fmt(meta['p_prop'])}; KS threshold "
        f"min(KS_PMIN, alpha/n) = {_fmt(meta['p_ks'])}; |z|_max = {meta['zmax']}",
        f"- wall time (test session): {meta['wall_seconds']:.0f} s",
        f"- **verdict: {results['verdict']}**",
        "",
        "## Controls",
        "",
        "| control | chi | expected | outcome | behaves |",
        "|---|---|---|---|---|",
    ]
    for c in results["controls"]:
        out.append(
            f"| {c['kind']} | {c['chi']} | {c['expected']} | {c['outcome']} "
            f"| {c['behaves']} |"
        )
    out += ["", "## Per-chi results (A = JAX, B = C++ unless stated)", ""]
    for chi_s, r in results["chi"].items():
        out += _chi_block(chi_s, r, summary)
    out += _power_blocks(results, summary)
    out += ["", "## Mapping and window verification", ""]
    out += [f"- {m}" for m in results["verification"]]
    return "\n".join(out) + "\n"


def _chi_block(chi_s: str, r: Dict[str, Any], summary: bool) -> List[str]:
    out = [
        f"### chi = {chi_s} (M = {r['physics']['m']}, chi_actual = "
        f"{r['physics']['chi_actual']:.6f}, K2 = {r['physics']['k2']:.5f})",
        "",
        f"**JAX vs C++: {_ok(r['real']['passed'])}**",
        "",
        *battery_table(r["real"]),
        "",
        f"False-alarm control (C++ 0000-0002 vs C++ 0010-0012): "
        f"**{_ok(r['false_alarm']['passed'])}**",
        "",
    ]
    if not summary:
        out += [*battery_table(r["false_alarm"]), ""]
    out += [
        f"- status counts (JAX): {r['status_counts']}; successes JAX "
        f"{r['n_success_jax']}/{r['real']['n_a']}, C++ {r['n_success_cpp']}/"
        f"{r['real']['n_b']}",
        f"- -log10 Phi among successes (median [Q1, Q3]): JAX "
        f"{r['logphi']['jax']}, C++ {r['logphi']['cpp']}",
        f"- timings: JAX relax {r['seconds_relax']:.1f} s "
        f"(cached: {r['cached']}), analysis {r['seconds_analysis']:.1f} s",
        "",
    ]
    return out


def _power_blocks(results: Dict[str, Any], summary: bool) -> List[str]:
    out = ["## Power control (JAX with K2 x 0.9 vs C++)", ""]
    for chi_s, p in results["power"].items():
        failed = [k for k, v in p["ks"].items() if not v["pass"]]
        out.append(
            f"- chi = {chi_s} ({p['n_a']} JAX configs): **{_ok(p['passed'])}** "
            f"(expected FAIL); success {p['frac_a']:.3f} vs {p['frac_b']:.3f} "
            f"(p {_fmt(p['prop']['p'])}); failing KS: {failed}; S(k) worst |z| = "
            f"{p['sk']['worst_abs_z']:.1f}"
        )
    return out + [""]
