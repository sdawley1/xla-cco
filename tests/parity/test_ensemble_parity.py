#!/usr/bin/env python3

"""
Ensemble parity (level 3): JAX L-BFGS vs the C++ NLopt L-BFGS ensemble.

For chi in {0.05, 0.15, 0.25, 0.35, 0.45} (d = 3, N = 100, phi = 0.05) the 1200
initial configurations of C++ tasks 0000-0002 are relaxed with the production
CLI (default L-BFGS: C++ 3-pass schedule, energy_tol 0), and the ensembles are
compared with the battery of ``tests/ensemble_stats.py``. Both calibrated
controls must behave before the JAX comparison counts:

- false alarm: C++ tasks 0000-0002 vs C++ tasks 0010-0012 must PASS;
- power: JAX with K2 x 0.9 (chi 0.25 and 0.45, 400 configs) must FAIL.

If the real comparison fails while both controls behave, escalate; do NOT tune
anything toward the C++ results.

Env: ``XLA_CCO_ENSEMBLE_LIMIT`` (configs per task, default 400 = full; smaller
values give a smoke run with no statistical assertions) and
``XLA_CCO_ENSEMBLE_DEVICE`` (``gpu`` default, ``cpu`` for debugging).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import datetime
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

# ----- numerics -----
import jax
import numpy as np
import pytest

# ----- project -----
import ensemble_report as er
import ensemble_run as run
import ensemble_stats as es
from tolerances import KS_PMIN, SK_ZMAX

pytestmark = [pytest.mark.della, pytest.mark.slow]

# ============================================================
# CONSTANTS
# ============================================================

ALPHA = 0.01
POWER_CHIS = (0.25, 0.45)
N_TESTS = 2 * len(run.CHIS) * (1 + len(es.KS_NAMES))  # real + false-alarm control
REPORT_DIR = Path(__file__).resolve().parents[2] / "parity_reports"

# ============================================================
# HELPERS
# ============================================================


def _git(*args: str) -> str:
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


def _logphi(sample: es.EnsembleSample) -> str:
    x = es.neg_log10_phi(sample.energy[sample.success])
    if x.size == 0:
        return "no successes"
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"


def _cpp_floor(cpp: es.EnsembleSample) -> Dict[str, Any]:
    """C++ Phi_final quantiles and success counts at fixed thresholds."""
    e = cpp.energy
    q = np.percentile(e, [50, 90, 99, 100])
    return {
        "quantiles": dict(
            zip(("median", "q90", "q99", "max"), map(float, q), strict=True)
        ),
        "counts": [int((e < tol).sum()) for tol in (1e-20, 2e-20, 5e-20)],
    }


def _verify_all() -> List[str]:
    """Verify init-config mapping and the K window for every task and chi."""
    lines = []
    for chi in run.CHIS:
        maps = [run.verify_mapping(chi, t) for t in (*run.REAL_TASKS, *run.CTRL_TASKS)]
        wins = [run.verify_window(chi, t) for t in (*run.REAL_TASKS, *run.CTRL_TASKS)]
        worst = max(m["max_wrapped_diff"] for m in maps)
        exact = all(m["bit_identical"] for m in maps)
        lines.append(
            f"chi {chi}: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 "
            f"configs: max wrapped diff {worst:.1e}, bit-identical {exact}; "
            f"K2 = {wins[0]['k2']:.6f} (log {wins[0]['k2_log']}), chi = "
            f"{wins[0]['chi_actual']:.6f} (log {wins[0]['chi_log']}), M = "
            f"{wins[0]['m']}: window and chi match all 6 engine.logs"
        )
        assert worst <= 1e-15, f"mapping mismatch at chi {chi}: {worst}"
    return lines


def _chi_block(chi: float, limit: int, p_prop: float, p_ks: float) -> Dict[str, Any]:
    t0 = time.perf_counter()
    an = run.make_analyzer(chi)
    cpp = run.cpp_sample(chi, run.REAL_TASKS, limit, an)
    ctrl = run.cpp_sample(chi, run.CTRL_TASKS, limit, an)
    t1 = time.perf_counter()
    jx, infos = run.jax_sample(chi, run.REAL_TASKS, 1.0, limit, an)
    t2 = time.perf_counter()
    real = es.run_battery(jx, cpp, p_prop, p_ks, SK_ZMAX, "less")
    fa = es.run_battery(cpp, ctrl, p_prop, p_ks, SK_ZMAX)
    return {
        "physics": run.physics(chi),
        "real": er.battery_dict(real, an.k_shell, N_TESTS),
        "false_alarm": er.battery_dict(fa, an.k_shell, N_TESTS),
        "status_counts": run.status_counts(infos),
        "cpp_floor": _cpp_floor(cpp),
        "n_success_jax": int(jx.success.sum()),
        "n_success_cpp": int(cpp.success.sum()),
        "logphi": {"jax": _logphi(jx), "cpp": _logphi(cpp)},
        "seconds_relax": sum(i["wall_seconds"] for i in infos),
        "cached": all(i["cached"] for i in infos),
        "seconds_analysis": (t1 - t0)
        + (t2 - t1)
        - sum(i["wall_seconds"] for i in infos)
        if not all(i["cached"] for i in infos)
        else (t2 - t0),
        "_cpp": cpp,
        "_an": an,
    }


def _power_block(
    chi: float, cpp: es.EnsembleSample, an: Any, limit: int, p_prop: float, p_ks: float
) -> Dict[str, Any]:
    jx, infos = run.jax_sample(chi, (0,), run.WRONG_K_SCALE, limit, an)
    res = es.run_battery(jx, cpp, p_prop, p_ks, SK_ZMAX)
    out = er.battery_dict(res, an.k_shell, N_TESTS)
    out["status_counts"] = run.status_counts(infos)
    return out


def _controls(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for chi_s, r in results["chi"].items():
        ok = r["false_alarm"]["passed"]
        rows.append(
            {
                "kind": "false alarm (C++ vs C++)",
                "chi": chi_s,
                "expected": "PASS",
                "outcome": er._ok(ok),
                "behaves": ok,
            }
        )
    for chi_s, p in results["power"].items():
        rows.append(
            {
                "kind": "power (JAX K2 x 0.9 vs C++)",
                "chi": chi_s,
                "expected": "FAIL",
                "outcome": er._ok(p["passed"]),
                "behaves": not p["passed"],
            }
        )
    return rows


def _write_report(results: Dict[str, Any]) -> None:
    date = results["meta"]["date"].replace("-", "")
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f"ensemble_{date}_tables.md").write_text(er.render(results))
    (REPORT_DIR / "ensemble_results.json").write_text(json.dumps(results, indent=1))
    (REPORT_DIR / "ensemble_summary.md").write_text(er.render(results, summary=True))
    print(er.render(results), flush=True)


@pytest.fixture(scope="module")
def results() -> Dict[str, Any]:
    """Compute (or load from cache) every ensemble and battery once."""
    t0 = time.perf_counter()
    limit = run.limit_from_env()
    p_prop, p_ks = es.bonferroni_thresholds(ALPHA, N_TESTS, KS_PMIN)
    verification = _verify_all()
    chis: Dict[str, Any] = {}
    keep: Dict[float, Any] = {}
    for chi in run.CHIS:
        blk = _chi_block(chi, limit, p_prop, p_ks)
        keep[chi] = (blk.pop("_cpp"), blk.pop("_an"))
        chis[f"{chi:g}"] = blk
    power = {
        f"{chi:g}": _power_block(chi, *keep[chi], limit, p_prop, p_ks)
        for chi in POWER_CHIS
    }
    res: Dict[str, Any] = {"chi": chis, "power": power, "verification": verification}
    res["controls"] = _controls(res)
    behave = all(c["behaves"] for c in res["controls"])
    real_ok = all(r["real"]["passed"] for r in chis.values())
    res["verdict"] = (
        ("PASS" if real_ok else "FAIL (controls behave: ESCALATE)")
        if behave
        else "CONTROLS MISBEHAVE (comparison not valid)"
    )
    res["meta"] = {
        "date": datetime.date.today().isoformat(),
        "git": _git("rev-parse", "--short", "HEAD"),
        "git_full": _git("rev-parse", "HEAD"),
        "src_tree": _git("rev-parse", "HEAD:src"),
        "src_dirty": bool(_git("status", "--porcelain", "--", "src") != "unknown"),
        "src_sha256": run.src_hash(),
        "job_id": __import__("os").environ.get("SLURM_JOB_ID", "none (interactive)"),
        "device": run.device_from_env(),
        "jax_version": jax.__version__,
        "limit": limit,
        "tol": run.CPP_PARITY_TOL,
        "alpha": ALPHA,
        "n_tests": N_TESTS,
        "p_prop": p_prop,
        "p_ks": p_ks,
        "zmax": SK_ZMAX,
        "wall_seconds": time.perf_counter() - t0,
    }
    _write_report(res)
    return res


def _full_run_only(results: Dict[str, Any]) -> None:
    if results["meta"]["limit"] < run.PER_TASK:
        pytest.skip("subset run (XLA_CCO_ENSEMBLE_LIMIT): smoke only")


# ============================================================
# TESTS
# ============================================================


def test_all_chi_computed(results: Dict[str, Any]) -> None:
    """Every chi produced a real, false-alarm and (where asked) power result."""
    assert set(results["chi"]) == {f"{c:g}" for c in run.CHIS}
    assert set(results["power"]) == {f"{c:g}" for c in POWER_CHIS}
    assert all("passed" in r["real"] for r in results["chi"].values())


@pytest.mark.parametrize("chi", run.CHIS)
def test_false_alarm_control_passes(results: Dict[str, Any], chi: float) -> None:
    """C++ 0000-0002 vs C++ 0010-0012 must pass the battery (no false alarm)."""
    _full_run_only(results)
    fa = results["chi"][f"{chi:g}"]["false_alarm"]
    assert fa["passed"], fa


@pytest.mark.parametrize("chi", POWER_CHIS)
def test_power_control_fails(results: Dict[str, Any], chi: float) -> None:
    """JAX relaxed with K2 x 0.9 must FAIL the battery against C++."""
    _full_run_only(results)
    p = results["power"][f"{chi:g}"]
    assert p["n_a"] >= run.PER_TASK
    detail = (
        f"per-statistic results at chi {chi}: success p={p['prop']['p']:.3g}, "
        f"KS {p['ks']}, S(k) {p['sk']}"
    )
    assert not p["passed"], f"battery has no power at chi {chi}: {detail}"
    # success fraction alone (0 successes) would fail trivially: demand more
    assert p["sk"]["worst_abs_z"] >= SK_ZMAX, f"S(k) did not detect: {detail}"
    assert any(not v["pass"] for v in p["ks"].values()), f"KS blind: {detail}"


@pytest.mark.parametrize("chi", run.CHIS)
def test_jax_matches_cpp(results: Dict[str, Any], chi: float) -> None:
    """JAX L-BFGS ensemble is statistically indistinguishable from C++ (gated)."""
    _full_run_only(results)
    bad = [c for c in results["controls"] if not c["behaves"]]
    assert not bad, f"controls misbehave, comparison invalid: {bad}"
    real = results["chi"][f"{chi:g}"]["real"]
    assert real["passed"], (
        f"REAL COMPARISON FAILED with controls behaving at chi {chi}: ESCALATE, "
        f"do not tune toward C++: {real}"
    )
