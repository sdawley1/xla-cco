#!/usr/bin/env python3

"""
Smoke test: the benchmark script runs end to end with tiny sizes on the CPU.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json
import os
import subprocess
import sys
from pathlib import Path

# ----- testing -----
import pytest

# ============================================================
# TESTS
# ============================================================

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_energy.py"


@pytest.mark.low
def test_bench_energy_smoke(tmp_path: Path) -> None:
    out = tmp_path / "res.json"
    cmd = [sys.executable, str(SCRIPT), "--device", "cpu", "--small"]
    cmd += ["--batch", "4", "--reps", "1", "--relax-batch", "2"]
    cmd += ["--relax-evals", "5", "--no-isolate", "--out", str(out)]
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr[-1500:]
    data = json.loads(out.read_text())
    rows = data["results"]
    assert {r["variant"] for r in rows} >= {"baseline", "tables"}
    assert all(r["peak_mem_mb"] > 0 for r in rows)
    vag = [r for r in rows if r["kind"] == "value_and_grad" and r["case"] == "A"]
    assert all(r["configs_per_s"] > 0 for r in vag)
    # table path agrees with the trig baseline on the same configurations
    by = {(r["chi"], r["variant"]): r["energy_mean"] for r in vag}
    for chi in (0.05, 0.45):
        assert abs(by[(chi, "tables")] - by[(chi, "baseline")]) <= 1e-10 * abs(
            by[(chi, "baseline")]
        )
