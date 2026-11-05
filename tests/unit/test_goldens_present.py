#!/usr/bin/env python3

"""
Every expected C++ golden file exists and validates against tests/golden.py.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json

# ----- numerics / testing -----
import numpy as np
import pytest

# ----- local modules -----
from golden import GOLDEN_DIR, load_golden

# ============================================================
# CONSTANTS
# ============================================================

EXPECTED = (
    [f"kernel_d{d}_chi{c}" for d in (1, 2, 3) for c in (0.05, 0.25, 0.45)]
    + ["kernel_d3_cpp_ic", "kernel_boundary_on_shell", "kernel_boundary_above_shell"]
    + ["traj_d2_n64_chi0.25", "traj_d3_n100_chi0.05", "sk_d3_chi0.05"]
    + ["lbfgs_d3_chi0.05", "lbfgs_d3_chi0.25"]
)

# ============================================================
# TESTS
# ============================================================


@pytest.mark.parametrize("name", EXPECTED)
def test_golden_present_and_valid(name: str) -> None:
    """File exists (never skipped here) and passes the schema validator."""
    assert (GOLDEN_DIR / f"{name}.npz").is_file(), f"{name}.npz missing"
    data = load_golden(name)
    params = data["params"]
    assert params["d"] in (1, 2, 3) and params["phi"] == 0.05


def test_manifest_lists_every_golden() -> None:
    """MANIFEST.json carries a SHA-256 for each expected file."""
    man = json.loads((GOLDEN_DIR / "MANIFEST.json").read_text())
    for name in EXPECTED:
        digest = man["files"][f"{name}.npz"]
        assert isinstance(digest, str) and len(digest) == 64


def test_kernel_d3_cpp_ic_matches_engine_log() -> None:
    """C++ engine.log of the cco run reports M = 13 and chi = 0.043771."""
    data = load_golden("kernel_d3_cpp_ic")
    assert int(data["m"]) == 13
    assert float(data["chi"]) == pytest.approx(0.043771, abs=5e-7)


def test_total_size_within_budget() -> None:
    """tests/data/cpp stays within the 5 MB budget."""
    total = sum(p.stat().st_size for p in GOLDEN_DIR.rglob("*") if p.is_file())
    assert total <= 5 * 1024 * 1024
    assert np.isfinite(total)
