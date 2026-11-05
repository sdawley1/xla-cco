#!/usr/bin/env python3

"""
Tests of the golden-file loader and schema validator.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import numpy as np
import pytest

# ----- project -----
import golden

# ============================================================
# TESTS
# ============================================================


def _kernel() -> dict:
    s = np.zeros((2, 3, 2))
    return {
        "params": np.array('{"d": 2}'),
        "n": np.ones((4, 2), dtype=np.int64),
        "m": np.array(4),
        "chi": np.array(0.1),
        "s": s,
        "energy": np.zeros(2),
        "forces": s.copy(),
    }


def test_valid_kernel_golden_passes() -> None:
    assert golden.validate_golden("kernel_a", _kernel()) == {"d": 2}


def test_missing_key_and_bad_dtype_rejected() -> None:
    d = _kernel()
    del d["chi"]
    with pytest.raises(AssertionError, match="missing"):
        golden.validate_golden("kernel_a", d)
    d = _kernel()
    d["s"] = d["s"].astype(np.float32)
    with pytest.raises(AssertionError, match="float64"):
        golden.validate_golden("kernel_a", d)


def test_inconsistent_m_rejected() -> None:
    d = _kernel()
    d["m"] = np.array(5)
    with pytest.raises(AssertionError, match="m != len"):
        golden.validate_golden("kernel_a", d)


def test_absent_file_skips() -> None:
    with pytest.raises(pytest.skip.Exception):
        golden.load_golden("kernel_does_not_exist")
