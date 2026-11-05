#!/usr/bin/env python3

"""
Loader and schema validator for C++ golden ``.npz`` files.

Schema (tests/data/cpp/):
  kernel_*.npz  params, n (M,d) int64, m, chi, s (B,N,d), energy (B,),
                forces (B,N,d)
  traj_*.npz    params, s0 (B,N,d), iters (T,), s (T,B,N,d), energy (T,B),
                horizon (B,)
  sk_*.npz      params, s (B,N,d), table (R,4)
  lbfgs_*.npz   params, s0, energy_final (B,), success (B,)

``params`` is a JSON string. A missing file skips the test.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json
from pathlib import Path
from typing import Any, Dict

# ----- numerics -----
import numpy as np
import pytest

# ============================================================
# CONSTANTS
# ============================================================

GOLDEN_DIR: Path = Path(__file__).parent / "data" / "cpp"
SCHEMAS: Dict[str, Dict[str, str]] = {
    "kernel": {
        "params": "str",
        "n": "i2",
        "m": "i0",
        "chi": "f0",
        "s": "f3",
        "energy": "f1",
        "forces": "f3",
    },
    "traj": {
        "params": "str",
        "s0": "f3",
        "iters": "i1",
        "s": "f4",
        "energy": "f2",
        "horizon": "i1",
    },
    "sk": {"params": "str", "s": "f3", "table": "f2"},
    "lbfgs": {
        "params": "str",
        "s0": "f3",
        "energy_final": "f1",
        "success": "b1",
    },
}

# ============================================================
# VALIDATION
# ============================================================


def _check_key(name: str, key: str, arr: np.ndarray, code: str) -> None:
    """Assert dtype kind and ndim of one array against its schema code."""
    kind, ndim = code[0], (0 if code == "str" else int(code[1:]))
    if code == "str":
        assert arr.ndim == 0 and arr.dtype.kind in "US", f"{name}:{key} not a string"
        return
    assert arr.ndim == ndim, f"{name}:{key} ndim {arr.ndim} != {ndim}"
    kinds = {"i": "iu", "f": "f", "b": "biu"}[kind]
    assert arr.dtype.kind in kinds, f"{name}:{key} dtype {arr.dtype}"
    if kind == "i" and key == "n":
        assert arr.dtype == np.int64, f"{name}:{key} must be int64"
    if kind == "f":
        assert arr.dtype == np.float64, f"{name}:{key} must be float64"


def validate_golden(name: str, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Validate arrays against the schema for ``name``; return parsed params.

    Raises
    ------
    AssertionError
        On a missing key, wrong dtype/ndim, or inconsistent shapes.
    """
    prefix = name.split("_", 1)[0]
    assert prefix in SCHEMAS, f"unknown golden kind {prefix!r}"
    schema = SCHEMAS[prefix]
    missing = set(schema) - set(data)
    assert not missing, f"{name}: missing keys {sorted(missing)}"
    for key, code in schema.items():
        _check_key(name, key, data[key], code)
    if prefix == "kernel":
        assert data["n"].shape[0] == int(data["m"]), f"{name}: m != len(n)"
        assert data["forces"].shape == data["s"].shape, f"{name}: forces shape"
        assert data["energy"].shape[0] == data["s"].shape[0], f"{name}: batch"
    if prefix == "traj":
        t, b = data["energy"].shape
        assert data["s"].shape[:2] == (t, b), f"{name}: s vs energy shape"
        assert data["iters"].shape == (t,), f"{name}: iters length"
        assert data["horizon"].shape == (b,), f"{name}: horizon length"
    if prefix == "sk":
        assert data["table"].shape[1] == 4, f"{name}: table needs 4 columns"
    if prefix == "lbfgs":
        assert data["energy_final"].shape == data["success"].shape, f"{name}: shapes"
    params = json.loads(str(data["params"]))
    assert isinstance(params, dict), f"{name}: params must be a JSON object"
    return params


def load_golden(name: str) -> Dict[str, Any]:
    """Load and validate ``tests/data/cpp/<name>.npz``.

    Returns a dict of arrays plus ``"params"`` replaced by the parsed JSON.
    Calls ``pytest.skip`` with a clear reason when the file is absent.
    """
    path = GOLDEN_DIR / f"{name}.npz"
    if not path.is_file():
        pytest.skip(
            f"golden file {path} absent; generate with tools/cpp_reference "
            "(needs the C++ binary on Della)"
        )
    with np.load(path, allow_pickle=False) as npz:
        data = {k: npz[k] for k in npz.files}
    params = validate_golden(name, data)
    out: Dict[str, Any] = dict(data)
    out["params"] = params
    return out
