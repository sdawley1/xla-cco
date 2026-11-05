#!/usr/bin/env python3

"""
Tab-separated function-table writer matching the C++ output format.

Reproduces ``WriteFunction`` (GeometryVector.cpp:146-172): no header, one
line per row, four fields (C++ MaxDimension) each ``%1.10e`` followed by a
TAB, then a newline.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path
from typing import Sequence

# ----- numerics -----
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================

N_FIELDS: int = 4  # C++ MaxDimension

# ============================================================
# I/O
# ============================================================


def write_function_table(path: Path, columns: Sequence[np.ndarray]) -> None:
    """Write equal-length 1-D columns as C++-format rows.

    Missing columns (up to four) are padded with 0.0.

    Raises
    ------
    ValueError
        On no columns, more than four, ragged or non-1-D columns, or
        non-finite values.
    """
    if not 1 <= len(columns) <= N_FIELDS:
        raise ValueError(f"need 1..{N_FIELDS} columns, got {len(columns)}")
    cols = [np.asarray(c, dtype=np.float64) for c in columns]
    if any(c.ndim != 1 for c in cols):
        raise ValueError("columns must be 1-D")
    n_rows = cols[0].size
    if any(c.size != n_rows for c in cols):
        raise ValueError("columns have unequal lengths")
    table = np.zeros((n_rows, N_FIELDS), dtype=np.float64)
    for j, c in enumerate(cols):
        table[:, j] = c
    if not np.isfinite(table).all():
        raise ValueError("non-finite value in table")
    line = "%1.10e\t" * N_FIELDS + "\n"
    text = (line * n_rows) % tuple(table.ravel().tolist())
    Path(path).write_bytes(text.encode("ascii"))
