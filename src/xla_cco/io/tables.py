#!/usr/bin/env python3

"""
Tab-separated function-table writer matching the C++ output format.

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
# I/O
# ============================================================


def write_function_table(path: Path, columns: Sequence[np.ndarray]) -> None:
    """Write equal-length columns as rows, format ``%1.10e``, tab-separated."""
    raise NotImplementedError
