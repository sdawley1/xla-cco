#!/usr/bin/env python3

"""
Hypothesis strategies for xla_cco property tests.

Shapes are drawn from fixed grids so jitted code does not recompile per
example.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any, Optional, Sequence, Tuple

# ----- numerics -----
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# ============================================================
# CONSTANTS
# ============================================================

N_GRID: Tuple[int, ...] = (1, 2, 7, 16, 40)
CHUNK_GRID: Tuple[Optional[int], ...] = (None, 1, 3, 8, 64)

# ============================================================
# STRATEGIES
# ============================================================


def dims() -> st.SearchStrategy[int]:
    """Spatial dimension d in {1, 2, 3}."""
    return st.sampled_from((1, 2, 3))


@st.composite
def fractional_configs(
    draw: Any,
    n_grid: Sequence[int] = N_GRID,
    dim: Optional[int] = None,
) -> np.ndarray:
    """Draw fractional positions (n, d) in [0, 1), n from ``n_grid``."""
    n = draw(st.sampled_from(tuple(n_grid)))
    d = draw(dims()) if dim is None else dim
    coord = st.floats(0.0, 1.0, exclude_max=True, allow_nan=False)
    return np.asarray(draw(hnp.arrays(np.float64, (n, d), elements=coord)))


@st.composite
def boxes(draw: Any, dim: Optional[int] = None) -> np.ndarray:
    """Draw a (d, d) basis: cubic scale or sheared lower-triangular, |det|>=0.1.

    Diagonal entries are in [0.5, 2] and off-diagonals in [-0.5, 0.5] so the
    cell stays reasonably reduced.
    """
    d = draw(dims()) if dim is None else dim
    diag = draw(
        hnp.arrays(np.float64, d, elements=st.floats(0.5, 2.0, allow_nan=False))
    )
    if draw(st.booleans()):
        return np.asarray(np.eye(d) * diag[0])
    off = draw(
        hnp.arrays(np.float64, (d, d), elements=st.floats(-0.5, 0.5, allow_nan=False))
    )
    basis = np.tril(off, -1) + np.diag(diag)
    assert abs(np.linalg.det(basis)) >= 0.1, "strategy bound violated"
    return np.asarray(basis)


@st.composite
def integer_kvectors(
    draw: Any,
    dim: Optional[int] = None,
    max_m: int = 30,
    max_abs: int = 4,
) -> np.ndarray:
    """Draw (m, d) int64 nonzero integer vectors with entries in [-max_abs, max_abs]."""
    d = draw(dims()) if dim is None else dim
    m = draw(st.integers(1, max_m))
    n = draw(hnp.arrays(np.int64, (m, d), elements=st.integers(-max_abs, max_abs)))
    n[~np.any(n != 0, axis=1)] = 1  # replace zero rows by (1, ..., 1)
    return np.asarray(n)


def chunk_sizes() -> st.SearchStrategy[Optional[int]]:
    """Chunk sizes for chunked k-sums (None = unchunked)."""
    return st.sampled_from(CHUNK_GRID)
