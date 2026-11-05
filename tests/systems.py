#!/usr/bin/env python3

"""
Small stealthy systems shared by the minimizer tests.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any, Tuple

# ----- numerics -----
import numpy as np

# ----- project -----
from xla_cco.box import cubic_box
from xla_cco.energy import StealthyEnergy, make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors

# ============================================================
# FACTORIES
# ============================================================


def small_system(n: int, d: int, k_outer: float) -> Tuple[Any, Any, StealthyEnergy]:
    """Return (box, kset, energy) for a cubic box with a (0, k_outer) window."""
    assert n >= 1 and d in (1, 2, 3), "bad system shape"
    box = cubic_box(n, d)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k_outer)
    return box, kset, make_stealthy_energy(box, kset)


def square_lattice(side: int) -> np.ndarray:
    """Return a side x side square lattice in fractional coordinates, (n, 2)."""
    g = np.arange(side) / side
    return np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
