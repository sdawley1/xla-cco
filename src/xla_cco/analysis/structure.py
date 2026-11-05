#!/usr/bin/env python3

"""
Structure factor S(k) = |rho(k)|^2 / N and its isotropic average.

Wavevectors are given as integer coefficients in the reciprocal basis.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Tuple

# ----- numerics -----
import numpy as np
from jax import Array
from jaxtyping import Float

# ----- project -----
from xla_cco._types import BatchPositions, Positions
from xla_cco.box import Box

# ============================================================
# STRUCTURE FACTOR
# ============================================================


def structure_factor(s: Positions, n_vectors: Float[Array, "m d"]) -> Float[Array, "m"]:
    """Return S(n) = |rho(n)|^2 / N for integer vectors ``n_vectors``."""
    raise NotImplementedError


def isotropic_structure_factor(
    s: BatchPositions,
    box: Box,
    k_dir_max: float = 5.0,
    k_max: float = 400.0,
    bin_width: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the binned isotropic S(|k|) averaged over a batch.

    Parameters
    ----------
    s : BatchPositions
        Fractional positions (b, n, d).
    box : Box
        Periodic box.
    k_dir_max : float
        Cutoff up to which individual directions are kept.
    k_max : float
        Maximum |k| (inverse length).
    bin_width : float
        Radial bin width (inverse length).
    """
    raise NotImplementedError
