#!/usr/bin/env python3

"""
Pair statistics: nearest-neighbour distances, histograms and g2.

Distances use the minimum-image convention in length units with density 1.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import numpy as np
from jax import Array
from jaxtyping import Float

# ----- project -----
from xla_cco._types import BatchPositions, Positions
from xla_cco.box import Box

# ============================================================
# PAIR STATISTICS
# ============================================================


def nearest_neighbor_distances(s: Positions, box: Box) -> Float[Array, "n"]:
    """Return minimum-image nearest-neighbour distance of every particle."""
    raise NotImplementedError


def pair_distance_histogram(
    s: BatchPositions, box: Box, edges: np.ndarray
) -> np.ndarray:
    """Return summed pair-distance counts over the batch for bin ``edges``."""
    raise NotImplementedError


def g2_from_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    n_particles: int,
    n_configs: int,
    volume: float,
    dim: int,
) -> np.ndarray:
    """Normalize pair counts to g2(r) (shell volume, density N / V)."""
    raise NotImplementedError
