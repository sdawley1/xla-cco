#!/usr/bin/env python3

"""
ConfigPack reader/writer (C++ binary format) and fractional stacking.

Positions in a file are relative (fractional) to a unit-volume basis; the
box is the unit basis times N^(1/d).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ----- numerics -----
import numpy as np

# ============================================================
# CONTAINERS
# ============================================================


@dataclasses.dataclass(frozen=True)
class PackedConfig:
    """One configuration as stored in a ConfigPack file.

    Attributes
    ----------
    basis : np.ndarray
        (d, d) float64 unit-volume lattice basis, rows = lattice vectors.
    relative : np.ndarray
        (N, d) float64 fractional positions.
    cell_size : float
        Cell size stored in the file.
    names : Tuple[bytes, ...]
        Per-particle name tags.
    """

    basis: np.ndarray
    relative: np.ndarray
    cell_size: float
    names: Tuple[bytes, ...]


# ============================================================
# I/O
# ============================================================


def read_configpack(
    prefix: Path, indices: Optional[Sequence[int]] = None
) -> List[PackedConfig]:
    """Read configurations from a ConfigPack file set.

    Parameters
    ----------
    prefix : Path
        File prefix of the pack.
    indices : Optional[Sequence[int]]
        Configurations to read (None = all).

    Raises
    ------
    ValueError
        On truncated or malformed files.
    """
    raise NotImplementedError


def write_configpack(prefix: Path, configs: Sequence[PackedConfig]) -> None:
    """Write configurations as a ConfigPack file set."""
    raise NotImplementedError


def stack_fractional(
    configs: Sequence[PackedConfig],
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack configurations into ((B, N, d) fractional, (d, d) unit basis).

    Raises
    ------
    ValueError
        On mixed N or d, or a non-unit-volume basis.
    TypeError
        On non-float64 positions.
    """
    raise NotImplementedError
