#!/usr/bin/env python3

"""
Batched relaxation with host-side compaction of converged configurations.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Union

# ----- project -----
from xla_cco._types import BatchPositions
from xla_cco.energy import AbstractEnergy
from xla_cco.minimize.descent import DescentSettings
from xla_cco.minimize.lbfgs import LBFGSSettings
from xla_cco.minimize.result import RelaxResult

# ============================================================
# BATCHED DRIVER
# ============================================================


def relax(
    energy: AbstractEnergy,
    s0: BatchPositions,
    settings: Union[DescentSettings, LBFGSSettings],
    chunk_iters: int = 2_000,
) -> RelaxResult:
    """Relax a batch of configurations; result leaves have a leading batch axis.

    Parameters
    ----------
    energy : AbstractEnergy
        Energy on fractional positions.
    s0 : BatchPositions
        Initial fractional positions, float64, shape (b, n, d).
    settings : Union[DescentSettings, LBFGSSettings]
        Minimizer settings; the type selects the minimizer.
    chunk_iters : int
        Device iterations between host-side compaction steps.
    """
    raise NotImplementedError
