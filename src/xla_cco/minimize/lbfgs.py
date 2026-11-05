#!/usr/bin/env python3

"""
Optax L-BFGS minimizer with the C++ three-pass rescale schedule.

Pass with rescale R optimizes x = R * s (R = 1e4, 1, 1e-2 by default,
matching RelaxStructure_NLOPT). Energies are dimensionless; positions are
fractional.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Tuple

# ----- numerics -----
import equinox as eqx

# ----- project -----
from xla_cco._types import Positions
from xla_cco.energy import AbstractEnergy
from xla_cco.minimize.result import RelaxResult

# ============================================================
# SETTINGS
# ============================================================


class LBFGSSettings(eqx.Module):
    """L-BFGS settings (every field static)."""

    memory: int = eqx.field(static=True, default=100)
    rescales: Tuple[float, ...] = eqx.field(static=True, default=(1e4, 1.0, 1e-2))
    max_evals_per_pass: int = eqx.field(static=True, default=500_000)
    energy_tol: float = eqx.field(static=True, default=0.0)
    grad_tol: float = eqx.field(static=True, default=0.0)
    stagnation_patience: int = eqx.field(static=True, default=50)
    max_linesearch_steps: int = eqx.field(static=True, default=30)


# ============================================================
# DRIVER
# ============================================================


def lbfgs(
    energy: AbstractEnergy, s0: Positions, settings: LBFGSSettings
) -> RelaxResult:
    """Relax one configuration with the three-pass L-BFGS schedule.

    Parameters
    ----------
    energy : AbstractEnergy
        Energy on fractional positions.
    s0 : Positions
        Initial fractional positions, float64, shape (n, d).
    settings : LBFGSSettings
        Static settings.
    """
    raise NotImplementedError
