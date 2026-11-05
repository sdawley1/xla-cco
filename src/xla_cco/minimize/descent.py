#!/usr/bin/env python3

"""
C++-faithful adaptive gradient descent in fractional coordinates.

Default rule halves the step on energy increase (grow = 1.0); the PRE 112,
045310 (2025) rule is grow = 2.0. Energies are dimensionless; the step is
in fractional-coordinate units.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import equinox as eqx

# ----- project -----
from xla_cco._types import Positions
from xla_cco.energy import AbstractEnergy
from xla_cco.minimize.result import RelaxResult

# ============================================================
# SETTINGS
# ============================================================


class DescentSettings(eqx.Module):
    """Adaptive-descent settings (every field static)."""

    step_init: float = eqx.field(static=True, default=1e-3)
    step_min: float = eqx.field(static=True, default=1e-15)
    shrink: float = eqx.field(static=True, default=0.5)
    grow: float = eqx.field(static=True, default=1.0)
    rel_tol: float = eqx.field(static=True, default=0.01)
    max_evals: int = eqx.field(static=True, default=10_000)
    energy_tol: float = eqx.field(static=True, default=0.0)
    record_every: int = eqx.field(static=True, default=0)
    max_records: int = eqx.field(static=True, default=0)


# ============================================================
# DRIVER
# ============================================================


def adaptive_descent(
    energy: AbstractEnergy, s0: Positions, settings: DescentSettings
) -> RelaxResult:
    """Relax one configuration by adaptive gradient descent.

    Parameters
    ----------
    energy : AbstractEnergy
        Energy on fractional positions.
    s0 : Positions
        Initial fractional positions, float64, shape (n, d).
    settings : DescentSettings
        Static settings.
    """
    raise NotImplementedError
