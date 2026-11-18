#!/usr/bin/env python3

"""
Result containers and status codes for the minimizers.

Status codes: 0 running, 1 energy below tolerance, 2 step below floor,
3 evaluation budget exhausted, 4 stagnated, 5 line-search failure,
6 gradient norm below tolerance (added in T7; L-BFGS only), 7 MD energy
drift above tolerance (xla_cco.dynamics).
Positions are fractional; energies are dimensionless (density 1).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Optional

# ----- numerics -----
import equinox as eqx
from jax import Array
from jaxtyping import Float, Int

# ----- project -----
from xla_cco._types import Positions

# ============================================================
# CONSTANTS
# ============================================================

STATUS_RUNNING: int = 0
STATUS_ENERGY_TOL: int = 1
STATUS_STEP_FLOOR: int = 2
STATUS_BUDGET: int = 3
STATUS_STAGNATED: int = 4
STATUS_LINESEARCH_FAIL: int = 5
STATUS_GRAD_TOL: int = 6
STATUS_DRIFT: int = 7  # MD: total-energy drift above MDSettings.drift_tol (NVE)

# ============================================================
# CONTAINERS
# ============================================================


class Trajectory(eqx.Module):
    """Recorded trajectory (fixed capacity t, ``count`` valid rows).

    Notes
    -----
    ``s`` rows are always wrapped into [0, 1) (``_loop._record``), for both
    descent and L-BFGS, although the iterates themselves are not wrapped
    during the run (the C++ parameters are not wrapped either).
    """

    s: Float[Array, "t n d"]
    energy: Float[Array, "t"]
    step: Float[Array, "t"]
    iteration: Int[Array, "t"]
    count: Int[Array, ""]


class RelaxResult(eqx.Module):
    """Outcome of one relaxation (batched: leaves gain a leading axis)."""

    s: Positions
    energy: Float[Array, ""]
    grad_norm: Float[Array, ""]
    n_evals: Int[Array, ""]
    n_iters: Int[Array, ""]
    status: Int[Array, ""]
    trajectory: Optional[Trajectory]
