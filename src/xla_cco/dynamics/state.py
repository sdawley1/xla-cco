#!/usr/bin/env python3

"""
Settings, tracked waves and per-lane state of the molecular dynamics (MD).

Conventions (frozen interface; see the MD plan):
- Cartesian, unwrapped positions r and velocities v; mass 1, k_B = 1.
  The stealthy energy is evaluated at fractional s = r @ inv(B) (never
  wrapped) and the force is F = -dPhi/ds @ inv(B)^T (Box.cartesian_forces).
- Phi = (1/V) sum_{half-space k} |rho_k|^2 (density 1; the C++ convention).
- One lane = one system at one temperature; batches vmap MDState over a
  leading axis. Every array leaf is float64 or integer.
- Winding of each tracked wave is accumulated on device in radians:
  wind_abs += |Delta theta|, wind_net += Delta theta, with
  Delta theta = atan2(Im(rho_new conj rho_old), Re(...)), counted only when
  S = |rho|^2 / N exceeds ``phase_floor`` at both ends (as
  experiments/scripts/batten2008_winding.cumulative_winding).
- Schemes: "verlet" (NVE velocity Verlet), "andersen" (velocity Verlet then
  Andersen resets), "baoab" (Langevin, Leimkuhler-Matthews splitting).
  Andersen "count" mode is the C++ rule (MD_System.cpp AndersonEvolve): an
  accumulator gains p per step and one uniformly chosen particle is redrawn
  from N(0, T) each time it reaches 1; "bernoulli" redraws one particle
  with probability p per step.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Optional, Tuple

# ----- numerics -----
import equinox as eqx
from jax import Array
from jaxtyping import Bool, Float, Int, UInt32

# ============================================================
# CONSTANTS
# ============================================================

SCHEMES: Tuple[str, ...] = ("verlet", "andersen", "baoab")
ANDERSEN_MODES: Tuple[str, ...] = ("count", "bernoulli")

# ============================================================
# CONTAINERS
# ============================================================


class MDSettings(eqx.Module):
    """Integration controls (all static).

    ``dt`` time step; ``scheme`` in SCHEMES; ``andersen_p`` reset rate per
    step and ``andersen_mode`` in ANDERSEN_MODES; ``gamma`` BAOAB friction
    (1/time); ``remove_com`` zeroes the total momentum when a segment begins
    (``run.begin_segment``) and at Maxwell-Boltzmann draws; ``chunk_steps``
    steps per jitted call; ``wind_every`` cadence (steps) of the winding
    update for the unconstrained waves (constrained waves: every step);
    ``record_steps`` absolute step numbers of the snapshots (sorted, at most
    256); ``record_rho_every`` / ``max_rho_records`` the dense rho trace (0:
    off); ``phase_floor`` on S = |rho|^2 / N; ``drift_tol`` aborts a "verlet"
    lane (STATUS_DRIFT) when |E - E0| exceeds drift_tol times the expected
    kinetic energy dof * T / 2. |E - E0| includes the bounded Verlet energy
    oscillation (~ (omega dt)^2 / 8 relative), not only secular drift, so dt
    must keep (omega_max dt)^2 well below drift_tol.
    """

    dt: float = eqx.field(static=True, default=0.01)
    scheme: str = eqx.field(static=True, default="verlet")
    andersen_p: float = eqx.field(static=True, default=0.01)
    andersen_mode: str = eqx.field(static=True, default="count")
    gamma: float = eqx.field(static=True, default=0.0)
    remove_com: bool = eqx.field(static=True, default=True)
    chunk_steps: int = eqx.field(static=True, default=10_000)
    wind_every: int = eqx.field(static=True, default=1)
    record_steps: Tuple[int, ...] = eqx.field(static=True, default=())
    record_rho_every: int = eqx.field(static=True, default=0)
    max_rho_records: int = eqx.field(static=True, default=0)
    phase_floor: float = eqx.field(static=True, default=1e-16)
    drift_tol: float = eqx.field(static=True, default=1e-3)


class Waves(eqx.Module):
    """Tracked density waves: the constrained set first, in KSet order.

    ``n_vectors`` (w, d) integer-valued reciprocal-basis coefficients,
    ``kvec`` (w, d) Cartesian wavevectors (n @ 2 pi inv(B)^T),
    ``constrained`` (w,) True for the first ``n_constrained`` rows, which
    equal the energy's unpadded ``n_vectors`` row by row (so their rho comes
    from the force pass at no extra cost).
    """

    n_vectors: Float[Array, "w d"]
    kvec: Float[Array, "w d"]
    constrained: Bool[Array, " w"]
    n_constrained: int = eqx.field(static=True)


class Snapshots(eqx.Module):
    """Fixed-capacity snapshots at ``MDSettings.record_steps`` (``count`` valid)."""

    step: Int[Array, " r"]
    time: Float[Array, " r"]
    wind_abs: Float[Array, "r w"]
    wind_net: Float[Array, "r w"]
    rho: Float[Array, "r w 2"]
    r: Float[Array, "r n d"]
    phi: Float[Array, " r"]
    kinetic: Float[Array, " r"]
    count: Int[Array, ""]


class RhoTrace(eqx.Module):
    """Dense rho of the tracked waves every ``record_rho_every`` steps."""

    rho: Float[Array, "q w 2"]
    time: Float[Array, " q"]
    count: Int[Array, ""]


class MDState(eqx.Module):
    """State of one lane (vmap over a leading axis for a batch).

    ``key`` is a raw uint32 PRNG key (jax.random.PRNGKey), so checkpoints
    are plain arrays. ``reset_acc`` is the Andersen count accumulator.
    ``rho_prev`` (w, 2) the last rho used for winding (COM-corrected).
    ``e0`` total energy at the segment start; ``max_drift`` the largest
    |E - e0| seen in the segment. ``sum_*`` running sums since the last
    ``begin_segment`` with reset_sums (``n_sum`` samples, one per step):
    kinetic, Phi, Phi^2 and S of every tracked wave. ``status`` is
    minimize.result.STATUS_RUNNING (0) or STATUS_DRIFT.
    """

    r: Float[Array, "n d"]
    v: Float[Array, "n d"]
    f: Float[Array, "n d"]
    phi: Float[Array, ""]
    temperature: Float[Array, ""]
    key: UInt32[Array, " 2"]
    step: Int[Array, ""]
    time: Float[Array, ""]
    reset_acc: Float[Array, ""]
    rho_prev: Float[Array, "w 2"]
    wind_abs: Float[Array, " w"]
    wind_net: Float[Array, " w"]
    n_valid: Int[Array, " w"]
    e0: Float[Array, ""]
    max_drift: Float[Array, ""]
    sum_kinetic: Float[Array, ""]
    sum_phi: Float[Array, ""]
    sum_phi2: Float[Array, ""]
    sum_s: Float[Array, " w"]
    n_sum: Int[Array, ""]
    status: Int[Array, ""]
    snap: Snapshots
    trace: RhoTrace


class MDResult(eqx.Module):
    """Final (batched) state, its snapshots and the dense rho trace."""

    state: MDState
    snapshots: Snapshots
    trace: Optional[RhoTrace]
