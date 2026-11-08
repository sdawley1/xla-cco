#!/usr/bin/env python3

"""
Low-temperature molecular dynamics for stealthy collective-coordinate potentials.

Velocity Verlet (NVE), Andersen resets (C++-faithful count mode) and
Langevin BAOAB in Cartesian coordinates (mass 1, k_B = 1), with on-device
winding of tracked density waves. See ``state`` for the conventions.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- project -----
from xla_cco.dynamics.checkpoint import load_state, save_state
from xla_cco.dynamics.integrators import andersen_reset, baoab_step, verlet_step
from xla_cco.dynamics.observables import (
    center_of_mass,
    com_correct,
    wave_densities,
    winding_update,
)
from xla_cco.dynamics.run import (
    begin_segment,
    init_state,
    make_force_fn,
    make_waves,
    md_chunk,
    select_dt,
    simulate,
)
from xla_cco.dynamics.state import (
    MDResult,
    MDSettings,
    MDState,
    RhoTrace,
    Snapshots,
    Waves,
)
from xla_cco.dynamics.thermal import (
    harmonic_sample,
    kinetic_energy,
    kinetic_temperature,
    maxwell_boltzmann,
)

__all__ = [
    "MDResult", "MDSettings", "MDState", "RhoTrace", "Snapshots", "Waves",
    "andersen_reset", "baoab_step", "begin_segment", "center_of_mass",
    "com_correct", "harmonic_sample", "init_state", "kinetic_energy",
    "kinetic_temperature", "load_state", "make_force_fn", "make_waves",
    "maxwell_boltzmann", "md_chunk", "save_state", "select_dt", "simulate",
    "verlet_step", "wave_densities", "winding_update",
]  # fmt: skip
