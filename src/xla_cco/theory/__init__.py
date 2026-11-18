#!/usr/bin/env python3

"""
Theory layer: configurational entropy of the stealthy ground-state manifold
under the pseudo-hard-sphere (PHS) ansatz and the low-temperature equations of
state (Torquato, Zhang & Stillinger, PRX 5, 021020 (2015)).

Modules: ``geometry`` (unit balls, alpha_d), ``hardsphere`` (Percus-Yevick
solver and analytic oracles), ``phs`` (b(d), c_K, S(q; chi), pair entropy
g(chi)), ``eos`` (fixed-K and fixed-chi equations of state). Pure
NumPy/SciPy; no JAX.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- project -----
from xla_cco.theory.eos import (
    compressibility_fixed_k,
    dp_dn_fixed_k,
    manifold_entropy_fixed_chi,
    pressure_fixed_k,
    z_fixed_chi,
)
from xla_cco.theory.geometry import intersection_fraction, unit_ball_volume
from xla_cco.theory.hardsphere import (
    PYSolution,
    c_py_3d,
    contact_value,
    g2_exact_1d,
    g2_low_density,
    solve_py,
)
from xla_cco.theory.phs import (
    b_coefficient,
    c_k,
    pair_entropy_derivs,
    pair_entropy_ratio,
    structure_factor,
)

__all__ = [
    "PYSolution",
    "b_coefficient",
    "c_k",
    "c_py_3d",
    "compressibility_fixed_k",
    "contact_value",
    "dp_dn_fixed_k",
    "g2_exact_1d",
    "g2_low_density",
    "intersection_fraction",
    "manifold_entropy_fixed_chi",
    "pair_entropy_derivs",
    "pair_entropy_ratio",
    "pressure_fixed_k",
    "solve_py",
    "structure_factor",
    "unit_ball_volume",
    "z_fixed_chi",
]
