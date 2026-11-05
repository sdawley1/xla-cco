#!/usr/bin/env python3

"""
Import-time contract tests for the xla_cco package.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import jax
import jax.numpy as jnp

# ----- package -----
import xla_cco

# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_API = [
    "AbstractEnergy",
    "Box",
    "DescentSettings",
    "KSet",
    "LBFGSSettings",
    "PackedConfig",
    "RelaxResult",
    "StealthyEnergy",
    "Trajectory",
    "adaptive_descent",
    "chi_from_count",
    "collective_density",
    "cubic_box",
    "enumerate_kvectors",
    "g2_from_histogram",
    "isotropic_structure_factor",
    "ka_from_chi",
    "k_outer_from_ka",
    "lbfgs",
    "make_stealthy_energy",
    "nearest_neighbor_distances",
    "pair_distance_histogram",
    "read_configpack",
    "relax",
    "require_float64",
    "stack_fractional",
    "structure_factor",
    "unit_ball_volume",
    "wrap_fractional",
    "write_configpack",
    "write_function_table",
]

# ============================================================
# TESTS
# ============================================================


def test_x64_enabled_on_import() -> None:
    """Importing the package enables float64 arrays."""
    assert jax.config.jax_enable_x64
    assert jnp.zeros(1).dtype == jnp.float64


def test_public_api_names() -> None:
    """The package exports exactly the frozen public names."""
    assert sorted(xla_cco.__all__) == sorted(EXPECTED_API)
    for name in EXPECTED_API:
        assert hasattr(xla_cco, name), name
