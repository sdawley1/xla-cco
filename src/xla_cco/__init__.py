#!/usr/bin/env python3

"""
xla_cco: pure-JAX collective-coordinate optimization of stealthy
hyperuniform ground states.

Importing this package enables float64 (``jax_enable_x64``) before any
submodule is imported; all computations are float64.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

# ----- project -----
from xla_cco._types import require_float64  # noqa: E402
from xla_cco.analysis.pairs import (  # noqa: E402
    g2_from_histogram,
    nearest_neighbor_distances,
    pair_distance_histogram,
)
from xla_cco.analysis.structure import (  # noqa: E402
    isotropic_structure_factor,
    structure_factor,
)
from xla_cco.box import Box, cubic_box, wrap_fractional  # noqa: E402
from xla_cco.energy import (  # noqa: E402
    AbstractEnergy,
    StealthyEnergy,
    collective_density,
    make_stealthy_energy,
)
from xla_cco.io.configpack import (  # noqa: E402
    PackedConfig,
    read_configpack,
    stack_fractional,
    write_configpack,
)
from xla_cco.io.tables import write_function_table  # noqa: E402
from xla_cco.kvectors import (  # noqa: E402
    KSet,
    chi_from_count,
    enumerate_kvectors,
    k_outer_from_ka,
    ka_from_chi,
    unit_ball_volume,
)
from xla_cco.minimize import relax  # noqa: E402
from xla_cco.minimize.descent import (  # noqa: E402
    DescentSettings,
    adaptive_descent,
)
from xla_cco.minimize.lbfgs import LBFGSSettings, lbfgs  # noqa: E402
from xla_cco.minimize.result import RelaxResult, Trajectory  # noqa: E402

# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
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
    "k_outer_from_ka",
    "ka_from_chi",
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
