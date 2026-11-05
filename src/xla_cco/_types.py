#!/usr/bin/env python3

"""
Shared jaxtyping aliases and dtype guards for xla_cco.

Positions are fractional coordinates (box basis rows = lattice vectors),
nominally in [0, 1); everything is float64.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import numpy as np
from jax import Array
from jaxtyping import ArrayLike, Float

# ============================================================
# TYPE ALIASES
# ============================================================

Positions = Float[Array, "n d"]
BatchPositions = Float[Array, "b n d"]

# ============================================================
# GUARDS
# ============================================================


def require_float64(x: ArrayLike, name: str) -> None:
    """Raise ``TypeError`` unless ``x`` has dtype float64.

    Parameters
    ----------
    x : ArrayLike
        Array (or array-like with a ``dtype``) to check.
    name : str
        Argument name used in the error message.

    Raises
    ------
    TypeError
        If the dtype of ``x`` is not float64.
    """
    assert isinstance(name, str), "name must be a string"
    dtype = getattr(x, "dtype", None)
    if dtype is None or dtype != np.float64:
        raise TypeError(f"{name} must be float64, got dtype {dtype}")
