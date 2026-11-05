#!/usr/bin/env python3

"""
Bounded lax.while_loop driver shared by both minimizers.

Provides per-lane freezing of converged configurations (bit-exact state,
history and step) and the trajectory recorder. Every loop condition carries
an explicit iteration counter.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any, Callable

# ============================================================
# DRIVER
# ============================================================


def run_bounded_loop(
    cond: Callable[[Any], Any],
    body: Callable[[Any], Any],
    init: Any,
    max_iters: int,
) -> Any:
    """Run ``body`` while ``cond`` holds, for at most ``max_iters`` iterations.

    Parameters
    ----------
    cond : Callable[[Any], Any]
        Scalar boolean predicate of the carry.
    body : Callable[[Any], Any]
        Carry-to-carry update.
    init : Any
        Initial carry PyTree.
    max_iters : int
        Static hard iteration bound (explicit counter in the loop condition).
    """
    raise NotImplementedError
