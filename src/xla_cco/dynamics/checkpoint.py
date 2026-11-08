#!/usr/bin/env python3

"""
Atomic checkpoints of (batched) MD states.

``save_state`` writes every MDState leaf (including the PRNG key and the
Andersen accumulator) plus a JSON metadata string to ``<path>.tmp`` and
renames it over ``path`` (os.replace), so a crash never leaves a partial
checkpoint. ``load_state`` rebuilds the state with the tree structure of a
template (``like``) and returns the metadata.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path
from typing import Any, Dict, Tuple

# ----- project -----
from xla_cco.dynamics.state import MDState

# ============================================================
# FUNCTIONS
# ============================================================


def save_state(path: Path, state: MDState, meta: Dict[str, Any]) -> None:
    """Write ``state`` and ``meta`` atomically to ``path`` (.npz)."""
    raise NotImplementedError


def load_state(path: Path, like: MDState) -> Tuple[MDState, Dict[str, Any]]:
    """Read a checkpoint into the structure of ``like``; returns (state, meta)."""
    raise NotImplementedError
