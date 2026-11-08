#!/usr/bin/env python3

"""
Low-temperature MD driver: python -m xla_cco.cli.md.

Loads ground states (an npz with ``s_final`` (B, N, d) fractional, ``basis``
(d, d), and a K value), builds the stealthy energy and the tracked waves
(|k| < --k-track), and runs the protocol: harmonic start, Andersen
thermalization (--n-equil steps), then --cycles of [Andersen --n-andersen,
NVE --n-nve] at each --temperature, writing ``<save>.npz`` (snapshots,
final winding, sums, statuses) and checkpoints. ``--device gpu`` fails fast
without a GPU (cli.relax.select_device); exit code 1 on any error.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import sys
from typing import List, Optional

# ============================================================
# FUNCTIONS
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the MD driver."""
    raise NotImplementedError


def main(argv: Optional[List[str]] = None) -> int:
    """Parse, validate, select the device, run; returns the exit code."""
    raise NotImplementedError


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    sys.exit(main())
