#!/usr/bin/env python3

"""
Fail-fast GPU guard for Della jobs (thin wrapper around ``xla_cco._device``).

Run as a SLURM guard step before any long computation. Exits 1 with the most
likely cause when JAX cannot see a GPU, so a billed allocation never silently
falls back to the CPU.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import sys

# ----- project -----
from xla_cco._device import require_gpu

# ============================================================
# __main__
# ============================================================


def main() -> int:
    """Run the guard; return the process exit code (0 ok, 1 no GPU)."""
    try:
        require_gpu()
    except Exception as exc:  # guard must catch all failure modes
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    print("GPU visible.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
