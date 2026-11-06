#!/usr/bin/env python3

"""
Build the initial-condition ensembles for the Dawley & Torquato (2025) runs.

- POS: Poisson (uniform random) point patterns, generated here from a fixed
  seed, unit-cell ConfigPack.
- GAU: hyperuniform patterns with Gaussian pair statistics,
  h(r) = -exp(-pi rho r^2), copied from an existing ConfigPack and
  normalized to a unit-volume basis (positions unchanged, wrapped).

Writes ``<out>/POS-2D-N<N>.ConfigPack``, ``<out>/GAU-2D-N<N>.ConfigPack`` and
``<out>/provenance.json``. All outputs are git-ignored data.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

# ----- numerics -----
import numpy as np

# ----- project -----
from xla_cco.io.configpack import PackedConfig, read_configpack, write_configpack

# ============================================================
# CONSTANTS
# ============================================================

NAME = b"a\x00\x00\x00"  # C++ default atom name

# ============================================================
# BUILDERS
# ============================================================


def poisson_ensemble(
    n_configs: int, n_particles: int, dim: int, seed: int
) -> List[PackedConfig]:
    """Return ``n_configs`` Poisson patterns in the unit cell (seeded)."""
    assert n_configs > 0 and n_particles > 0, "need positive sizes"
    assert dim in (1, 2, 3), "dim must be 1, 2 or 3"
    rng = np.random.default_rng(seed)
    basis = np.eye(dim)
    return [
        PackedConfig(basis, rng.random((n_particles, dim)), 1.0, (NAME,) * n_particles)
        for _ in range(n_configs)
    ]


def unit_cell_copy(configs: List[PackedConfig]) -> List[PackedConfig]:
    """Return copies with the basis scaled to unit volume, positions wrapped."""
    assert len(configs) > 0, "need at least one configuration"
    out = []
    for c in configs:
        det = abs(float(np.linalg.det(c.basis)))
        if det <= 0.0:
            raise ValueError("degenerate basis in source pack")
        d = c.basis.shape[0]
        unit = c.basis / det ** (1.0 / d)
        rel = np.mod(c.relative, 1.0)
        out.append(PackedConfig(unit, rel, 1.0, (NAME,) * rel.shape[0]))
    return out


def sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    assert path.is_file(), f"{path} is not a file"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(
    pos_prefix: Path, gau_prefix: Path, gau_source: Path, seed: int
) -> Dict[str, object]:
    """Return a provenance record for both ensembles."""
    src = Path(f"{gau_source}.ConfigPack")
    return {
        "POS": {
            "file": str(pos_prefix),
            "generator": "numpy default_rng",
            "seed": seed,
        },
        "GAU": {
            "file": str(gau_prefix),
            "source": str(gau_source),
            "source_sha256": sha256(src),
            "transform": "basis scaled to unit volume; positions wrapped mod 1",
        },
    }


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # ----- inputs -----
    parser.add_argument(
        "--gau-source",
        type=Path,
        default=Path(
            "/home/sd2402/torquato-scratch-sd2402/CONFIGURATIONS/"
            "N1000_configs_GaussianPairStat/GAUSSIAN-2D-N_1000"
        ),
    )

    # ----- ensemble sizes -----
    parser.add_argument("--n-pos", type=int, default=100)
    parser.add_argument("--n-particles-pos", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20251106)

    # ----- output -----
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/initial_conditions")
    )

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # POISSON
    # =========================================================
    pos = poisson_ensemble(args.n_pos, args.n_particles_pos, 2, args.seed)
    pos_prefix = args.out / f"POS-2D-N{args.n_particles_pos}"
    write_configpack(pos_prefix, pos)
    print(f"wrote {len(pos)} POS configs -> {pos_prefix}", flush=True)

    # =========================================================
    # GAUSSIAN PAIR STATISTICS
    # =========================================================
    gau = unit_cell_copy(read_configpack(args.gau_source))
    gau_prefix = args.out / f"GAU-2D-N{gau[0].relative.shape[0]}"
    write_configpack(gau_prefix, gau)
    print(f"wrote {len(gau)} GAU configs -> {gau_prefix}", flush=True)

    # =========================================================
    # PROVENANCE
    # =========================================================
    record = provenance(pos_prefix, gau_prefix, args.gau_source, args.seed)
    (args.out / "provenance.json").write_text(json.dumps(record, indent=1))
    print("Done.", flush=True)
