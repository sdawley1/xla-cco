#!/usr/bin/env python3

"""
Build the initial-condition ensembles for the Dawley & Torquato (2025) runs.

- POS: Poisson (uniform random) point patterns, generated here from a fixed
  seed, unit-cell ConfigPack.
- GAU: hyperuniform patterns with Gaussian pair statistics,
  h(r) = -exp(-pi rho r^2), copied from an existing ConfigPack and
  normalized to a unit-volume basis (positions unchanged, wrapped).
- URL: uniformly randomized triangular lattices (each site of a triangular
  lattice displaced uniformly within a fundamental triangle, in one of six
  orientations), copied the same way; the cell is rectangular, 32 x 16 sqrt(3)
  lattice constants for N = 1024.

Writes ``<out>/<NAME>-2D-N<N>.ConfigPack`` for the ensembles named in
--ensembles (default POS and GAU) and updates ``<out>/provenance.json``.
All outputs are git-ignored data.

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


def copied_record(prefix: Path, source: Path) -> Dict[str, object]:
    """Provenance of an ensemble copied from an existing ConfigPack."""
    return {
        "file": str(prefix),
        "source": str(source),
        "source_sha256": sha256(Path(f"{source}.ConfigPack")),
        "transform": "basis scaled to unit volume; positions wrapped mod 1",
    }


def copy_ensemble(source: Path, name: str, out: Path) -> Path:
    """Unit-cell copy of a ConfigPack as <out>/<name>-2D-N<N>; returns the prefix."""
    configs = unit_cell_copy(read_configpack(source))
    prefix = out / f"{name}-2D-N{configs[0].relative.shape[0]}"
    write_configpack(prefix, configs)
    print(f"wrote {len(configs)} {name} configs -> {prefix}", flush=True)
    return prefix


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # ----- inputs -----
    parser.add_argument(
        "--ensembles", nargs="+", choices=("pos", "gau", "url"), default=["pos", "gau"]
    )
    parser.add_argument(
        "--url-source",
        type=Path,
        default=None,
        help="ConfigPack prefix of the uniformly randomized triangular lattices "
        "(the group copy is CONFIGURATIONS/TRIANGULAR_URL/"
        "TRIANGULAR_URL-N_1024-rho_1.0/TRIANGULAR_URL-N_1024-rho_1.0; read-only)",
    )
    parser.add_argument(
        "--gau-source",
        type=Path,
        default=None,
        help="ConfigPack prefix of the Gaussian-pair-statistics configurations "
        "(the group copy is CONFIGURATIONS/N1000_configs_GaussianPairStat/"
        "GAUSSIAN-2D-N_1000 under the TORQUATO scratch space; read-only)",
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
    # SLURM writes the trajectory-job logs here, so it must exist up front.
    (args.out.parent / "trajectories").mkdir(parents=True, exist_ok=True)

    record_path = args.out / "provenance.json"
    record: Dict[str, object] = (
        json.loads(record_path.read_text()) if record_path.exists() else {}
    )

    # =========================================================
    # POISSON
    # =========================================================
    if "pos" in args.ensembles:
        pos = poisson_ensemble(args.n_pos, args.n_particles_pos, 2, args.seed)
        pos_prefix = args.out / f"POS-2D-N{args.n_particles_pos}"
        write_configpack(pos_prefix, pos)
        print(f"wrote {len(pos)} POS configs -> {pos_prefix}", flush=True)
        record["POS"] = {"file": str(pos_prefix), "generator": "numpy default_rng",
                         "seed": args.seed}  # fmt: skip

    # =========================================================
    # COPIED ENSEMBLES (GAUSSIAN PAIR STATISTICS, URL)
    # =========================================================
    for name, source in (("GAU", args.gau_source), ("URL", args.url_source)):
        if name.lower() not in args.ensembles:
            continue
        if source is None:
            parser.error(f"--{name.lower()}-source is required for {name}")
        prefix = copy_ensemble(source, name, args.out)
        record[name] = copied_record(prefix, source)

    # =========================================================
    # PROVENANCE
    # =========================================================
    record_path.write_text(json.dumps(record, indent=1))
    print("Done.", flush=True)
