#!/usr/bin/env python3

"""
Uniformly randomized triangular lattices (URL) of any size, with the kick
law of the group's N = 1024 pack.

The group's pack (experiments/initial_conditions/URL-2D-N1024, copied from
CONFIGURATIONS/TRIANGULAR_URL) holds 32 x 32 sites of a triangular lattice
in a rectangular cell 32a x 16 sqrt(3) a, particle n at site
(i a + j a / 2, j a sqrt(3) / 2) with i, j = divmod(n, 32), plus an
independent kick per site. The kicks are not those of the group's
triangular_lattice.py (a hexagon of circumradius 2a / sqrt(3)): measured
from the pack they lie within 0.4997a, fill a disc of radius about 0.49a
uniformly with a soft edge, and are uncorrelated between neighbouring sites
(|r| < 0.003). No other N exists in that size range (only N = 10000), so
this script builds an nx x ny lattice in the same cell shape and draws each
kick from the empirical kicks of the reference pack (seeded resampling with
replacement), which reproduces the single-site law exactly.

Writes <out>/URL-2D-N<N>.ConfigPack (unit-volume rectangular basis, positions
wrapped) and the "URL" entry of <out>/provenance.json. --check compares the
structure factor of a resampled N = 1024 ensemble with the reference pack.
NumPy only.

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
from typing import List, Tuple

# ----- numerics -----
import numpy as np

# ----- project -----
from xla_cco.io.configpack import PackedConfig, read_configpack, write_configpack

# ============================================================
# CONSTANTS
# ============================================================

NAME = b"a\x00\x00\x00"  # C++ default atom name, as in make_initial_conditions
REF_SIDE = 32  # sites per row and number of rows of the reference pack
SEED = 20261109

# ============================================================
# LATTICE AND KICKS
# ============================================================


def cell(nx: int, ny: int) -> np.ndarray:
    """Rectangular cell lengths (Lx, Ly) in units of the lattice constant."""
    assert nx > 0 and ny > 0 and ny % 2 == 0, "ny must be even (periodic rows)"
    out = np.array([float(nx), ny * np.sqrt(3.0) / 2.0])
    assert out.shape == (2,), "two lengths"
    return out


def sites(nx: int, ny: int) -> np.ndarray:
    """(nx ny, 2) triangular-lattice sites, particle n at (i, j) = divmod(n, ny)."""
    n = np.arange(nx * ny)
    i, j = n // ny, n % ny
    out = np.stack([i + 0.5 * j, j * np.sqrt(3.0) / 2.0], axis=1)
    out = np.mod(out, cell(nx, ny))
    assert out.shape == (nx * ny, 2), "one site per particle"
    return out


def reference_kicks(prefix: Path) -> np.ndarray:
    """(B N, 2) kicks of the reference pack, in units of the lattice constant."""
    configs = read_configpack(prefix)
    assert configs and configs[0].relative.shape[0] == REF_SIDE**2, "32 x 32 pack"
    lengths = cell(REF_SIDE, REF_SIDE)
    site = sites(REF_SIDE, REF_SIDE)
    out: List[np.ndarray] = []
    for c in configs:
        ratio = np.diag(c.basis)[0] / np.diag(c.basis)[1]
        assert abs(ratio - lengths[0] / lengths[1]) < 1e-6, "same cell shape"
        d = np.mod(c.relative, 1.0) * lengths - site
        out.append(d - np.round(d / lengths) * lengths)
    kicks = np.concatenate(out)
    assert np.max(np.hypot(kicks[:, 0], kicks[:, 1])) < 0.5, "kicks inside a/2"
    return kicks


def ensemble(
    kicks: np.ndarray, nx: int, ny: int, n_configs: int, seed: int
) -> List[PackedConfig]:
    """n_configs URL patterns with kicks resampled from ``kicks`` (seeded)."""
    assert n_configs > 0 and kicks.ndim == 2, "need configs and (K, 2) kicks"
    rng = np.random.default_rng(seed)
    lengths, site = cell(nx, ny), sites(nx, ny)
    basis = np.diag(lengths) / np.sqrt(np.prod(lengths))  # unit volume
    out = []
    for _ in range(n_configs):
        r = site + kicks[rng.integers(0, kicks.shape[0], site.shape[0])]
        rel = np.mod(r / lengths, 1.0)
        out.append(PackedConfig(basis, rel, 1.0, (NAME,) * rel.shape[0]))
    assert len(out) == n_configs, "one pattern per config"
    return out


# ============================================================
# CHECK
# ============================================================


def structure_factor(configs: List[PackedConfig], n_max: int) -> np.ndarray:
    """(B, K) per-config S(k) on the cell vectors 0 < |n| <= n_max."""
    nv = np.array([(a, b) for a in range(-n_max, n_max + 1)
                   for b in range(-n_max, n_max + 1) if a * a + b * b > 0])  # fmt: skip
    rows = []
    for c in configs:
        phase = 2.0 * np.pi * np.mod(c.relative, 1.0) @ nv.T
        rho = np.exp(-1j * phase).sum(axis=0)
        rows.append(np.abs(rho) ** 2 / c.relative.shape[0])
    out = np.array(rows)
    assert out.shape == (len(configs), nv.shape[0]) and np.all(out >= 0.0), "S >= 0"
    return out


def check(prefix: Path, kicks: np.ndarray) -> Tuple[float, float]:
    """Two-sample z of mean S(k), reference vs resampled N = 1024: (sd, max |z|).

    Under the null (same kick law) the z values are about standard normal, so
    sd ~ 1 and max |z| ~ 3.5 over the ~600 vectors with |n| <= 12.
    """
    ref = read_configpack(prefix)
    new = ensemble(kicks, REF_SIDE, REF_SIDE, 4 * len(ref), SEED + 1)
    a, b = structure_factor(ref, 12), structure_factor(new, 12)
    se = np.sqrt(a.var(0, ddof=1) / a.shape[0] + b.var(0, ddof=1) / b.shape[0])
    z = (a.mean(0) - b.mean(0)) / se
    out = (float(z.std()), float(np.max(np.abs(z))))
    assert np.isfinite(out[0]) and out[1] >= 0.0, "finite"
    return out


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ic = Path("experiments/initial_conditions")
    parser.add_argument("--reference", type=Path, default=ic / "URL-2D-N1024")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--n-configs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=ic / "n4096")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    k_ref = reference_kicks(args.reference)
    r = np.hypot(k_ref[:, 0], k_ref[:, 1])
    print(f"reference kicks: {k_ref.shape[0]}, max |d|/a {r.max():.4f}, "
          f"<|d|>/a {r.mean():.4f}", flush=True)  # fmt: skip
    if args.check:
        sd_z, max_z = check(args.reference, k_ref)
        print(f"S(k), |n| <= 12, pack vs resampled: z sd {sd_z:.2f}, "
              f"max |z| {max_z:.2f} (null: ~1 and ~3.5)", flush=True)  # fmt: skip
    pats = ensemble(k_ref, args.nx, args.ny, args.n_configs, args.seed)
    n = args.nx * args.ny
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = args.out / f"URL-2D-N{n}"
    write_configpack(prefix, pats)
    ref_pack = Path(f"{args.reference}.ConfigPack")
    record_path = args.out / "provenance.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    record["URL"] = {
        "file": str(prefix),
        "generator": "experiments/scripts/url_resample.py",
        "lattice": f"{args.nx} x {args.ny} triangular, cell nx a x ny sqrt(3)/2 a",
        "kicks": "resampled with replacement from the reference pack's kicks",
        "reference": str(args.reference),
        "reference_sha256": hashlib.sha256(ref_pack.read_bytes()).hexdigest(),
        "seed": args.seed,
        "n_configs": args.n_configs,
    }
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.n_configs} URL configs (N = {n}) -> {prefix}", flush=True)
