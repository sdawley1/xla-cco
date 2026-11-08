#!/usr/bin/env python3

"""
Saturated random sequential addition (RSA) of hard disks in a periodic square.

A NumPy port of the voxel algorithm in the group's saturated_rsa.py
(CONFIGURATIONS/SATURATED_RSA; after Zhang & Torquato, PRE 88, 053312
(2013)):

1. Random insertion: disks of diameter D are proposed uniformly and kept if
   no center lies within D (minimum image), until fewer than STALL[0] of
   STALL[1] consecutive trials succeed.
2. Voxel refinement: the square is tiled with voxels of side about D/4; a
   voxel is dropped once a single exclusion disk (radius D about a center)
   covers it entirely, since no new center can land there. Points are then
   proposed only in the remaining voxels. When insertions stall again,
   every voxel is split into four and covered ones dropped, until no voxel
   is left: the pattern is saturated (no further disk fits anywhere), up to
   voxels of side min_voxel D around exact tangencies.

One correction to the original: it tests coverage with the particle radius
D/2, which leaves voxels that are excluded (within D of a center) but not
inside a particle, so it can stop on its iteration cap before saturation.
Here coverage uses the exclusion radius D.

CLI: generate --n-configs patterns (seeded, one process per CPU) and write
them to an npz shard; --assemble reads shards, keeps the configurations
whose N is the most common value, and writes the first --keep of them as
<out>/RSA-2D-N<N>.ConfigPack (unit-volume basis), updating provenance.json.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import json
import math
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

# ----- numerics -----
import numpy as np

# ----- project -----
from xla_cco.io.configpack import PackedConfig, write_configpack

# ============================================================
# CONSTANTS
# ============================================================

PHI_SAT = 0.5472  # 2D saturated RSA packing fraction (Zhang & Torquato 2013)
STALL = (3, 750)  # stop a stage when fewer than 3 of 750 trials succeed
NAME = b"A\x00\x00\x00"  # ConfigPack particle name

# ============================================================
# HELPERS
# ============================================================


class CellList:
    """Centers in a periodic square, binned in cells of side >= D for overlap tests."""

    def __init__(self, side: float, diameter: float) -> None:
        assert side > 2.0 * diameter > 0.0, "box must exceed two diameters"
        self.side, self.d = side, diameter
        self.n = max(3, int(side // diameter))
        self.cells: Dict[Tuple[int, int], List[int]] = {}
        self.pts: List[np.ndarray] = []

    def cell(self, p: np.ndarray) -> Tuple[int, int]:
        """Cell index of point p."""
        c = (p / self.side * self.n).astype(int) % self.n
        return int(c[0]), int(c[1])

    def neighbors(self, p: np.ndarray) -> List[int]:
        """Indices of centers in the 3 x 3 cells around p."""
        cx, cy = self.cell(p)
        out: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out += self.cells.get(((cx + dx) % self.n, (cy + dy) % self.n), [])
        return out

    def fits(self, p: np.ndarray) -> bool:
        """True if a disk centered at p overlaps no disk (all centers > D away)."""
        idx = self.neighbors(p)
        if not idx:
            return True
        q = np.asarray([self.pts[i] for i in idx]) - p
        q -= self.side * np.round(q / self.side)
        return bool(np.min(np.einsum("ij,ij->i", q, q)) > self.d**2)

    def add(self, p: np.ndarray) -> None:
        """Insert center p."""
        self.cells.setdefault(self.cell(p), []).append(len(self.pts))
        self.pts.append(p)

    def array(self) -> np.ndarray:
        """Centers (N, 2)."""
        return np.asarray(self.pts).reshape(-1, 2)


def covered(
    corners: np.ndarray, centers: np.ndarray, side: float, diameter: float
) -> np.ndarray:
    """Voxels (V, 4, 2 corner coordinates) inside one exclusion disk of ``centers``.

    A square voxel lies inside the disk of radius D about c iff all four
    corners do (the disk is convex). Checks every center, in chunks.
    """
    assert corners.ndim == 3 and corners.shape[1:] == (4, 2), "corners (V, 4, 2)"
    out = np.zeros(corners.shape[0], dtype=bool)
    if centers.shape[0] == 0:
        return out
    for a in range(0, corners.shape[0], 2048):
        q = corners[a : a + 2048, :, None, :] - centers[None, None, :, :]  # (v,4,P,2)
        q -= side * np.round(q / side)
        inside = np.einsum("vcpk,vcpk->vcp", q, q) <= diameter**2
        out[a : a + 2048] = np.any(np.all(inside, axis=1), axis=1)
    return out


def voxel_corners(lo: np.ndarray, size: float) -> np.ndarray:
    """Corners (V, 4, 2) of square voxels with lower-left corners lo (V, 2)."""
    off = np.array([[0.0, 0.0], [size, 0.0], [0.0, size], [size, size]])
    out: np.ndarray = lo[:, None, :] + off[None, :, :]
    return out


def insert_until_stall(rng: np.random.Generator, cl: CellList, propose: object) -> int:
    """Propose points (``propose(rng)``) until fewer than STALL[0] of STALL[1] fit."""
    added = 0
    while True:
        hits = 0
        for _ in range(STALL[1]):
            p = propose(rng)  # type: ignore[operator]
            if cl.fits(p):
                cl.add(p)
                hits += 1
        added += hits
        if hits < STALL[0]:
            return added


def saturated_rsa(
    rng: np.random.Generator, side: float, diameter: float, min_voxel: float = 1e-7
) -> np.ndarray:
    """Centers (N, 2) of a saturated RSA packing of disks of diameter D.

    Stage 1 inserts uniformly; stage 2 inserts in uncovered voxels, splitting
    them as insertions stall, until none remain or they shrink below
    min_voxel * D.
    """
    cl = CellList(side, diameter)
    insert_until_stall(rng, cl, lambda r: r.random(2) * side)
    n_v = int(math.ceil(side / (diameter / 4.0)))
    size = side / n_v
    gx, gy = np.meshgrid(np.arange(n_v) * size, np.arange(n_v) * size, indexing="ij")
    lo = np.stack([gx.ravel(), gy.ravel()], axis=1)
    lo = lo[~covered(voxel_corners(lo, size), cl.array(), side, diameter)]
    while lo.shape[0] and size > min_voxel * diameter:
        n_before = len(cl.pts)
        box = lo  # voxels sampled this round

        def propose(
            r: np.random.Generator, box: np.ndarray = box, size: float = size
        ) -> np.ndarray:
            return (box[r.integers(box.shape[0])] + r.random(2) * size) % side

        insert_until_stall(rng, cl, propose)
        new = cl.array()[n_before:]
        if new.shape[0]:
            lo = lo[~covered(voxel_corners(lo, size), new, side, diameter)]
        size /= 2.0  # split the rest into four and drop the covered children
        lo = lo[:, None, :] + size * np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        lo = lo.reshape(-1, 2)
        lo = lo[~covered(voxel_corners(lo, size), cl.array(), side, diameter)]
    return cl.array()


def one_pattern(args: Tuple[int, float, float]) -> np.ndarray:
    """Worker: the saturated pattern for one seed."""
    seed, side, diameter = args
    return saturated_rsa(np.random.default_rng(seed), side, diameter)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-target", type=int, default=1000,
                        help="expected N; the box side is sqrt(n_target)")  # fmt: skip
    parser.add_argument("--n-configs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", type=Path, default=None, help="npz to write")
    parser.add_argument("--assemble", nargs="+", type=Path, default=None,
                        help="shards to combine into a ConfigPack")  # fmt: skip
    parser.add_argument("--keep", type=int, default=20)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/initial_conditions")
    )
    args = parser.parse_args()
    side = math.sqrt(args.n_target)  # density about 1
    diameter = 2.0 * math.sqrt(PHI_SAT / math.pi)

    if args.assemble is None:
        assert args.shard is not None and args.n_configs > 0, "need --shard"
        t0 = time.time()
        jobs = [(args.seed + i, side, diameter) for i in range(args.n_configs)]
        with Pool(args.workers) as pool:
            pats = pool.map(one_pattern, jobs)
        counts = np.array([p.shape[0] for p in pats])
        flat = np.concatenate(pats)
        np.savez(args.shard, centers=flat, counts=counts,
                 seeds=np.array([j[0] for j in jobs]), side=np.array(side),
                 diameter=np.array(diameter))  # fmt: skip
        span = f"N {counts.min()}-{counts.max()} (mean {counts.mean():.1f})"
        print(f"{args.n_configs} patterns, {span}; {time.time() - t0:.0f} s",
              flush=True)  # fmt: skip
    else:
        pats, seeds = [], []
        for path in args.assemble:
            with np.load(path) as z:
                ends = np.cumsum(z["counts"])
                pats += np.split(z["centers"], ends[:-1])
                seeds += z["seeds"].tolist()
                side, diameter = float(z["side"]), float(z["diameter"])
        tally = Counter(p.shape[0] for p in pats)
        n_mode, n_have = tally.most_common(1)[0]
        assert n_have >= args.keep, f"only {n_have} patterns with N = {n_mode}"
        chosen = [i for i, p in enumerate(pats) if p.shape[0] == n_mode][: args.keep]
        configs = [PackedConfig(np.eye(2), np.mod(pats[i] / side, 1.0),
                                diameter / side, (NAME,) * n_mode)
                   for i in chosen]  # fmt: skip
        prefix = args.out / f"RSA-2D-N{n_mode}"
        write_configpack(prefix, configs)
        record_path = args.out / "provenance.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        record["RSA"] = {
            "file": str(prefix),
            "generator": "experiments/scripts/saturated_rsa.py (NumPy port of "
            "CONFIGURATIONS/SATURATED_RSA/saturated_rsa.py)",
            "seeds": [seeds[i] for i in chosen],
            "pool": len(pats),
            "pool_N_counts": {str(k): v for k, v in sorted(tally.items())},
            "box_side": side,
            "diameter": diameter,
            "packing_fraction": n_mode * math.pi * diameter**2 / 4.0 / side**2,
        }
        record_path.write_text(json.dumps(record, indent=1))
        print(f"kept {len(chosen)} of {len(pats)} patterns with N = {n_mode} "
              f"({n_have} available) -> {prefix}", flush=True)  # fmt: skip
