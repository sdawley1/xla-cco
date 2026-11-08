#!/usr/bin/env python3

"""
Point patterns from the 2D isotropic hyperplane (line) intersection process.

A vectorized port of the group's hpp.py (CONFIGURATIONS/HYPERPLANE_
INTERSECTION; after Klatt et al., Nat. Commun. 10, 811 (2019)): a Poisson
number of lines with mean 2 gamma R, each at a uniform signed distance in
[-R, R] from the origin and a uniform orientation in [0, pi), where
R = m sqrt(2) is the half-diagonal of the window [-m, m]^2 and
gamma = sqrt(pi) gives unit intensity; the points are the pairwise line
intersections inside the window. The process is antihyperuniform: S(k)
diverges as k -> 0.

The pattern is a window sample, not periodic: lines end at the window
edge, and the flows treat the window as a periodic cell (as the group's
HPP-2D-N_1024 pack does).

N varies widely between samples (its spread is about +-200 at N ~ 1000),
so the CLI generates samples until --keep of them have exactly --n-target
points (the process conditioned on N) and writes them as
<out>/HIP-2D-N<N>.ConfigPack (unit-volume basis), updating provenance.json.

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
from pathlib import Path

# ----- numerics -----
import numpy as np

# ----- project -----
from xla_cco.io.configpack import PackedConfig, write_configpack

# ============================================================
# CONSTANTS
# ============================================================

GAMMA = math.sqrt(math.pi)  # line density for unit point intensity in 2D
NAME = b"A\x00\x00\x00"  # ConfigPack particle name

# ============================================================
# HELPERS
# ============================================================


def hip_2d(rng: np.random.Generator, side: float) -> np.ndarray:
    """Intersections (N, 2) of a 2D isotropic line process inside [-m, m]^2.

    m = side / 2; the mean number of points is side^2 (unit intensity).
    """
    assert side > 0.0, "window side must be positive"
    m = side / 2.0
    r = m * math.sqrt(2.0)
    n_lines = rng.poisson(2.0 * GAMMA * r)
    dist = rng.uniform(-r, r, n_lines)
    theta = rng.uniform(0.0, math.pi, n_lines)
    i, j = np.triu_indices(n_lines, k=1)
    nx, ny = np.cos(theta), np.sin(theta)
    det = nx[i] * ny[j] - ny[i] * nx[j]  # Cramer's rule for the 2 x 2 systems
    ok = np.abs(det) > 1e-14
    i, j, det = i[ok], j[ok], det[ok]
    x = (dist[i] * ny[j] - ny[i] * dist[j]) / det
    y = (nx[i] * dist[j] - dist[i] * nx[j]) / det
    inside = (np.abs(x) < m) & (np.abs(y) < m)
    return np.stack([x[inside], y[inside]], axis=1)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-target", type=int, default=1000)
    parser.add_argument("--keep", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/initial_conditions")
    )
    args = parser.parse_args()
    assert args.n_target > 0 and args.keep > 0, "bad arguments"
    t0 = time.time()
    side = math.sqrt(args.n_target)
    kept, seeds, counts = [], [], []
    for k in range(args.max_samples):
        pts = hip_2d(np.random.default_rng(args.seed + k), side)
        counts.append(pts.shape[0])
        if pts.shape[0] == args.n_target:
            kept.append(pts)
            seeds.append(args.seed + k)
            if len(kept) == args.keep:
                break
    assert len(kept) == args.keep, f"only {len(kept)} samples with N = {args.n_target}"
    configs = [PackedConfig(np.eye(2), np.mod(p / side + 0.5, 1.0), 1.0,
                            (NAME,) * args.n_target) for p in kept]  # fmt: skip
    prefix = args.out / f"HIP-2D-N{args.n_target}"
    write_configpack(prefix, configs)
    c = np.array(counts)
    record_path = args.out / "provenance.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    record["HIP"] = {
        "file": str(prefix),
        "generator": "experiments/scripts/hip.py (vectorized port of "
        "CONFIGURATIONS/HYPERPLANE_INTERSECTION/hpp.py)",
        "conditioning": f"N = {args.n_target} exactly",
        "seeds": seeds,
        "samples_drawn": int(c.size),
        "sample_N_mean": float(c.mean()),
        "sample_N_std": float(c.std()),
        "window_side": side,
    }
    record_path.write_text(json.dumps(record, indent=1))
    print(f"kept {len(kept)} of {c.size} samples with N = {args.n_target} "
          f"(sample N mean {c.mean():.0f}, sd {c.std():.0f}) -> {prefix}; "
          f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip
