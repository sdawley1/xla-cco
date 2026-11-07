#!/usr/bin/env python3

"""
Exact Wasserstein-2 at the frames where Sinkhorn is not certified.

The Sinkhorn pass (wasserstein_sinkhorn.py) is certified exact only where
the row-argmax rounding of its plan is a bijection; at colliding frames the
greedy repair can return a suboptimal matching (D-014). This script solves
the exact assignment (Hungarian algorithm) at every frame and reference
whose rounding was not bijective, in parallel over trajectories, and writes
<traj>/wasserstein/fallback_<ENS>.npz with per-entry arrays

    traj (trajectory index), frame, col (0: vs r(0), 1: vs r(inf)),
    w (exact W_2, scaled like p_0), n_re (labels re-matched),

which analyze_wasserstein.py substitutes for the Sinkhorn values.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ----- numerics -----
import numpy as np

# ----- project -----
from xla_cco.io.trajstore import open_trajectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_wasserstein import load_sinkhorn  # noqa: E402
from wasserstein_exact import exact_w2  # noqa: E402

# ============================================================
# PER TRAJECTORY
# ============================================================


def solve(job: Tuple[Path, int, np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
    """Exact W_2 at the listed (frame, col) entries of one trajectory."""
    prefix, traj, frames_idx, cols = job
    assert frames_idx.shape == cols.shape, "one column per frame entry"
    frames, _, meta = open_trajectory(prefix)
    basis = np.asarray(meta["basis"], dtype=np.float64)
    rho = float(meta["density"])
    first, last = np.array(frames[0]), np.array(frames[-1])
    w = np.empty(frames_idx.shape[0])
    n_re = np.empty(frames_idx.shape[0], dtype=np.int64)
    for j, (k, c) in enumerate(zip(frames_idx, cols, strict=True)):
        cur = np.array(frames[int(k)])
        # Same orientation as the Sinkhorn pass: rows are the reference.
        ref = first if c == 0 else last
        dist, moved = exact_w2(ref, cur, basis)
        w[j], n_re[j] = np.sqrt(rho) * dist, moved
    return {
        "traj": np.full(frames_idx.shape[0], traj, dtype=np.int64),
        "frame": frames_idx.astype(np.int64),
        "col": cols.astype(np.int64),
        "w": w,
        "n_re": n_re,
    }


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # ----- inputs -----
    parser.add_argument(
        "--traj", type=Path, default=Path("experiments/trajectories/chi0.1")
    )
    parser.add_argument("--ensembles", nargs="+", default=["POS", "GAU"])
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()
    assert args.workers > 0, "need at least one worker"
    wdir = args.traj / "wasserstein"

    # =========================================================
    # ENSEMBLES
    # =========================================================
    for ens in args.ensembles:
        sk = load_sinkhorn(wdir, ens)
        off = sk["offsets"]
        bad = sk["is_perm"] != 1
        jobs: List[Tuple[Path, int, np.ndarray, np.ndarray]] = []
        for i, name in enumerate(sk["names"]):
            rows, cols = np.nonzero(bad[off[i] : off[i + 1]])
            if rows.size:
                jobs.append((args.traj / ens / str(name), i, rows, cols))
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers) as pool:
            parts = pool.map(solve, jobs, chunksize=1)
        keys = ("traj", "frame", "col", "w", "n_re")
        if parts:
            result = {k: np.concatenate([p[k] for p in parts]) for k in keys}
        else:
            result = {k: np.empty(0, dtype=np.int64) for k in keys}
            result["w"] = np.empty(0)
        result["names"] = sk["names"]
        np.savez(wdir / f"fallback_{ens}.npz", **result)
        print(
            f"{ens}: {int(bad.sum())} uncertified (frame, reference) entries "
            f"of {bad.size} ({bad.mean():.2%}) solved exactly in "
            f"{time.time() - t0:.0f} s",
            flush=True,
        )
    print("Done.", flush=True)
