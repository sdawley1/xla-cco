#!/usr/bin/env python3

"""
Exact periodic Wasserstein-2 distances along recorded descent trajectories.

For each trajectory, about 200 frames are chosen log-spaced in t/T (plus
t = 0). At each chosen frame r(t), the exact optimal matching (Hungarian
algorithm, scipy.optimize.linear_sum_assignment) on the minimum-image cost
gives

    W_0(t)   = W_2(r(0), r(t))     (label-free analogue of p_0(t)),
    W_inf(t) = W_2(r(t), r(inf))   (distance to the ground state reached),

both scaled like p_0, rho^{1/2} [min_sigma sum_i |r_i - r'_sigma(i)|^2]^{1/2},
and the number of particles the optimal matching assigns away from their own
label. The labelled counterparts p_0(t) and q_inf(t) = rho^{1/2}|r(t) - r(inf)|
(unwrapped paths) are stored on the same frames. Trajectories are processed
in parallel worker processes. Output: <traj>/wasserstein/exact_<ENS>.npz.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ----- numerics -----
import numpy as np
from scipy.optimize import linear_sum_assignment

# ----- project -----
from xla_cco.analysis.trajectory import proximity_metric, unwrap_fractional
from xla_cco.io.trajstore import open_trajectory

# ============================================================
# CONSTANTS
# ============================================================

KEYS = ("u", "frame", "energy", "p0", "q_inf", "w0", "w_inf", "n_re0", "n_re_inf")

# ============================================================
# HELPERS
# ============================================================


def frame_times(arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-frame t/T and Phi (frame 0 = initial condition)."""
    acc = arrays["accepted"].astype(bool)
    total = float(arrays["time"][-1])
    assert total > 0.0, "trajectory has no accepted moves"
    u = np.concatenate([[0.0], arrays["time"][acc]]) / total
    energy = np.concatenate([[float(arrays["energy0"])], arrays["energy"][acc]])
    assert np.all(np.diff(u) >= 0.0), "time must be non-decreasing"
    return u, energy


def select_frames(u: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return, for each target t/T, the first frame with u >= target."""
    assert u.ndim == 1 and targets.ndim == 1, "1-D inputs required"
    idx = np.searchsorted(u, targets, side="left")
    return np.asarray(np.clip(idx, 0, u.shape[0] - 1), dtype=np.int64)


def exact_w2(a: np.ndarray, b: np.ndarray, basis: np.ndarray) -> Tuple[float, int]:
    """Return (rho = 1 scaled) exact W_2 between a and b and #re-matched labels."""
    assert a.shape == b.shape and basis.shape == (a.shape[1],) * 2, "bad shapes"
    ds = a[:, None, :] - b[None, :, :]
    ds -= np.round(ds)
    assert np.allclose(basis, np.diag(np.diag(basis))), "orthogonal cell assumed"
    r = ds * np.diag(basis)
    cost = np.sum(r * r, axis=-1)
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].sum())), int(np.sum(cols != rows))


# ============================================================
# PER TRAJECTORY
# ============================================================


def process(job: Tuple[Path, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute the KEYS arrays of one trajectory on its selected frames."""
    prefix, targets = job
    frames, arrays, meta = open_trajectory(prefix)
    basis = np.asarray(meta["basis"], dtype=np.float64)
    rho = float(meta["density"])
    u, energy = frame_times(arrays)
    sel = select_frames(u, targets)
    paths = unwrap_fractional(np.asarray(frames, dtype=np.float64))
    p0 = proximity_metric(paths[sel], paths[0], basis, rho)
    q_inf = proximity_metric(paths[sel], paths[-1], basis, rho)
    first, last = np.array(frames[0]), np.array(frames[-1])
    cache: Dict[int, Tuple[float, int, float, int]] = {}
    for k in np.unique(sel):
        cur = np.array(frames[k])
        w0, n0 = exact_w2(first, cur, basis)
        wi, ni = exact_w2(cur, last, basis)
        cache[int(k)] = (w0, n0, wi, ni)
    vals = np.array([cache[int(k)] for k in sel])
    return {
        "u": u[sel],
        "frame": sel,
        "energy": energy[sel],
        "p0": p0,
        "q_inf": q_inf,
        "w0": np.sqrt(rho) * vals[:, 0],
        "w_inf": np.sqrt(rho) * vals[:, 2],
        "n_re0": vals[:, 1].astype(np.int64),
        "n_re_inf": vals[:, 3].astype(np.int64),
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
    parser.add_argument("--limit", type=int, default=None)

    # ----- sampling -----
    parser.add_argument("--n-times", type=int, default=200)
    parser.add_argument("--tmin", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()
    assert args.n_times > 1 and 0.0 < args.tmin < 1.0, "bad sampling"
    targets = np.concatenate(
        [[0.0], np.logspace(np.log10(args.tmin), 0.0, args.n_times)]
    )
    out_dir = args.traj / "wasserstein"
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # ENSEMBLES
    # =========================================================
    for ens in args.ensembles:
        prefixes = sorted(
            p.with_suffix("") for p in (args.traj / ens).glob("traj_*.json")
        )[: args.limit]
        assert prefixes, f"no trajectories under {args.traj / ens}"
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows: List[Dict[str, np.ndarray]] = pool.map(
                process, [(p, targets) for p in prefixes], chunksize=1
            )
        result = {k: np.stack([r[k] for r in rows]) for k in KEYS}
        result["targets"] = targets
        result["names"] = np.array([p.name for p in prefixes])
        np.savez(out_dir / f"exact_{ens}.npz", **result)
        w_inf0 = result["w0"][:, -1]
        p_inf0 = result["p0"][:, -1]
        print(
            f"{ens}: {len(rows)} trajectories in {time.time() - t0:.0f} s; "
            f"<W(r0, r_inf)> = {w_inf0.mean():.4f}, <p0(inf)> = {p_inf0.mean():.4f}, "
            f"max re-matched = {result['n_re0'].max()}",
            flush=True,
        )
    print("Done.", flush=True)
