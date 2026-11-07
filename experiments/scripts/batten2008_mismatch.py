#!/usr/bin/env python3

"""
How well the 2D landscape slices report the energy along the 2008 flows.

The contour panels of batten2008_plot.py show Phi on a plane, and the flow is
drawn as its projection onto that plane. The value under the drawn path is
therefore Phi at the projected point, not Phi(r(t)). This script measures the
gap, frame by frame, for every N > 3 flow and each plane of
batten2008_slices.py:

- start-end and PCA: Phi(origin + a(t) e1 + b(t) e2), with (a, b) the
  projection of r(t);
- two-particle slice: Phi with all particles at r(inf) except i, j, which sit
  at x_i(t), x_j(t).

Phi is evaluated exactly at those points (the contour grids sample the same
function). Output: <root>/N<N>/mismatch_chi<tag>.npz with, per flow
(NaN padded), t, the true energy, and <view>_phi_plane and <view>_offplane.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

# ----- numerics -----
import numpy as np

# ----- project -----
from xla_cco.cli.relax import select_device

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import build_energy, chi_tag  # noqa: E402
from batten2008_slices import (  # noqa: E402
    Evaluator,
    make_evaluator,
    make_to_s,
    pair_axes,
    pca_axes,
    project,
    start_end_axes,
)
from batten2008_wasserstein import pad  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

VIEWS = ("se", "pca", "pair")
FLOOR = 1e-20  # Phi below this is ground-state round-off (same floor as the plots)

# ============================================================
# HELPERS
# ============================================================


def plane_points(x: np.ndarray) -> Dict[str, np.ndarray]:
    """For each view, the in-plane point under every frame of x (F, N)."""
    assert x.ndim == 2 and x.shape[0] >= 2, "need a recorded path"
    c, p1, p2, _ = pca_axes(x)
    origin, _, _, i, j = pair_axes(x)
    out: Dict[str, np.ndarray] = {}
    for view, axes in (("se", start_end_axes(x)), ("pca", (c, p1, p2))):
        ab, off = project(x, axes)
        out[view] = axes[0][None, :] + ab[:, :1] * axes[1] + ab[:, 1:] * axes[2]
        out[f"{view}_off"] = off
    pts = np.repeat(origin[None, :], x.shape[0], axis=0)
    pts[:, i], pts[:, j] = x[:, i], x[:, j]
    out["pair"] = pts
    out["pair_off"] = np.linalg.norm(x - pts, axis=1)
    return out


def analyze(x: np.ndarray, to_s: object, ev: Evaluator) -> Dict[str, np.ndarray]:
    """Phi at the projected points and the off-plane distance, per view."""
    pts = plane_points(x)
    out: Dict[str, np.ndarray] = {}
    for view in VIEWS:
        out[f"{view}_phi_plane"] = ev(to_s(pts[view]))  # type: ignore[operator]
        out[f"{view}_offplane"] = pts[f"{view}_off"]
    return out


def log_ratio(plane: np.ndarray, true: np.ndarray) -> np.ndarray:
    """log10 of the plane-to-true energy ratio, both floored at FLOOR."""
    return np.log10(np.maximum(plane, FLOOR)) - np.log10(np.maximum(true, FLOOR))


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/trajectories/batten2008")
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000])
    parser.add_argument("--chi", nargs="+", default=["1/3", "2/3"])
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    assert all(n > 3 for n in args.sizes), "N = 3 needs no slice (exact landscape)"
    select_device(args.device)

    for n in args.sizes:
        root = args.root / f"N{n}"
        for chi_str in args.chi:
            t0 = time.time()
            tag = chi_tag(Fraction(chi_str))
            with np.load(root / f"flows_chi{tag}.npz") as z:
                flows = {k: z[k] for k in z.files}
            energy, basis, _ = build_energy(n, Fraction(chi_str))
            side = float(basis[0, 0])
            to_s, ev = make_to_s(n, side), make_evaluator(energy, args.chunk)
            per: Dict[str, List[np.ndarray]] = {}
            for b in range(flows["s0"].shape[0]):
                cnt = int(flows["traj_count"][b])
                x = flows["traj_s"][b, :cnt, :, 0] * side  # Cartesian, unwrapped
                res = analyze(x, to_s, ev)
                res["t"] = flows["traj_t"][b, :cnt]
                res["energy"] = flows["traj_energy"][b, :cnt]
                for k, v in res.items():
                    per.setdefault(k, []).append(v)
            np.savez(
                root / f"mismatch_chi{tag}.npz", **{k: pad(v) for k, v in per.items()}
            )
            msg = [f"N={n} chi={chi_str}: median log10(Phi_plane/Phi)"]
            for view in VIEWS:
                lr = [
                    log_ratio(p, e)
                    for p, e in zip(
                        per[f"{view}_phi_plane"], per["energy"], strict=True
                    )
                ]
                late = np.concatenate([r[r.shape[0] // 2 :] for r in lr])
                msg.append(
                    f"{view} all {np.median(np.concatenate(lr)):+.2f}, "
                    f"2nd half {np.median(late):+.2f}"
                )
            print("; ".join(msg) + f"; {time.time() - t0:.0f} s", flush=True)
    print("Done.", flush=True)
