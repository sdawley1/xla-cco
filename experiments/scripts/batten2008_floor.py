#!/usr/bin/env python3

"""
Valley-floor contours for the 2008 landscape planes (N > 3).

A flat slice shows Phi(origin + a e1 + b e2). Here each grid point (a, b)
instead reports the floor of the valley above it: Phi minimized over the
N - 2 directions orthogonal to the plane, with (a, b) held fixed,

    Phi_floor(a, b) = min_y Phi(origin + a e1 + b e2 + y),   y ⊥ e1, e2,

found by L-BFGS (xla_cco.minimize.relax) on FiberEnergy, whose gradient has
its in-plane part removed. Each grid point starts from the off-plane offset y
of the nearest flow frame (in plane coordinates), so near the path it starts
from configurations the flow visits. Selected flow frames are also relaxed
from r(t) itself, which gives Phi_floor <= Phi(r(t)) there by construction.

Planes: start-end and PCA of batten2008_slices.py for the chosen flows,
on the extent of the existing contour panels (views_chi<tag>.npz).
Output: <root>/N<N>/floor_chi<tag>_<view>.npz (view = se or pca).

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
from typing import Dict, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

# ----- project -----
from xla_cco.cli.relax import select_device
from xla_cco.energy import AbstractEnergy, StealthyEnergy
from xla_cco.minimize import relax
from xla_cco.minimize.lbfgs import LBFGSSettings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import build_energy, chi_tag  # noqa: E402
from batten2008_slices import (  # noqa: E402
    make_evaluator,
    pca_axes,
    project,
    start_end_axes,
)

# ============================================================
# CONSTANTS
# ============================================================

FLOOR = 1e-20  # stop once Phi is below this (ground state to round-off)

# ============================================================
# ENERGY ON A FIBER
# ============================================================


class FiberEnergy(AbstractEnergy):
    """Phi with derivatives restricted to the complement of span(q).

    The value is Phi(s); the gradient (analytic and autodiff) has its
    components along the orthonormal rows of ``q`` removed, so a gradient
    minimizer keeps the in-plane coordinates q . s fixed.
    """

    base: StealthyEnergy
    q: Float[Array, "p n d"]

    def _project(self, g: Array) -> Array:
        coef = jnp.tensordot(self.q, g, axes=((1, 2), (0, 1)))
        return g - jnp.tensordot(coef, self.q, axes=1)

    def __call__(self, s: Array) -> Array:
        # value of Phi(s); d frozen / ds is the projector, so autodiff sees
        # only the off-plane part of s
        delta = s - jax.lax.stop_gradient(s)  # zero value, identity derivative
        frozen = s - (delta - self._project(delta))
        return self.base(frozen)

    def value_and_grad(self, s: Array) -> Tuple[Array, Array]:
        value, grad = self.base.value_and_grad(s)
        return value, self._project(grad)


# ============================================================
# HELPERS
# ============================================================


def plane_offsets(
    x: np.ndarray, axes: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """In-plane coordinates (F, 2) of the frames and their off-plane offsets (F, N)."""
    origin, e1, e2 = axes
    ab, _ = project(x, axes)
    off = x - origin - ab[:, :1] * e1 - ab[:, 1:] * e2
    return ab, off


def grid_starts(
    ra: np.ndarray, rb: np.ndarray, ab: np.ndarray, off: np.ndarray, axes: Tuple
) -> np.ndarray:
    """Start points (G*G, N): each grid point plus its nearest frame's offset."""
    origin, e1, e2 = axes
    aa, bb = np.meshgrid(ra, rb)
    grid = np.stack([aa.ravel(), bb.ravel()], axis=1)
    scale = np.array([np.ptp(ra), np.ptp(rb)])
    assert np.all(scale > 0), "degenerate grid"
    d2 = (((grid[:, None, :] - ab[None, :, :]) / scale) ** 2).sum(-1)
    near = np.argmin(d2, axis=1)
    return origin + grid[:, :1] * e1 + grid[:, 1:] * e2 + off[near]


def relax_points(
    fiber: FiberEnergy,
    pts: np.ndarray,
    side: float,
    settings: LBFGSSettings,
    batch: int,
) -> Dict[str, np.ndarray]:
    """L-BFGS on the fiber from Cartesian points (P, N), ``batch`` lanes per call."""
    rows: Dict[str, list] = {"phi": [], "status": [], "evals": [], "s": []}
    assert batch >= 1, "batch must be positive"
    for a in range(0, pts.shape[0], batch):
        chunk = pts[a : a + batch]
        s0 = np.zeros((batch, pts.shape[1], 1))  # fixed shape: one compile
        s0[: chunk.shape[0], :, 0] = chunk / side
        s0[chunk.shape[0] :] = s0[:1]
        res = relax(fiber, jnp.asarray(s0), settings)
        k = chunk.shape[0]
        rows["phi"].append(np.asarray(res.energy)[:k])
        rows["status"].append(np.asarray(res.status)[:k])
        rows["evals"].append(np.asarray(res.n_evals)[:k])
        rows["s"].append(np.asarray(res.s)[:k, :, 0])
    out = {k: np.concatenate(v) for k, v in rows.items()}
    d = out.pop("s") - pts / side
    d -= np.round(d)  # minimum image (result is wrapped into [0, 1))
    disp = d * side
    q = np.asarray(fiber.q)[:, :, 0]
    out["drift"] = np.linalg.norm(disp @ q.T, axis=1)  # in-plane motion
    out["moved"] = np.linalg.norm(disp, axis=1)
    return out


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/trajectories/batten2008")
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--chi", nargs="+", default=["1/3", "2/3"])
    parser.add_argument("--flow", type=int, default=0)
    parser.add_argument("--grid", type=int, default=41)
    parser.add_argument("--path-points", type=int, default=120)
    parser.add_argument("--max-evals", type=int, default=20_000, help="per pass")
    parser.add_argument("--batch", type=int, default=256, help="lanes per call")
    parser.add_argument(
        "--views", nargs="+", default=["se", "pca"], choices=("se", "pca")
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    assert args.n > 3 and args.grid >= 2 and args.path_points >= 2, "bad arguments"
    select_device(args.device)
    settings = LBFGSSettings(energy_tol=FLOOR, max_evals_per_pass=args.max_evals)

    root = args.root / f"N{args.n}"
    for chi_str in args.chi:
        tag = chi_tag(Fraction(chi_str))
        with np.load(root / f"flows_chi{tag}.npz") as z:
            cnt = int(z["traj_count"][args.flow])
            x = z["traj_s"][args.flow, :cnt, :, 0]
            t = z["traj_t"][args.flow, :cnt]
            e_true = z["traj_energy"][args.flow, :cnt]
        with np.load(root / f"views_chi{tag}.npz") as z:
            extent = {v: (z[f"{v}_a"][args.flow], z[f"{v}_b"][args.flow])
                      for v in ("se", "pca")}  # fmt: skip
        energy, basis, _ = build_energy(args.n, Fraction(chi_str))
        side = float(basis[0, 0])
        evaluate = make_evaluator(energy, 256)
        x = x * side
        u = t / t[-1]
        k_path = np.unique(np.searchsorted(u, np.concatenate(
            [[0.0], np.logspace(np.log10(u[1]), 0.0, args.path_points - 1)])
        ).clip(0, cnt - 1))  # fmt: skip
        c, p1, p2, _ = pca_axes(x)
        base: Dict[str, np.ndarray] = {"path_idx": k_path, "path_u": u[k_path],
                                      "path_true": e_true[k_path]}  # fmt: skip
        planes = {"se": start_end_axes(x), "pca": (c, p1, p2)}
        for view in args.views:
            axes = planes[view]
            out = dict(base)
            t0 = time.time()
            ab, off = plane_offsets(x, axes)
            q = jnp.asarray(np.stack([axes[1], axes[2]])[:, :, None])
            fiber = FiberEnergy(base=energy, q=q)
            ra0, rb0 = extent[view]
            ra = np.linspace(ra0[0], ra0[-1], args.grid)
            rb = np.linspace(rb0[0], rb0[-1], args.grid)
            starts = grid_starts(ra, rb, ab, off, axes)
            both = relax_points(
                fiber, np.concatenate([starts, x[k_path]]), side, settings, args.batch
            )
            n_grid = starts.shape[0]
            grid_res = {k: v[:n_grid] for k, v in both.items()}
            path_res = {k: v[n_grid:] for k, v in both.items()}
            aa, bb = np.meshgrid(ra, rb)
            flat_pts = (
                axes[0] + aa.reshape(-1, 1) * axes[1] + bb.reshape(-1, 1) * axes[2]
            )
            flat = evaluate(flat_pts[:, :, None] / side)
            shape = (args.grid, args.grid)
            out.update({f"{view}_a": ra, f"{view}_b": rb,
                        f"{view}_flat": flat.reshape(shape),
                        f"{view}_path_ab": ab[k_path], f"{view}_ab": ab})  # fmt: skip
            for key, val in grid_res.items():
                out[f"{view}_{key}"] = val.reshape(shape)
            for key, val in path_res.items():
                out[f"{view}_path_{key}"] = val
            below = np.mean(grid_res["phi"] < 1e-18)
            codes, counts = np.unique(grid_res["status"], return_counts=True)
            drift = max(grid_res["drift"].max(), path_res["drift"].max())
            ok = path_res["phi"] <= e_true[k_path] * (1 + 1e-9) + FLOOR  # round-off
            print(
                f"chi={chi_str} {view}: grid {args.grid}^2, floor < 1e-18 at "
                f"{below:.0%}; floor median {np.median(grid_res['phi']):.1e}, "
                f"max {np.max(grid_res['phi']):.1e}; statuses "
                f"{dict(zip(codes.tolist(), counts.tolist(), strict=True))}; "
                f"max in-plane drift {drift:.1e}; path floor <= true at "
                f"{np.mean(ok):.0%}; {time.time() - t0:.0f} s",
                flush=True,
            )
            np.savez(root / f"floor_chi{tag}_{view}.npz", **out)
    print("Done.", flush=True)
