#!/usr/bin/env python3

"""
Energy landscapes on 2D slices of configuration space for the 2008 flows.

N = 3 (the appendix setting): Phi(x2, x3) on the unit square of fractional
positions with x1 = 0, exact by translation invariance; flows are drawn in
the relative coordinates (x2 - x1, x3 - x1) mod 1.

N > 3: for every flow, three planes through the Cartesian configuration
space X in R^(N d), each with Phi on a grid and the flow projected onto it:

- start-end: through X(0), spanned by e1 = (X(inf) - X(0))/|.| and e2, the
  direction of the flow's largest excursion perpendicular to e1. Both ends
  lie in the plane; `offplane` stores each recorded point's distance from it.
- pca: through the path mean, spanned by its first two principal components
  (`explained` = their variance fractions).
- pair: all particles frozen at X(inf) except the two that moved most, whose
  positions span the plane (as in the 3-particle figures). The flow overlay
  is the projection (x_i(t), x_j(t)); the others are not frozen along it.

Output: <dir>/landscape_chi<tag>.npz (N = 3) or views_chi<tag>.npz.

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
from typing import Callable, Dict

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.cli.relax import select_device

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import build_energy, chi_tag  # noqa: E402

# ============================================================
# HELPERS
# ============================================================

Evaluator = Callable[[np.ndarray], np.ndarray]


def make_evaluator(energy: object, chunk: int) -> Evaluator:
    """Return f(s (B, N, d) fractional) -> Phi (B,), evaluated in chunks."""
    assert chunk > 0, "chunk must be positive"
    batched = jax.jit(jax.vmap(energy))  # type: ignore[arg-type]

    def evaluate(s: np.ndarray) -> np.ndarray:
        out = np.empty(s.shape[0])
        for a in range(0, s.shape[0], chunk):
            out[a : a + chunk] = np.asarray(batched(jnp.asarray(s[a : a + chunk])))
        return out

    return evaluate


def make_to_s(n: int, side: float) -> Callable[[np.ndarray], np.ndarray]:
    """Map flattened Cartesian points (P, N) of a 1D cell to (P, N, 1) fractional."""
    assert n > 0 and side > 0.0, "positive size"

    def to_s(pts: np.ndarray) -> np.ndarray:
        return pts.reshape(-1, n, 1) / side

    return to_s


def plane_grid(
    origin: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    ra: np.ndarray,
    rb: np.ndarray,
    to_s: Callable[[np.ndarray], np.ndarray],
    evaluate: Evaluator,
) -> np.ndarray:
    """Phi(origin + a e1 + b e2) on the (len(rb), len(ra)) grid."""
    aa, bb = np.meshgrid(ra, rb)
    pts = origin[None, :] + aa.reshape(-1, 1) * e1 + bb.reshape(-1, 1) * e2
    return evaluate(to_s(pts)).reshape(aa.shape)


def span(lo: float, hi: float, pad: float, min_half: float, n: int) -> np.ndarray:
    """Grid covering [lo, hi] with relative padding and a minimum half-width."""
    c, half = 0.5 * (lo + hi), max(0.5 * (hi - lo) * (1 + pad), min_half)
    return np.linspace(c - half, c + half, n)


# ============================================================
# VIEWS (N > 3)
# ============================================================


def start_end_view(x: np.ndarray, to_s: Callable, ev: Evaluator, g: int) -> Dict:
    """Plane through X(0) spanned by the net displacement and max excursion."""
    y = x - x[0]
    dist = float(np.linalg.norm(y[-1]))
    assert dist > 0.0, "flow did not move"
    e1 = y[-1] / dist
    a = y @ e1
    perp = y - a[:, None] * e1
    k = int(np.argmax(np.linalg.norm(perp, axis=1)))
    pk = perp[k]
    if np.linalg.norm(pk) < 1e-12 * dist:  # straight path: any orthogonal e2
        pk = np.random.default_rng(0).normal(size=e1.shape)
        pk -= (pk @ e1) * e1
    e2 = pk / np.linalg.norm(pk)
    b = y @ e2
    ra = span(min(a.min(), 0.0), max(a.max(), dist), 0.3, 0.0, g)
    rb = span(min(b.min(), 0.0), max(b.max(), 0.0), 0.6, 0.25 * dist, g)
    off = np.linalg.norm(y - a[:, None] * e1 - b[:, None] * e2, axis=1)
    phi = plane_grid(x[0], e1, e2, ra, rb, to_s, ev)
    return {"a": ra, "b": rb, "phi": phi, "path": np.stack([a, b], 1), "offplane": off}


def pca_view(x: np.ndarray, to_s: Callable, ev: Evaluator, g: int) -> Dict:
    """Plane through the path mean spanned by its first two principal axes."""
    c = x.mean(axis=0)
    _, sv, vt = np.linalg.svd(x - c, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    p = (x - c) @ np.stack([e1, e2], 1)
    ra = span(p[:, 0].min(), p[:, 0].max(), 0.3, 1e-3, g)
    rb = span(p[:, 1].min(), p[:, 1].max(), 0.3, 1e-3, g)
    phi = plane_grid(c, e1, e2, ra, rb, to_s, ev)
    explained = sv[:2] ** 2 / np.sum(sv**2)
    return {"a": ra, "b": rb, "phi": phi, "path": p, "explained": explained}


def pair_view(x: np.ndarray, to_s: Callable, ev: Evaluator, g: int) -> Dict:
    """Two particles free (largest displacements), the rest at X(inf); d = 1."""
    disp = np.abs(x[-1] - x[0])
    i, j = (int(v) for v in np.argsort(disp)[::-1][:2])
    ra = span(x[:, i].min(), x[:, i].max(), 0.3, 0.75, g)
    rb = span(x[:, j].min(), x[:, j].max(), 0.3, 0.75, g)
    ei, ej = np.zeros_like(x[0]), np.zeros_like(x[0])
    ei[i], ej[j] = 1.0, 1.0
    origin = x[-1].copy()
    origin[i], origin[j] = 0.0, 0.0
    phi = plane_grid(origin, ei, ej, ra, rb, to_s, ev)
    path = np.stack([x[:, i], x[:, j]], 1)
    return {"a": ra, "b": rb, "phi": phi, "path": path, "pair": np.array([i, j])}


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", type=Path, required=True, help="flows directory")
    parser.add_argument("--chi", nargs="+", default=["1/3", "2/3"])
    parser.add_argument("--grid", type=int, default=101)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    select_device(args.device)

    for chi_str in args.chi:
        tag = chi_tag(Fraction(chi_str))
        with np.load(args.dir / f"flows_chi{tag}.npz") as npz:
            data = {k: npz[k] for k in npz.files}
        n = data["s0"].shape[1]
        energy, basis, _ = build_energy(n, Fraction(chi_str))
        ev = make_evaluator(energy, args.chunk)
        t0 = time.time()
        if n == 3:
            # x1 = 0; grid over the fractional positions of particles 2 and 3.
            u = np.linspace(0.0, 1.0, 4 * args.grid + 1)
            xx2, xx3 = np.meshgrid(u, u)
            s = np.stack([np.zeros_like(xx2), xx2, xx3], -1).reshape(-1, 3, 1)
            phi = ev(s).reshape(xx2.shape)
            np.savez(args.dir / f"landscape_chi{tag}.npz", x=u, phi=phi)
            print(
                f"N=3 chi={chi_str}: landscape {phi.shape}, "
                f"Phi in [{phi.min():.2e}, {phi.max():.3f}]",
                flush=True,
            )
            continue
        side = float(basis[0, 0])
        to_s = make_to_s(n, side)
        out: Dict[str, list] = {}
        for b in range(data["s0"].shape[0]):
            cnt = int(data["traj_count"][b])
            x = data["traj_s"][b, :cnt].reshape(cnt, -1) * side  # Cartesian, unwrapped
            for name, fn in (
                ("se", start_end_view),
                ("pca", pca_view),
                ("pair", pair_view),
            ):
                view = fn(x, to_s, ev, args.grid)
                for k, v in view.items():
                    out.setdefault(f"{name}_{k}", []).append(v)
        packed: Dict[str, np.ndarray] = {}
        for k, vals in out.items():
            if vals[0].ndim == 2 and k.endswith(("path",)):  # ragged (count, 2)
                width = max(v.shape[0] for v in vals)
                arr = np.full((len(vals), width, 2), np.nan)
                for i, v in enumerate(vals):
                    arr[i, : v.shape[0]] = v
                packed[k] = arr
            elif k.endswith("offplane"):
                width = max(v.shape[0] for v in vals)
                arr = np.full((len(vals), width), np.nan)
                for i, v in enumerate(vals):
                    arr[i, : v.shape[0]] = v
                packed[k] = arr
            else:
                packed[k] = np.stack(vals)
        np.savez(args.dir / f"views_chi{tag}.npz", **packed)
        print(
            f"N={n} chi={chi_str}: 3 views x {data['s0'].shape[0]} flows on "
            f"{args.grid}^2 grids in {time.time() - t0:.0f} s",
            flush=True,
        )
    print("Done.", flush=True)
