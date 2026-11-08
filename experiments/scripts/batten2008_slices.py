#!/usr/bin/env python3

"""
Energy landscapes on 2D slices of configuration space for the 2008 flows.

N = 3 (the appendix setting): Phi(x2, x3) on the unit square of fractional
positions with x1 = 0, exact by translation invariance; flows are drawn in
the relative coordinates (x2 - x1, x3 - x1) mod 1.

N > 3 (any dimension d; the energy is rebuilt from the flows' basis and K):
for every flow, three planes through the Cartesian configuration space
X in R^(N d), each with Phi on a grid and the flow projected onto it:

- start-end: through X(0), spanned by e1 = (X(inf) - X(0))/|.| and e2, the
  direction of the flow's largest excursion perpendicular to e1. Both ends
  lie in the plane; `offplane` stores each recorded point's distance from it.
- pca: through the path mean, spanned by its first two principal components
  (`explained` = their variance fractions).
- pair: all particles frozen at X(inf) except the two that moved most, whose
  positions span the plane (as in the 3-particle figures; in d > 1 each moves
  along its own net displacement). The flow overlay is the projection of the
  pair's positions; the others are not frozen along it.

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
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.cli.relax import select_device

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import (  # noqa: E402
    build_energy,
    cell_lengths,
    chi_tag,
    load_system,
)

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


def make_to_s(n: int, side: object, d: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    """Map Cartesian points (P, N d) of a cubic cell to fractional (P, N, d)."""
    assert n > 0 and np.all(np.asarray(side) > 0.0) and d >= 1, "positive size"

    def to_s(pts: np.ndarray) -> np.ndarray:
        return pts.reshape(-1, n, d) / side

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


Axes = Tuple[np.ndarray, np.ndarray, np.ndarray]  # origin, e1, e2 (flattened R^N)


def start_end_axes(x: np.ndarray) -> Axes:
    """Plane through X(0) spanned by the net displacement and max excursion."""
    y = x - x[0]
    dist = float(np.linalg.norm(y[-1]))
    assert dist > 0.0, "flow did not move"
    e1 = y[-1] / dist
    perp = y - (y @ e1)[:, None] * e1
    pk = perp[int(np.argmax(np.linalg.norm(perp, axis=1)))]
    if np.linalg.norm(pk) < 1e-12 * dist:  # straight path: any orthogonal e2
        pk = np.random.default_rng(0).normal(size=e1.shape)
        pk -= (pk @ e1) * e1
    return x[0], e1, pk / np.linalg.norm(pk)


def pca_axes(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Path mean, its first two principal axes and their variance fractions."""
    c = x.mean(axis=0)
    _, sv, vt = np.linalg.svd(x - c, full_matrices=False)
    return c, vt[0], vt[1], sv[:2] ** 2 / np.sum(sv**2)


def pair_axes(
    x: np.ndarray, d: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """X(inf) with the two most-displaced particles i, j free.

    ``x`` is (F, N d), particle-major. Particle i moves along the unit vector
    of its net displacement (along +x in 1D), so the plane coordinates are
    x_i in 1D and the position along that direction in general.
    """
    assert x.shape[1] % d == 0, "x must be (F, N d)"
    net = (x[-1] - x[0]).reshape(-1, d)
    i, j = (int(v) for v in np.argsort(np.linalg.norm(net, axis=1))[::-1][:2])
    ei, ej = np.zeros_like(x[0]), np.zeros_like(x[0])
    for e, p in ((ei, i), (ej, j)):
        u = net[p] / np.linalg.norm(net[p]) if d > 1 else np.ones(1)
        e[p * d : (p + 1) * d] = u
    origin = x[-1] - (x[-1] @ ei) * ei - (x[-1] @ ej) * ej
    return origin, ei, ej, i, j


def project(z: np.ndarray, axes: Axes) -> Tuple[np.ndarray, np.ndarray]:
    """In-plane coordinates (P, 2) of points z (P, N) and their distance from it."""
    origin, e1, e2 = axes
    y = z - origin
    coords = np.stack([y @ e1, y @ e2], axis=1)
    off = np.linalg.norm(y - coords[:, :1] * e1 - coords[:, 1:] * e2, axis=1)
    return coords, off


def start_end_view(x: np.ndarray, to_s: Callable, ev: Evaluator, g: int) -> Dict:
    """Start-end plane view: Phi grid, projected path and off-plane distance."""
    axes = start_end_axes(x)
    dist = float(np.linalg.norm(x[-1] - x[0]))
    path, off = project(x, axes)
    a, b = path[:, 0], path[:, 1]
    ra = span(min(a.min(), 0.0), max(a.max(), dist), 0.3, 0.0, g)
    rb = span(min(b.min(), 0.0), max(b.max(), 0.0), 0.6, 0.25 * dist, g)
    phi = plane_grid(*axes, ra, rb, to_s, ev)
    return {"a": ra, "b": rb, "phi": phi, "path": path, "offplane": off}


def pca_view(x: np.ndarray, to_s: Callable, ev: Evaluator, g: int) -> Dict:
    """PCA plane view: Phi grid, projected path and variance fractions."""
    c, e1, e2, explained = pca_axes(x)
    p, _ = project(x, (c, e1, e2))
    ra = span(p[:, 0].min(), p[:, 0].max(), 0.3, 1e-3, g)
    rb = span(p[:, 1].min(), p[:, 1].max(), 0.3, 1e-3, g)
    phi = plane_grid(c, e1, e2, ra, rb, to_s, ev)
    return {"a": ra, "b": rb, "phi": phi, "path": p, "explained": explained}


def pair_view(x: np.ndarray, to_s: Callable, ev: Evaluator, g: int, d: int = 1) -> Dict:
    """Two-particle slice: Phi grid with the rest at X(inf), and the pair's path."""
    origin, ei, ej, i, j = pair_axes(x, d)
    path = np.stack([x @ ei, x @ ej], 1)
    ra = span(path[:, 0].min(), path[:, 0].max(), 0.3, 0.75, g)
    rb = span(path[:, 1].min(), path[:, 1].max(), 0.3, 0.75, g)
    phi = plane_grid(origin, ei, ej, ra, rb, to_s, ev)
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
        n, d = data["s0"].shape[1], data["s0"].shape[2]
        if n == 3:
            energy, basis, _ = build_energy(n, Fraction(chi_str))
        else:  # rebuilt from the stored basis and K (1D or 2D flows)
            energy, basis, _, _ = load_system(args.dir, tag)
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
        side = cell_lengths(basis)  # (d,): rectangular cells
        to_s = make_to_s(n, side, d)
        out: Dict[str, list] = {}
        for b in range(data["s0"].shape[0]):
            cnt = int(data["traj_count"][b])
            x = (data["traj_s"][b, :cnt] * side).reshape(cnt, -1)  # Cartesian
            for name, fn in (
                ("se", start_end_view),
                ("pca", pca_view),
                ("pair", partial(pair_view, d=d)),
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
