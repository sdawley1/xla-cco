#!/usr/bin/env python3

"""
Exact Wasserstein-2 analysis of the Batten, Stillinger & Torquato (2008) flows.

For every recorded frame r(t) of every gradient flow (batten2008_flows.py),
the exact optimal matching on the circle (periodic_assignment_1d) is
computed against r(0) and against r(inf). This gives:

- W_0(t) = W(r(0), r(t)) and W_inf(t) = W(r(t), r(inf)), scaled like the
  proximity metric (rho = 1), with the labelled p_0(t) = |r(t) - r(0)| and
  q_inf(t) = |r(t) - r(inf)| (minimum image) and the number of re-matched
  particles n_re;
- the relabelled flows r(0) + d_0(t) and r(inf) + d_inf(t), i.e. each frame
  relabelled into the reference's order (d: optimal minimum-image
  displacements);
- the W_2 geodesic g(s) = r(0) + s d_0(inf), s in [0, 1] (displacement
  interpolation), with Phi along it.

N = 3: curves in the relative coordinates (x2 - x1, x3 - x1) / L (exact).
N > 3: curves projected onto the start-end, PCA and two-particle planes of
batten2008_slices.py (the same planes as the landscape figures), with each
point's distance from the plane.

Outputs per N: <root>/N<N>/wasserstein_chi<tag>.npz (per flow, NaN padded)
and <root>/N<N>/w2_summary.npz (ensemble bands on a t/T grid, in the layout
plot_wasserstein.py reads; ensembles "chi1-3" and "chi2-3").

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
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.analysis.transport import periodic_assignment_1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_wasserstein import band  # noqa: E402
from batten2008_flows import build_energy, chi_tag  # noqa: E402
from batten2008_slices import pair_axes, pca_axes, project, start_end_axes  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

CURVES = ("flow", "w0", "winf")  # labelled flow, relabelled to r(0), to r(inf)
VIEWS = ("se", "pca", "pair")
CHUNK = 500  # frames per exact-assignment call
GEO_BATCH = 16  # geodesic points per energy call

# ============================================================
# HELPERS
# ============================================================


def min_image(d: np.ndarray, length: float) -> np.ndarray:
    """Minimum-image displacement on a circle of circumference ``length``."""
    out: np.ndarray = d - length * np.round(d / length)
    return out


def match(reference: np.ndarray, x: np.ndarray, length: float) -> Dict[str, np.ndarray]:
    """Exact assignment of every frame of ``x`` (F, N) to ``reference``."""
    parts = [
        periodic_assignment_1d(reference, x[a : a + CHUNK], length)
        for a in range(0, x.shape[0], CHUNK)
    ]
    perm = np.concatenate([p.perm for p in parts])
    return {
        "w": np.sqrt(np.concatenate([p.cost for p in parts])),
        "disp": np.concatenate([p.disp for p in parts]),
        "n_re": np.sum(perm != np.arange(x.shape[1]), axis=1),
    }


def analyze_flow(
    x: np.ndarray, length: float, n_geo: int, energy_of: object
) -> Dict[str, np.ndarray]:
    """W_2 quantities, relabelled flows and geodesic of one Cartesian path x."""
    m0, mi = match(x[0], x, length), match(x[-1], x, length)
    assert np.isclose(m0["w"][-1], mi["w"][0], rtol=1e-10, atol=1e-12), (
        "W not symmetric"
    )
    s = np.linspace(0.0, 1.0, n_geo)
    geo = x[0][None, :] + s[:, None] * m0["disp"][-1][None, :]
    # Batched evaluation materializes (batch, M, N) phases: keep batches small.
    phi_geo = np.concatenate(
        [
            np.asarray(energy_of(jnp.asarray(geo[a : a + GEO_BATCH, :, None] / length)))  # type: ignore[operator]
            for a in range(0, n_geo, GEO_BATCH)
        ]
    )
    return {
        "w0": m0["w"],
        "w_inf": mi["w"],
        "p0": np.linalg.norm(min_image(x - x[0], length), axis=1),
        "q_inf": np.linalg.norm(min_image(x - x[-1], length), axis=1),
        "n_re0": m0["n_re"],
        "n_re_inf": mi["n_re"],
        "curve_flow": x,
        "curve_w0": x[0][None, :] + m0["disp"],
        "curve_winf": x[-1][None, :] + mi["disp"],
        "geo": geo,
        "geo_energy": phi_geo,
        "geo_s": s,
    }


def relative(z: np.ndarray, length: float) -> np.ndarray:
    """N = 3 relative fractional coordinates ((x2 - x1)/L, (x3 - x1)/L), unwrapped."""
    assert z.shape[1] == 3, "relative coordinates are for N = 3"
    return (z[:, 1:] - z[:, :1]) / length


def project_all(x: np.ndarray, res: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Project the curves and the geodesic onto the three planes of one flow."""
    c, e1, e2, _ = pca_axes(x)
    origin, ei, ej, _, _ = pair_axes(x)
    planes = {"se": start_end_axes(x), "pca": (c, e1, e2), "pair": (origin, ei, ej)}
    out: Dict[str, np.ndarray] = {}
    for view, axes in planes.items():
        for name in (*CURVES, "geo"):
            z = res["geo"] if name == "geo" else res[f"curve_{name}"]
            coords, off = project(z, axes)
            out[f"{view}_{name}"] = coords
            out[f"{view}_{name}_off"] = off
    return out


def pad(rows: List[np.ndarray]) -> np.ndarray:
    """Stack ragged leading dimensions into a NaN-padded array."""
    width = max(r.shape[0] for r in rows)
    arr = np.full((len(rows), width) + rows[0].shape[1:], np.nan)
    for i, r in enumerate(rows):
        arr[i, : r.shape[0]] = r
    return arr


def summary_bands(per_flow: Dict[str, List[np.ndarray]], grid: np.ndarray) -> Dict:
    """Interpolate per-flow series onto the t/T grid; mean and 10/90 % bands."""
    curves: Dict[str, List[np.ndarray]] = {}
    for i, t in enumerate(per_flow["t"]):
        u = t / t[-1]
        series = {
            "w0_dist": per_flow["w0"][i][-1] - per_flow["w0"][i],
            "p0_dist": per_flow["p0"][i][-1] - per_flow["p0"][i],
            "w_inf": per_flow["w_inf"][i],
            "q_inf": per_flow["q_inf"][i],
            "n_re0": per_flow["n_re0"][i].astype(float),
            "n_re_inf": per_flow["n_re_inf"][i].astype(float),
            "log10_energy": np.log10(per_flow["energy"][i]),
        }
        for k, v in series.items():
            curves.setdefault(k, []).append(np.interp(grid, u, v))
    out: Dict[str, np.ndarray] = {}
    for k, v in curves.items():
        out.update(band(v, k))
    return out


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/trajectories/batten2008")
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[3, 1000])
    parser.add_argument("--chi", nargs="+", default=["1/3", "2/3"])
    parser.add_argument("--geodesic-points", type=int, default=201)
    parser.add_argument("--grid-points", type=int, default=300)
    args = parser.parse_args()
    assert args.geodesic_points >= 2 and args.grid_points >= 2, "bad sizes"

    for n in args.sizes:
        root = args.root / f"N{n}"
        summary: Dict[str, np.ndarray] = {}
        for chi_str in args.chi:
            t0 = time.time()
            tag = chi_tag(Fraction(chi_str))
            with np.load(root / f"flows_chi{tag}.npz") as z:
                flows = {k: z[k] for k in z.files}
            energy, basis, _ = build_energy(n, Fraction(chi_str))
            length = float(basis[0, 0])
            energy_of = jax.jit(jax.vmap(energy))
            per: Dict[str, List[np.ndarray]] = {}
            for b in range(flows["s0"].shape[0]):
                cnt = int(flows["traj_count"][b])
                x = flows["traj_s"][b, :cnt, :, 0] * length  # Cartesian, unwrapped
                res = analyze_flow(x, length, args.geodesic_points, energy_of)
                keep = {
                    k: res[k]
                    for k in (
                        "w0",
                        "w_inf",
                        "p0",
                        "q_inf",
                        "n_re0",
                        "n_re_inf",
                        "geo_energy",
                    )
                }
                keep["t"] = flows["traj_t"][b, :cnt]
                keep["length"] = flows["traj_length"][b, :cnt]
                keep["energy"] = flows["traj_energy"][b, :cnt]
                if n == 3:
                    for name in (*CURVES, "geo"):
                        z = res["geo"] if name == "geo" else res[f"curve_{name}"]
                        keep[f"rel_{name}"] = relative(z, length)
                else:
                    keep.update(project_all(x, res))
                for k, v in keep.items():
                    per.setdefault(k, []).append(v)
            out = {k: pad(v) for k, v in per.items()}
            out["geo_s"] = np.linspace(0.0, 1.0, args.geodesic_points)
            np.savez(root / f"wasserstein_chi{tag}.npz", **out)

            # ----- ensemble bands for plot_wasserstein.py -----
            u_min = min(float(t[1] / t[-1]) for t in per["t"])
            grid = np.concatenate(
                [[0.0], np.logspace(np.log10(u_min), 0.0, args.grid_points)]
            )
            e = f"chi{tag}"
            summary[f"{e}/grid"] = grid
            summary[f"{e}/w0_inf"] = np.array([w[-1] for w in per["w0"]])
            summary[f"{e}/p0_inf"] = np.array([p[-1] for p in per["p0"]])
            for k, v in summary_bands(per, grid).items():
                summary[f"{e}/{k}"] = v
            n_re = max(int(np.max(v)) for v in per["n_re0"] + per["n_re_inf"])
            gap = max(
                float(np.max(np.abs(w - p)))
                for w, p in zip(per["w0"], per["p0"], strict=True)
            )
            print(
                f"N={n} chi={chi_str}: {len(per['t'])} flows, "
                f"{sum(t.shape[0] for t in per['t'])} frames; max n_re {n_re}; "
                f"max |W0 - p0| {gap:.1e}; geodesic max Phi median "
                f"{np.median([g.max() for g in per['geo_energy']]):.3g}; "
                f"{time.time() - t0:.0f} s",
                flush=True,
            )
        np.savez(root / "w2_summary.npz", **summary)
    print("Done.", flush=True)
