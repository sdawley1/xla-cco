#!/usr/bin/env python3

"""
Hessian of Phi along the 2008 flows: stiffness spectra and valley cross-sections.

At sample times t/T along every flow (N > 3) the exact Cartesian Hessian
H = d^2 Phi / dx dx (closed form, StealthyEnergy.hessian / L^2) is
diagonalized. Stored per flow and sample: the sorted eigenvalues, Phi and
|grad Phi|.

For flow 0 at a few of those samples, Phi is also evaluated on the plane
through r(t) spanned by

- tau = -grad Phi / |grad Phi|, the direction the flow is moving, and
- v, the eigenvector of the largest eigenvalue, made orthogonal to tau,

with the flow's own frames projected onto it. Along v, Phi rises as
Phi + lambda beta^2 / 2 (v is orthogonal to the gradient), so the valley
half-width at which Phi grows by a factor f is sqrt(2 (f - 1) Phi / lambda).
The plane spans beta up to f = 100 and, along tau, the distance the flow
covers while Phi falls tenfold.

Output: <root>/N<N>/hessian_chi<tag>.npz.

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
from typing import Dict, List, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.cli.relax import select_device

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import build_energy, chi_tag  # noqa: E402
from batten2008_slices import make_evaluator  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

SAMPLES = (0.0, 1e-5, 1e-3, 1e-2, 1e-1, 1.0)  # t/T (nearest frame at or after)
SECTIONS = (1e-5, 1e-2, 1e-1)  # t/T of the flow-0 cross-sections
GROWTH = 100.0  # cross-section half-width along v: Phi grows by this factor
ZERO_REL = (1e-9, 1e-6)  # |lambda| < z * lambda_max counts as a zero mode

# ============================================================
# HELPERS
# ============================================================


def sample_frames(u: np.ndarray, times: Tuple[float, ...]) -> np.ndarray:
    """Index of the first frame with u >= each time (last frame for 1.0)."""
    idx = np.searchsorted(u, np.array(times), side="left")
    return np.minimum(idx, u.shape[0] - 1)


def section_axes(grad: np.ndarray, vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """tau = -grad/|grad| and the stiffest eigenvector made orthogonal to tau."""
    gn = float(np.linalg.norm(grad))
    assert gn > 0.0, "zero gradient: no flow direction"
    tau = -grad / gn
    v = vecs[:, -1] - (vecs[:, -1] @ tau) * tau
    assert np.linalg.norm(v) > 1e-6, "stiffest mode parallel to the flow"
    return tau, v / np.linalg.norm(v)


def tenfold_distance(x: np.ndarray, e: np.ndarray, k: int) -> float:
    """Distance from frame k to the first later frame with Phi <= Phi_k / 10."""
    later = np.flatnonzero(e[k:] <= e[k] / 10.0)
    j = k + (int(later[0]) if later.size else e.shape[0] - 1 - k)
    return max(float(np.linalg.norm(x[j] - x[k])), 1e-9)


def cross_section(
    x: np.ndarray, e: np.ndarray, k: int, grad: np.ndarray, lam: np.ndarray,
    vecs: np.ndarray, ev: object, side: float, g: int,
) -> Dict[str, np.ndarray]:  # fmt: skip
    """Phi on the (tau, v) plane through frame k, with the path projected."""
    tau, v = section_axes(grad, vecs)
    half_b = np.sqrt(2.0 * (GROWTH - 1.0) * e[k] / lam[-1])
    half_a = tenfold_distance(x, e, k)
    ra = np.linspace(-0.25 * half_a, 1.25 * half_a, g)
    rb = np.linspace(-half_b, half_b, g)
    aa, bb = np.meshgrid(ra, rb)
    pts = x[k][None, :] + aa.reshape(-1, 1) * tau + bb.reshape(-1, 1) * v
    phi = ev(pts.reshape(-1, x.shape[1], 1) / side).reshape(aa.shape)  # type: ignore[operator]
    y = x - x[k]
    path = np.stack([y @ tau, y @ v], axis=1)
    off = np.linalg.norm(y - path[:, :1] * tau - path[:, 1:] * v, axis=1)
    return {"a": ra, "b": rb, "phi": phi, "path": path, "off": off,
            "half_b": np.array(half_b), "half_a": np.array(half_a)}  # fmt: skip


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
    parser.add_argument("--grid", type=int, default=81)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    assert all(n > 3 for n in args.sizes) and args.grid >= 3, "bad arguments"
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
            hess = jax.jit(lambda s: energy.hessian(s))  # noqa: B023
            vag = jax.jit(lambda s: energy.value_and_grad(s))  # noqa: B023
            ev = make_evaluator(energy, 512)
            out: Dict[str, List[np.ndarray]] = {}
            sections: Dict[str, np.ndarray] = {}
            for b in range(flows["s0"].shape[0]):
                cnt = int(flows["traj_count"][b])
                t = flows["traj_t"][b, :cnt]
                u, x = t / t[-1], flows["traj_s"][b, :cnt, :, 0] * side
                e = flows["traj_energy"][b, :cnt]
                rows = {"lam": [], "phi": [], "gnorm": [], "u": []}
                for i, k in enumerate(sample_frames(u, SAMPLES)):
                    s = jnp.asarray(x[k][:, None] / side)
                    h = np.asarray(hess(s)) / side**2  # Cartesian Hessian
                    lam, vecs = np.linalg.eigh(0.5 * (h + h.T))
                    phi, g = vag(s)
                    grad = np.asarray(g)[:, 0] / side  # Cartesian gradient
                    rows["lam"].append(lam)
                    rows["phi"].append(np.array(float(phi)))
                    rows["gnorm"].append(np.array(np.linalg.norm(grad)))
                    rows["u"].append(np.array(u[k]))
                    if b == 0 and SAMPLES[i] in SECTIONS:
                        j = SECTIONS.index(SAMPLES[i])
                        sec = cross_section(
                            x, e, k, grad, lam, vecs, ev, side, args.grid
                        )
                        for name, val in sec.items():
                            sections[f"sec{j}_{name}"] = val
                        sections[f"sec{j}_u"] = np.array(u[k])
                        sections[f"sec{j}_lam_max"] = np.array(lam[-1])
                for key, val in rows.items():
                    out.setdefault(key, []).append(np.stack(val))
            packed = {k: np.stack(v) for k, v in out.items()}
            lam = packed["lam"]
            for z_rel, name in zip(ZERO_REL, ("n_zero", "n_zero_loose"), strict=True):
                packed[name] = (np.abs(lam) < z_rel * lam[..., -1:]).sum(axis=-1)
            zero = np.abs(lam) < ZERO_REL[0] * lam[..., -1:]
            packed["n_negative"] = ((lam < 0) & ~zero).sum(axis=-1)
            np.savez(root / f"hessian_chi{tag}.npz", samples=np.array(SAMPLES),
                     **packed, **sections)  # fmt: skip
            nz, nl = packed["n_zero"][:, -1], packed["n_zero_loose"][:, -1]
            nn = packed["n_negative"]
            print(
                f"N={n} chi={chi_str}: zero modes at the end {nz.min()}-{nz.max()} "
                f"(loose {nl.min()}-{nl.max()}); "
                f"negative modes per sample (median) {np.median(nn, axis=0)}; "
                f"lambda_max per sample (median) "
                f"{np.median(lam[..., -1], axis=0).round(2)}; "
                f"sections {sorted({k.split('_')[0] for k in sections})}; "
                f"{time.time() - t0:.0f} s",
                flush=True,
            )
    print("Done.", flush=True)
