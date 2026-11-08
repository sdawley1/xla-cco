#!/usr/bin/env python3

"""
Hessian of Phi along the 2008 flows: stiffness spectra and valley cross-sections.

At energy levels along every flow (N > 3) the exact Cartesian Hessian
H = d^2 Phi / dx dx (closed form, StealthyEnergy.hessian / L^2) is
diagonalized. Stored per flow and sample: the sorted eigenvalues, Phi,
|grad Phi|, the curvature of Phi along each density wave n = 1..M
(density_wave_stiffness; k_n^2 at a decoupled ground state), and the mean
wavelength power of the negative-curvature eigenvectors and of as many of
the stiffest ones (wavelength_power).

The clock is the energy itself: Phi decreases monotonically along a gradient
flow (dPhi/dt = -|grad Phi|^2), so the samples are the first frames with
log10 Phi at or below LEVELS (every quarter decade from 1e3 to 1e-20), and
every flow is compared at equal energy. (t/T is not used: the stored flows'
last few Rosenbrock steps are huge, so T is set by round-off-level motion.)
If tail_chi<tag>.npz exists (batten2008_tail.py), its capped-step frames
replace the stored ones below its restart energy, which resolves the last
decades; without it, deep levels fall on a few coarse frames.

For flow 0 at a few of those levels, Phi is also evaluated on the plane
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
from batten2008_flows import cell_lengths, chi_tag, load_system  # noqa: E402
from batten2008_slices import make_evaluator  # noqa: E402
from batten2008_tail import tail_frames  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

# log10 Phi levels (first frame at or below each): every quarter decade
LEVELS = tuple(round(3.0 - 0.25 * i, 2) for i in range(93))  # 3.0 ... -20.0
SECTIONS = (1.0, -1.0, -8.0)  # log10 Phi of the flow-0 cross-sections
GROWTH = 100.0  # cross-section half-width along v: Phi grows by this factor
ZERO_REL = (1e-9, 1e-6)  # |lambda| < z * lambda_max counts as a zero mode

# ============================================================
# HELPERS
# ============================================================


def density_wave_stiffness(
    x: np.ndarray, h: np.ndarray, kvec: np.ndarray
) -> np.ndarray:
    """Rayleigh quotients of H along the density-wave directions, (M, 2).

    ``x`` (N, d) Cartesian, ``h`` (N d, N d) Cartesian Hessian, ``kvec`` (M, d).
    For wavevector k the directions are the gradients of Re rho(k) and
    Im rho(k) with respect to r: -sin(k . r_j) k and cos(k . r_j) k,
    normalized. At a ground state they are eigenvectors with eigenvalue
    |k|^2 when the modes decouple; elsewhere the quotient is the curvature
    of Phi along that density wave.
    """
    n, d = x.shape
    m = kvec.shape[0]
    assert h.shape == (n * d, n * d) and kvec.shape[1] == d and m >= 1, "shapes"
    phase = x @ kvec.T  # (N, M)
    out = np.empty((m, 2))
    for c, wave in enumerate((-np.sin(phase), np.cos(phase))):
        g = (wave[:, None, :] * kvec.T[None, :, :]).reshape(n * d, m)
        g = g / np.linalg.norm(g, axis=0, keepdims=True)
        out[:, c] = np.einsum("im,im->m", g, h @ g)
    return out


def qvectors(n: int, d: int, side: object) -> np.ndarray:
    """Wavevectors 2 pi q / L for the wavelength spectra, sorted by |q|.

    1D: q = 1..N/2. d > 1: half-space integer q with every |q_a| <= L_a, which
    reaches past K for the 2D chi values used here. ``side`` is the cell
    length, a scalar (square cell) or one per axis (rectangular cell).
    """
    sides = np.broadcast_to(np.asarray(side, dtype=np.float64), (d,))
    assert np.all(sides > 0.0), "cell lengths must be positive"
    if d == 1:
        ints = np.arange(1, n // 2 + 1, dtype=np.float64)[:, None]
    else:
        lims = [int(round(v)) for v in sides]
        axes = np.meshgrid(*[np.arange(-v, v + 1) for v in lims], indexing="ij")
        grid = np.stack([a.ravel() for a in axes], axis=1)
        first = np.array([row[np.flatnonzero(row)[0]] if row.any() else 0
                          for row in grid])  # fmt: skip
        ints = grid[first > 0].astype(np.float64)  # half space, no q = 0
    q = 2.0 * np.pi * ints / sides
    return q[np.argsort(np.einsum("qa,qa->q", q, q), kind="stable")]


def wavelength_power(x: np.ndarray, vecs: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Mean Fourier power sum_a |sum_j v_ja exp(i q . r_j)|^2 of unit vectors v.

    ``x`` (N, d), ``vecs`` (N d, K) unit columns, ``q`` (Q, d); averages over
    the K columns. Only relative values (across q, or between sets of
    eigenvectors) are used.
    """
    n, d = x.shape
    if vecs.shape[1] == 0:
        return np.full(q.shape[0], np.nan)
    e = jnp.exp(1j * (jnp.asarray(x) @ jnp.asarray(q).T))  # (N, Q)
    v = jnp.asarray(vecs).reshape(n, d, -1)  # (N, d, K)
    power = sum(jnp.abs(e.T @ v[:, a, :]) ** 2 for a in range(d))  # (Q, K)
    return np.asarray(jnp.mean(power, axis=1))


def level_frames(energy: np.ndarray, levels: Tuple[float, ...]) -> np.ndarray:
    """Index of the first frame with log10 Phi <= each level (else the last).

    Levels above the start map to frame 0. Along a gradient flow ``energy``
    decreases (up to round-off near the end), so this is a monotone clock.
    """
    assert energy.ndim == 1 and energy.shape[0] >= 1, "one energy per frame"
    assert energy[-1] <= energy[0], "energy must decrease along the flow"
    below = np.log10(np.maximum(energy, 1e-300))[None, :] <= np.array(levels)[:, None]
    first = np.argmax(below, axis=1)
    return np.where(below.any(axis=1), first, energy.shape[0] - 1)


def section_axes(grad: np.ndarray, vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """tau = -grad/|grad| and the stiffest eigenvector made orthogonal to tau."""
    gn = float(np.linalg.norm(grad))
    assert gn > 0.0, "zero gradient: no flow direction"
    tau = -grad / gn
    v = vecs[:, -1] - (vecs[:, -1] @ tau) * tau
    assert np.linalg.norm(v) > 1e-6, "stiffest mode parallel to the flow"
    v = v * np.sign(v[np.argmax(np.abs(v))])  # eigenvector sign is arbitrary
    return tau, v / np.linalg.norm(v)


def tenfold_distance(x: np.ndarray, e: np.ndarray, k: int) -> float:
    """Distance from frame k to the first later frame with Phi <= Phi_k / 10."""
    later = np.flatnonzero(e[k:] <= e[k] / 10.0)
    j = k + (int(later[0]) if later.size else e.shape[0] - 1 - k)
    return max(float(np.linalg.norm(x[j] - x[k])), 1e-9)


def cross_section(
    x: np.ndarray, e: np.ndarray, k: int, grad: np.ndarray, lam: np.ndarray,
    vecs: np.ndarray, ev: object, side: np.ndarray, g: int, n_part: int,
) -> Dict[str, np.ndarray]:  # fmt: skip
    """Phi on the (tau, v) plane through frame k of x (F, N d), with the path."""
    tau, v = section_axes(grad, vecs)
    half_b = np.sqrt(2.0 * (GROWTH - 1.0) * e[k] / lam[-1])
    half_a = tenfold_distance(x, e, k)
    ra = np.linspace(-0.25 * half_a, 1.25 * half_a, g)
    rb = np.linspace(-half_b, half_b, g)
    aa, bb = np.meshgrid(ra, rb)
    pts = x[k][None, :] + aa.reshape(-1, 1) * tau + bb.reshape(-1, 1) * v
    phi = ev(pts.reshape(-1, n_part, x.shape[1] // n_part) / side).reshape(aa.shape)  # type: ignore[operator]
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
            tail_path = root / f"tail_chi{tag}.npz"
            tail: Dict[str, np.ndarray] = {}
            if tail_path.exists():
                with np.load(tail_path) as z:
                    tail = {k: z[k] for k in z.files}
            energy, basis, nvec, kmag = load_system(root, tag)
            side, d = cell_lengths(basis), basis.shape[0]
            kvec = nvec @ (2.0 * np.pi * np.linalg.inv(basis)).T  # Cartesian k
            q = qvectors(n, d, side)
            eigh = jax.jit(jnp.linalg.eigh)
            hess = jax.jit(lambda s: energy.hessian(s))  # noqa: B023
            vag = jax.jit(lambda s: energy.value_and_grad(s))  # noqa: B023
            ev = make_evaluator(energy, 512)
            out: Dict[str, List[np.ndarray]] = {}
            sections: Dict[str, np.ndarray] = {}
            worst = 0.0  # largest |log10 Phi(frame) - level| inside a flow's range
            for b in range(flows["s0"].shape[0]):
                t, s_fr, e = tail_frames(flows, tail, b)
                cnt = t.shape[0]
                u, x3 = t / t[-1], s_fr * side  # (F, N, d)
                x = x3.reshape(cnt, -1)
                rows: Dict[str, list] = {
                    "lam": [], "phi": [], "gnorm": [], "u": [], "t": [],
                    "dw_stiff": [], "neg_power": [], "pos_power": [],
                }  # fmt: skip
                frames = level_frames(e, LEVELS)
                inside = (np.array(LEVELS) <= np.log10(e[0])) & (
                    np.array(LEVELS) >= np.log10(e[-1])
                )
                gap = np.abs(np.log10(e[frames]) - np.array(LEVELS))[inside]
                worst = max(worst, float(gap.max(initial=0.0)))
                cache: Dict[int, Dict[str, np.ndarray]] = {}
                for i, k in enumerate(frames):
                    if int(k) in cache:  # levels above the start share frame 0
                        for key, val in cache[int(k)].items():
                            rows[key].append(val)
                        continue
                    s = jnp.asarray(x3[k] / side)
                    scale = np.tile(side, n)  # fractional -> Cartesian, per axis
                    h = np.asarray(hess(s)) / np.outer(scale, scale)  # Cartesian
                    lam, vecs = (np.asarray(a) for a in eigh(0.5 * (h + h.T)))
                    phi, g = vag(s)
                    grad = np.asarray(g).reshape(-1) / scale  # Cartesian gradient
                    rows["lam"].append(lam)
                    rows["phi"].append(np.array(float(phi)))
                    rows["gnorm"].append(np.array(np.linalg.norm(grad)))
                    rows["u"].append(np.array(u[k]))
                    rows["t"].append(np.array(t[k]))
                    rows["dw_stiff"].append(density_wave_stiffness(x3[k], h, kvec))
                    neg = lam < -ZERO_REL[1] * lam[-1]
                    rows["neg_power"].append(wavelength_power(x3[k], vecs[:, neg], q))
                    stiff = vecs[:, -max(1, int(neg.sum())) :]  # as many, stiffest
                    rows["pos_power"].append(wavelength_power(x3[k], stiff, q))
                    cache[int(k)] = {key: val[-1] for key, val in rows.items()}
                    if b == 0 and LEVELS[i] in SECTIONS:
                        j = SECTIONS.index(LEVELS[i])
                        sec = cross_section(
                            x, e, k, grad, lam, vecs, ev, side, args.grid, n
                        )
                        for name, val in sec.items():
                            sections[f"sec{j}_{name}"] = val
                        sections[f"sec{j}_u"] = np.array(u[k])
                        sections[f"sec{j}_phi0"] = np.array(e[k])
                        sections[f"sec{j}_lam_max"] = np.array(lam[-1])
                for key, val in rows.items():
                    out.setdefault(key, []).append(np.stack(val))
            missing = [x for j, x in enumerate(SECTIONS) if f"sec{j}_a" not in sections]
            assert not missing, f"section levels {missing} share a frame with another"
            packed = {k: np.stack(v) for k, v in out.items()}
            lam = packed["lam"]
            for z_rel, name in zip(ZERO_REL, ("n_zero", "n_zero_loose"), strict=True):
                packed[name] = (np.abs(lam) < z_rel * lam[..., -1:]).sum(axis=-1)
            zero = np.abs(lam) < ZERO_REL[0] * lam[..., -1:]
            packed["n_negative"] = ((lam < 0) & ~zero).sum(axis=-1)
            np.savez(root / f"hessian_chi{tag}.npz", samples=np.array(LEVELS),
                     clock=np.array("log10_phi"), tail=np.array(bool(tail)),
                     kmag=kmag, nvec=nvec, qmag=np.linalg.norm(q, axis=1),
                     dim=np.array(d),
                     **packed, **sections)  # fmt: skip
            nz, nl = packed["n_zero"][:, -1], packed["n_zero_loose"][:, -1]
            nn = packed["n_negative"]
            print(
                f"N={n} chi={chi_str}: tail {'yes' if tail else 'no'}; "
                f"max |log10 Phi - level| {worst:.2f}; "
                f"zero modes at the end {nz.min()}-{nz.max()} "
                f"(dN - 2M = {n * d - 2 * nvec.shape[0]}) "
                f"(loose {nl.min()}-{nl.max()}); "
                f"negative modes every 2 decades (median) "
                f"{np.median(nn, axis=0)[::8]}; "
                f"sections {sorted({k.split('_')[0] for k in sections})}; "
                f"{time.time() - t0:.0f} s",
                flush=True,
            )
    print("Done.", flush=True)
