#!/usr/bin/env python3

"""
Shape of each flow's departure from its start-end plane, per flow and per particle.

The off-plane figure (batten2008_w2_plot.py) shows the distance of every
flow from its start-end plane rising and falling twice. This script resolves
that curve into its parts. Each flow is resampled at G points uniform in
arclength, s = ell / ell(inf) in [0, 1], as y(s) = r(s) - r(0) (Nd vector).
With the chord e1 = y(1) / |y(1)| and P(s) = y - (y . e1) e1 the part
perpendicular to it, the start-end plane is spanned by e1 and
e2 = P(s*) / |P(s*)|, s* the point farthest from the chord. Stored:

- the off-plane residual W(s) = P - (P . e2) e2, as |W| / |y(1)| (the
  off-plane curve), and its signed coordinate along e3, the principal
  direction of W (sign: positive over the first half), plus what e3 misses;
- the path in (e1, e2, e3) coordinates over |y(1)|;
- per particle j (W_j = particle j's d components of W): the amplitude
  |W_j| sqrt(N) / |y(1)|, whose root-mean-square over particles is the
  off-plane curve, and the signed W_j . e3_j / |e3_j|. Flow 0: every
  particle; all flows: the mean and 10/50/90 % of each particle's profile
  scaled to its own maximum, a histogram of those scaled profiles, and the
  participation ratio (sum |W_j|^2)^2 / (N sum |W_j|^4);
- a plane-free version: P with its sin(pi s) component removed from every
  coordinate ("bow-free"), as |.| / |y(1)| and per particle;
- the Karhunen-Loeve (principal-component) modes of P over s: variance
  fractions, the temporal eigenfunctions phi_k(s) (sign: positive overlap
  with sin(k pi s)) and the path coordinates sqrt(lambda_k) phi_k / |y(1)|;
- the tangent autocorrelation <t(s) . t(s + lag)> against lag / G, and the
  cumulative turning angle (sum of angles between successive tangents) over
  its total;
- where the displacement lives, from the density-wave directions at r(0)
  (the unit gradients of Re rho(k) and Im rho(k), k in the k-set of
  modes_chi<tag>.npz): the net displacement's power on each constrained
  wave, the linear prediction 2 S(k, 0) / |k|^2 (the least squared motion
  that cancels rho(k) from the start), and the share of the remaining
  displacement y(1) - y(s) on the longest-wave shell |k| = k_1;
- a linear relaxation model with no fitted parameter: wave n carries
  displacement sqrt(2 S(k_n, 0)) / |k_n| along its own orthogonal direction
  and relaxes as exp(-|k_n|^2 t) (the linearized flow); its off-plane curve,
  cumulative turning and KL modes, computed exactly as for the flow.

The plane pins W, and so every W_j, to zero at s = 0, s* and 1; the bow-free
residual has no such pins.

Needs flows_chi<tag>.npz and modes_chi<tag>.npz (batten2008_modes.py).
Output: <root>/N<N>/offplane_chi<tag>.npz. NumPy only (CPU).

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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import cell_lengths, chi_tag  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

G = 400  # arclength samples per flow
N_KL = 10  # KL variance fractions stored
N_PHI = 6  # KL eigenfunctions stored
HIST_BINS = 40  # bins of the scaled per-particle profiles, [0, 1]

# ============================================================
# HELPERS
# ============================================================


def arclength_path(s: np.ndarray, side: object, g: int) -> Tuple[np.ndarray, float]:
    """Displacement y(s) = r(s) - r(0), (g, N d), at g points uniform in arclength.

    ``s`` (F, N, d) fractional frames; steps between frames use the minimum
    image, so wrapped coordinates are fine. Returns y and the path length.
    """
    assert s.ndim == 3 and s.shape[0] >= 2 and g >= 2, "need (F, N, d), F >= 2"
    step = np.diff(s, axis=0)
    step -= np.round(step)
    y = np.concatenate([np.zeros((1,) + s.shape[1:]), np.cumsum(step, axis=0)])
    y = (y * np.asarray(side)).reshape(s.shape[0], -1)  # side: scalar or (d,)
    seg = np.linalg.norm(np.diff(y, axis=0), axis=1)
    ell = np.concatenate([[0.0], np.cumsum(seg)])
    assert ell[-1] > 0.0, "the path does not move"
    u = np.linspace(0.0, ell[-1], g)
    k = np.clip(np.searchsorted(ell, u, side="right") - 1, 0, s.shape[0] - 2)
    w = ((u - ell[k]) / np.maximum(seg[k], 1e-300))[:, None]
    return y[k] * (1.0 - w) + y[k + 1] * w, float(ell[-1])


def plane_residual(y: np.ndarray) -> Dict[str, np.ndarray]:
    """Start-end plane of path y (g, D) and the residual off it.

    Returns chord, P (perpendicular to the chord), the index of s*, W (off
    the plane), coordinates (g, 3) along e1, e2, e3 over the chord, and e3.
    """
    chord = float(np.linalg.norm(y[-1]))
    assert chord > 0.0 and np.allclose(y[0], 0.0), "path must start at 0 and move"
    e1 = y[-1] / chord
    p = y - np.outer(y @ e1, e1)
    star = int(np.argmax(np.linalg.norm(p, axis=1)))
    e2 = p[star] / np.linalg.norm(p[star])
    w = p - np.outer(p @ e2, e2)
    _, _, vt = np.linalg.svd(w, full_matrices=False)
    e3 = vt[0]
    half = w.shape[0] // 2
    if np.sum(w[:half] @ e3) < 0.0:  # first hump positive
        e3 = -e3
    coords = np.stack([y @ e1, p @ e2, w @ e3], axis=1) / chord
    return {"chord": np.array(chord), "p": p, "star": np.array(star), "w": w,
            "coords": coords, "e3": e3}  # fmt: skip


def kl_modes(p: np.ndarray, n_keep: int) -> Dict[str, np.ndarray]:
    """Karhunen-Loeve modes of the path p (g, D) over its g samples.

    The temporal eigenfunctions are unit vectors over the samples, signed so
    that their first lobe is positive (as for sin(k pi s); the first sample
    reaching 30 % of the maximum is positive); "overlap" is |<phi_k, sin(k pi
    s)>| for unit vectors, and "coord" is sqrt(lambda_k) phi_k.
    """
    g = p.shape[0]
    assert g >= n_keep >= 1, "need at least n_keep samples"
    lam, vec = np.linalg.eigh(p @ p.T)
    lam, vec = np.maximum(lam[::-1], 0.0), vec[:, ::-1][:, :n_keep]
    sg = np.linspace(0.0, 1.0, g)
    sines = np.stack([np.sin((k + 1) * np.pi * sg) for k in range(n_keep)], axis=1)
    sines /= np.linalg.norm(sines, axis=0, keepdims=True)
    overlap = np.einsum("gk,gk->k", vec, sines)
    first = np.argmax(np.abs(vec) >= 0.3 * np.abs(vec).max(axis=0), axis=0)
    vec = vec * np.where(vec[first, np.arange(n_keep)] < 0.0, -1.0, 1.0)
    coord = vec * np.sqrt(lam[:n_keep])
    frac = lam / max(lam.sum(), 1e-300)
    return {"frac": frac, "phi": vec, "overlap": np.abs(overlap), "coord": coord}


def remove_bow(p: np.ndarray) -> np.ndarray:
    """p (g, D) with the sin(pi s) component removed from every coordinate."""
    g = p.shape[0]
    bow = np.sin(np.pi * np.linspace(0.0, 1.0, g))
    bow /= np.linalg.norm(bow)
    return p - np.outer(bow, bow @ p)


def per_particle(w: np.ndarray, n: int, chord: float) -> np.ndarray:
    """Particle amplitudes |W_j| sqrt(N) / chord, (g, N); RMS over j is |W| / chord."""
    g = w.shape[0]
    assert w.shape[1] % n == 0, "D must be N d"
    return np.linalg.norm(w.reshape(g, n, -1), axis=2) * np.sqrt(n) / chord


def signed_per_particle(
    w: np.ndarray, e3: np.ndarray, n: int, chord: float
) -> np.ndarray:
    """W_j . e3_j / |e3_j| sqrt(N) / chord, (g, N): each particle's signed share."""
    g = w.shape[0]
    wj, ej = w.reshape(g, n, -1), e3.reshape(n, -1)
    unit = ej / np.maximum(np.linalg.norm(ej, axis=1, keepdims=True), 1e-300)
    return np.einsum("gnd,nd->gn", wj, unit) * np.sqrt(n) / chord


def scaled_profiles(amp: np.ndarray) -> np.ndarray:
    """Each particle's profile (column of amp, (g, N)) over its own maximum."""
    peak = amp.max(axis=0, keepdims=True)
    out: np.ndarray = amp / np.maximum(peak, 1e-300)
    return out


def participation(amp: np.ndarray) -> np.ndarray:
    """(sum_j a_j^2)^2 / (N sum_j a_j^4) at every sample: 1 if all share equally."""
    a2 = amp**2
    num = a2.sum(axis=1) ** 2
    den = amp.shape[1] * np.maximum((a2**2).sum(axis=1), 1e-300)
    out: np.ndarray = num / den
    return out


def tangent_correlation(y: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean t(s) . t(s + lag) of the unit tangents of y (g, D), lag = 0..max_lag."""
    t = np.diff(y, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
    assert 0 <= max_lag < t.shape[0], "lag beyond the path"
    return np.array([np.mean(np.sum(t[: t.shape[0] - k] * t[k:], axis=1))
                     for k in range(max_lag + 1)])  # fmt: skip


def wave_axes(x0: np.ndarray, kvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unit density-wave directions at x0 (N, d), (N d, 2M), and S(k) there, (M,).

    Columns m and M + m are the gradients of Re rho(k_m) and Im rho(k_m):
    -sin(k . r_j) k and cos(k . r_j) k.
    """
    n, d = x0.shape
    m = kvec.shape[0]
    assert kvec.shape[1] == d and m >= 1, "k-set must be (M, d)"
    phase = x0 @ kvec.T  # (N, M)
    cols = [(w[:, None, :] * kvec.T[None, :, :]).reshape(n * d, m)
            for w in (-np.sin(phase), np.cos(phase))]  # fmt: skip
    axes = np.concatenate(cols, axis=1)
    axes /= np.linalg.norm(axes, axis=0, keepdims=True)
    sk = (np.cos(phase).sum(axis=0) ** 2 + np.sin(phase).sum(axis=0) ** 2) / n
    return axes, sk


def wave_power(u: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Power of displacements u (..., N d) on each wave: Re^2 + Im^2, (..., M)."""
    proj = u @ axes
    m = axes.shape[1] // 2
    out: np.ndarray = proj[..., :m] ** 2 + proj[..., m:] ** 2
    return out


def cumulative_turning(y: np.ndarray) -> np.ndarray:
    """Turning angle accumulated along y (g, D), over its total, (g - 1,)."""
    t = np.diff(y, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
    cos = np.clip(np.sum(t[:-1] * t[1:], axis=1), -1.0, 1.0)
    turn = np.concatenate([[0.0], np.cumsum(np.arccos(cos))])
    out: np.ndarray = turn / max(float(turn[-1]), 1e-300)
    return out


def linear_model_path(
    weight: np.ndarray, rate: np.ndarray, g: int, n_t: int = 4000
) -> np.ndarray:
    """Path y(t) = sum_n sqrt(w_n) (1 - exp(-rate_n t)) e_n at g arclength points.

    The e_n are orthonormal, so y is (g, M) in mode coordinates; t runs on a
    log grid from 0.01 / max(rate) to 40 / min(rate), plus t = 0.
    """
    assert weight.shape == rate.shape and np.all(rate > 0.0), "one rate per mode"
    lo, hi = np.log10(0.01 / rate.max()), np.log10(40.0 / rate.min())
    t = np.concatenate([[0.0], np.logspace(lo, hi, n_t)])
    y = np.sqrt(weight)[None, :] * -np.expm1(-np.outer(t, rate))
    seg = np.linalg.norm(np.diff(y, axis=0), axis=1)
    ell = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.linspace(0.0, ell[-1], g)
    k = np.clip(np.searchsorted(ell, u, side="right") - 1, 0, t.shape[0] - 2)
    a = ((u - ell[k]) / np.maximum(seg[k], 1e-300))[:, None]
    out: np.ndarray = y[k] * (1.0 - a) + y[k + 1] * a
    return out


def share1(power: np.ndarray, kmag: np.ndarray) -> np.ndarray:
    """Share of power (..., M) on the longest-wave shell |k| = k_1."""
    out: np.ndarray = power[..., np.isclose(kmag, kmag[0])].sum(-1) / power.sum(-1)
    return out


def analyze_flow(
    s: np.ndarray, side: np.ndarray, g: int, kvec: np.ndarray, kmag: np.ndarray
) -> Dict[str, np.ndarray]:
    """Every stored quantity of one flow (frames s (F, N, d), k-set (M, d))."""
    n = s.shape[1]
    y, length = arclength_path(s, side, g)
    axes, sk0 = wave_axes(s[0] * side, kvec)
    remaining = y[-1][None, :] - y
    rem_power = wave_power(remaining, axes)
    shell1 = np.isclose(kmag, kmag[0])
    rem_norm2 = np.maximum(np.sum(remaining**2, axis=1), 1e-300)
    lin_y = linear_model_path(2.0 * sk0 / kmag**2, kmag**2, g)
    lin_pr = plane_residual(lin_y)
    lin_kl = kl_modes(lin_pr["p"], max(N_KL, N_PHI))
    pr = plane_residual(y)
    chord = float(pr["chord"])
    kl = kl_modes(pr["p"], max(N_KL, N_PHI))
    bowfree = remove_bow(pr["p"])
    amp = per_particle(pr["w"], n, chord)
    amp_bf = per_particle(bowfree, n, chord)
    scaled = scaled_profiles(amp)
    edges = np.linspace(0.0, 1.0, HIST_BINS + 1)
    hist = np.stack([np.histogram(row, bins=edges)[0] for row in scaled])
    return {
        "length": np.array(length),
        "chord": pr["chord"],
        "s_star": np.array(int(pr["star"]) / (g - 1)),
        "off": np.linalg.norm(pr["w"], axis=1) / chord,
        "coords": pr["coords"],
        "rest": np.linalg.norm(pr["w"] - np.outer(pr["w"] @ pr["e3"], pr["e3"]), axis=1)
        / chord,  # fmt: skip
        "bowfree": np.linalg.norm(bowfree, axis=1) / chord,
        "kl_frac": kl["frac"][:N_KL],
        "kl_phi": kl["phi"][:, :N_PHI],
        "kl_overlap": kl["overlap"][:N_PHI],
        "kl_coord": kl["coord"][:, :3] / chord,
        "pp_mean": scaled.mean(axis=1),
        "pp_q": np.percentile(scaled, [10.0, 50.0, 90.0], axis=1),
        "pp_hist": hist,
        "pp_bowfree_mean": scaled_profiles(amp_bf).mean(axis=1),
        "pp_participation": participation(amp),
        "tangent_corr": tangent_correlation(y, g // 2),
        "turning": cumulative_turning(y),
        "net_power": wave_power(y[-1], axes),
        "lin_power": 2.0 * sk0 / kmag**2,
        "rem_shell1": rem_power[:, shell1].sum(axis=1) / rem_norm2,
        "lin_off": np.linalg.norm(lin_pr["w"], axis=1) / float(lin_pr["chord"]),
        "lin_turning": cumulative_turning(lin_y),
        "lin_kl_phi": lin_kl["phi"][:, :N_PHI],
        "lin_kl_frac": lin_kl["frac"][:N_KL],
        "_amp": amp,
        "_signed": signed_per_particle(pr["w"], pr["e3"], n, chord),
        "_bowfree": amp_bf,
    }


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
    parser.add_argument("--grid", type=int, default=G)
    args = parser.parse_args()
    assert args.grid >= 2 * N_KL, "grid too coarse for the KL modes"

    for n in args.sizes:
        root = args.root / f"N{n}"
        for chi_str in args.chi:
            t0 = time.time()
            tag = chi_tag(Fraction(chi_str))
            with np.load(root / f"flows_chi{tag}.npz") as z:
                traj, counts = z["traj_s"], z["traj_count"]
                basis = np.asarray(z["basis"])
            side = cell_lengths(basis)
            with np.load(root / f"modes_chi{tag}.npz") as z:  # the k-set
                kmag = np.asarray(z["kmag"])
                kvec = np.asarray(z["nvec"]) @ (2.0 * np.pi * np.linalg.inv(basis)).T
            out: Dict[str, List[np.ndarray]] = {}
            flow0: Dict[str, np.ndarray] = {}
            for b in range(traj.shape[0]):
                res = analyze_flow(traj[b, : int(counts[b])], side, args.grid,
                                   kvec, kmag)  # fmt: skip
                for key, val in res.items():
                    if key.startswith("_"):
                        if b == 0:
                            flow0[f"flow0{key}"] = val
                        continue
                    out.setdefault(key, []).append(val)
            packed = {k: np.stack(v) for k, v in out.items()}
            np.savez(root / f"offplane_chi{tag}.npz",
                     s=np.linspace(0.0, 1.0, args.grid), kmag=kmag,
                     dim=np.array(basis.shape[0]), **packed, **flow0)  # fmt: skip
            off, sg = packed["off"], np.linspace(0.0, 1.0, args.grid)
            half = args.grid // 2
            print(
                f"N={n} chi={chi_str}: {traj.shape[0]} flows; s* median "
                f"{np.median(packed['s_star']):.2f}; humps at s = "
                f"{np.median(sg[np.argmax(off[:, :half], axis=1)]):.2f}, "
                f"{np.median(sg[half + np.argmax(off[:, half:], axis=1)]):.2f}; "
                f"KL fractions {np.median(packed['kl_frac'], axis=0)[:4].round(3)}; "
                f"k1-shell share of the net motion (measured / linear) "
                f"{np.median(share1(packed['net_power'], kmag)):.2f} / "
                f"{np.median(share1(packed['lin_power'], kmag)):.2f}; "
                f"participation at the humps "
                f"{np.median(packed['pp_participation'][:, half // 2]):.2f}; "
                f"{time.time() - t0:.0f} s",
                flush=True,
            )
    print("Done.", flush=True)
