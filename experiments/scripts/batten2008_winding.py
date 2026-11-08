#!/usr/bin/env python3

"""
Winding of the density waves along the flows, and its source in the potential.

Phi = (1/N) sum_k |rho_k|^2 depends only on the moduli |rho_k|, so a change
of the phase arg rho_k costs that wave no energy. Along the flow
dr_j/dt = -grad_j Phi, differentiating rho_k = sum_j exp(i k . r_j) gives
exactly

    drho_k/dt = (1/N) sum_q (k . q) [rho_q^* rho_{k+q} - rho_q rho_{k-q}],

summed over the constrained half-space set q (the q = k term with
rho_0 = N is the linear decay -|k|^2 rho_k). The phase rate
theta_k' = Im(rho_k^* drho_k/dt) / |rho_k|^2 therefore comes only from the
three-wave (triad) terms k, q, k +- q, many of them through unconstrained
wavevectors |k +- q| > K. In the harmonic limit it vanishes.

For every flow (stored frames spliced with tail_chi<tag>.npz when present)
this script stores:

- at the log10 Phi levels of batten2008_hessian.LEVELS: S(k_n), the
  cumulative winding W_n = (1/2 pi) sum |d arg rho_n| up to that frame
  (steps with S > 1e-16 at both ends, as phase_stats), the exact rates
  theta_n' and d ln|rho_n|/dt from the gradient (no finite differences),
  |grad arg rho_n| and |grad Phi|. Since theta_n is a function of the
  configuration, theta_n' = -grad theta_n . grad Phi = -|grad theta_n|
  |grad Phi| cos alpha_n: the phase is easy to move when S is small
  (|grad theta_n| ~ |k_n| / sqrt(2 S_n)) and moves as far as the force
  aligns with it (cos alpha_n, zero for separable, harmonic relaxation);
- for every flow at every frame (NaN-padded): the total winding
  sum_n W_n over all constrained waves, the arclength, Phi and the time;
- for flow 0, W_n and S(k_n) at every frame, with the flow time;
- for flow 0 at TRIAD_LEVELS: drho/dt from the gradient and from the triad
  sum (an exact identity: a check), and the share of each wave's phase
  torque |Im(rho_k^* term)| carried by triads through unconstrained
  wavevectors.

Output: <root>/N<N>/winding_chi<tag>.npz.

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
from xla_cco.energy import collective_density

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import cell_lengths, chi_tag, load_system  # noqa: E402
from batten2008_hessian import LEVELS, level_frames  # noqa: E402
from batten2008_tail import tail_frames  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

PHASE_FLOOR = 1e-16  # S below this: the phase is round-off (as phase_stats)
TRIAD_LEVELS = (1.0, -1.0, -3.0, -6.0, -9.0)  # log10 Phi of the triad checks
FRAME_BATCH = 128

# ============================================================
# HELPERS
# ============================================================


def rho_of(s: np.ndarray, nvec: np.ndarray, fn: object) -> np.ndarray:
    """Complex rho (F, M) of fractional frames s (F, N, d) on integer rows nvec."""
    out = []
    for a in range(0, s.shape[0], FRAME_BATCH):
        re, im = fn(jnp.asarray(s[a : a + FRAME_BATCH]), jnp.asarray(nvec))  # type: ignore[operator]
        out.append(np.asarray(re) + 1j * np.asarray(im))
    return np.concatenate(out)


def cumulative_winding(rho: np.ndarray, n: int) -> np.ndarray:
    """W (F, M): turns of arg rho accumulated up to each frame (resolved steps)."""
    sk = np.abs(rho) ** 2 / n
    dth = np.abs(np.angle(rho[1:] * np.conj(rho[:-1])))
    ok = (sk[1:] > PHASE_FLOOR) & (sk[:-1] > PHASE_FLOOR)
    step = np.where(ok, dth, 0.0) / (2.0 * np.pi)
    zero = np.zeros((1, rho.shape[1]))
    out: np.ndarray = np.concatenate([zero, np.cumsum(step, axis=0)])
    return out


def exact_rhodot(x: np.ndarray, vel: np.ndarray, kvec: np.ndarray) -> np.ndarray:
    """drho_k/dt = sum_j i (k . v_j) exp(i k . r_j) for velocities vel (N, d)."""
    assert x.shape == vel.shape and kvec.shape[1] == x.shape[1], "shapes"
    phase = np.exp(1j * (x @ kvec.T))  # (N, M)
    out: np.ndarray = 1j * np.sum((vel @ kvec.T) * phase, axis=0)
    return out


def triad_rhodot(
    s: np.ndarray, nvec: np.ndarray, kvec: np.ndarray, fn: object
) -> Tuple[np.ndarray, np.ndarray]:
    """drho/dt (M,) from the triad sum, and the unconstrained share of the torque.

    rho is tabulated on every integer vector with |n_a| <= 2 max|nvec_a|, so
    rho_{k +- q} is a lookup. The share is sum |Im(rho_k^* term)| over triads
    whose partner k +- q lies outside the constrained set, over the sum for
    all triads, per wave k.
    """
    n_part, d = s.shape
    lim = 2 * np.abs(nvec).max(axis=0).astype(int)
    axes = np.meshgrid(*[np.arange(-v, v + 1) for v in lim], indexing="ij")
    table_n = np.stack([a.ravel() for a in axes], axis=1)
    table = rho_of(s[None], table_n.astype(np.float64), fn)[0]
    width = 2 * lim + 1

    def index(nn: np.ndarray) -> np.ndarray:
        flat = np.zeros(nn.shape[:-1], dtype=np.int64)
        for a in range(d):
            flat = flat * width[a] + (nn[..., a].astype(np.int64) + lim[a])
        return flat

    ni = nvec.astype(np.int64)
    rho_q = table[index(ni)]  # (M,)
    plus, minus = ni[:, None, :] + ni[None, :, :], ni[:, None, :] - ni[None, :, :]
    kq = kvec @ kvec.T  # (M, M): k . q
    t_plus = kq * np.conj(rho_q)[None, :] * table[index(plus)]
    t_minus = -kq * rho_q[None, :] * table[index(minus)]
    rhodot = (t_plus + t_minus).sum(axis=1) / n_part
    known = {tuple(v) for v in ni} | {tuple(-v) for v in ni} | {(0,) * d}
    unc_p = np.array([[tuple(v) not in known for v in row] for row in plus])
    unc_m = np.array([[tuple(v) not in known for v in row] for row in minus])
    rk = np.conj(rho_q)[:, None]
    tq_p, tq_m = np.abs(np.imag(rk * t_plus)), np.abs(np.imag(rk * t_minus))
    total = np.maximum(tq_p.sum(axis=1) + tq_m.sum(axis=1), 1e-300)
    share = ((tq_p * unc_p).sum(axis=1) + (tq_m * unc_m).sum(axis=1)) / total
    return rhodot, share


def phase_gradient_norm(x: np.ndarray, kvec: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """|grad_r arg rho_k| (M,): grad_j theta_k = k Re(exp(i k . r_j) / rho_k)."""
    phase = np.exp(1j * (x @ kvec.T)) / rho[None, :]  # (N, M)
    out: np.ndarray = np.linalg.norm(kvec, axis=1) * np.sqrt(
        np.sum(np.real(phase) ** 2, axis=0)
    )
    return out


def rates(rho: np.ndarray, rhodot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(theta', d ln|rho|/dt) from rho and drho/dt."""
    z = np.conj(rho) * rhodot / np.maximum(np.abs(rho) ** 2, 1e-300)
    return np.imag(z), np.real(z)


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
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    select_device(args.device)
    fn = jax.jit(jax.vmap(collective_density, in_axes=(0, None)))

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
            side = cell_lengths(basis)
            kvec = nvec @ (2.0 * np.pi * np.linalg.inv(basis)).T
            vag = jax.jit(lambda s: energy.value_and_grad(s))  # noqa: B023
            out: Dict[str, List[np.ndarray]] = {}
            series: Dict[str, List[np.ndarray]] = {}  # per frame, ragged
            extra: Dict[str, np.ndarray] = {}
            for b in range(flows["s0"].shape[0]):
                t, s_fr, e = tail_frames(flows, tail, b)
                rho = rho_of(s_fr, nvec, fn)
                wind = cumulative_winding(rho, n)
                xc = (s_fr * side).reshape(s_fr.shape[0], -1)  # Cartesian, unwrapped
                ell = np.concatenate(
                    [[0.0], np.cumsum(np.linalg.norm(np.diff(xc, axis=0), axis=1))]
                )
                series.setdefault("total_wind", []).append(wind.sum(axis=1))
                series.setdefault("frame_length", []).append(ell)
                series.setdefault("frame_phi", []).append(e)
                series.setdefault("frame_t", []).append(t)
                frames = level_frames(e, LEVELS)
                th, lr, gth, gn = [], [], [], []
                for level, k in zip(LEVELS, frames, strict=True):
                    _, g = vag(jnp.asarray(s_fr[k]))
                    vel = -np.asarray(g) / side  # Cartesian velocity -grad_r Phi
                    rd = exact_rhodot(s_fr[k] * side, vel, kvec)
                    a, c = rates(rho[k], rd)
                    th.append(a)
                    lr.append(c)
                    gth.append(phase_gradient_norm(s_fr[k] * side, kvec, rho[k]))
                    gn.append(np.array(np.linalg.norm(vel)))
                    if b == 0 and level in TRIAD_LEVELS:
                        j = TRIAD_LEVELS.index(level)
                        pred, share = triad_rhodot(s_fr[k], nvec, kvec, fn)
                        extra[f"triad{j}_exact"] = rd
                        extra[f"triad{j}_pred"] = pred
                        extra[f"triad{j}_unc_share"] = share
                        extra[f"triad{j}_S"] = np.abs(rho[k]) ** 2 / n
                rows = {"wind": wind[frames], "S": np.abs(rho[frames]) ** 2 / n,
                        "theta_rate": np.array(th), "lnrho_rate": np.array(lr),
                        "grad_theta": np.array(gth), "grad_phi": np.array(gn),
                        "t": t[frames], "phi": e[frames]}  # fmt: skip
                for key, val in rows.items():
                    out.setdefault(key, []).append(val)
                if b == 0:
                    extra.update(flow0_wind=wind, flow0_S=np.abs(rho) ** 2 / n,
                                 flow0_t=t, flow0_phi=e)  # fmt: skip
            packed = {k: np.stack(v) for k, v in out.items()}
            for key, rows_k in series.items():  # NaN-padded (flows, frames)
                width = max(r.shape[0] for r in rows_k)
                arr = np.full((len(rows_k), width), np.nan)
                for b_i, r in enumerate(rows_k):
                    arr[b_i, : r.shape[0]] = r
                packed[key] = arr
            np.savez(root / f"winding_chi{tag}.npz", levels=np.array(LEVELS),
                     triad_levels=np.array(TRIAD_LEVELS), kmag=kmag, nvec=nvec,
                     tau1=np.array(1.0 / (2.0 * kmag[0] ** 2)), n_part=np.array(n),
                     **packed, **extra)  # fmt: skip
            done = [j for j in range(len(TRIAD_LEVELS)) if f"triad{j}_pred" in extra]
            err = max(
                float(
                    np.max(np.abs(extra[f"triad{j}_pred"] - extra[f"triad{j}_exact"]))
                )
                / float(np.max(np.abs(extra[f"triad{j}_exact"])))
                for j in done
            )
            shares = [float(np.median(extra[f"triad{j}_unc_share"])) for j in done]
            print(
                f"N={n} chi={chi_str}: M={nvec.shape[0]}; triad identity max rel. "
                f"error {err:.1e}; unconstrained torque share (median) by level "
                f"{np.round(shares, 2)}; final winding median "
                f"{np.median(packed['wind'][:, -1]):.2f}; {time.time() - t0:.0f} s",
                flush=True,
            )
    print("Done.", flush=True)
