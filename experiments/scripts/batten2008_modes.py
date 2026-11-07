#!/usr/bin/env python3

"""
Collective coordinates along the 2008 flows: every constrained mode, exactly.

The stealthy energy is a sum over the constrained modes,
Phi = (1/V) sum_n |rho(k_n)|^2 = sum_n S(k_n) (V = N), with
rho(k_n) = sum_j exp(i k_n . r_j) over the M half-space wavevectors of the
k-set, sorted by |k| (in 1D, k_n = 2 pi n / L, n = 1..M; any dimension works,
the energy is rebuilt from the flows' basis and K). So, unlike a
2D slice, these coordinates give the energy along the path exactly, mode by
mode. For every recorded frame of every flow (N > 3) this script stores:

- S(k_n, t) on a log t/T grid (nearest recorded frame at or after each grid
  time), shape (flows, grid, M);
- t_relax: for thresholds eps, the last t/T at which S(k_n) > eps (0 if never),
  shape (flows, M, len(eps));
- rho(k_n, t) / sqrt(N) at every frame for rows n = 1-5, 10 and the rows with
  |k| nearest K/4, K/2, 3K/4, K (phase planes; "picks", 1-based rows);
- the k-set itself ("nvec", integer coefficients, and "kmag" = |k|);
- per mode: phase turns, S at which the phase locks, unresolved-step
  fraction (phase_stats);
- the largest relative gap between sum_n S(k_n) and the recorded energy.

Output: <root>/N<N>/modes_chi<tag>.npz.

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
from batten2008_flows import chi_tag, load_system  # noqa: E402
from batten2008_wasserstein import pad  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

EPS = (1e-2, 1e-8, 1e-14)  # S(k) thresholds for the relaxation times
SMALL_N = (1, 2, 3, 4, 5, 10)  # modes always stored in full (phase planes)
PHASE_FLOOR = 1e-16  # S below this: the phase of rho is round-off
LOCK_RAD = 0.1  # phase "locked" once less than this remains to turn
FRAME_BATCH = 128  # frames per collective-density call

# ============================================================
# HELPERS
# ============================================================


def mode_indices(kmag: np.ndarray, picks: Tuple[float, ...]) -> np.ndarray:
    """1-based rows of the |k|-sorted k-set: SMALL_N plus |k| nearest f K.

    In 1D row n is k_n = 2 pi n / L, so this is n = 1-5, 10 and n = f M.
    """
    m = kmag.shape[0]
    assert m >= 1 and all(0.0 <= p <= 1.0 for p in picks), "bad picks"
    assert np.all(np.diff(kmag) >= 0.0), "k-set must be sorted by |k|"
    near = {int(np.argmin(np.abs(kmag - p * kmag[-1]))) + 1 for p in picks}
    out = sorted(near | {n for n in SMALL_N if n <= m})
    return np.array(out, dtype=np.int64)


def structure_factors(
    s: np.ndarray, m: int, rho_fn: object
) -> Tuple[np.ndarray, np.ndarray]:
    """(Re rho, Im rho) / sqrt(N) of frames s (F, N, 1), modes n = 1..m."""
    n = s.shape[1]
    re, im = [], []
    for a in range(0, s.shape[0], FRAME_BATCH):
        r, i = rho_fn(jnp.asarray(s[a : a + FRAME_BATCH]))  # type: ignore[operator]
        re.append(np.asarray(r)[:, :m])
        im.append(np.asarray(i)[:, :m])
    scale = np.sqrt(n)
    return np.concatenate(re) / scale, np.concatenate(im) / scale


def relax_times(u: np.ndarray, sk: np.ndarray, eps: Tuple[float, ...]) -> np.ndarray:
    """Last u = t/T with S(k_n) > eps, per mode and threshold (0 if never)."""
    assert u.shape[0] == sk.shape[0], "one time per frame"
    out = np.zeros((sk.shape[1], len(eps)))
    for e, thr in enumerate(eps):
        above = sk > thr  # (F, M)
        last = sk.shape[0] - 1 - np.argmax(above[::-1], axis=0)
        out[:, e] = np.where(above.any(axis=0), u[last], 0.0)
    return out


def phase_stats(re: np.ndarray, im: np.ndarray) -> Dict[str, np.ndarray]:
    """Per mode: phase turns, S when the phase locks, unresolved-step fraction.

    Uses consecutive frames where S > PHASE_FLOOR at both ends. ``turns`` is
    sum |d arg rho| / 2 pi; ``s_lock`` is S at the first frame after which
    less than LOCK_RAD of turning remains (NaN if S never exceeds the floor);
    ``unresolved`` is the fraction of steps with |d arg rho| > pi / 2, where the
    frame spacing may alias the phase.
    """
    assert re.shape == im.shape and re.ndim == 2, "need (frames, modes)"
    sk = re**2 + im**2
    dth = np.angle(np.exp(1j * np.diff(np.arctan2(im, re), axis=0)))
    ok = (sk[1:] > PHASE_FLOOR) & (sk[:-1] > PHASE_FLOOR)
    step = np.where(ok, np.abs(dth), 0.0)
    remaining = np.cumsum(step[::-1], axis=0)[::-1]  # turning still to come
    locked = remaining < LOCK_RAD
    first = np.where(locked.any(axis=0), np.argmax(locked, axis=0), step.shape[0])
    s_lock = sk[np.minimum(first, sk.shape[0] - 1), np.arange(sk.shape[1])]
    n_ok = np.maximum(ok.sum(axis=0), 1)
    return {
        "turns": step.sum(axis=0) / (2.0 * np.pi),
        "s_lock": np.where(ok.any(axis=0), s_lock, np.nan),
        "unresolved": (ok & (np.abs(dth) > 0.5 * np.pi)).sum(axis=0) / n_ok,
    }


def on_grid(u: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of the first recorded frame with u >= each grid time."""
    idx = np.searchsorted(u, grid, side="left")
    out: np.ndarray = np.minimum(idx, u.shape[0] - 1)
    return out


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
    parser.add_argument("--grid-points", type=int, default=300)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    assert all(n > 3 for n in args.sizes), "N = 3 has a single mode per chi"
    select_device(args.device)

    for n in args.sizes:
        root = args.root / f"N{n}"
        for chi_str in args.chi:
            t0 = time.time()
            tag = chi_tag(Fraction(chi_str))
            with np.load(root / f"flows_chi{tag}.npz") as z:
                flows = {k: z[k] for k in z.files}
            _, _, nvec, kmag = load_system(root, tag)  # sorted by |k|
            m = nvec.shape[0]
            nv = jnp.asarray(nvec)
            rho_fn = jax.jit(jax.vmap(lambda s, nv=nv: collective_density(s, nv)))
            picks = mode_indices(kmag, (0.0, 0.25, 0.5, 0.75, 1.0))
            n_flow = flows["s0"].shape[0]
            u_min = min(
                float(flows["traj_t"][b, 1] / flows["traj_t"][b, int(c) - 1])
                for b, c in enumerate(flows["traj_count"])
            )
            grid = np.concatenate(
                [[0.0], np.logspace(np.log10(u_min), 0.0, args.grid_points)]
            )
            out: Dict[str, List[np.ndarray]] = {}
            worst = 0.0
            for b in range(n_flow):
                cnt = int(flows["traj_count"][b])
                t = flows["traj_t"][b, :cnt]
                u = t / t[-1]
                re, im = structure_factors(flows["traj_s"][b, :cnt], m, rho_fn)
                sk = re**2 + im**2  # S(k_n) = |rho|^2 / N; Phi = sum_n S
                e = flows["traj_energy"][b, :cnt]
                gap = np.abs(sk.sum(axis=1) - e) / np.maximum(e, 1e-300)
                worst = max(worst, float(np.max(gap[e > 1e-12])))
                out.setdefault("sk_grid", []).append(sk[on_grid(u, grid)])
                out.setdefault("t_relax", []).append(relax_times(u, sk, EPS))
                out.setdefault("rho_re", []).append(re[:, picks - 1])
                out.setdefault("rho_im", []).append(im[:, picks - 1])
                out.setdefault("u", []).append(u)
                for key, val in phase_stats(re, im).items():
                    out.setdefault(key, []).append(val)
                out.setdefault("energy", []).append(e)
            packed = {k: pad(v) for k, v in out.items()}
            packed.update(grid=grid, eps=np.array(EPS), picks=picks, m=np.array(m),
                          nvec=nvec, kmag=kmag)  # fmt: skip
            np.savez(root / f"modes_chi{tag}.npz", **packed)
            print(
                f"N={n} chi={chi_str}: M={m}, {n_flow} flows; "
                f"max |sum S - Phi|/Phi (Phi > 1e-12) {worst:.1e}; "
                f"{time.time() - t0:.0f} s",
                flush=True,
            )
    print("Done.", flush=True)
