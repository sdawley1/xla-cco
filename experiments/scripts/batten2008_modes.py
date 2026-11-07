#!/usr/bin/env python3

"""
Collective coordinates along the 2008 flows: every constrained mode, exactly.

The stealthy energy is a sum over the constrained modes,
Phi = (1/V) sum_n |rho(k_n)|^2 = sum_n S(k_n) (V = N), with
rho(k_n) = sum_j exp(i k_n x_j) and k_n = 2 pi n / L, n = 1..M. So, unlike a
2D slice, these coordinates give the energy along the path exactly, mode by
mode. For every recorded frame of every flow (N > 3) this script stores:

- S(k_n, t) on a log t/T grid (nearest recorded frame at or after each grid
  time), shape (flows, grid, M);
- t_relax: for thresholds eps, the last t/T at which S(k_n) > eps (0 if never),
  shape (flows, M, len(eps));
- rho(k_n, t) / sqrt(N) at every frame for a few modes (phase planes);
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
from batten2008_flows import build_energy, chi_tag  # noqa: E402
from batten2008_wasserstein import pad  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

EPS = (1e-2, 1e-8, 1e-14)  # S(k) thresholds for the relaxation times
FRAME_BATCH = 128  # frames per collective-density call

# ============================================================
# HELPERS
# ============================================================


def mode_indices(m: int, picks: Tuple[float, ...]) -> np.ndarray:
    """Distinct mode numbers n in 1..m at fractions ``picks`` of m (n = 1 first)."""
    assert m >= 1 and all(0.0 <= p <= 1.0 for p in picks), "bad picks"
    out = sorted({min(m, max(1, round(p * m))) for p in picks} | {1, 2})
    return np.array([n for n in out if n <= m], dtype=np.int64)


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
            energy, _, m = build_energy(n, Fraction(chi_str))
            nvec = np.asarray(energy.n_vectors)[:m]
            assert np.array_equal(nvec[:, 0], np.arange(1, m + 1)), "k_n = 2 pi n/L"
            nv = jnp.asarray(nvec)
            rho_fn = jax.jit(jax.vmap(lambda s, nv=nv: collective_density(s, nv)))
            picks = mode_indices(m, (0.0, 0.25, 0.5, 0.75, 1.0))
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
                out.setdefault("energy", []).append(e)
            packed = {k: pad(v) for k, v in out.items()}
            packed.update(grid=grid, eps=np.array(EPS), picks=picks, m=np.array(m))
            np.savez(root / f"modes_chi{tag}.npz", **packed)
            print(
                f"N={n} chi={chi_str}: M={m}, {n_flow} flows; "
                f"max |sum S - Phi|/Phi (Phi > 1e-12) {worst:.1e}; "
                f"{time.time() - t0:.0f} s",
                flush=True,
            )
    print("Done.", flush=True)
