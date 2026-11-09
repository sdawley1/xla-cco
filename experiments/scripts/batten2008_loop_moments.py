#!/usr/bin/env python3

"""
Loop decomposition of the vibrational spectrum of stealthy ground states.

In the complex wave basis Q = {+q, -q} (2M states, |q| < K) the coupling is
H_ab = (Q_a . Q_b) rho(Q_a - Q_b) / V with rho(0) = N; it has the spectrum
of K = (2/V) J J^T (omega^2 of the nonzero normal modes). Write H = D + W,
D = diag(|Q_a|^2 N / V). Every moment of W is a sum over closed polygons of
nonzero momentum transfers k_i = Q_(a_i) - Q_(a_(i+1)):

  w_n = (1/2M) tr (W/K^2)^n
      = sum over closed walks of prod (Q_i . Q_(i+1)) rho(k_i) / (V K^2)^n.

Three parts are separated exactly:
- non-crossing pairings: the self-consistent Born (SCBA) moments, from the
  recursion g_n = sum_(a+b=n-2) eta(g_a) g_b, eta(g)_a = sum_b s_ab g_b, with
  the variance profile s_ab = (Q_a . Q_b)^2 |rho(Q_a - Q_b)|^2 / (V K^2)^2;
- all pairings: the mean of w_n over R random-phase surrogates (every
  |rho(k)| kept, phases uniform; terms survive iff their phases cancel);
  crossing pairings = all pairings - non-crossing;
- irreducible loops: ground state - surrogate mean (odd n: entirely
  irreducible; n = 3 is the triad / bispectrum term).

It also stores the spectra of H/K^2 (ground states and surrogates) and the
SCBA density of states of H, -(1/pi) Im <G(x + i eta)>, from the vector
Dyson equation G_a = 1 / (z - D_a - sum_b s_ab G_b) (the S(k)-only theory),
with s_ab averaged over the ground states. Rectangular cells use
Q = 2 pi (n_x / L_x, n_y / L_y) and V = L_x L_y (batten2008_lattice.py).
Output: <root>/loops_chi<tag>.npz for every chi tag given (default: every tag
present in each root). NumPy (CPU); N = 4096 needs a SLURM CPU job.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_lattice import (  # noqa: E402
    cell_lengths,
    full_waves,
    h_matrix,
    k_matrix,
    lattice_limits,
    lattice_rho,
    phase_randomized,
    present_tags,
    scaled,
    waves,
)

# ============================================================
# CONSTANTS
# ============================================================

N_MAX = 12  # highest moment order
SCBA_GRID = np.linspace(-0.6, 2.4, 601)  # x = lambda / K^2
SCBA_ETA = (0.05, 0.02, 0.01, 0.005, 0.0025)  # continuation in eta (units K^2)
SCBA_TOL = 1e-9
SCBA_MAX_ITERS = 4000
SEED = 2026
TAGS = ("1-10", "1-4", "7-20", "2-5", "9-20", "49-100")  # increasing chi

# ============================================================
# MATRICES
# ============================================================


def moments(w: np.ndarray, scale: float) -> np.ndarray:
    """Raw moments (1/2M) tr (W/scale)^n, n = 2..N_MAX, from the spectrum."""
    ev = np.linalg.eigvalsh(w) / scale
    out = np.array([np.mean(ev**n) for n in range(2, N_MAX + 1)])
    assert out.shape == (N_MAX - 1,), "one value per order"
    return out


# ============================================================
# S(k)-ONLY THEORY (SCBA)
# ============================================================


def nc_moments(s: np.ndarray) -> np.ndarray:
    """Non-crossing-pairing moments of W from the operator-valued recursion."""
    m = s.shape[0]
    g: List[np.ndarray] = [np.ones(m), np.zeros(m)]
    for n in range(2, N_MAX + 1):
        acc = np.zeros(m)
        for a in range(0, n - 1):
            acc += (s @ g[a]) * g[n - 2 - a]
        g.append(acc)
    out = np.array([g[n].mean() for n in range(2, N_MAX + 1)])
    assert np.all(out[::2] >= 0.0), "even non-crossing moments are positive"
    return out


def scba_dos(s: np.ndarray, diag: np.ndarray) -> Dict[str, np.ndarray]:
    """SCBA density of states of D + W on SCBA_GRID (continuation in eta)."""
    g = 1.0 / (SCBA_GRID[None, :] + 1j * SCBA_ETA[0] - diag[:, None])
    iters = 0
    for eta in SCBA_ETA:
        z = SCBA_GRID[None, :] + 1j * eta
        for _ in range(SCBA_MAX_ITERS):
            new = 1.0 / (z - diag[:, None] - s @ g)
            change = float(np.max(np.abs(new - g)))
            g = 0.5 * g + 0.5 * new
            iters += 1
            if change < SCBA_TOL:
                break
    dos = -np.imag(g.mean(axis=0)) / np.pi
    assert dos.shape == SCBA_GRID.shape and np.all(dos > -1e-9), "a density"
    return {"x": SCBA_GRID.copy(), "dos": dos, "eta": np.array(SCBA_ETA[-1]),
            "iters": np.array(iters)}  # fmt: skip


# ============================================================
# ONE SYSTEM
# ============================================================


def run(root: Path, tag: str, n_states: int, n_sur: int) -> Dict[str, np.ndarray]:
    """Moments, spectra and the SCBA baseline for one (root, chi tag)."""
    meta = json.loads((root / f"flows_chi{tag}.json").read_text())
    side = cell_lengths(meta["basis"])
    k_cut, n = float(meta["K"]), int(meta["N"])
    nfull = full_waves(side, k_cut)
    nmax = lattice_limits(side, k_cut)
    scale = k_cut**2
    diag = uncoupled(nfull, side, n)
    rng = np.random.default_rng(SEED)
    with np.load(root / f"flows_chi{tag}.npz", mmap_mode="r") as f:
        states = np.asarray(f["s_final"][:n_states]) % 1.0 * side
    rows: Dict[str, List[np.ndarray]] = {"x_gs": [], "x_sur": [], "w_gs": [],
                                         "w_sur": [], "nc": []}  # fmt: skip
    s_mean = np.zeros((nfull.shape[0], nfull.shape[0]))
    t0 = time.time()
    for b, x in enumerate(states):
        rho = lattice_rho(x, side, nmax)
        h = h_matrix(rho, nfull, side)
        rows["x_gs"].append(np.linalg.eigvalsh(h) / scale)
        w = h - np.diag(np.diag(h))
        rows["w_gs"].append(moments(w, scale))
        s = np.abs(w / scale) ** 2
        s_mean += s / len(states)
        rows["nc"].append(nc_moments(s))
        for _ in range(n_sur):
            hs = h_matrix(phase_randomized(rho, rng), nfull, side)
            rows["x_sur"].append(np.linalg.eigvalsh(hs) / scale)
            rows["w_sur"].append(moments(hs - np.diag(np.diag(hs)), scale))
        print(f"  {root.name} chi{tag}: state {b + 1}/{len(states)} "
              f"({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
    out = {k: np.asarray(v) for k, v in rows.items()}
    sc = scba_dos(s_mean, diag / scale)
    out.update(scba_x=sc["x"], scba_dos=sc["dos"], scba_eta=sc["eta"],
               chi=np.array(float(meta["chi_actual"])), n=np.array(n),
               orders=np.arange(2, N_MAX + 1))  # fmt: skip
    print(f"{root.name} chi{tag}: SCBA iterations {int(sc['iters'])}; "
          f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip
    return out


def uncoupled(nfull: np.ndarray, side: np.ndarray, n: int) -> np.ndarray:
    """Diagonal D_a = |Q_a|^2 N / V of H (the uncoupled spectrum omega^2)."""
    out: np.ndarray = (np.linalg.norm(scaled(nfull, side), axis=1) * 2.0 * np.pi
                       / side[0]) ** 2 * n / (side[0] * side[1])  # fmt: skip
    assert out.shape == (nfull.shape[0],) and np.all(out > 0.0), "one per wave"
    return out


def check_basis(root: Path, tag: str) -> Tuple[float, float, float]:
    """Spectrum of H (complex basis) against K (Re/Im basis), and diag H against D.

    Returns the largest eigenvalue difference, the largest eigenvalue and the
    largest relative difference of diag H from |Q|^2 N / V.
    """
    meta = json.loads((root / f"flows_chi{tag}.json").read_text())
    side, k_cut = cell_lengths(meta["basis"]), float(meta["K"])
    nmax = lattice_limits(side, k_cut)
    with np.load(root / f"flows_chi{tag}.npz", mmap_mode="r") as f:
        x = np.asarray(f["s_final"][0]) % 1.0 * side
    rho = lattice_rho(x, side, nmax)
    nfull = full_waves(side, k_cut)
    h = h_matrix(rho, nfull, side)
    a = np.linalg.eigvalsh(h)
    b = np.linalg.eigvalsh(k_matrix(rho, waves(side, k_cut), side))
    d = uncoupled(nfull, side, int(meta["N"]))
    out = (float(np.max(np.abs(a - b))), float(np.max(np.abs(b))),
           float(np.max(np.abs(np.real(np.diag(h)) - d) / d)))  # fmt: skip
    assert out[0] < 1e-9 * out[1], f"bases disagree ({out[0]:.1e})"
    assert out[2] < 1e-9, f"diag H is not |Q|^2 N / V ({out[2]:.1e})"
    return out


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", default=None,
                        help="default: every chi tag present in each root")  # fmt: skip
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--surrogates", type=int, default=8)
    args = parser.parse_args()
    assert args.states >= 1 and args.surrogates >= 2, "states, surrogates"
    for root in args.roots:
        for tag in present_tags(root, args.tags or TAGS):
            err, top, diag_err = check_basis(root, tag)
            print(f"{root.name} chi{tag}: complex vs Re/Im basis max |dlambda| "
                  f"{err:.1e} (of {top:.1e}); diag H vs |Q|^2 N/V {diag_err:.1e}",
                  flush=True)  # fmt: skip
            res = run(root, tag, args.states, args.surrogates)
            np.savez(root / f"loops_chi{tag}.npz", **res)
    print("Done.", flush=True)
