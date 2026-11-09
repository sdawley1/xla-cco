#!/usr/bin/env python3

"""
Can single density waves inside the exclusion region be excited independently?

At a ground state r0 the constrained waves u = (Re rho_q, Im rho_q) (2M
numbers) have gradients J (2M, N d) and coupling K = (2/V) J J^T; to linear
order a displacement dx changes u by J dx, Phi = |u|^2 / V, and the
dynamics are du/dt = -K u (gradient flow) or d^2u/dt^2 = -K u (MD, mass 1).
For flows 0..B-1 of each system this stores, per wave (averaged over its Re
and Im coordinates):

- cost: |k|^2 (K^-1)_kk, the least squared displacement that changes only
  wave k, relative to an independent well (1); its random-matrix mean is
  1 / (1 - c), c = 2M / (N d) (Marchenko-Pastur);
- s_thermal: (2/V) (J H^+ J^T)_kk, i.e. <S(k)>/T of a harmonic canonical
  ensemble (H = (2/V) J^T J; equipartition gives 1 for every k);
- modes: participation number of the wave over the eigenvectors of K (how
  many collective modes its fluctuation is spread over); om_mean, om_std:
  mean and spread of the collective frequencies sqrt(lambda) it carries
  (MD), over |k|;
- retention(x): the share of a gradient-flow excitation of wave k alone
  that is still in wave k at |k|^2 t = x (1 for independent wells);
- soft_waves: the number of waves the softest collective mode of K spans
  (participation number over waves);
- for flow 0: the frequency weights of three waves (|k|/K = 0.25, 0.5, 0.9)
  and the full eigensystem of K (eig_lam (2M,), eig_q (2M, 2M); rows are
  the coordinates Re rho_1..M, Im rho_1..M).

Output: <root>/exclusion_chi<tag>.npz. NumPy (one eigh of K per ground state).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import sys
from pathlib import Path
from typing import Dict, List

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import cell_lengths  # noqa: E402
from batten2008_plateau import hessian_parts  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

X_GRID = np.geomspace(1e-2, 30.0, 60)  # |k|^2 t of the retention curves
EXAMPLE_FRACS = (0.25, 0.5, 0.9)  # |k|/K of the example spectra (flow 0)

# ============================================================
# HELPERS
# ============================================================


def wave_stats(
    jac: np.ndarray, kmag: np.ndarray, volume: float
) -> Dict[str, np.ndarray]:
    """Per-wave cost, thermal S/T, mode count and frequency moments at one state."""
    m = kmag.shape[0]
    assert jac.shape[0] == 2 * m and volume > 0.0, "J must have 2M rows"
    k_mat = (2.0 / volume) * jac @ jac.T
    lam, q = np.linalg.eigh(k_mat)
    assert lam[0] > 0.0, "K must be positive definite (chi < 1/2)"
    u_svd, sv, _ = np.linalg.svd(jac, full_matrices=False)
    rank = int(np.sum(sv > 1e-10 * sv[0]))
    s_th = np.sum(u_svd[:, :rank] ** 2, axis=1)  # diag (2/V) J H^+ J^T
    w = q**2
    om = np.sqrt(lam)
    k2 = np.concatenate([kmag, kmag]) ** 2
    mean = (w * om).sum(axis=1)
    per_coord = {
        "cost": k2 * (w / lam).sum(axis=1),
        "s_thermal": s_th,
        "modes": 1.0 / (w**2).sum(axis=1),
        "om_mean": mean / np.sqrt(k2),
        "om_std": np.sqrt(np.maximum((w * om**2).sum(axis=1) - mean**2, 0.0))
        / np.sqrt(k2),
    }
    out = {key: 0.5 * (v[:m] + v[m:]) for key, v in per_coord.items()}
    out["retention"] = retention(q, lam, k2, m)
    p_soft = q[:m, 0] ** 2 + q[m:, 0] ** 2  # softest collective mode over waves
    out["soft_waves"] = np.array(1.0 / np.sum(p_soft**2))
    out["lam"], out["w"], out["q"] = lam, w, q
    return out


def retention(q: np.ndarray, lam: np.ndarray, k2: np.ndarray, m: int) -> np.ndarray:
    """(M, X) share of a single-coordinate excitation left in its own wave."""
    assert q.shape == (2 * m, 2 * m) and lam.shape == (2 * m,), "K eigenpairs"
    out = np.zeros((2 * m, X_GRID.shape[0]))
    for j, x in enumerate(X_GRID):
        e = np.exp(-np.outer(x / k2, lam))  # (coord, mode)
        u_self = (q**2 * e).sum(axis=1)  # own coordinate
        partner = (q * e) @ q.T  # u(t) of each excited coordinate, (coord, coord)
        idx = np.arange(2 * m)
        twin = np.where(idx < m, idx + m, idx - m)  # the other quadrature of wave k
        own = u_self**2 + partner[idx, twin] ** 2
        out[:, j] = own / np.maximum((partner**2).sum(axis=1), 1e-300)
    return 0.5 * (out[:m] + out[m:])


def run_system(root: Path, tag: str, n_flows: int) -> Dict[str, np.ndarray]:
    """Statistics of flows 0..n_flows-1 at their ground states."""
    with np.load(root / f"hessian_chi{tag}.npz") as z:
        nvec, kmag = np.asarray(z["nvec"]), np.asarray(z["kmag"])
    with np.load(root / f"flows_chi{tag}.npz") as f:
        s_final, basis = np.asarray(f["s_final"][:n_flows]), np.asarray(f["basis"])
    side = cell_lengths(basis)
    kvec = nvec @ (2.0 * np.pi * np.linalg.inv(basis)).T
    volume = float(np.prod(side))
    rows: Dict[str, List[np.ndarray]] = {}
    for b in range(s_final.shape[0]):
        jac, _, _, phi = hessian_parts(s_final[b] * side, kvec, volume)
        assert phi < 1e-15, f"flow {b} is not at a ground state (Phi = {phi:.1e})"
        st = wave_stats(jac, kmag, volume)
        if b == 0:
            for frac in EXAMPLE_FRACS:
                row = int(np.argmin(np.abs(kmag / kmag[-1] - frac)))
                rows.setdefault("ex_rows", []).append(np.array(row))
                w_row = 0.5 * (st["w"][row] + st["w"][row + kmag.shape[0]])
                rows.setdefault("ex_w", []).append(w_row)
            rows["ex_om"] = [np.sqrt(st["lam"])]
            rows["eig_lam"], rows["eig_q"] = [st["lam"]], [st["q"]]
        keys = ("cost", "s_thermal", "modes", "om_mean", "om_std", "retention",
                "soft_waves")  # fmt: skip
        for key in keys:
            rows.setdefault(key, []).append(st[key])
    out = {k: np.asarray(v) for k, v in rows.items()}
    out.update(kmag=kmag, x_grid=X_GRID,
               c=np.array(2.0 * kmag.shape[0] / s_final[0].size))  # fmt: skip
    return out


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", default=["1-10", "2-5"])
    parser.add_argument("--flows", type=int, default=5)
    args = parser.parse_args()
    assert args.flows >= 1, "need at least one ground state"
    for root in args.roots:
        for tag in args.tags:
            res = run_system(root, tag, args.flows)
            np.savez(root / f"exclusion_chi{tag}.npz", **res)
            print(f"{root} chi{tag}: c = {float(res['c']):.3f}, median cost "
                  f"{np.median(res['cost']):.2f} (MP {1 / (1 - float(res['c'])):.2f}), "
                  f"S/T thermal in [{res['s_thermal'].min():.12f}, "
                  f"{res['s_thermal'].max():.12f}], modes/wave "
                  f"{np.median(res['modes']):.0f}", flush=True)  # fmt: skip
