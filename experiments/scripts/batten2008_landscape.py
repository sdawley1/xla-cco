#!/usr/bin/env python3

"""
The wave landscape along the 2D flows: independent wells or a coupled valley.

With u = (Re rho_q, Im rho_q) over the constrained set C (2M numbers) and J
(2M, N d) their gradients, Phi = |u|^2 / V and grad Phi = (2/V) J^T u, so
along the flow dr/dt = -grad Phi the waves obey exactly

  du/dt = -K u,   K = (2/V) J J^T   (2M x 2M, positive semi-definite).

K_qq ~ |q|^2 is the stiffness of wave q's own well; the off-diagonal K_pq
couple the waves (through the density fluctuations at q -+ p). Independent
quadratic wells means K diagonal. K is also the Hessian of Phi seen from
the waves (G = (2/V) J^T J has the same non-zero spectrum).

For flows 0..B-1 of each system, at log10 Phi in LEVELS (every half
decade), this stores:
- k_eig: eigenvalues of K; c_eig: eigenvalues of C = D^-1/2 K D^-1/2,
  D = diag K (all ones without coupling; for random wave gradients the
  Marchenko-Pastur law with ratio 2M / (N d));
- coupling: median over waves of sqrt(sum_{p != q} C_pq^2);
- spread: per eigenvector of K, the standard deviation of ln|k| over the
  waves it occupies (0 if it is one shell);
- weight: the residual's share (v_i . u)^2 / |u|^2 on each eigenvector;
- ray_k = u^T K u / u^T u (= -1/2 d ln Phi / dt exactly) and
  ray_d = u^T D u / u^T u (the same rate for independent wells);
- soft subspace (eigenvectors of the n_soft smallest eigenvalues, n_soft =
  the coordinates of the waves with |k| <= the 10 % quantile, whole
  shells): its overlap with those waves' own coordinates (o_wells; 1 for
  independent wells), with the next level (o_next) and with the last level
  (o_end) of the same flow.

Output: <root>/landscape_chi<tag>.npz. NumPy on the CPU.

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
from pathlib import Path
from typing import Dict, List

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import cell_lengths  # noqa: E402
from batten2008_hessian import level_frames  # noqa: E402
from batten2008_plateau import hessian_parts  # noqa: E402
from batten2008_tail import tail_frames  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

LEVELS = tuple(round(2.5 - 0.5 * i, 2) for i in range(26))  # log10 Phi 2.5 .. -10
SOFT_QUANTILE = 0.1  # soft subspace: waves with |k| up to this quantile

# ============================================================
# HELPERS
# ============================================================


def soft_count(kmag: np.ndarray) -> int:
    """Re and Im coordinates of the waves up to the SOFT_QUANTILE of |k|."""
    assert kmag.ndim == 1 and np.all(np.diff(kmag) >= -1e-12), "kmag sorted"
    k_cut = kmag[int(SOFT_QUANTILE * (kmag.shape[0] - 1))]
    return 2 * int(np.sum(kmag <= k_cut + 1e-9))


def wave_matrix(x: np.ndarray, kvec: np.ndarray, volume: float) -> Dict:
    """K = (2/V) J J^T, u = (Re rho, Im rho) and Phi at positions x."""
    jac, _, _, phi = hessian_parts(x, kvec, volume)
    phase = x @ kvec.T
    u = np.concatenate([np.cos(phase).sum(axis=0), np.sin(phase).sum(axis=0)])
    return {"k": (2.0 / volume) * jac @ jac.T, "u": u, "phi": phi}


def spectra(k: np.ndarray, u: np.ndarray, logk: np.ndarray, n_soft: int) -> Dict:
    """Spectra, residual weights, eigenvector spread and the soft subspace of K."""
    assert k.shape == (u.shape[0], u.shape[0]) and n_soft <= u.shape[0], "shapes"
    d = np.diag(k).copy()
    lam, vec = np.linalg.eigh(k)
    c = k / np.sqrt(np.outer(d, d))
    off = c - np.diag(np.diag(c))
    p = vec**2  # (2M, 2M): eigenvector i's weight on coordinate q
    mean = logk @ p
    spread = np.sqrt(np.maximum((logk**2) @ p - mean**2, 0.0))
    soft = vec[:, :n_soft]
    m, half = d.shape[0] // 2, n_soft // 2
    own = np.r_[0:half, m : m + half]  # Re and Im coordinates of the soft waves
    uu = float(u @ u)
    return {
        "k_eig": lam,
        "c_eig": np.linalg.eigvalsh(c),
        "coupling": float(np.median(np.sqrt((off**2).sum(axis=1)))),
        "spread": spread,
        "weight": (vec.T @ u) ** 2 / uu,
        "ray_k": float(u @ k @ u) / uu,
        "ray_d": float(u @ (d * u)) / uu,
        "o_wells": float(np.sum(soft[own] ** 2) / n_soft),
        "_soft": soft,
    }


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Subspace overlap ||a^T b||_F^2 / n of two orthonormal bases (columns)."""
    assert a.shape == b.shape, "same subspace size"
    return float(np.sum((a.T @ b) ** 2) / a.shape[1])


def run_flow(
    s_fr: np.ndarray, e: np.ndarray, side: np.ndarray, kvec: np.ndarray,
    logk: np.ndarray, n_soft: int,
) -> Dict[str, List[object]]:  # fmt: skip
    """Every level of one flow, with the level-to-level soft-subspace overlaps."""
    volume = float(np.prod(side))
    rows: Dict[str, List[object]] = {}
    for k in level_frames(e, LEVELS):
        wm = wave_matrix(s_fr[k] * side, kvec, volume)
        assert np.isclose(wm["phi"], e[k], rtol=1e-6), "energy does not match"
        out = spectra(wm["k"], wm["u"], logk, n_soft)
        out["phi"] = wm["phi"]
        for key, val in out.items():
            rows.setdefault(key, []).append(val)
    soft = rows.pop("_soft")
    rows["o_next"] = [
        overlap(a, b) for a, b in zip(soft[:-1], soft[1:], strict=True)
    ] + [1.0]
    rows["o_end"] = [overlap(a, soft[-1]) for a in soft]
    return rows


def run_system(root: Path, tag: str, n_flows: int) -> Dict[str, np.ndarray]:
    """All levels of flows 0..n_flows-1 of one stored system."""
    with np.load(root / f"hessian_chi{tag}.npz") as z:
        nvec, kmag = np.asarray(z["nvec"]), np.asarray(z["kmag"])
    tail_path = root / f"tail_chi{tag}.npz"
    tail = dict(np.load(tail_path)) if tail_path.exists() else {}
    with np.load(root / f"flows_chi{tag}.npz") as f:
        flows = {k: f[k] for k in ("traj_t", "traj_s", "traj_energy", "traj_count")}
        basis = np.asarray(f["basis"])
    side = cell_lengths(basis)
    kvec = nvec @ (2.0 * np.pi * np.linalg.inv(basis)).T
    logk = np.log(np.concatenate([kmag, kmag]))
    n_soft = soft_count(kmag)
    rows: Dict[str, List[object]] = {}
    for b in range(n_flows):
        _, s_fr, e = tail_frames(flows, tail, b)
        for key, val in run_flow(s_fr, e, side, kvec, logk, n_soft).items():
            rows.setdefault(key, []).append(val)
    res = {k: np.asarray(v) for k, v in rows.items()}
    res.update(levels=np.array(LEVELS), nvec=nvec, kmag=kmag, n_soft=np.array(n_soft),
               n_part=np.array(s_fr.shape[1]))  # fmt: skip
    return res


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", default=["1-10", "2-5"])
    parser.add_argument("--flows", type=int, default=10)
    args = parser.parse_args()
    assert args.flows >= 1, "need at least one flow"
    for root in args.roots:
        for tag in args.tags:
            t0 = time.time()
            res = run_system(root, tag, args.flows)
            np.savez(root / f"landscape_chi{tag}.npz", **res)
            lv = list(res["levels"])
            j = lv.index(-4.0)
            print(f"{root} chi{tag}: n_soft {int(res['n_soft'])}; at Phi = 1e-4 "
                  f"(median) min c_eig {np.median(res['c_eig'][:, j, 0]):.3f}, "
                  f"coupling {np.median(res['coupling'][:, j]):.3f}, ray_k/ray_d "
                  f"{np.median(res['ray_k'][:, j] / res['ray_d'][:, j]):.3f}, o_wells "
                  f"{np.median(res['o_wells'][:, j]):.3f}; {time.time() - t0:.0f} s",
                  flush=True)  # fmt: skip
