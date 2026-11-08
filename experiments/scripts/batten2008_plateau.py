#!/usr/bin/env python3

"""
Why half the soft Hessian directions are negative near the ground-state manifold.

For Phi = (1/V) sum_{q in C} |rho_q|^2 (C the constrained half-space set),
with rho_q = a_q + i b_q = sum_j exp(i q . r_j), the Cartesian Hessian splits
exactly into

  H = G + R,  G = (2/V) J^T J,  R = (2/V) sum_q (a_q dd a_q + b_q dd b_q),

where dd is the Hessian of a single wave and J (2M, N d) stacks grad a_q
and grad b_q. G (Gauss-Newton) is positive
semi-definite with rank 2M; R is the curvature carried by the residual
waves. R is block-diagonal: particle j has the d x d block

  R_j = -(2/V) sum_q w_{jq} q q^T,   w_{jq} = Re(rho_q^* exp(i q . r_j)),

and sum_j w_{jq} = |rho_q|^2, so the weights of each wave sum to (almost)
zero near a ground state. On the N d - 2M directions G does not see (the
null space P of J, tangent to the manifold), the soft eigenvalues of H are
those of P R P to first order in |rho| ~ sqrt(Phi).

For flows 0..B-1 of each system, at log10 Phi in LEVELS, this stores:
- lam: eigenvalues of H; lam_pr: eigenvalues of P R P; lam_r: eigenvalues of
  R (the 2 x 2 block eigenvalues);
- trace_j: tr R_j per particle; balance: median over q of
  |sum_j w_jq| / sum_j |w_jq|;
- phi, n_part, rank (of J), volume.

Output: <root>/plateau_chi<tag>.npz. NumPy on the CPU (one eigh of N d per
sample, plus an SVD of J).

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
from typing import Dict, List, Tuple

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import cell_lengths  # noqa: E402
from batten2008_hessian import level_frames  # noqa: E402
from batten2008_tail import tail_frames  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

LEVELS = (2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -10.0)  # log10 Phi
RANK_REL = 1e-10  # singular values of J below this x max count as zero

# ============================================================
# HELPERS
# ============================================================


def hessian_parts(
    x: np.ndarray, kvec: np.ndarray, volume: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(J (2M, N d), R blocks (N, d, d), weights w (N, M), Phi) at positions x.

    G = (2 / V) J^T J and R = blockdiag(R_j) sum to the Hessian of Phi.
    """
    n, d = x.shape
    assert kvec.ndim == 2 and kvec.shape[1] == d and volume > 0.0, "shapes"
    phase = x @ kvec.T  # (N, M)
    c, s = np.cos(phase), np.sin(phase)
    a, b = c.sum(axis=0), s.sum(axis=0)  # Re, Im rho_q
    ja = -(s[:, :, None] * kvec[None]).transpose(1, 0, 2).reshape(-1, n * d)
    jb = (c[:, :, None] * kvec[None]).transpose(1, 0, 2).reshape(-1, n * d)
    w = a[None, :] * c + b[None, :] * s  # Re(rho_q^* e^{i q r_j})
    qq = kvec[:, :, None] * kvec[:, None, :]  # (M, d, d)
    blocks = -(2.0 / volume) * np.einsum("jq,qab->jab", w, qq)
    phi = float((a @ a + b @ b) / volume)
    return np.concatenate([ja, jb]), blocks, w, phi


def block_diag(blocks: np.ndarray) -> np.ndarray:
    """Dense (N d, N d) matrix from N blocks (N, d, d)."""
    n, d, _ = blocks.shape
    out = np.zeros((n * d, n * d))
    for a in range(d):
        for b in range(d):
            out[a::d, b::d][np.diag_indices(n)] = blocks[:, a, b]
    return out


def analyze(x: np.ndarray, kvec: np.ndarray, volume: float) -> Dict[str, object]:
    """Spectra of H, of P R P and of R, and the weight balance, at x."""
    jac, blocks, w, phi = hessian_parts(x, kvec, volume)
    r = block_diag(blocks)
    h = (2.0 / volume) * jac.T @ jac + r
    _, sv, vt = np.linalg.svd(jac, full_matrices=True)
    rank = int(np.sum(sv > RANK_REL * sv[0]))
    q = vt[rank:].T  # null space of J: tangent to the manifold
    assert q.shape[1] == x.size - rank, "null-space size"
    balance = np.abs(w.sum(axis=0)) / np.maximum(np.abs(w).sum(axis=0), 1e-300)
    return {
        "lam": np.linalg.eigvalsh(h),
        "lam_pr": np.linalg.eigvalsh(q.T @ r @ q),
        "lam_r": np.sort(np.linalg.eigvalsh(blocks).ravel()),
        "trace_j": np.trace(blocks, axis1=1, axis2=2),
        "balance": float(np.median(balance)),
        "phi": phi,
        "rank": rank,
    }


def run_system(root: Path, tag: str, n_flows: int) -> Dict[str, np.ndarray]:
    """All samples of flows 0..n_flows-1 of one stored system."""
    with np.load(root / f"hessian_chi{tag}.npz") as z:
        nvec = np.asarray(z["nvec"])
    tail_path = root / f"tail_chi{tag}.npz"
    tail = dict(np.load(tail_path)) if tail_path.exists() else {}
    rows: Dict[str, List[object]] = {}
    with np.load(root / f"flows_chi{tag}.npz") as f:
        flows = {k: f[k] for k in ("traj_t", "traj_s", "traj_energy", "traj_count")}
        basis = np.asarray(f["basis"])
    side = cell_lengths(basis)
    kvec = nvec @ (2.0 * np.pi * np.linalg.inv(basis)).T
    volume = float(np.prod(side))
    for b in range(n_flows):
        _, s_fr, e = tail_frames(flows, tail, b)
        for level, k in zip(LEVELS, level_frames(e, LEVELS), strict=True):
            out = analyze(s_fr[k] * side, kvec, volume)
            assert np.isclose(out["phi"], e[k], rtol=1e-6), "energy does not match"
            out["level"] = level
            for key, val in out.items():
                rows.setdefault(key, []).append(val)
    shape = (n_flows, len(LEVELS))
    res = {k: np.asarray(v).reshape(shape + np.asarray(v[0]).shape)
           for k, v in rows.items()}  # fmt: skip
    res.update(levels=np.array(LEVELS), nvec=nvec, volume=np.array(volume),
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
            np.savez(root / f"plateau_chi{tag}.npz", **res)
            nd = int(res["n_part"]) * res["nvec"].shape[1]
            m2 = 2 * res["nvec"].shape[0]
            neg = (res["lam_pr"] < 0).sum(axis=-1)
            print(f"{root} chi{tag}: Nd - 2M = {nd - m2}, rank J = "
                  f"{int(np.median(res['rank']))}; negative P R P per level "
                  f"(median) {np.median(neg, axis=0).astype(int).tolist()}; "
                  f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip
