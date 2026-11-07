#!/usr/bin/env python3

"""
Ensemble statistics of the Wasserstein-2 and proximity-metric curves.

Reads <traj>/wasserstein/exact_<ENS>.npz and sinkhorn_<ENS>[_<start>].npz,
interpolates every trajectory's curves onto a common t/T grid (linear plus
log-spaced), keeping
the ensemble mean and 10-90 % band of

- exact frames:    w0_dist = W_0(inf) - W_0(t), w_inf = W_inf(t),
                   p0_dist = p_0(inf) - p_0(t), q_inf = q_inf(t),
                   n_re0, n_re_inf (labels re-matched), log10 Phi;
- Sinkhorn frames: sk_w0_dist, sk_w_inf (rounded matching), certified
                   frames only (bijective rounding or converged plan).

Per-trajectory scalars (W_0(inf), p_0(inf)) and the Sinkhorn-vs-exact
discrepancy on the exact frames are also stored. Writes
<traj>/wasserstein/summary.npz, consumed by plot_wasserstein.py.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
from pathlib import Path
from typing import Dict, List

# ----- numerics -----
import numpy as np

# ============================================================
# HELPERS
# ============================================================


def load_sinkhorn(wdir: Path, ens: str) -> Dict[str, np.ndarray]:
    """Load sinkhorn_<ENS>.npz, or merge its slice files in trajectory order."""
    parts = sorted(wdir.glob(f"sinkhorn_{ens}_*.npz"))
    whole = wdir / f"sinkhorn_{ens}.npz"
    assert parts or whole.exists(), f"no Sinkhorn output for {ens} in {wdir}"
    if not parts:
        parts = [whole]
    loaded = []
    for path in parts:
        with np.load(path) as npz:
            loaded.append({k: npz[k] for k in npz.files})
    out = {k: v for k, v in loaded[0].items() if k not in ("offsets", "names")}
    for k in out:
        if out[k].ndim > 0 and k != "schedule":
            out[k] = np.concatenate([d[k] for d in loaded])
    lengths = np.concatenate([np.diff(d["offsets"]) for d in loaded])
    out["offsets"] = np.concatenate([[0], np.cumsum(lengths)])
    out["names"] = np.concatenate([d["names"] for d in loaded])
    assert out["offsets"][-1] == out["u"].shape[0], "inconsistent parts"
    return out


def band(curves: List[np.ndarray], prefix: str) -> Dict[str, np.ndarray]:
    """Return mean and 10/90 % quantiles of equal-length curves."""
    assert curves and all(c.shape == curves[0].shape for c in curves), "ragged"
    arr = np.stack(curves)
    return {
        f"{prefix}_mean": arr.mean(axis=0),
        f"{prefix}_q10": np.quantile(arr, 0.10, axis=0),
        f"{prefix}_q90": np.quantile(arr, 0.90, axis=0),
    }


def exact_curves(ex: Dict[str, np.ndarray], grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Interpolate the exact-frame quantities of every trajectory onto ``grid``."""
    assert ex["u"].ndim == 2 and grid.ndim == 1, "expected (traj, frames) arrays"
    per: Dict[str, List[np.ndarray]] = {}
    for i in range(ex["u"].shape[0]):
        _, keep = np.unique(ex["frame"][i], return_index=True)
        u = ex["u"][i, keep]
        series = {
            "w0_dist": ex["w0"][i, -1] - ex["w0"][i, keep],
            "p0_dist": ex["p0"][i, -1] - ex["p0"][i, keep],
            "w_inf": ex["w_inf"][i, keep],
            "q_inf": ex["q_inf"][i, keep],
            "n_re0": ex["n_re0"][i, keep].astype(float),
            "n_re_inf": ex["n_re_inf"][i, keep].astype(float),
            "log10_energy": np.log10(ex["energy"][i, keep]),
        }
        for k, v in series.items():
            per.setdefault(k, []).append(np.interp(grid, u, v))
    out: Dict[str, np.ndarray] = {}
    for k, v in per.items():
        out.update(band(v, k))
    return out


def certified(sk: Dict[str, np.ndarray]) -> np.ndarray:
    """Frames whose Sinkhorn matching is trusted: bijective rounding or a
    converged plan (marginal error < tol, i.e. a genuine near-tie that the
    greedy repair resolves). Frames still unconverged at the iteration cap
    are rejected."""
    assert sk["is_perm"].shape == sk["err"].shape, "shape mismatch"
    ok: np.ndarray = (sk["is_perm"] == 1) | (sk["err"] < float(sk["tol"]))
    return ok


def sinkhorn_curves(
    sk: Dict[str, np.ndarray], w0_inf: np.ndarray, grid: np.ndarray
) -> Dict[str, np.ndarray]:
    """Interpolate certified per-frame Sinkhorn distances onto ``grid``.

    W_0(inf) is taken from the exact pass (``w0_inf``, one per trajectory).
    """
    off = sk["offsets"]
    assert off[0] == 0 and off[-1] == sk["u"].shape[0], "bad offsets"
    assert w0_inf.shape == (off.shape[0] - 1,), "one W_0(inf) per trajectory"
    ok = certified(sk)
    per: Dict[str, List[np.ndarray]] = {"sk_w0_dist": [], "sk_w_inf": []}
    for i, (a, b) in enumerate(zip(off[:-1], off[1:], strict=True)):
        u, w = sk["u"][a:b], sk["w_round"][a:b]
        m0, m1 = ok[a:b, 0], ok[a:b, 1]
        per["sk_w0_dist"].append(np.interp(grid, u[m0], w0_inf[i] - w[m0, 0]))
        per["sk_w_inf"].append(np.interp(grid, u[m1], w[m1, 1]))
    out: Dict[str, np.ndarray] = {}
    for k, v in per.items():
        out.update(band(v, k))
    return out


def discrepancy(
    ex: Dict[str, np.ndarray], sk: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Max relative |Sinkhorn - exact| on the exact frames, by frame class.

    ``rel_certified`` covers the frames kept by ``certified``; ``rel_rejected``
    the rejected ones; ``rel_ent`` the entropic-plan estimate on all frames.
    ``frac_rejected`` is the rejected fraction of all Sinkhorn frames.
    """
    off = sk["offsets"]
    assert off.shape[0] - 1 == ex["u"].shape[0], "trajectory count mismatch"
    ok = certified(sk)
    worst = {"rel_certified": 0.0, "rel_rejected": 0.0, "rel_ent": 0.0}
    for i in range(ex["u"].shape[0]):
        rows = off[i] + ex["frame"][i]
        for col, key in ((0, "w0"), (1, "w_inf")):
            ref = ex[key][i]
            big = ref > 1e-6
            rel = np.abs(sk["w_round"][rows, col] - ref) / np.maximum(ref, 1e-300)
            ent = np.abs(sk["w_ent"][rows, col] - ref) / np.maximum(ref, 1e-300)
            for name, mask in (
                ("rel_certified", big & ok[rows, col]),
                ("rel_rejected", big & ~ok[rows, col]),
                ("rel_ent", big),
            ):
                if mask.any():
                    vals = ent if name == "rel_ent" else rel
                    worst[name] = max(worst[name], float(vals[mask].max()))
    worst["frac_rejected"] = float(1.0 - ok.mean())
    return worst


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # ----- inputs -----
    parser.add_argument(
        "--traj", type=Path, default=Path("experiments/trajectories/chi0.1")
    )
    parser.add_argument("--ensembles", nargs="+", default=["POS", "GAU"])

    # ----- grid -----
    parser.add_argument("--n-grid", type=int, default=1001)
    parser.add_argument("--grid-tmin", type=float, default=1e-6)

    args = parser.parse_args()
    wdir = args.traj / "wasserstein"
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, args.n_grid),
                np.logspace(np.log10(args.grid_tmin), 0.0, args.n_grid),
            ]
        )
    )
    result: Dict[str, np.ndarray] = {}

    # =========================================================
    # ENSEMBLES
    # =========================================================
    for ens in args.ensembles:
        with np.load(wdir / f"exact_{ens}.npz") as npz:
            ex = {k: npz[k] for k in npz.files}
        sk = load_sinkhorn(wdir, ens)
        assert np.array_equal(sk["names"], ex["names"]), "exact/Sinkhorn mismatch"
        result[f"{ens}/grid"] = grid
        for k, v in {
            **exact_curves(ex, grid),
            **sinkhorn_curves(sk, ex["w0"][:, -1], grid),
        }.items():
            result[f"{ens}/{k}"] = v
        result[f"{ens}/w0_inf"] = ex["w0"][:, -1]
        result[f"{ens}/p0_inf"] = ex["p0"][:, -1]
        result[f"{ens}/n_re_final"] = ex["n_re0"][:, -1]
        worst = discrepancy(ex, sk)
        for k, v in worst.items():
            result[f"{ens}/disc_{k}"] = np.asarray(v)
        result[f"{ens}/n_nonperm"] = np.asarray(int(np.sum(sk["is_perm"] == 0)))
        result[f"{ens}/eps"] = sk["eps"]
        gap = 1.0 - ex["w0"][:, -1] / ex["p0"][:, -1]
        print(
            f"{ens}: <W0(inf)> = {ex['w0'][:, -1].mean():.4f}, "
            f"<p0(inf)> = {ex['p0'][:, -1].mean():.4f}, max gap {gap.max():.2e}, "
            f"<n re-matched> = {ex['n_re0'][:, -1].mean():.1f}; Sinkhorn vs exact: "
            + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()),
            flush=True,
        )

    # =========================================================
    # SAVE
    # =========================================================
    np.savez(wdir / "summary.npz", **result)
    print(f"wrote {wdir / 'summary.npz'}", flush=True)
    print("Done.", flush=True)
