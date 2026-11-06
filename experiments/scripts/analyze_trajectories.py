#!/usr/bin/env python3

"""
Proximity-metric analysis of recorded descent trajectories.

For every trajectory (Dawley & Torquato, PRE 112, 045310 (2025), Eq. (10)):
p_0(t) = rho^{1/2} [sum_j |r_j(t) - r_j(0)|^2]^{1/2} on unwrapped paths, the
configurational distance from the ground-state manifold p_0(inf) - p_0(t),
and the energy Phi(t), all versus the dimensionless time t/T. Ensemble
statistics (mean and 10-90 % band) are formed on a common t/T grid. The
initial-condition structure factors S(k) are computed for Fig. 2(a)-style
plots. Writes one ``summary.npz`` (git-ignored) consumed by plot_figure.py.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

# ----- numerics -----
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.analysis.structure import isotropic_structure_factor
from xla_cco.analysis.trajectory import proximity_metric, unwrap_fractional
from xla_cco.box import Box
from xla_cco.io.configpack import read_configpack, stack_fractional
from xla_cco.io.trajstore import open_trajectory

# ============================================================
# PER TRAJECTORY
# ============================================================


def trajectory_curves(prefix: Path) -> Dict[str, np.ndarray]:
    """Return per-frame t/T, p0(inf) - p0(t), Phi(t) and scalars for one run."""
    frames, arrays, meta = open_trajectory(prefix)
    assert frames.shape[0] >= 2, f"{prefix}: trajectory has no accepted moves"
    basis = np.asarray(meta["basis"], dtype=np.float64)
    paths = unwrap_fractional(np.asarray(frames, dtype=np.float64))
    p0 = proximity_metric(paths, paths[0], basis, float(meta["density"]))
    acc = arrays["accepted"].astype(bool)
    t_frame = np.concatenate([[0.0], arrays["time"][acc]])
    e_frame = np.concatenate([[float(arrays["energy0"])], arrays["energy"][acc]])
    total = float(arrays["time"][-1])
    assert t_frame.shape[0] == p0.shape[0] and total > 0.0, "frame/time mismatch"
    return {
        "u": t_frame / total,
        "dist": p0[-1] - p0,
        "energy": e_frame,
        "p0_inf": np.asarray(p0[-1]),
        "T": np.asarray(total),
        "n_iters": np.asarray(meta["n_iters"]),
        "final_energy": np.asarray(meta["final_energy"]),
    }


# ============================================================
# ENSEMBLE
# ============================================================


def ensemble_stats(
    curves: List[Dict[str, np.ndarray]], grid: np.ndarray
) -> Dict[str, np.ndarray]:
    """Interpolate curves onto ``grid`` (t/T) and return mean and 10/90 %."""
    assert len(curves) > 0 and grid.ndim == 1, "need curves and a 1-D grid"
    dist = np.stack([np.interp(grid, c["u"], c["dist"]) for c in curves])
    loge = np.stack([np.interp(grid, c["u"], np.log10(c["energy"])) for c in curves])
    out: Dict[str, np.ndarray] = {"grid": grid}
    for name, arr in (("dist", dist), ("log10_energy", loge)):
        out[f"{name}_mean"] = arr.mean(axis=0)
        out[f"{name}_q10"] = np.quantile(arr, 0.10, axis=0)
        out[f"{name}_q90"] = np.quantile(arr, 0.90, axis=0)
    for key in ("p0_inf", "T", "n_iters", "final_energy"):
        out[key] = np.array([c[key] for c in curves])
    return out


def initial_sk(prefix: Path, bin_width: float, k_max: float) -> Tuple[np.ndarray, ...]:
    """Return (k, S, S_sem) of an initial-condition ensemble."""
    assert bin_width > 0.0 and k_max > 0.0, "positive binning required"
    s, unit = stack_fractional(read_configpack(prefix))
    n, d = s.shape[1], s.shape[2]
    box = Box(basis=jnp.asarray(unit * n ** (1.0 / d)))
    k, sk, _, sem = isotropic_structure_factor(
        jnp.asarray(s), box, k_dir_max=5.0, k_max=k_max, bin_width=bin_width
    )
    return k, sk, sem


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # ----- inputs -----
    parser.add_argument(
        "--traj", type=Path, default=Path("experiments/trajectories/chi0.1")
    )
    parser.add_argument(
        "--ic", type=Path, default=Path("experiments/initial_conditions")
    )
    parser.add_argument("--ensembles", nargs="+", default=["POS", "GAU"])
    parser.add_argument(
        "--ic-names", nargs="+", default=["POS-2D-N1024", "GAU-2D-N1000"]
    )

    # ----- grids -----
    parser.add_argument("--n-grid", type=int, default=1001)
    parser.add_argument("--sk-bin", type=float, default=0.1)
    parser.add_argument("--sk-kmax", type=float, default=10.0)

    # ----- output -----
    parser.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()
    out_path = args.out or args.traj / "summary.npz"
    grid = np.linspace(0.0, 1.0, args.n_grid)
    result: Dict[str, np.ndarray] = {}

    # =========================================================
    # TRAJECTORIES AND INITIAL CONDITIONS
    # =========================================================
    for ens, ic_name in zip(args.ensembles, args.ic_names, strict=True):
        manifest = json.loads((args.traj / ens / "manifest.json").read_text())
        prefixes = sorted(
            {
                p.with_suffix("").with_suffix("")
                for p in (args.traj / ens).glob("traj_*.json")
            }
        )
        curves = [trajectory_curves(p) for p in prefixes]
        for key, val in ensemble_stats(curves, grid).items():
            result[f"{ens}/{key}"] = val
        k, sk, sem = initial_sk(args.ic / ic_name, args.sk_bin, args.sk_kmax)
        result.update({f"{ens}/sk_k": k, f"{ens}/sk_S": sk, f"{ens}/sk_sem": sem})
        for key in ("K", "chi_actual", "M", "N"):
            result[f"{ens}/{key}"] = np.asarray(manifest[key])
        print(
            f"{ens}: {len(curves)} trajectories, <p0(inf)> = "
            f"{result[f'{ens}/p0_inf'].mean():.3f}, <T> = "
            f"{result[f'{ens}/T'].mean():.3f}",
            flush=True,
        )

    # =========================================================
    # SAVE
    # =========================================================
    np.savez(out_path, **result)
    print(f"wrote {out_path}", flush=True)
    print("Done.", flush=True)
