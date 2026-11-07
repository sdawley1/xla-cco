#!/usr/bin/env python3

"""
Gradient flows to 2D stealthy ground states from Poisson initial conditions.

The 2D counterpart of batten2008_flows.py, on the PRE 2025 setting
(Dawley & Torquato, PRE 112, 045310): the first --n-traj Poisson
configurations of experiments/initial_conditions/POS-2D-N1024 (N = 1024,
density 1, square cell L = 32) and the same stealthy energies as
run_trajectories.py (K from chi; chi = 0.1 gives M = 210, chi = 0.4 gives
M = 820). Unlike those descent runs, this integrates the true gradient flow
dr/dt = -grad_r Phi (Rosenbrock ROS2 with the exact Hessian), so times are
flow times and the analyses of the 1D flows apply unchanged.

Resumable and chunked exactly like batten2008_flows.py; output is
<out>/flows_chi<tag>.{npz,json} with the same keys (traj_s is (B, rows, N, 2)),
plus "K", "d" and "N" in the JSON, which the analysis scripts read.

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
from fractions import Fraction
from pathlib import Path
from typing import Dict

# ----- numerics -----
import numpy as np

# ----- project -----
from xla_cco.cli.relax import select_device
from xla_cco.minimize.flow import FlowSettings
from xla_cco.minimize.result import STATUS_RUNNING

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import (  # noqa: E402
    STATUS_NAMES,
    chi_tag,
    consolidate,
    init_state_from,
    integrate,
)
from run_trajectories import build_problem  # noqa: E402

# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ic", type=Path, default=Path("experiments/initial_conditions/POS-2D-N1024")
    )
    parser.add_argument("--chi", nargs="+", default=["0.1", "0.4"])
    parser.add_argument("--n-traj", type=int, default=20)
    parser.add_argument(
        "--method", choices=("rosenbrock", "dopri5"), default="rosenbrock"
    )
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--energy-tol", type=float, default=1e-20)
    parser.add_argument("--grad-tol", type=float, default=1e-12)
    parser.add_argument("--max-steps", type=int, default=2_000_000)
    parser.add_argument("--chunk-steps", type=int, default=1000)
    parser.add_argument("--wall-budget", type=float, default=2700.0)
    parser.add_argument("--max-records", type=int, default=2000)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/trajectories/flows2d/N1024")
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    device = select_device(args.device)
    assert args.n_traj > 0 and args.chunk_steps > 0 and args.max_records >= 2
    started = time.time()

    for chi_str in args.chi:
        tag = chi_tag(Fraction(chi_str))
        s0_all, box, energy, phys = build_problem(args.ic, float(chi_str))
        assert s0_all.shape[0] >= args.n_traj, "not enough initial conditions"
        basis = np.asarray(box.basis)
        run_dir = args.out / f"chi{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / "state.npz"
        if state_path.exists():
            with np.load(state_path) as z:
                state: Dict[str, np.ndarray] = {k: z[k] for k in z.files}
        else:
            state = init_state_from(s0_all[: args.n_traj].copy(), 1e-3)
        fs = FlowSettings(
            method=args.method,
            rtol=args.rtol,
            atol=args.atol,
            energy_tol=args.energy_tol,
            grad_tol=args.grad_tol,
            max_steps=args.chunk_steps,
            record_every=1,
            max_records=args.chunk_steps + 2,
        )
        done = integrate(state, energy, basis, fs, run_dir, args.max_steps,
                         started + args.wall_budget, f"chi={chi_str}")  # fmt: skip
        if not done:
            continue
        paths = consolidate(run_dir, state, args.max_records)
        out = {
            "s0": state["s0"],
            "s_final": state["s"],
            "energy_final": state["energy"],
            "t_final": state["t"],
            "length_final": state["length"],
            "n_steps": state["steps"],
            "status": state["status"],
            "basis": basis,
            **paths,
        }
        np.savez(args.out / f"flows_chi{tag}.npz", **out)
        status = state["status"]
        meta = {
            **phys,
            "chi_str": chi_str,
            "n_traj": args.n_traj,
            "ic": str(args.ic),
            "method": args.method,
            "rtol": args.rtol,
            "atol": args.atol,
            "energy_tol": args.energy_tol,
            "grad_tol": args.grad_tol,
            "chunk_steps": args.chunk_steps,
            "device": device,
            "wall_seconds": float(state["wall"]),
            "status_counts": {
                STATUS_NAMES.get(int(c), str(c)): int(np.sum(status == c))
                for c in np.unique(status)
            },
        }
        (args.out / f"flows_chi{tag}.json").write_text(json.dumps(meta, indent=2))
        print(
            f"N={phys['N']} d=2 chi={chi_str} M={phys['M']}: status "
            f"{meta['status_counts']}; steps {int(state['steps'].min())}-"
            f"{int(state['steps'].max())}; Phi_final max "
            f"{np.max(state['energy']):.2e}; wall {float(state['wall']):.0f} s",
            flush=True,
        )
        assert not np.any(status == STATUS_RUNNING), "consolidated a running flow"
    print("Done.", flush=True)
