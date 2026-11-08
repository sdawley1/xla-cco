#!/usr/bin/env python3

"""
Re-integrate the final approach of stored flows with a capped step.

Once the remaining motion falls below the absolute tolerance, the L-stable
Rosenbrock steps of the stored flows grow without bound: the last three
steps cover most of the flow time and the energy drops 2-4 decades per
recorded frame below Phi ~ 1e-13. That is harmless for the end state but
leaves the endgame (the last negative-curvature directions and the last
zero modes) with a handful of frames.

This script restarts every flow from its last stored frame with
Phi >= --phi-start and integrates the same gradient flow (same method,
tolerances and stops as the stored flows' JSON) with the step capped at
dt_max = --dt-frac * tau_1, where tau_1 = 1 / (2 |k_1|^2) is the relaxation
time of the slowest constrained mode (S(k_1) ~ exp(-t / tau_1) at the end).
Every accepted step is recorded.

Output: <root>/N<N>/tail_chi<tag>.npz with, per flow, the restart row
("start_index") of flows_chi<tag>.npz and the tail ("traj_s", "traj_t",
"traj_length", "traj_energy": absolute clocks; "traj_count" valid rows; row 0
is the restart frame), plus "status", "dt_max" and "tau1". Analyses splice
the stored rows up to start_index with tail rows 1.. (tail_frames).

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
from typing import Dict, Tuple

# ----- numerics -----
import numpy as np

# ----- project -----
from xla_cco.cli.relax import select_device
from xla_cco.minimize.flow import FlowSettings
from xla_cco.minimize.result import STATUS_BUDGET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_flows import STATUS_NAMES, chi_tag, load_system, run_chunk  # noqa: E402

# ============================================================
# HELPERS
# ============================================================


def restart_index(energy: np.ndarray, phi_start: float) -> int:
    """Last row with Phi >= phi_start (0 if none); energy decreases along a flow."""
    assert energy.ndim == 1 and energy.shape[0] >= 1, "one energy per frame"
    assert phi_start > 0.0, "phi_start must be positive"
    above = np.flatnonzero(energy >= phi_start)
    return int(above[-1]) if above.size else 0


def tail_frames(
    flows: Dict[str, np.ndarray], tail: Dict[str, np.ndarray], b: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, s, Phi) of flow b: stored rows up to the restart, then the tail.

    ``flows`` is a flows_chi<tag>.npz, ``tail`` a tail_chi<tag>.npz (or {} to
    use the stored rows alone). The tail's row 0 repeats the restart frame.
    """
    cnt = int(flows["traj_count"][b])
    t, s = flows["traj_t"][b, :cnt], flows["traj_s"][b, :cnt]
    e = flows["traj_energy"][b, :cnt]
    if not tail:
        return t, s, e
    k, m = int(tail["start_index"][b]), int(tail["traj_count"][b])
    assert 0 <= k < cnt and m >= 1, "tail does not match the stored flow"
    assert np.isclose(tail["traj_t"][b, 0], t[k]), "tail restarts at another time"
    return (
        np.concatenate([t[: k + 1], tail["traj_t"][b, 1:m]]),
        np.concatenate([s[: k + 1], tail["traj_s"][b, 1:m]]),
        np.concatenate([e[: k + 1], tail["traj_energy"][b, 1:m]]),
    )


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
    parser.add_argument("--phi-start", type=float, default=1e-9)
    parser.add_argument("--dt-frac", type=float, default=0.125)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    assert args.dt_frac > 0.0 and args.max_steps > 0, "bad arguments"
    select_device(args.device)

    for n in args.sizes:
        root = args.root / f"N{n}"
        for chi_str in args.chi:
            t0 = time.time()
            tag = chi_tag(Fraction(chi_str))
            meta = json.loads((root / f"flows_chi{tag}.json").read_text())
            with np.load(root / f"flows_chi{tag}.npz") as z:
                flows = {k: z[k] for k in z.files}
            energy, basis, _, kmag = load_system(root, tag)
            tau1 = 1.0 / (2.0 * float(kmag[0]) ** 2)
            n_flow = flows["s0"].shape[0]
            start = np.array([
                restart_index(flows["traj_energy"][b, : int(c)], args.phi_start)
                for b, c in enumerate(flows["traj_count"])
            ])  # fmt: skip
            lanes = np.arange(n_flow)
            s = flows["traj_s"][lanes, start]
            gaps = flows["traj_t"][lanes, start + 1] - flows["traj_t"][lanes, start]
            fs = FlowSettings(
                method=meta["method"],
                rtol=meta["rtol"],
                atol=meta["atol"],
                energy_tol=meta["energy_tol"],
                grad_tol=meta["grad_tol"],
                dt_max=args.dt_frac * tau1,
                max_steps=args.max_steps,
                record_every=1,
                max_records=args.max_steps + 2,
            )
            res = run_chunk(energy, s, np.minimum(gaps, fs.dt_max), basis, fs)
            tr = res.trajectory
            assert tr is not None, "the tail records every step"
            count = np.asarray(tr.count)
            width = int(count.max())
            status = np.asarray(res.status)
            t_start = flows["traj_t"][lanes, start, None]
            l_start = flows["traj_length"][lanes, start, None]
            np.savez(
                root / f"tail_chi{tag}.npz",
                start_index=start,
                traj_s=np.asarray(tr.s)[:, :width],
                traj_t=np.asarray(tr.t)[:, :width] + t_start,
                traj_length=np.asarray(tr.length)[:, :width] + l_start,
                traj_energy=np.asarray(tr.energy)[:, :width],
                traj_count=count,
                status=status,
                dt_max=np.array(fs.dt_max),
                tau1=np.array(tau1),
                phi_start=np.array(args.phi_start),
            )
            names = {STATUS_NAMES.get(int(c), str(c)): int(np.sum(status == c))
                     for c in np.unique(status)}  # fmt: skip
            final = np.asarray(res.energy)
            print(
                f"N={n} chi={chi_str}: tau_1 {tau1:.4g}, dt_max {fs.dt_max:.4g}; "
                f"restart Phi {np.median(flows['traj_energy'][lanes, start]):.1e} "
                f"(median); tail steps {count.min() - 1}-{count.max() - 1}; "
                f"status {names}; final Phi max {final.max():.1e}; "
                f"{time.time() - t0:.0f} s",
                flush=True,
            )
            assert not np.any(status == STATUS_BUDGET), "raise --max-steps"
    print("Done.", flush=True)
