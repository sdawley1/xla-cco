#!/usr/bin/env python3

"""
Record full collective-coordinate-optimization trajectories to stealthy
ground states with the library's adaptive gradient descent (paper rule).

Follows Dawley & Torquato, PRE 112, 045310 (2025), Sec. IV:

- normalized-gradient descent; the step doubles after an accepted downhill
  move and halves otherwise (acceptance as in the C++ LocalGradientDescent,
  rel_tol 0.01);
- initial step gamma rho^{1/2} = 1e-3 in Cartesian units (fractional step
  1e-3 / L in a square box of side L);
- a ground state is reached when Phi < 1e-22.

Every iteration is kept losslessly: per-iteration arrays (energy, step,
accept flag, time) plus the positions after every ACCEPTED iteration
(rejected iterations leave positions unchanged; ``frame_index`` maps each
iteration to its frame). Time follows the paper: an iteration contributes
t_m = gamma rho^{1/2} (Cartesian step) when its move is accepted, 0 otherwise.

Output per trajectory (git-ignored): ``<out>/<ENS>/traj_XXXX.{positions.f64,
npz,json}`` via ``xla_cco.io.trajstore``.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import functools
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----- numerics -----
import equinox as eqx
import jax
import numpy as np

# ----- project -----
from xla_cco.box import Box
from xla_cco.cli.relax import select_device
from xla_cco.energy import StealthyEnergy, make_stealthy_energy
from xla_cco.io.configpack import read_configpack, stack_fractional
from xla_cco.io.trajstore import TrajectoryWriter
from xla_cco.kvectors import (
    chi_from_count,
    enumerate_kvectors,
    k_outer_from_ka,
    ka_from_chi,
)
from xla_cco.minimize import _loop
from xla_cco.minimize.descent import (
    DescentSettings,
    DescentState,
    chunk_batch,
    init_batch,
)
from xla_cco.minimize.result import STATUS_ENERGY_TOL

# ============================================================
# CONSTANTS
# ============================================================

PHI_FOR_KA: float = 0.05  # cancels in K (only fixes the Ka <-> K convention)
STATUS_NAMES = {0: "RUNNING", 1: "ENERGY_TOL", 2: "STEP_FLOOR", 3: "BUDGET"}
# The descent stops at Phi < 1e-22 (paper) or at the step floor, where the
# float64 acceptance test can no longer resolve energy changes (Phi ~ 1e-22);
# a terminal Phi below this tolerance counts as a stealthy ground state.
GROUND_STATE_TOL: float = 1e-20

# ============================================================
# SETUP
# ============================================================


def build_problem(
    prefix: Path, chi: float
) -> Tuple[np.ndarray, Box, StealthyEnergy, Dict[str, Any]]:
    """Return (s0 (B,N,d), box, energy, physics metadata) for one ensemble."""
    s0, unit = stack_fractional(read_configpack(prefix))
    n, d = s0.shape[1], s0.shape[2]
    assert d == 2, "the paper's ensembles are two-dimensional"
    box = Box(basis=np.asarray(unit * n ** (1.0 / d), dtype=np.float64))
    k2 = k_outer_from_ka(ka_from_chi(chi, PHI_FOR_KA, d), PHI_FOR_KA, d)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k2)
    m = int(kset.n.shape[0])
    phys = {
        "N": n,
        "d": d,
        "L": float(n ** (1.0 / d)),
        "density": 1.0,
        "basis": np.asarray(box.basis).tolist(),
        "chi_target": chi,
        "chi_actual": chi_from_count(m, n, d),
        "M": m,
        "K": k2,
    }
    return s0, box, make_stealthy_energy(box, kset), phys


def paper_settings(length: float, chunk: int, max_iters: int) -> DescentSettings:
    """Return the paper's descent rule with per-iteration recording."""
    assert length > 0.0 and chunk > 0 and max_iters > 0, "positive args required"
    return DescentSettings(
        step_init=1e-3 / length,
        grow=2.0,
        shrink=0.5,
        rel_tol=0.01,
        max_evals=2 * max_iters,
        energy_tol=1e-22,
        record_every=1,
        max_records=chunk + 1,
    )


def reset_buffers(state: DescentState, settings: DescentSettings) -> DescentState:
    """Return ``state`` with a fresh per-lane recording buffer (row 0 = now)."""
    init = functools.partial(
        _loop.init_trajectory,
        every=settings.record_every,
        max_records=settings.max_records,
    )
    fresh = jax.vmap(init)(state.s, state.f, state.step)
    return eqx.tree_at(lambda st: st.traj, state, fresh)


# ============================================================
# STREAMING
# ============================================================


class LaneLog:
    """Per-trajectory accumulator for per-iteration arrays and frames."""

    def __init__(
        self, writer: TrajectoryWriter, s0: np.ndarray, step0: float, energy0: float
    ) -> None:
        """Start a lane at iteration 0 (frame 0 = initial condition)."""
        assert s0.ndim == 2 and step0 > 0.0, "bad initial lane state"
        self.writer = writer
        writer.append(np.mod(s0, 1.0)[None])
        self.step_prev = step0
        self.energy0 = energy0
        self.cols: Dict[str, List[Any]] = {
            "iteration": [],
            "energy": [],
            "step_used": [],
            "accepted": [],
            "frame_index": [],
        }

    def ingest(self, rows: Dict[str, np.ndarray]) -> None:
        """Consume new recorder rows (iterations m >= 1) for this lane."""
        assert rows["step"].ndim == 1, "rows must be per-iteration 1-D arrays"
        steps = np.concatenate([[self.step_prev], rows["step"]])
        accepted = steps[1:] > steps[:-1]  # x2 on accept, x0.5 on reject
        frame = (self.writer.n_frames - 1) + np.cumsum(accepted)  # state after m
        self.writer.append(np.ascontiguousarray(rows["s"][accepted]))
        self.cols["iteration"].append(rows["iteration"])
        self.cols["energy"].append(rows["energy"])
        self.cols["step_used"].append(steps[:-1])
        self.cols["accepted"].append(accepted)
        self.cols["frame_index"].append(frame.astype(np.int64))
        self.step_prev = float(steps[-1])

    def close(self, length: float, meta: Dict[str, Any]) -> None:
        """Assemble arrays (incl. Cartesian time) and finalize the files."""
        arrays = {
            k: np.concatenate(v) if v else np.empty(0) for k, v in self.cols.items()
        }
        step_cart = arrays["step_used"] * length
        arrays["step_used_cartesian"] = step_cart
        arrays["time"] = np.cumsum(np.where(arrays["accepted"], step_cart, 0.0))
        arrays["energy0"] = np.asarray(self.energy0)
        self.writer.close(arrays, meta)


def drain(state: DescentState, logs: List[LaneLog]) -> None:
    """Move every lane's new recorder rows (1..count-1) into its LaneLog."""
    traj = jax.device_get(state.traj)
    assert traj is not None and len(logs) == traj.count.shape[0], "lane mismatch"
    for b, log in enumerate(logs):
        c = int(traj.count[b])
        if c > 1:
            log.ingest(
                {
                    "s": traj.s[b, 1:c],
                    "energy": traj.energy[b, 1:c],
                    "step": traj.step[b, 1:c],
                    "iteration": traj.iteration[b, 1:c],
                }
            )


# ============================================================
# DRIVER
# ============================================================


def run_ensemble(
    s0: np.ndarray,
    energy: StealthyEnergy,
    settings: DescentSettings,
    logs: List[LaneLog],
    chunk: int,
    max_rounds: int,
) -> DescentState:
    """Advance all lanes chunk by chunk, streaming records after each chunk."""
    assert s0.shape[0] == len(logs) and chunk > 0, "inconsistent driver inputs"
    state = init_batch(energy, jax.numpy.asarray(s0), settings)
    for r in range(max_rounds):
        state = chunk_batch(energy, settings, state, chunk)
        drain(state, logs)
        state = reset_buffers(state, settings)
        status = np.asarray(state.status)
        if r % 10 == 0 or not np.any(status == 0):
            print(
                f"round {r}: running {int(np.sum(status == 0))}/{len(logs)}, "
                f"max it {int(np.max(np.asarray(state.it)))}",
                flush=True,
            )
        if not np.any(status == 0):
            break
    return state


def git_commit() -> str:
    """Return the current git commit (or 'unknown')."""
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # ----- required -----
    parser.add_argument("--ensemble", required=True, help="e.g. POS or GAU")
    parser.add_argument("--load", type=Path, required=True, help="ConfigPack prefix")

    # ----- optional -----
    parser.add_argument("--chi", type=float, default=0.1)
    parser.add_argument("--chunk", type=int, default=250)
    parser.add_argument("--max-iters", type=int, default=1_000_000)
    parser.add_argument("--limit", type=int, default=0, help="first k configs (0=all)")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--out", type=Path, default=Path("experiments/trajectories"))

    args = parser.parse_args()
    platform = select_device(args.device)

    # =========================================================
    # SETUP
    # =========================================================
    s0, box, energy, phys = build_problem(args.load, args.chi)
    if args.limit > 0:
        s0 = s0[: args.limit]
    settings = paper_settings(phys["L"], args.chunk, args.max_iters)
    outdir = args.out / f"chi{args.chi:g}" / args.ensemble
    outdir.mkdir(parents=True, exist_ok=True)
    print(
        f"{args.ensemble}: B={s0.shape[0]} N={phys['N']} M={phys['M']} "
        f"chi={phys['chi_actual']:.6f} K={phys['K']:.6f} on {platform}",
        flush=True,
    )

    # =========================================================
    # RUN
    # =========================================================
    t0 = time.time()
    e0 = np.asarray(jax.vmap(energy)(jax.numpy.asarray(s0)))
    logs = [
        LaneLog(
            TrajectoryWriter(outdir / f"traj_{i:04d}", phys["N"], phys["d"]),
            s0[i],
            settings.step_init,
            float(e0[i]),
        )
        for i in range(s0.shape[0])
    ]
    rounds = -(-args.max_iters // args.chunk)
    state = run_ensemble(s0, energy, settings, logs, args.chunk, rounds)
    wall = time.time() - t0

    # =========================================================
    # FINALIZE
    # =========================================================
    status = np.asarray(state.status)
    common = {
        "ensemble": args.ensemble,
        "source": str(args.load),
        **phys,
        "settings": {
            k: getattr(settings, k)
            for k in (
                "step_init",
                "step_min",
                "shrink",
                "grow",
                "rel_tol",
                "max_evals",
                "energy_tol",
            )
        },
        "time_definition": "sum over accepted iterations of the Cartesian step",
        "git_commit": git_commit(),
        "jax_version": jax.__version__,
        "device": platform,
        "ensemble_wall_seconds": wall,
    }
    its, fin = np.asarray(state.it), np.asarray(state.f)
    for i, log in enumerate(logs):
        name = STATUS_NAMES.get(int(status[i]), str(status[i]))
        meta = dict(
            common,
            config_index=i,
            status=name,
            n_iters=int(its[i]),
            final_energy=float(fin[i]),
        )
        log.close(phys["L"], meta)
    n_ok = int(np.sum(status == STATUS_ENERGY_TOL))
    n_gs = int(np.sum(fin < GROUND_STATE_TOL))
    manifest = dict(
        common,
        n_trajectories=len(logs),
        n_below_1e22=n_ok,
        n_ground_states=n_gs,
        ground_state_tol=GROUND_STATE_TOL,
    )
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        f"{args.ensemble}: {n_ok}/{len(logs)} reached Phi < 1e-22, "
        f"{n_gs}/{len(logs)} below {GROUND_STATE_TOL:g}, in {wall:.0f} s",
        flush=True,
    )
    print("Done.", flush=True)
