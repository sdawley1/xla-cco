#!/usr/bin/env python3

"""
JAX replica of the C++ MD reference protocol for the parity test (plan V2).

Replays ``CollectiveCoordinateMD`` (main.cpp:725-890, recorded machine-readably
under ``protocol`` in ``tests/data/md_cxx_reference.json``) with
``xla_cco.dynamics`` in the same box (side ``L = 1000^(1/d)``, D-004), the same
K, N, d, T, dt and Andersen rule, many lanes vmapped:

1. a uniform random start per lane, relaxed with L-BFGS (``prerelax_evals``
   energy evaluations, one rescale pass); velocities 0;
2. ``nc_equi`` x 2 x [every velocity component ~ N(0, T), no COM removal;
   NVE velocity Verlet ``nve_steps_equi`` steps];
3. ``nc`` x [count-mode Andersen ``andersen_steps`` steps; NVE ``nve_steps``
   steps; record Phi and KE].

Andersen schedule: C++ resets its ``ResetCount`` to 0 in every call and fires
while it is > 0 after each step (501 resets at steps 1, 100, ..., 50000 per
call, D-050). The library count mode fires when ``reset_acc + p >= 1`` and
carries the accumulator across calls; starting every Andersen phase at
``reset_acc = ANDERSEN_ACC_START = 1.0`` (the C++ counter plus one) reproduces
the C++ steps exactly (checked in tests/parity/test_md_cxx_parity.py against
``md_reference.andersen_reset_steps``). D-051 (the C++ first half-kick with a
zero acceleration) is not replicated: it is one kick from a relaxed start
2.2e7 steps before the first sample.

Subcommands
-----------
``run --task I``
    Lanes ``I * 2 * lanes_per_t ...``: ``lanes_per_t`` at T and as many at
    ``CONTROL_FACTOR`` T (the power control), one vmapped batch; writes
    ``task<I>.json`` under ``--run-root``. Meant for
    ``slurm/md_jax_replica_cpu.slurm`` (one CPU per task).
``collect``
    Merges the task files into ``tests/data/md_jax_replica.json``.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import dataclasses
import datetime
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.box import Box
from xla_cco.dynamics import run
from xla_cco.dynamics.state import MDSettings, MDState, Waves
from xla_cco.dynamics.thermal import kinetic_energy, maxwell_boltzmann
from xla_cco.energy import StealthyEnergy, make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors
from xla_cco.minimize import relax
from xla_cco.minimize.lbfgs import LBFGSSettings
from xla_cco.minimize.result import STATUS_RUNNING

# ============================================================
# CONSTANTS
# ============================================================

ROOT: Path = Path(__file__).resolve().parents[2]
REF_JSON: Path = ROOT / "tests" / "data" / "md_cxx_reference.json"
OUT_JSON: Path = ROOT / "tests" / "data" / "md_jax_replica.json"
RUN_ROOT: Path = Path("/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/md_jax_replica")
ANDERSEN_ACC_START: float = 1.0  # C++ ResetCount = 0 at each call, plus one
CONTROL_FACTOR: float = 1.1  # power control temperature T_control / T
CHUNK_STEPS: int = 10_000
N_TASKS: int = 10
LANES_PER_T: int = 4  # per task and temperature
N_SAMPLES: int = 20  # per lane
BASE_SEED: int = 20261108
EQUI_REDRAWS: int = 2  # velocity redraws per equilibration cycle (main.cpp:803)

# ============================================================
# SYSTEM AND PROTOCOL
# ============================================================


@dataclasses.dataclass(frozen=True)
class Protocol:
    """Step counts of the replicated protocol (C++ values from the JSON)."""

    dt: float
    andersen_p: float
    equi_segments: int  # [redraw, NVE] segments = nc_equi * 2
    equi_nve_steps: int
    andersen_steps: int
    nve_steps: int
    n_samples: int
    prerelax_evals: int


class System(NamedTuple):
    """Energy, constrained waves and box of the C++ reference system."""

    energy: StealthyEnergy
    waves: Waves
    basis: jax.Array
    n: int
    d: int
    m: int
    temperature: float


def protocol_from_reference(ref: Dict[str, Any], n_samples: int) -> Protocol:
    """The C++ protocol of ``ref['protocol']`` with ``n_samples`` per lane."""
    p = ref["protocol"]
    assert p["andersen"]["mode"] == "count" and p["remove_com"] is False, p
    assert p["velocity_redraws_per_equi_cycle"] == EQUI_REDRAWS and n_samples >= 1
    return Protocol(
        dt=float(p["dt"]), andersen_p=float(p["andersen_p"]),
        equi_segments=int(p["nc_equi"]) * EQUI_REDRAWS,
        equi_nve_steps=int(p["nve_steps_equi"]),
        andersen_steps=int(p["andersen_steps"]), nve_steps=int(p["nve_steps"]),
        n_samples=int(n_samples), prerelax_evals=int(p["prerelax_evals"]),
    )  # fmt: skip


def build_system(ref: Dict[str, Any]) -> System:
    """StealthyEnergy, constrained-only waves and basis in the C++ box and K."""
    basis = np.asarray(ref["basis"], dtype=np.float64)
    k = float(ref["K"])
    energy = make_stealthy_energy(
        Box(basis=jnp.asarray(basis)), enumerate_kvectors(basis, 0.0, k)
    )
    waves = run.make_waves(basis, k, k)
    assert waves.n_constrained == int(ref["M"]), "M differs from the C++ k-set"
    vol = float(ref["volume"])
    assert abs(float(np.linalg.det(basis)) - vol) < 1e-9 * vol, "C++ box volume"
    return System(energy, waves, jnp.asarray(basis), int(ref["N"]), int(ref["d"]),
                  waves.n_constrained, float(ref["T"]))  # fmt: skip


def settings_pair(proto: Protocol) -> Tuple[MDSettings, MDSettings]:
    """(NVE, count-mode Andersen) settings; no COM removal, no recording."""
    nve = MDSettings(dt=proto.dt, scheme="verlet", remove_com=False,
                     chunk_steps=CHUNK_STEPS)  # fmt: skip
    ander = dataclasses.replace(nve, scheme="andersen", andersen_mode="count",
                                andersen_p=proto.andersen_p)  # fmt: skip
    assert nve.record_steps == ander.record_steps == (), "no snapshots"
    assert not (nve.remove_com or ander.remove_com), "C++ never removes the COM"
    return nve, ander


# ============================================================
# LANES
# ============================================================


def relaxed_starts(
    system: System, seeds: List[int], evals: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Cartesian (L, n, d) L-BFGS-relaxed uniform starts and their Phi (L,)."""
    assert len(seeds) >= 1 and evals >= 1, "need lanes and an evaluation budget"
    s0 = np.stack([np.random.default_rng(sd).uniform(size=(system.n, system.d))
                   for sd in seeds])  # fmt: skip
    res = relax(system.energy, jnp.asarray(s0),
                LBFGSSettings(rescales=(1.0,), max_evals_per_pass=evals))  # fmt: skip
    r = np.asarray(res.s) @ np.asarray(system.basis)
    phi = np.asarray(res.energy)
    assert r.shape == s0.shape and np.all(np.isfinite(r)), "bad relaxed starts"
    assert phi.shape == (len(seeds),) and np.all(phi >= 0.0), "bad start energies"
    return r, phi


def init_lanes(
    system: System, r0: np.ndarray, v0: np.ndarray, temps: np.ndarray,
    seeds: List[int], st: MDSettings,
) -> MDState:  # fmt: skip
    """Batched MDState at (r0, v0), one temperature and Andersen key per lane."""
    assert r0.shape == v0.shape and r0.shape[0] == temps.size == len(seeds)
    keys = jnp.stack([jax.random.PRNGKey(sd) for sd in seeds])

    def one(r: jax.Array, v: jax.Array, t: jax.Array, k: jax.Array) -> MDState:
        return run.init_state(system.energy, system.waves, system.basis, r, v,
                              t, k, st)  # fmt: skip

    lanes: MDState = jax.vmap(one)(jnp.asarray(r0), jnp.asarray(v0),
                                   jnp.asarray(temps, jnp.float64), keys)  # fmt: skip
    assert lanes.r.shape == r0.shape, "one lane per start"
    return lanes


def _begin(state: MDState, st: MDSettings) -> MDState:
    """begin_segment of every lane (e0, sums, pending rho_prev; v untouched)."""
    assert not st.remove_com, "the replica never removes the COM momentum"
    out: MDState = jax.vmap(lambda s: run.begin_segment(s, st, True, True))(state)
    assert out.v.shape == state.v.shape, "lanes keep their shape"
    return out


def redraw_velocities(state: MDState, vel_keys: jax.Array, i: int) -> MDState:
    """C++ SetRandomSpeed: every component of every lane's v ~ N(0, T_lane)."""
    n, d = state.r.shape[1:]
    keys = jax.vmap(lambda k: jax.random.fold_in(k, i))(vel_keys)
    v = jax.vmap(lambda k, t: maxwell_boltzmann(k, n, d, t, False))(
        keys, state.temperature
    )
    assert v.shape == state.v.shape and i >= 0, "one draw per lane"
    return dataclasses.replace(state, v=v)


def phase(system: System, st: MDSettings, state: MDState, n_steps: int) -> MDState:
    """One protocol phase of n_steps; Andersen phases restart the counter."""
    if st.scheme == "andersen":
        acc = jnp.full_like(state.reset_acc, ANDERSEN_ACC_START)
        state = dataclasses.replace(state, reset_acc=acc)
    out = run.simulate(system.energy, system.waves, system.basis, st,
                       _begin(state, st), n_steps).state  # fmt: skip
    assert np.all(np.asarray(out.status) == STATUS_RUNNING), "a lane aborted"
    assert np.all(np.asarray(out.n_sum) == n_steps), "every step ran"
    return out


def _energies(state: MDState) -> Tuple[np.ndarray, np.ndarray]:
    """(Phi, KE) of every lane, float64 (L,)."""
    phi = np.asarray(state.phi, dtype=np.float64)
    ke = np.asarray(jax.vmap(kinetic_energy)(state.v), dtype=np.float64)
    assert phi.shape == ke.shape == state.step.shape, "one value per lane"
    assert np.all(np.isfinite(phi)) and np.all(np.isfinite(ke)), "non-finite"
    return phi, ke


def run_protocol(
    system: System, proto: Protocol, state: MDState, vel_keys: jax.Array
) -> Dict[str, np.ndarray]:
    """Stage 1 (equilibration) and stage 2 (samples) of the C++ protocol.

    Returns per-lane arrays: ``phi``, ``ke`` (L, n_samples) after each NVE
    sample phase and ``equil_phi``, ``equil_ke`` (L, cycles) after each
    equilibration cycle (two redraws).
    """
    st_nve, st_and = settings_pair(proto)
    eq: List[Tuple[np.ndarray, np.ndarray]] = []
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    t0 = time.monotonic()
    for i in range(proto.equi_segments):
        state = redraw_velocities(state, vel_keys, i)
        state = phase(system, st_nve, state, proto.equi_nve_steps)
        if (i + 1) % EQUI_REDRAWS == 0:
            eq.append(_energies(state))
    for j in range(proto.n_samples):
        state = phase(system, st_and, state, proto.andersen_steps)
        state = phase(system, st_nve, state, proto.nve_steps)
        out.append(_energies(state))
        print(f"  sample {j + 1}/{proto.n_samples}: {time.monotonic() - t0:.0f} s, "
              f"<Phi>/(M T) = {np.mean(out[-1][0] / state.temperature) / system.m:.4f}",
              flush=True)  # fmt: skip
    res = {
        "phi": np.stack([o[0] for o in out], axis=1),
        "ke": np.stack([o[1] for o in out], axis=1),
        "equil_phi": np.stack([e[0] for e in eq], axis=1) if eq else np.zeros((0,)),
        "equil_ke": np.stack([e[1] for e in eq], axis=1) if eq else np.zeros((0,)),
    }
    assert res["phi"].shape == (state.step.size, proto.n_samples), "sample array"
    return res


# ============================================================
# TASKS
# ============================================================


def task_lanes(task: int, lanes_per_t: int) -> Tuple[List[int], np.ndarray]:
    """(global lane ids, temperature factors) of one task: lanes_per_t at 1,
    then lanes_per_t at CONTROL_FACTOR."""
    assert task >= 0 and lanes_per_t >= 1, "bad task layout"
    first = task * 2 * lanes_per_t
    ids = list(range(first, first + 2 * lanes_per_t))
    factors = np.repeat(np.array([1.0, CONTROL_FACTOR]), lanes_per_t)
    assert len(ids) == factors.size, "one factor per lane"
    return ids, factors


def _git_commit() -> str:
    """HEAD of the repository (or 'unknown')."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    assert out.stdout.strip(), "empty git output"
    return out.stdout.strip()


def run_task(task: int, lanes_per_t: int, n_samples: int, root: Path) -> Path:
    """Run one task's lanes through the full protocol; write task<I>.json."""
    ref = json.loads(REF_JSON.read_text())
    system, proto = build_system(ref), protocol_from_reference(ref, n_samples)
    ids, factors = task_lanes(task, lanes_per_t)
    seeds = [BASE_SEED + i for i in ids]
    t0 = time.monotonic()
    r0, phi0 = relaxed_starts(system, seeds, proto.prerelax_evals)
    temps = factors * system.temperature
    state = init_lanes(system, r0, np.zeros_like(r0), temps, seeds,
                       settings_pair(proto)[0])  # fmt: skip
    vel_keys = jnp.stack([jax.random.PRNGKey(sd + 7_919_000) for sd in seeds])
    res = run_protocol(system, proto, state, vel_keys)
    rec = {
        "task": task, "lane_ids": ids, "seeds": seeds, "t_factor": factors.tolist(),
        "phi_start": phi0.tolist(), **{k: v.tolist() for k, v in res.items()},
        "wall_seconds": time.monotonic() - t0, "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "jax_backend": jax.default_backend(), "git_commit": _git_commit(),
        "protocol": dataclasses.asdict(proto),
    }  # fmt: skip
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"task{task}.json"
    path.write_text(json.dumps(rec) + "\n")
    assert path.exists() and len(rec["phi"]) == len(ids), "task file written"
    print(f"wrote {path} ({rec['wall_seconds']:.0f} s)", flush=True)
    return path


def collect(root: Path, n_tasks: int, out: Path) -> None:
    """Merge task0..task<n-1>.json into ``out`` (lane arrays concatenated)."""
    assert n_tasks >= 1, "need tasks"
    recs = [json.loads((root / f"task{i}.json").read_text()) for i in range(n_tasks)]
    protos = {json.dumps(r["protocol"], sort_keys=True) for r in recs}
    assert len(protos) == 1, "tasks ran different protocols"
    keys = ("lane_ids", "seeds", "t_factor", "phi_start", "phi", "ke",
            "equil_phi", "equil_ke")  # fmt: skip
    walls = [r["wall_seconds"] for r in recs]
    merged: Dict[str, Any] = {
        "description": "JAX replica of the C++ MD reference protocol (plan V2)",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/cpp_reference/md_jax_replica.py collect",
        "reference": "tests/data/md_cxx_reference.json",
        "protocol": recs[0]["protocol"], "andersen_acc_start": ANDERSEN_ACC_START,
        "control_factor": CONTROL_FACTOR, "base_seed": BASE_SEED,
        **{k: [x for r in recs for x in r[k]] for k in keys},
        "cost": {"tasks": n_tasks, "cpu_hours": sum(walls) / 3600.0,
                 "max_task_seconds": max(walls),
                 "hosts": sorted({r["host"] for r in recs}),
                 "slurm_array_job_id": recs[0]["slurm_array_job_id"],
                 "jax_backend": recs[0]["jax_backend"],
                 "git_commit": recs[0]["git_commit"]},
    }  # fmt: skip
    assert len(merged["phi"]) == len(merged["lane_ids"]), "lane arrays align"
    out.write_text(json.dumps(merged, indent=1) + "\n")
    print(f"wrote {out}: {len(merged['phi'])} lanes, "
          f"{merged['cost']['cpu_hours']:.2f} CPU-h", flush=True)  # fmt: skip


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX replica of the C++ MD (V2)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run one task's lanes")
    p_run.add_argument("--task", type=int, required=True)
    p_run.add_argument("--lanes-per-t", type=int, default=LANES_PER_T)
    p_run.add_argument("--samples", type=int, default=N_SAMPLES)
    p_col = sub.add_parser("collect", help="merge the task files")
    p_col.add_argument("--tasks", type=int, default=N_TASKS)
    p_col.add_argument("--out", type=Path, default=OUT_JSON)
    for p in (p_run, p_col):
        p.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()

    # =========================================================
    # DISPATCH
    # =========================================================
    print("cmd: " + " ".join(sys.argv), flush=True)
    if args.cmd == "run":
        run_task(args.task, args.lanes_per_t, args.samples, args.run_root)
    else:
        collect(args.run_root, args.tasks, args.out)

    print("Done.", flush=True)
