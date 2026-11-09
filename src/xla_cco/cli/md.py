#!/usr/bin/env python3

"""
Low-temperature MD driver: python -m xla_cco.cli.md.

Loads ground states (an npz with ``s_final`` (B, N, d) fractional, ``basis``
(d, d), and a K value), builds the stealthy energy and the tracked waves
(|k| < --k-track), and runs the protocol: harmonic start, Andersen
thermalization (--n-equil steps), then --cycles of [Andersen --n-andersen,
NVE --n-nve] at each --temperature, writing ``<save>.npz`` (snapshots,
final winding, sums, statuses) and checkpoints. ``--device gpu`` fails fast
without a GPU (cli.relax.select_device); exit code 1 on any error.

Details:

- The cell is the Cartesian basis of the input (density 1, |det| = N, as
  ``flows_chi<tag>.npz`` stores it). K and M come from the sibling
  ``<input>.json`` ("K", "M") unless ``--k-outer`` is given.
- Lanes are (ground state x temperature), ground state major. Each lane
  starts at ``thermal.harmonic_sample`` positions and Maxwell-Boltzmann
  velocities (zero total momentum) drawn from ``--seed``.
- Equilibration (Andersen, or BAOAB) runs in two halves; begin_segment
  starts the second, so the gate <Phi>/(M T) (E1: within GATE_TOL = 5 %)
  sees only the second half. If any lane fails, equilibration is extended
  once by --n-equil steps (again in halves) and re-checked; the run never
  aborts on the gate, which is recorded per lane (``equil_gate*``).
- ``--scheme andersen-nve``: equilibration, then per cycle Andersen
  ``--n-andersen`` steps, ``run.begin_segment`` (COM removal, e0, sums and
  winding reset) and NVE ``--n-nve`` steps. ``--scheme baoab``: BAOAB
  throughout; each cycle is begin_segment then ``--n-nve`` BAOAB steps
  (``--n-andersen`` is not used). A phase that does not begin a segment
  (first equilibration half, Andersen phases) starts with
  begin_segment(reset_sums=False, reset_winding=True) and no COM removal,
  so its winding uses its own COM rule from the first step (D-062).
- Snapshots: ``--n-snapshots`` record steps (at most 256), shared between
  the cycles' production segments and log-spaced in the steps since each
  segment began (first step and last step included). They are distinct
  integers by construction, so a segment gets exactly its share unless it
  is shorter than that (``segment_offsets``).
- ``--checkpoint PATH`` writes the state at the start and after every
  phase (atomic, see ``dynamics.checkpoint``) and passes
  ``PATH.inphase.npz`` to ``run.simulate``, which only WRITES checkpoints
  there. The CLI deletes that file before each simulate call and after each
  phase checkpoint; on resume a newer in-phase file is folded into PATH.
  A checkpoint is resumed only if its fingerprint (all run settings, the
  input's SHA-256, M and K) matches; anything else is an error.
- Exit codes: 0 on success and --help; 1 on a usage error or any
  ValueError / TypeError / RuntimeError / OSError (message on stderr).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.box import Box
from xla_cco.cli.relax import STATUS_NAMES, select_device
from xla_cco.dynamics import run as md_run
from xla_cco.dynamics import thermal
from xla_cco.dynamics.checkpoint import load_state, read_meta, save_state
from xla_cco.dynamics.state import MDResult, MDSettings, MDState, Waves
from xla_cco.energy import StealthyEnergy, make_stealthy_energy
from xla_cco.kvectors import chi_from_count, enumerate_kvectors
from xla_cco.minimize.result import STATUS_RUNNING

# ============================================================
# CONSTANTS
# ============================================================

MAX_SNAPSHOTS: int = 256
MAX_TOTAL_STEPS: int = 2**31 - 1  # int32-safe step counters
DENSITY_RTOL: float = 1e-9  # |det(basis)| = N (density 1)
RECORD_FIELDS: Tuple[str, ...] = (
    "wind_abs", "wind_net", "n_valid", "sum_kinetic", "sum_phi", "sum_phi2",
    "sum_s", "n_sum", "status", "max_drift", "step", "time",
)  # fmt: skip
SNAP_FIELDS: Tuple[str, ...] = (
    "step", "time", "wind_abs", "wind_net", "rho", "r", "phi", "kinetic", "count",
)  # fmt: skip
FINGERPRINT_ARGS: Tuple[str, ...] = (
    "indices", "temperature", "dt", "n_equil", "n_andersen", "n_nve", "cycles",
    "andersen_p", "andersen_mode", "scheme", "gamma", "wind_every",
    "n_snapshots", "chunk_steps", "seed", "k_outer", "k_track",
    "record_rho_every", "max_rho_records", "phase_floor", "drift_tol", "k_chunk",
)  # fmt: skip
META_FIELDS: Tuple[str, ...] = ("fingerprint", "next_phase", "partial", "records")
RECORDS_FIELDS: Tuple[str, ...] = ("cycles", "phase_seconds", "gate", "extended")
GATE_TOL: float = 0.05  # E1 gate: <Phi>/(M T) within 5 % after equilibration
MAX_PHASES: int = 10_000  # bound of the phase loop (2 * cycles + 4 phases)
HASH_BLOCK: int = 1 << 20
MAX_HASH_BLOCKS: int = 1 << 24  # inputs up to 16 TiB


@dataclasses.dataclass(frozen=True)
class Phase:
    """One protocol phase: ``scheme`` for ``n_steps``; production if cycle >= 0."""

    name: str
    scheme: str
    n_steps: int
    begin: bool
    cycle: int
    gate: bool = False


# ============================================================
# ARGUMENTS
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the MD driver."""
    p = argparse.ArgumentParser(
        prog="python -m xla_cco.cli.md",
        description=(
            "Low-temperature MD of stealthy ground states: harmonic start, "
            "Andersen equilibration, then cycles of [Andersen, NVE] (or BAOAB "
            "throughout) at each temperature, with on-device winding of the "
            "tracked waves |k| < --k-track. Writes SAVE.npz and SAVE.json."
        ),
    )
    _add_system_args(p)
    _add_protocol_args(p)
    _add_output_args(p)
    assert p.prog.endswith("cli.md"), "parser identity"
    assert p.get_default("scheme") == "andersen-nve", "default protocol"
    return p


def _add_system_args(p: argparse.ArgumentParser) -> None:
    """Input file, ground-state selection and wave cutoffs."""
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="npz with s_final (B, N, d) fractional and basis (d, d)",
    )
    p.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="ground states (rows of s_final) to run; default all",
    )
    p.add_argument(
        "--k-outer",
        type=float,
        default=None,
        help="constraint radius K; default: 'K' of the sibling .json",
    )
    p.add_argument(
        "--k-track",
        type=float,
        default=None,
        help="track all waves with |k| < k-track (default 2K)",
    )
    p.add_argument("--k-chunk", type=int, default=None, help="energy k-chunk")
    assert p.get_default("indices") is None, "default: all ground states"
    assert p.get_default("k_track") is None, "default: 2K"


def _add_protocol_args(p: argparse.ArgumentParser) -> None:
    """Temperatures, time step, scheme and phase lengths."""
    p.add_argument("--temperature", type=float, nargs="+", required=True)
    p.add_argument("--dt", type=float, required=True, help="time step")
    p.add_argument(
        "--scheme", choices=("andersen-nve", "baoab"), default="andersen-nve"
    )
    p.add_argument("--n-equil", type=int, default=200_000)
    p.add_argument("--n-andersen", type=int, default=100_000)
    p.add_argument(
        "--n-nve",
        type=int,
        default=1_000_000,
        help="steps per production segment (NVE, or BAOAB)",
    )
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--andersen-p", type=float, default=0.01)
    p.add_argument("--andersen-mode", choices=("count", "bernoulli"), default="count")
    p.add_argument(
        "--gamma", type=float, default=0.0, help="BAOAB friction (--scheme baoab only)"
    )
    p.add_argument("--seed", type=int, default=0)
    assert p.get_default("n_equil") > 0, "equilibration on by default"
    assert p.get_default("cycles") >= 1, "at least one production segment"


def _add_output_args(p: argparse.ArgumentParser) -> None:
    """Recording, execution and files."""
    # ----- recording -----
    p.add_argument("--wind-every", type=int, default=1)
    p.add_argument(
        "--n-snapshots",
        type=int,
        default=200,
        help=f"log-spaced snapshots (<= {MAX_SNAPSHOTS})",
    )
    p.add_argument("--record-rho-every", type=int, default=0)
    p.add_argument("--max-rho-records", type=int, default=0)
    p.add_argument("--phase-floor", type=float, default=1e-16)
    p.add_argument("--drift-tol", type=float, default=1e-3)
    # ----- execution and files -----
    p.add_argument("--chunk-steps", type=int, default=10_000)
    p.add_argument("--save", type=Path, required=True, help="output prefix")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint file (.npz); resumed when present",
    )
    p.add_argument("--checkpoint-every-s", type=float, default=1800.0)
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    assert p.get_default("n_snapshots") <= MAX_SNAPSHOTS, "default within cap"
    assert p.get_default("device") == "gpu", "GPU unless asked otherwise"


def _require(ok: bool, message: str) -> None:
    """Raise ValueError(message) unless ``ok``."""
    if not ok:
        raise ValueError(message)


def validate_args(args: argparse.Namespace) -> None:
    """Raise ValueError with a clear message on inconsistent arguments."""
    temps = list(args.temperature)
    _require(
        all(math.isfinite(t) and t > 0.0 for t in temps),
        f"--temperature must be finite and > 0, got {temps}",
    )
    _require(len(set(temps)) == len(temps), "--temperature values must differ")
    _require(math.isfinite(args.dt) and args.dt > 0.0, "--dt must be > 0")
    _require(
        min(args.n_equil, args.n_andersen) >= 0,
        "--n-equil and --n-andersen must be >= 0",
    )
    _require(args.n_nve >= 1 and args.cycles >= 1, "--n-nve and --cycles must be >= 1")
    _require(0.0 < args.andersen_p <= 1.0, "--andersen-p must be in (0, 1]")
    if args.scheme == "baoab":
        _require(
            math.isfinite(args.gamma) and args.gamma > 0.0,
            "--scheme baoab needs --gamma > 0",
        )
    else:
        _require(args.gamma == 0.0, "--gamma applies to --scheme baoab only")
    for name in ("wind_every", "chunk_steps", "n_snapshots"):
        _require(getattr(args, name) >= 1, f"--{name.replace('_', '-')} must be >= 1")
    _require(
        args.n_snapshots <= MAX_SNAPSHOTS, f"--n-snapshots must be <= {MAX_SNAPSHOTS}"
    )
    _require(
        min(args.record_rho_every, args.max_rho_records) >= 0
        and (args.record_rho_every > 0) == (args.max_rho_records > 0),
        "--record-rho-every and --max-rho-records: both 0 or both > 0",
    )
    for name in ("k_outer", "k_track"):
        val = getattr(args, name)
        _require(
            val is None or (math.isfinite(val) and val > 0.0),
            f"--{name.replace('_', '-')} must be finite and > 0",
        )
    _require(args.k_chunk is None or args.k_chunk >= 1, "--k-chunk must be >= 1")
    _require(
        args.phase_floor >= 0.0 and args.drift_tol > 0.0,
        "need --phase-floor >= 0 and --drift-tol > 0",
    )
    _require(
        args.seed >= 0 and args.checkpoint_every_s > 0.0,
        "need --seed >= 0 and --checkpoint-every-s > 0",
    )
    if args.indices is not None:
        _require(
            min(args.indices) >= 0 and len(set(args.indices)) == len(args.indices),
            f"--indices must be distinct and >= 0, got {args.indices}",
        )
    _require(args.cycles <= MAX_PHASES // 4, f"--cycles must be <= {MAX_PHASES // 4}")
    total = sum(ph.n_steps for ph in protocol_phases(args, extended=True))
    _require(total <= MAX_TOTAL_STEPS, f"{total} steps exceed {MAX_TOTAL_STEPS}")


# ============================================================
# SCHEDULE
# ============================================================


def _equil_phases(args: argparse.Namespace, extended: bool) -> List[Phase]:
    """Equilibration halves (a: transient, b: gated), twice when extended."""
    assert args.n_equil >= 0, "n_equil must be >= 0"
    if args.n_equil == 0:
        return []
    scheme = "baoab" if args.scheme == "baoab" else "andersen"
    half = args.n_equil // 2
    out: List[Phase] = []
    for tag in ("equil", "ext")[: 2 if extended else 1]:
        if half > 0:
            out.append(Phase(f"{tag}_a", scheme, half, False, -1))
        out.append(Phase(f"{tag}_b", scheme, args.n_equil - half, True, -1, True))
    assert sum(ph.n_steps for ph in out) == args.n_equil * (2 if extended else 1)
    return out


def protocol_phases(args: argparse.Namespace, extended: bool = False) -> List[Phase]:
    """The phases in run order (empty equilibration/Andersen phases dropped).

    Equilibration is split into a first half and a second half that starts
    with begin_segment (sums reset), so the gate sees only the second half;
    ``extended`` appends one more equilibration of the same length.
    """
    assert args.cycles >= 1 and args.n_nve >= 1, "need a production segment"
    baoab = args.scheme == "baoab"
    phases = _equil_phases(args, extended)
    for c in range(args.cycles):
        if not baoab and args.n_andersen > 0:
            phases.append(Phase(f"andersen{c}", "andersen", args.n_andersen, False, -1))
        prod = "baoab" if baoab else "verlet"
        phases.append(Phase(f"{prod}{c}", prod, args.n_nve, True, c))
    assert sum(ph.cycle >= 0 for ph in phases) == args.cycles, "one per cycle"
    return phases


def segment_offsets(length: int, n: int) -> np.ndarray:
    """min(n, length) distinct log-spaced steps in [1, length], ending at length.

    Greedy: each step is the geometric step that reaches ``length`` in the
    remaining number of points, rounded, but at least one more than the
    previous step and leaving room for the points still to come.
    """
    assert length >= 1 and n >= 0, "need length >= 1 and n >= 0"
    count = min(n, length)
    if count == length:
        return np.arange(1, length + 1, dtype=np.int64)
    out = np.empty(count, dtype=np.int64)
    prev = 0
    for i in range(count):
        if i == 0:
            x = 1 if count > 1 else length
        else:
            target = prev * (length / prev) ** (1.0 / (count - i))
            x = max(prev + 1, int(round(target)))
        x = min(x, length - (count - 1 - i))
        out[i], prev = x, x
    assert count == 0 or (out[-1] == length and np.all(np.diff(out) > 0))
    return out


def record_schedule(phases: List[Phase], n_snapshots: int) -> Tuple[int, ...]:
    """Absolute record steps, log-spaced within every production segment.

    The ``n_snapshots`` are split as evenly as possible between the
    production segments (earlier segments get the remainder); a segment of
    length L starting at step s0 gets s0 + segment_offsets(L, n_c), i.e.
    exactly min(n_c, L) distinct steps (fewer than requested only when the
    segment is shorter than its share). Segments are disjoint and in run
    order, so the concatenation is strictly increasing (asserted).
    """
    assert 1 <= n_snapshots <= MAX_SNAPSHOTS, "n_snapshots out of range"
    starts = np.cumsum([0] + [ph.n_steps for ph in phases])[:-1]
    prod = [(int(s0), ph.n_steps) for s0, ph in zip(starts, phases, strict=True)
            if ph.cycle >= 0]  # fmt: skip
    assert len(prod) >= 1, "need a production segment"
    parts: List[np.ndarray] = []
    expected = 0
    for c, (s0, length) in enumerate(prod):
        n_c = n_snapshots // len(prod) + (1 if c < n_snapshots % len(prod) else 0)
        parts.append(s0 + segment_offsets(length, n_c))
        expected += min(n_c, length)
    steps = np.concatenate(parts)  # segments are disjoint and in order
    assert np.all(np.diff(steps) > 0), "record steps are distinct and sorted"
    assert steps.size == expected <= n_snapshots, "as many steps as asked"
    return tuple(int(x) for x in steps)


def make_settings(
    args: argparse.Namespace, record_steps: Tuple[int, ...]
) -> Dict[str, MDSettings]:
    """MDSettings per scheme used by the protocol (they differ only in scheme)."""
    assert len(record_steps) <= MAX_SNAPSHOTS, "too many record steps"
    base = MDSettings(
        dt=float(args.dt), scheme="verlet", andersen_p=float(args.andersen_p),
        andersen_mode=args.andersen_mode, gamma=float(args.gamma),
        remove_com=True, chunk_steps=int(args.chunk_steps),
        wind_every=int(args.wind_every), record_steps=tuple(record_steps),
        record_rho_every=int(args.record_rho_every),
        max_rho_records=int(args.max_rho_records),
        phase_floor=float(args.phase_floor), drift_tol=float(args.drift_tol),
    )  # fmt: skip
    schemes = {ph.scheme for ph in protocol_phases(args)}
    assert schemes, "the protocol has phases"
    return {s: dataclasses.replace(base, scheme=s) for s in sorted(schemes)}


def apply_schedule(ctx: Dict[str, Any], extended: bool) -> None:
    """Set ctx phases, record steps and settings (extended equilibration or not).

    Extending shifts the production segments, so the record steps move by
    n_equil; their number (the snapshot capacity) does not change.
    """
    args = ctx["args"]
    phases = protocol_phases(args, extended)
    steps = record_schedule(phases, args.n_snapshots)
    if "record_steps" in ctx:
        assert len(steps) == len(ctx["record_steps"]), "snapshot capacity is fixed"
    settings = make_settings(args, steps)
    any_settings = next(iter(settings.values()))
    settings_json = {
        f.name: getattr(any_settings, f.name)
        for f in dataclasses.fields(any_settings)
        if f.name != "scheme"
    }
    ctx.update(phases=phases, record_steps=steps, settings=settings,
               settings_json=json.loads(json.dumps(settings_json)))  # fmt: skip
    assert len(phases) >= 1 and len(steps) >= 1, "empty protocol"


# ============================================================
# INPUT AND PHYSICS
# ============================================================


def load_input(
    path: Path, indices: Optional[List[int]]
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return (s (G, N, d) selected ground states, basis (d, d), indices).

    Raises
    ------
    ValueError
        On missing arrays, bad shapes, non-finite values, a cell that is not
        at density 1 (|det| = N) or out-of-range indices.
    TypeError
        If ``s_final`` or ``basis`` is not float64.
    """
    with np.load(Path(path), allow_pickle=False) as z:
        _require("s_final" in z.files and "basis" in z.files,
                 f"{path} needs arrays 's_final' and 'basis'")  # fmt: skip
        s_all, basis = np.asarray(z["s_final"]), np.asarray(z["basis"])
    for name, a in (("s_final", s_all), ("basis", basis)):
        if a.dtype != np.float64:
            raise TypeError(f"{name} must be float64, got {a.dtype}")
        _require(bool(np.all(np.isfinite(a))), f"{name} has non-finite values")
    _require(s_all.ndim == 3 and s_all.shape[0] >= 1 and s_all.shape[1] >= 2,
             f"s_final must be (B, N >= 2, d), got {s_all.shape}")  # fmt: skip
    b, n, d = s_all.shape
    _require(
        d in (1, 2, 3) and basis.shape == (d, d),
        f"basis must be ({d}, {d}) with d in 1..3, got {basis.shape}",
    )
    det = abs(float(np.linalg.det(basis)))
    _require(
        abs(det / n - 1.0) <= DENSITY_RTOL,
        f"cell volume |det(basis)| = {det!r} must equal N = {n} (density 1)",
    )
    chosen = list(range(b)) if indices is None else list(indices)
    _require(max(chosen) < b, f"--indices {chosen} out of range for B = {b}")
    return np.ascontiguousarray(s_all[chosen]), basis, chosen


def resolve_k(args: argparse.Namespace, basis: np.ndarray) -> Tuple[float, int]:
    """Return (K, M): --k-outer, or "K" (checked against "M") of <input>.json."""
    sibling = Path(args.input).with_suffix(".json")
    expect_m: Optional[int] = None
    if args.k_outer is not None:
        k_outer = float(args.k_outer)
    else:
        _require(sibling.is_file(), f"no --k-outer and no {sibling} with 'K'")
        meta = json.loads(sibling.read_text())
        _require(isinstance(meta, dict) and "K" in meta, f"{sibling} lacks 'K'")
        k_outer = float(meta["K"])
        expect_m = int(meta["M"]) if "M" in meta else None
    _require(math.isfinite(k_outer) and k_outer > 0.0, f"bad K = {k_outer!r}")
    m = int(enumerate_kvectors(basis, 0.0, k_outer).n.shape[0])
    _require(m >= 1, f"empty k-set for K = {k_outer!r}")
    _require(
        expect_m is None or m == expect_m,
        f"K = {k_outer!r} gives M = {m}, but {sibling} says M = {expect_m}",
    )
    return k_outer, m


def build_system(
    args: argparse.Namespace, basis: np.ndarray, n: int
) -> Tuple[StealthyEnergy, Waves, Dict[str, Any]]:
    """Energy (0 < |k| < K), tracked waves (|k| < k_track) and system info."""
    k_outer, m = resolve_k(args, basis)
    k_track = 2.0 * k_outer if args.k_track is None else float(args.k_track)
    _require(k_track >= k_outer, f"--k-track {k_track!r} must be >= K {k_outer!r}")
    kset = enumerate_kvectors(basis, 0.0, k_outer)
    box = Box(basis=jnp.asarray(basis))
    energy = make_stealthy_energy(box, kset, chunk_size=args.k_chunk)
    waves = md_run.make_waves(basis, k_outer, k_track)
    w, n_c = int(waves.n_vectors.shape[0]), int(waves.n_constrained)
    if n_c != m or w < m:
        raise RuntimeError(f"make_waves returned {n_c} constrained of {w}; M = {m}")
    d = int(basis.shape[0])
    info = {
        "N": n,
        "d": d,
        "M": m,
        "K": k_outer,
        "k_track": k_track,
        "chi": chi_from_count(m, n, d),
        "n_waves": w,
        "n_constrained": n_c,
    }
    return energy, waves, info  # fmt: skip


# ============================================================
# LANES
# ============================================================


def lane_table(n_states: int, temps: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """(row into the selected states, temperature) of every lane, state major."""
    assert n_states >= 1 and len(temps) >= 1, "need states and temperatures"
    rows = np.repeat(np.arange(n_states), len(temps))
    lane_t = np.tile(np.asarray(temps, dtype=np.float64), n_states)
    assert rows.shape == lane_t.shape, "one row per lane"
    return rows, lane_t


def _stack(states: List[MDState]) -> MDState:
    """Stack single-lane states along a new leading axis."""
    assert len(states) >= 1, "need at least one lane"
    out: MDState = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *states)
    assert np.shape(out.step) == (len(states),), "leading lane axis"
    return out


def init_lanes(
    energy: StealthyEnergy, waves: Waves, basis: np.ndarray, s_gs: np.ndarray,
    temps: List[float], seed: int, settings: MDSettings,
) -> MDState:  # fmt: skip
    """Batched lanes: harmonic positions, MB velocities, run.init_state each."""
    rows, lane_t = lane_table(s_gs.shape[0], temps)
    n, d = int(s_gs.shape[1]), int(s_gs.shape[2])
    seqs = np.random.SeedSequence(seed).spawn(rows.size)
    keys = jax.random.split(jax.random.PRNGKey(seed), 2 * rows.size)
    basis_j = jnp.asarray(basis)
    lanes: List[MDState] = []
    for lane in range(rows.size):
        t = float(lane_t[lane])
        r0 = thermal.harmonic_sample(np.random.default_rng(seqs[lane]), energy,
                                     s_gs[rows[lane]], basis, t)  # fmt: skip
        if np.shape(r0) != (n, d) or not np.all(np.isfinite(r0)):
            raise RuntimeError(f"lane {lane}: bad harmonic sample {np.shape(r0)}")
        v0 = thermal.maxwell_boltzmann(keys[lane], n, d, jnp.asarray(t), True)
        lanes.append(md_run.init_state(
            energy, waves, basis_j, jnp.asarray(r0, dtype=jnp.float64),
            jnp.asarray(v0, dtype=jnp.float64), jnp.asarray(t),
            keys[rows.size + lane], settings))  # fmt: skip
    assert len(lanes) == rows.size, "one state per lane"
    return _stack(lanes)


def lane_template(ctx: Dict[str, Any]) -> MDState:
    """A state with the batch's structure, shapes and dtypes (for load_state)."""
    n_lanes = len(ctx["indices"]) * len(ctx["args"].temperature)
    assert n_lanes >= 1, "need at least one lane"
    basis = ctx["basis"]
    r0 = jnp.asarray(ctx["s_gs"][0] @ basis)
    settings = ctx["settings"][ctx["phases"][0].scheme]
    one = md_run.init_state(ctx["energy"], ctx["waves"], jnp.asarray(basis), r0,
                            jnp.zeros_like(r0), jnp.asarray(1.0),
                            jax.random.PRNGKey(0), settings)  # fmt: skip
    out: MDState = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_lanes, *jnp.shape(x))), one
    )
    assert np.shape(out.step) == (n_lanes,), "leading lane axis"
    return out


# ============================================================
# PROTOCOL
# ============================================================


def _begin(state: MDState, settings: MDSettings) -> MDState:
    """begin_segment on every lane (COM removal, e0, sums and winding reset);
    stopped lanes (status != STATUS_RUNNING) are left unchanged (D-082)."""
    out: MDState = jax.vmap(lambda s: md_run.begin_segment(s, settings, True, True))(
        state
    )
    assert isinstance(out, MDState), "begin_segment returns an MDState"
    assert np.shape(out.step) == np.shape(state.step), "lanes preserved"
    return out


def _rewind(state: MDState, settings: MDSettings) -> MDState:
    """Winding restart at a phase without begin_segment (D-062).

    begin_segment(reset_sums=False, reset_winding=True) with the COM
    momentum left as it is: rho_prev is recomputed under the phase's COM
    rule (an Andersen phase after NVE no longer compares raw with corrected
    rho); sums, velocities and step are unchanged, e0 is reset (unused
    outside NVE). Stopped lanes are left unchanged (D-082).
    """
    keep = dataclasses.replace(settings, remove_com=False)

    def one(lane: MDState) -> MDState:
        return md_run.begin_segment(lane, keep, False, True)

    out: MDState = jax.vmap(one)(state)
    assert isinstance(out, MDState), "begin_segment returns an MDState"
    assert np.shape(out.step) == np.shape(state.step), "lanes preserved"
    return out


def _capture(state: MDState) -> Dict[str, Any]:
    """JSON-able copy of the per-lane end-of-segment observables."""
    assert isinstance(state, MDState), "need an MDState"
    out = {k: np.asarray(getattr(state, k)).tolist() for k in RECORD_FIELDS}
    assert len(out) == len(RECORD_FIELDS), "one entry per field"
    return out


def inphase_path(checkpoint: Path) -> Path:
    """Path handed to run.simulate for checkpoints inside a phase."""
    return Path(str(checkpoint) + ".inphase.npz")


def _max_step(state: MDState) -> int:
    """Largest lane step (the step of the running lanes)."""
    return int(np.max(np.asarray(state.step)))


def gate_values(state: MDState, m: int) -> np.ndarray:
    """<Phi> / (M T) per lane over the sums since the last begin_segment."""
    assert m >= 1, "need M >= 1"
    n_sum = np.asarray(state.n_sum)
    mean = np.asarray(state.sum_phi) / np.maximum(n_sum, 1)
    out = np.where(n_sum > 0, mean, np.nan) / (m * np.asarray(state.temperature))
    assert out.shape == n_sum.shape, "one value per lane"
    return np.asarray(out, dtype=np.float64)


def gate_pass(values: np.ndarray) -> np.ndarray:
    """|gate - 1| <= GATE_TOL per lane (False for NaN)."""
    assert 0.0 < GATE_TOL < 1.0, "gate tolerance is a fraction"
    out = np.abs(np.asarray(values, dtype=np.float64) - 1.0) <= GATE_TOL
    assert out.shape == np.shape(values), "one flag per lane"
    return out


def _clear_inphase(ctx: Dict[str, Any]) -> None:
    """Remove the in-phase file (consumed or stale): simulate only writes it."""
    path = ctx["inphase"]
    assert path is None or path.suffix == ".npz", "in-phase checkpoints are npz"
    if path is not None and path.is_file():
        path.unlink()
    assert path is None or not path.exists(), "in-phase file removed"


def _save_phase(
    ctx: Dict[str, Any], state: MDState, next_phase: int, partial: int,
    records: Dict[str, Any],
) -> None:  # fmt: skip
    """Write the checkpoint (if any), then drop the now-older in-phase file."""
    args = ctx["args"]
    assert 0 <= next_phase <= len(ctx["phases"]) and partial >= 0, "bad position"
    if args.checkpoint is None:
        return
    meta = {"fingerprint": ctx["fingerprint"], "next_phase": next_phase,
            "partial": partial, "records": records}  # fmt: skip
    save_state(args.checkpoint, state, meta)
    assert Path(args.checkpoint).is_file(), "checkpoint written"
    _clear_inphase(ctx)


def _check_gate(ctx: Dict[str, Any], state: MDState, records: Dict[str, Any]) -> None:
    """Record the equilibration gate; extend equilibration once if it fails."""
    assert any(ph.gate for ph in ctx["phases"]), "a gated phase just ran"
    values = gate_values(state, ctx["info"]["M"])
    records["gate"].append(values.tolist())
    ok = gate_pass(values)
    print(f"gate <Phi>/(MT): {np.round(values, 4).tolist()}", flush=True)
    if not ok.all() and not records["extended"]:
        records["extended"] = True
        apply_schedule(ctx, True)
        print(f"gate failed in {int((~ok).sum())} lanes: extending", flush=True)
    assert len(records["gate"]) <= 2, "at most one extension"


def _run_phase(ctx: Dict[str, Any], state: MDState, ph: Phase, done: int) -> MDResult:
    """begin_segment (unless resumed inside the phase), then run.simulate."""
    assert 0 <= done < ph.n_steps, "partial progress inside the phase"
    st = ctx["settings"][ph.scheme]
    if ph.begin and done == 0:
        state = _begin(state, st)
    elif done == 0:
        state = _rewind(state, st)
    _clear_inphase(ctx)  # never hand simulate an existing file
    result = md_run.simulate(
        ctx["energy"], ctx["waves"], ctx["basis_j"], st, state, ph.n_steps - done,
        checkpoint=ctx["inphase"], checkpoint_every_s=ctx["args"].checkpoint_every_s,
    )  # fmt: skip
    assert isinstance(result, MDResult), "simulate returns an MDResult"
    print(f"phase {ph.name} ({ph.n_steps - done} steps) done", flush=True)
    return result


def run_phases(
    ctx: Dict[str, Any], state: MDState, start: int, partial: int,
    records: Dict[str, Any],
) -> Tuple[MDState, Optional[MDResult]]:  # fmt: skip
    """Run phases[start:] (the first one already ``partial`` steps in).

    Updates ``records`` (cycle observables, gate values, phase timings),
    may extend the equilibration (ctx phases change) and writes the
    checkpoint after every phase.
    """
    assert 0 <= start <= len(ctx["phases"]), "phase index out of range"
    result: Optional[MDResult] = None
    i = start
    for _ in range(MAX_PHASES):
        if i >= len(ctx["phases"]):
            break
        ph, t0 = ctx["phases"][i], time.perf_counter()
        result = _run_phase(ctx, state, ph, partial if i == start else 0)
        state = result.state
        if ph.cycle >= 0:
            records["cycles"].append(_capture(state))
        if ph.gate:
            _check_gate(ctx, state, records)
        records["phase_seconds"].append([ph.name, time.perf_counter() - t0])
        i += 1
        _save_phase(ctx, state, i, 0, records)
    assert i == len(ctx["phases"]), "every phase ran"
    return state, result


def input_digest(path: Path) -> str:
    """SHA-256 of the input file's bytes (hex)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for _ in range(MAX_HASH_BLOCKS):
            block = fh.read(HASH_BLOCK)
            if not block:
                break
            h.update(block)
        else:
            raise ValueError(f"{path} is too large to hash")
    out = h.hexdigest()
    assert len(out) == 64, "sha256 hex digest"
    return out


def fingerprint(
    args: argparse.Namespace, info: Dict[str, Any], digest: str
) -> Dict[str, Any]:
    """Run settings a checkpoint must match (JSON-normalized)."""
    assert len(digest) == 64, "need the input's sha256"
    fp = {k: getattr(args, k) for k in FINGERPRINT_ARGS}
    fp.update(input_sha256=digest, M=info["M"], K=info["K"])
    out: Dict[str, Any] = json.loads(json.dumps(fp))
    assert out.keys() == fp.keys(), "fingerprint keys survive JSON"
    return out


def _check_meta(path: Path, meta: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Raise ValueError unless the checkpoint metadata belongs to this run."""
    if meta.get("fingerprint") != ctx["fingerprint"]:
        raise ValueError(f"{path} was written by a run with other settings")
    missing = [k for k in META_FIELDS if k not in meta]
    _require(not missing, f"{path}: checkpoint metadata lacks {missing}")
    records = meta["records"]
    lacking = [k for k in RECORDS_FIELDS if k not in records]
    _require(isinstance(records, dict) and not lacking,
             f"{path}: checkpoint records lack {lacking}")  # fmt: skip


def resume(ctx: Dict[str, Any]) -> Tuple[MDState, int, int, Dict[str, Any]]:
    """Load (state, next phase, steps done in it, records) from the checkpoint.

    A newer in-phase checkpoint (ahead by fewer steps than remain in the
    phase) is folded into the phase checkpoint and removed.
    """
    path = ctx["args"].checkpoint
    meta = read_meta(path)
    _check_meta(path, meta, ctx)
    records = meta["records"]
    apply_schedule(ctx, bool(records["extended"]))
    like = lane_template(ctx)
    state, _ = load_state(path, like)
    phases, start, partial = (
        ctx["phases"],
        int(meta["next_phase"]),
        int(meta["partial"]),
    )
    _require(0 <= start <= len(phases) and partial >= 0
             and (partial == 0 or partial < phases[start].n_steps),
             f"{path}: bad position phase {start} + {partial} steps")  # fmt: skip
    if start < len(phases) and ctx["inphase"].is_file():
        later, _ = load_state(ctx["inphase"], like)
        ahead = _max_step(later) - _max_step(state)
        if ahead > 0 and partial + ahead < phases[start].n_steps:
            state, partial = later, partial + ahead
            _save_phase(ctx, state, start, partial, records)
    _clear_inphase(ctx)
    assert np.shape(state.step) == np.shape(like.step), "resumed lanes"
    print(f"resuming {path} at phase {start} (+{partial} steps)", flush=True)
    return state, start, partial, records


# ============================================================
# OUTPUT
# ============================================================


def _gate_arrays(records: Dict[str, Any], n_lanes: int) -> Dict[str, np.ndarray]:
    """First and final gate values per lane, pass flags, extension flag."""
    nan = np.full(n_lanes, np.nan)
    assert n_lanes >= 1 and len(records["gate"]) <= 2, "at most two gate checks"
    gates = [np.asarray(g, dtype=np.float64) for g in records["gate"]]
    first, last = (gates[0], gates[-1]) if gates else (nan, nan)
    assert first.shape == last.shape == (n_lanes,), "one gate value per lane"
    return {
        "equil_gate_first": first, "equil_gate": last,
        "equil_gate_pass": gate_pass(last),
        "equil_extended": np.asarray(bool(records["extended"])),
    }  # fmt: skip


def output_arrays(
    state: MDState, result: Optional[MDResult], records: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Dict[str, np.ndarray]:  # fmt: skip
    """All arrays of ``<save>.npz`` (documented in the JSON 'npz_keys')."""
    assert len(records["cycles"]) == ctx["args"].cycles, "one record per cycle"
    snaps = state.snap if result is None else result.snapshots
    trace = state.trace if result is None else result.trace
    out = {f"snap_{k}": np.asarray(getattr(snaps, k)) for k in SNAP_FIELDS}
    for k in RECORD_FIELDS:
        out[k] = np.asarray(getattr(state, k))
        out[f"cycle_{k}"] = np.stack(
            [np.asarray(c[k], dtype=out[k].dtype) for c in records["cycles"]]
        )
    if trace is not None and ctx["args"].max_rho_records > 0:
        out.update(trace_rho=np.asarray(trace.rho), trace_time=np.asarray(trace.time),
                   trace_count=np.asarray(trace.count))  # fmt: skip
    rows, lane_t = lane_table(len(ctx["indices"]), list(ctx["args"].temperature))
    waves = ctx["waves"]
    out.update(_gate_arrays(records, rows.size))
    out.update(
        r_final=np.asarray(state.r), v_final=np.asarray(state.v),
        temperatures=np.asarray(ctx["args"].temperature, dtype=np.float64),
        indices=np.asarray(ctx["indices"], dtype=np.int64),
        lane_index=np.asarray(ctx["indices"], dtype=np.int64)[rows],
        lane_temperature=lane_t, basis=np.asarray(ctx["basis"]),
        waves_n_vectors=np.asarray(waves.n_vectors),
        waves_kvec=np.asarray(waves.kvec),
        waves_constrained=np.asarray(waves.constrained),
        record_steps=np.asarray(ctx["record_steps"], dtype=np.int64),
        settings=np.asarray(json.dumps(ctx["settings_json"])),
    )  # fmt: skip
    assert out["lane_index"].shape == out["status"].shape, "one entry per lane"
    return out


def write_outputs(
    prefix: Path, arrays: Dict[str, Any], summary: Dict[str, Any]
) -> None:
    """Write ``<prefix>.npz`` and ``<prefix>.json``."""
    assert arrays and summary, "nothing to write"
    assert set(summary["npz_keys"]) == set(arrays), "summary lists the npz keys"
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    with open(str(prefix) + ".npz", "wb") as fh:
        np.savez(fh, **arrays)
    Path(str(prefix) + ".json").write_text(json.dumps(summary, indent=2) + "\n")


def summarize(
    arrays: Dict[str, np.ndarray], records: Dict[str, Any], ctx: Dict[str, Any],
    timings: Dict[str, float],
) -> Dict[str, Any]:  # fmt: skip
    """JSON metadata: system, settings, protocol, statuses, gate, timings."""
    status = arrays["status"]
    counts = {STATUS_NAMES.get(int(c), f"STATUS_{int(c)}"): int((status == c).sum())
              for c in np.unique(status)}  # fmt: skip
    args = ctx["args"]
    assert arrays["equil_gate"].shape == status.shape, "one gate value per lane"
    assert sum(counts.values()) == status.size, "every lane counted"
    return {
        **ctx["info"], "input": str(args.input), "indices": ctx["indices"],
        "input_sha256": ctx["fingerprint"]["input_sha256"],
        "temperatures": list(args.temperature), "n_lanes": int(status.size),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "settings": ctx["settings_json"],
        "phases": [dataclasses.asdict(ph) for ph in ctx["phases"]],
        "n_record_steps": len(ctx["record_steps"]),
        "status_counts": counts,
        "n_running": int((status == STATUS_RUNNING).sum()),
        "gate_tol": GATE_TOL, "equil_gate": records["gate"],
        "equil_gate_pass": arrays["equil_gate_pass"].tolist(),
        "equil_extended": bool(records["extended"]),
        "phase_seconds": records["phase_seconds"], "wall_seconds": timings,
        "npz_keys": sorted(arrays), "device": ctx["platform"],
        "jax_version": jax.__version__,
    }  # fmt: skip


# ============================================================
# DRIVER
# ============================================================


def setup(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate, select the device, load the input, build energy and settings."""
    validate_args(args)
    platform = select_device(args.device)  # before touching the input
    s_gs, basis, indices = load_input(args.input, args.indices)
    energy, waves, info = build_system(args, basis, int(s_gs.shape[1]))
    ctx: Dict[str, Any] = {
        "args": args, "platform": platform, "s_gs": s_gs, "basis": basis,
        "basis_j": jnp.asarray(basis), "indices": indices, "energy": energy,
        "waves": waves, "info": info,
        "fingerprint": fingerprint(args, info, input_digest(args.input)),
        "inphase": None if args.checkpoint is None else inphase_path(args.checkpoint),
    }  # fmt: skip
    apply_schedule(ctx, False)
    assert ctx["phases"] and ctx["record_steps"], "empty protocol"
    assert platform in ("cpu", "gpu", "cuda", "rocm"), f"platform {platform}"
    return ctx


def start_state(ctx: Dict[str, Any]) -> Tuple[MDState, int, int, Dict[str, Any]]:
    """Fresh lanes (checkpointed at once), or the resumed run's position."""
    args = ctx["args"]
    assert ctx["phases"], "a protocol to start"
    if args.checkpoint is not None and Path(args.checkpoint).is_file():
        return resume(ctx)
    first = ctx["settings"][ctx["phases"][0].scheme]
    n_lanes = len(ctx["indices"]) * len(args.temperature)
    state = init_lanes(ctx["energy"], ctx["waves"], ctx["basis"], ctx["s_gs"],
                       list(args.temperature), args.seed, first)  # fmt: skip
    records = {"cycles": [], "phase_seconds": [], "gate": [], "extended": False}
    assert np.asarray(state.step).shape == (n_lanes,), "batched lanes"
    _save_phase(ctx, state, 0, 0, records)  # also drops a stale in-phase file
    return state, 0, 0, records


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the driver; write ``<save>.npz`` / ``<save>.json``; return the summary."""
    assert isinstance(args, argparse.Namespace), "parsed arguments"
    t0 = time.perf_counter()
    ctx = setup(args)
    resumed = args.checkpoint is not None and Path(args.checkpoint).is_file()
    t1 = time.perf_counter()
    state, start, partial, records = start_state(ctx)
    info = ctx["info"]
    print(f"MD: {np.asarray(state.step).size} lanes, N={info['N']} M={info['M']} "
          f"w={info['n_waves']} on {ctx['platform']}", flush=True)  # fmt: skip
    t2 = time.perf_counter()
    state, result = run_phases(ctx, state, start, partial, records)
    t3 = time.perf_counter()
    arrays = output_arrays(state, result, records, ctx)
    timings = {"setup": t1 - t0, "init": t2 - t1, "run": t3 - t2}
    summary = summarize(arrays, records, ctx, timings)
    summary["resumed"] = resumed
    timings["output"] = time.perf_counter() - t3
    write_outputs(args.save, arrays, summary)
    assert len(records["cycles"]) == args.cycles, "one record per cycle"
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """Parse, validate, select the device, run; returns the exit code.

    0 on success and for --help; 1 on a usage error (argparse) or a
    ValueError / TypeError / RuntimeError / OSError (message on stderr).
    """
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # --help (code 0) or a usage error (code 2)
        return 0 if exc.code in (0, None) else 1
    try:
        summary = run(args)
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    assert isinstance(summary, dict), "run returns the summary"
    brief = {k: summary[k] for k in ("N", "M", "n_lanes", "status_counts")}
    assert all(v is not None for v in brief.values()), "summary is complete"
    print(json.dumps(brief), flush=True)
    return 0


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    sys.exit(main())
