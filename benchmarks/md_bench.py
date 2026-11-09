#!/usr/bin/env python3

"""
V3 benchmark of the MD library: throughput, dt and wind_every selection,
softest constrained mode, and the production cost projection.

Modes (``--mode``):

- ``throughput``: one (system, chi, lanes, waves, k-chunk, scheme) group in
  this process; for every ``--wind-every`` it times one compile call and a
  steady run of ``--n-steps`` steps of ``run.simulate`` with the production
  recording buffers (200 snapshots, a dense rho trace of all tracked waves
  every ``--record-rho-every`` steps with ``--max-rho-records`` rows) and
  appends one JSON line per wind_every to ``--out``. The peak device memory
  is the process peak, so every group runs in a fresh process.
- ``grid``: runs the throughput grid, one subprocess per group (skipping
  groups already in ``--out``).
- ``calib``: for one chi, canonical lanes (harmonic start + Maxwell-
  Boltzmann velocities, Andersen p = 1 thermalization) of flows2d and hip2d,
  4 ground states x the four temperatures each; the dt ladder: NVE energy
  drift over 10 decile windows, std(E) and max |E - e0| (first 2 ground
  states, --n-drift steps), and per-wave unsigned winding against dt/2 from
  the same state over the same time --t-wind (all lanes, |k| < 2K), judged
  per |k|/K bin with a matched-pair standard error (``compare_pooled``); a
  full-length (--n-confirm) drift trial at the chosen dt; then wind_every
  from a dense every-step rho trace at that dt (exact subsampling of one
  trajectory). Raw winding goes to ``<out stem>_wind.npz``.
- ``omega``: smallest and largest nonzero eigenvalue of K = (2/V) J J^T per
  ground state (CPU), with a Cartesian-Hessian cross-check.
- ``project``: merges the result files into ``bench.json`` with the chosen
  settings and the projected GPU-hours of E1, E1b and E4.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import dataclasses
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco._device import require_gpu
from xla_cco.box import Box
from xla_cco.cli import md as md_cli
from xla_cco.dynamics import run as md_run
from xla_cco.dynamics.state import MDSettings, MDState
from xla_cco.dynamics.thermal import _cartesian_hessian, maxwell_boltzmann
from xla_cco.energy import make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors
from xla_cco.minimize.result import STATUS_RUNNING

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

# ============================================================
# CONSTANTS
# ============================================================

DATA_ROOT: Path = Path("/scratch/gpfs/TORQUATO/sd2402/xla-cco/experiments/trajectories")
SYSTEMS: Dict[str, str] = {
    "flows2d": "flows2d/N1024", "gau2d": "gau2d/N1000", "url2d": "url2d/N1024",
    "rsa2d": "rsa2d/N999", "hip2d": "hip2d/N1000",
}  # fmt: skip
CHI_TAGS: Dict[str, float] = {"1-10": 0.1, "2-5": 0.4}
TEMPS: Tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3)
LADDER: Tuple[float, ...] = (0.04, 0.02, 0.01, 0.005)
WIND_EVERY: Tuple[int, ...] = (1, 2, 5, 10)
CALIB_SYSTEMS: Tuple[str, ...] = ("flows2d", "hip2d")
# ----- pass criteria (plan V3) -----
DRIFT_REL: float = 1e-4  # |E_last - E_first| / <KE>, decile window means
STD_REL: float = 1e-2  # std(E) / <KE>
WIND_DT_REL: float = 1e-2  # per-bin mean unsigned winding, dt against dt/2
WIND_EVERY_REL: float = 1e-2  # per-wave W_abs against wind_every = 1
N_BINS: int = 20  # |k|/K bins over (0, 2) (as the E2 analysis)
MIN_BIN_WAVES: int = 8  # bins with fewer (wave, lane) entries are not judged
NOISE_SE: float = 0.005  # SE_b above which a bin cannot resolve 1 % at 2 sigma
N_LANE_GS: int = 4  # ground states per system in the winding (c) trials
N_DRIFT_GS: int = 2  # ground states per system in the drift trials
N_WINDOWS: int = 10  # drift windows (deciles)
# ----- production protocol (plan E1 / E1b / E4) -----
N_EQUIL: int = 200_000
N_ANDERSEN: int = 100_000
N_NVE: int = 1_000_000
CYCLES: int = 2
TRACE_SAMPLES: int = 500  # Amendment A4: >= 500 rho samples per NVE segment
N_SNAPSHOTS: int = 200
SPECTRA_T: float = 200.0
SPECTRA_EVERY: int = 5
BUDGET_PROD: float = 60.0
BUDGET_LANGEVIN: float = 12.0
OVERHEAD_FRAC: float = 0.05  # checkpoints, host transfers, output writing
# Throughput was timed on an A100-SXM4-80GB; the A100-PCIE-40GB sanity run of
# the heaviest case was 17 % slower (210 vs 253 steps/s), so projected times
# are multiplied by 1.25 for any full (non-MIG) Della A100.
HARDWARE_MARGIN: float = 1.25

# ============================================================
# SYSTEMS AND LANES
# ============================================================


def load_system(
    system: str, tag: str, waves: str, k_chunk: Optional[int]
) -> Dict[str, Any]:
    """Energy, tracked waves ("constrained" or "2K"), basis, ground states, K, M.

    Raises
    ------
    ValueError
        On an unknown system, chi tag or waves choice.
    """
    if (
        system not in SYSTEMS
        or tag not in CHI_TAGS
        or waves not in ("constrained", "2K")
    ):
        raise ValueError(f"unknown system/chi/waves {system} {tag} {waves}")
    root = DATA_ROOT / SYSTEMS[system]
    meta = json.loads((root / f"flows_chi{tag}.json").read_text())
    with np.load(root / f"flows_chi{tag}.npz") as z:
        s_final = np.asarray(z["s_final"], dtype=np.float64)
        basis = np.asarray(z["basis"], dtype=np.float64)
    k_outer, m = float(meta["K"]), int(meta["M"])
    kset = enumerate_kvectors(basis, 0.0, k_outer)
    assert int(kset.n.shape[0]) == m, "k-set does not reproduce M"
    energy = make_stealthy_energy(Box(basis=jnp.asarray(basis)), kset,
                                  chunk_size=k_chunk)  # fmt: skip
    k_track = k_outer if waves == "constrained" else 2.0 * k_outer
    wv = md_run.make_waves(basis, k_outer, k_track)
    assert wv.n_constrained == m, "constrained rows equal the energy's k-set"
    return {
        "system": system, "tag": tag, "chi": CHI_TAGS[tag], "energy": energy,
        "waves": wv, "basis": basis, "basis_j": jnp.asarray(basis),
        "s_final": s_final, "K": k_outer, "M": m, "N": int(s_final.shape[1]),
        "d": int(s_final.shape[2]), "W": int(wv.kvec.shape[0]),
    }  # fmt: skip


def quick_lanes(
    sysd: Dict[str, Any], n_gs: int, temps: List[float], settings: MDSettings,
    seed: int,
) -> MDState:  # fmt: skip
    """Batched lanes (ground state major) for timing only: ground state plus
    sqrt(T) Gaussian jitter, Maxwell-Boltzmann velocities (no eigh)."""
    assert 1 <= n_gs <= sysd["s_final"].shape[0] and temps, "lane table"
    rng = np.random.default_rng(seed)
    keys = jax.random.split(jax.random.PRNGKey(seed), 2 * n_gs * len(temps))
    lanes: List[MDState] = []
    n, d = sysd["N"], int(sysd["basis"].shape[0])
    for i in range(n_gs * len(temps)):
        t = float(temps[i % len(temps)])
        r0 = sysd["s_final"][i // len(temps)] @ sysd["basis"]
        r0 = r0 + 0.3 * math.sqrt(t) * rng.standard_normal((n, d))
        v0 = maxwell_boltzmann(keys[i], n, d, jnp.asarray(t), True)
        lanes.append(md_run.init_state(
            sysd["energy"], sysd["waves"], sysd["basis_j"], jnp.asarray(r0), v0,
            jnp.asarray(t), keys[n_gs * len(temps) + i], settings))  # fmt: skip
    out: MDState = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *lanes)
    assert np.shape(out.step) == (len(lanes),), "leading lane axis"
    return out


def restart_lanes(
    sysd: Dict[str, Any], state: MDState, settings: MDSettings
) -> MDState:
    """Fresh lanes (init_state, step 0) at the r, T and key of ``state``, with
    the COM velocity removed (so the NVE dof is d (N - 1))."""
    assert state.step.ndim == 1, "batched state"

    def one(r: jax.Array, v: jax.Array, t: jax.Array, k: jax.Array) -> MDState:
        v = v - jnp.mean(v, axis=0)
        return md_run.init_state(sysd["energy"], sysd["waves"], sysd["basis_j"],
                                 r, v, t, k, settings)  # fmt: skip

    out: MDState = jax.vmap(one)(state.r, state.v, state.temperature, state.key)
    assert out.step.shape == state.step.shape, "one restart per lane"
    return out


def canonical_lanes(
    sysd: Dict[str, Any], gs: List[int], n_therm: int, seed: int
) -> MDState:
    """Harmonic start + MB velocities (cli.md.init_lanes), then n_therm steps of
    count-mode Andersen with p = 1 (one particle redrawn per step) at dt 0.01."""
    settings = MDSettings(dt=0.01, scheme="andersen", andersen_p=1.0,
                          chunk_steps=10_000)  # fmt: skip
    state = md_cli.init_lanes(sysd["energy"], sysd["waves"], sysd["basis"],
                              sysd["s_final"][gs], list(TEMPS), seed,
                              settings)  # fmt: skip
    out = md_run.simulate(sysd["energy"], sysd["waves"], sysd["basis_j"],
                          settings, state, n_therm).state  # fmt: skip
    assert np.all(np.asarray(out.status) == STATUS_RUNNING), "Andersen never drifts"
    assert out.step.shape == (len(gs) * len(TEMPS),), "lanes"
    return out


def device_peak_bytes() -> Optional[int]:
    """Peak bytes in use on device 0 (None on CPU)."""
    dev = jax.devices()[0]
    stats = dev.memory_stats() if dev.platform != "cpu" else None
    assert dev is not None, "a device"
    assert stats is None or isinstance(stats, dict), "memory_stats is a dict"
    return None if not stats else int(stats.get("peak_bytes_in_use", -1))


# ============================================================
# THROUGHPUT
# ============================================================


def production_records(n_nve: int) -> Tuple[Tuple[int, ...], int, int]:
    """(record_steps, record_rho_every, max_rho_records) of an E1 job: the
    CLI snapshot schedule (extended equilibration) and the A4 trace."""
    assert n_nve >= TRACE_SAMPLES, "n_nve must allow the trace samples"
    ns = argparse.Namespace(n_equil=N_EQUIL, n_andersen=N_ANDERSEN, n_nve=n_nve,
                            cycles=CYCLES, scheme="andersen-nve")  # fmt: skip
    phases = md_cli.protocol_phases(ns, extended=True)
    steps = md_cli.record_schedule(phases, N_SNAPSHOTS)
    every = n_nve // TRACE_SAMPLES
    total = sum(ph.n_steps for ph in phases)
    cap = total // every
    assert cap >= CYCLES * TRACE_SAMPLES and len(steps) <= N_SNAPSHOTS, "caps"
    return steps, every, cap


def time_group(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Throughput rows of one group (one per wind_every), timed in this process."""
    sysd = load_system(args.system, args.chi, args.waves, args.k_chunk)
    rec, every, cap = production_records(args.n_nve)
    n_gs = 10 if args.lanes == 40 else 5
    temps = list(TEMPS) if args.lanes == 40 else [1e-6, 1e-4]
    assert n_gs * len(temps) == args.lanes, "lanes = ground states x T"
    rows: List[Dict[str, Any]] = []
    for we in args.wind_every:
        settings = MDSettings(
            dt=0.02, scheme=args.scheme, gamma=0.03 if args.scheme == "baoab"
            else 0.0, chunk_steps=args.chunk_steps, wind_every=int(we),
            record_steps=rec, record_rho_every=every, max_rho_records=cap,
            drift_tol=math.inf,
        )  # fmt: skip
        state = quick_lanes(sysd, n_gs, temps, settings, args.seed)
        jax.block_until_ready(state.r)  # type: ignore[no-untyped-call]
        t0 = time.perf_counter()
        state = md_run.simulate(sysd["energy"], sysd["waves"], sysd["basis_j"],
                                settings, state, 1).state  # fmt: skip
        jax.block_until_ready(state.r)  # type: ignore[no-untyped-call]
        t1 = time.perf_counter()
        out = md_run.simulate(sysd["energy"], sysd["waves"], sysd["basis_j"],
                              settings, state, args.n_steps).state  # fmt: skip
        jax.block_until_ready(out.r)  # type: ignore[no-untyped-call]
        t2 = time.perf_counter()
        assert np.all(np.isfinite(np.asarray(out.phi))), "finite energies"
        rows.append(_throughput_row(args, sysd, we, t1 - t0, t2 - t1, every, cap))
        print(json.dumps(rows[-1]), flush=True)
    assert len(rows) == len(args.wind_every), "one row per wind_every"
    return rows


def _throughput_row(
    args: argparse.Namespace, sysd: Dict[str, Any], we: int, t_compile: float,
    t_run: float, every: int, cap: int,
) -> Dict[str, Any]:  # fmt: skip
    """JSON row of one timed configuration."""
    assert t_run > 0.0 and args.n_steps >= 1, "timed steps"
    assert every >= 1 and cap >= 1, "production trace"
    trace_bytes = args.lanes * cap * sysd["W"] * 2 * 8
    return {
        "system": args.system, "chi": sysd["chi"], "tag": args.chi, "N": sysd["N"],
        "M": sysd["M"], "W": sysd["W"], "waves": args.waves, "lanes": args.lanes,
        "k_chunk": args.k_chunk, "wind_every": int(we), "scheme": args.scheme,
        "chunk_steps": args.chunk_steps, "n_steps": args.n_steps,
        "compile_s": t_compile, "run_s": t_run,
        "steps_per_s": args.n_steps / t_run,
        "lane_steps_per_s": args.n_steps * args.lanes / t_run,
        "peak_bytes": device_peak_bytes(), "trace_bytes": trace_bytes,
        "record_rho_every": every, "max_rho_records": cap,
        "device": str(jax.devices()[0].device_kind),
    }  # fmt: skip


def grid_groups() -> List[List[str]]:
    """Argument lists of the throughput grid (one subprocess each)."""
    out: List[List[str]] = []
    for system in ("flows2d", "hip2d"):
        for tag in CHI_TAGS:
            for lanes in (10, 40):
                for waves in ("constrained", "2K"):
                    for kc in ("none", "64", "256"):
                        we = (
                            ["1"]
                            if waves == "constrained"
                            else (["1", "2", "5", "10"] if lanes == 40 else ["1", "5"])
                        )
                        out.append(["--system", system, "--chi", tag, "--lanes",
                                    str(lanes), "--waves", waves, "--k-chunk", kc,
                                    "--wind-every", *we])  # fmt: skip
    for tag in CHI_TAGS:  # thermostat phases (Andersen) and E4 (BAOAB)
        out.append(["--system", "flows2d", "--chi", tag, "--lanes", "40", "--waves",
                    "2K", "--k-chunk", "none", "--wind-every", "1", "--scheme",
                    "andersen"])  # fmt: skip
        out.append(["--system", "flows2d", "--chi", tag, "--lanes", "10", "--waves",
                    "2K", "--k-chunk", "none", "--wind-every", "1", "--scheme",
                    "baoab"])  # fmt: skip
    assert len(out) == 2 * 2 * 2 * 2 * 3 + 4, "grid size"
    assert all(len(g) % 2 == 1 for g in out), "flag-value pairs, then wind_every"
    return out


def run_grid(args: argparse.Namespace) -> None:
    """Run every grid group not yet in ``args.out`` in a fresh subprocess."""
    done = set()
    if args.out.is_file():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["system"], r["tag"], r["lanes"], r["waves"],
                      str(r["k_chunk"]).lower(), r["scheme"]))  # fmt: skip
    groups = grid_groups()
    assert groups and args.out.suffix == ".jsonl", "grid writes JSON lines"
    assert len(done) <= len(groups), "done groups come from the grid"
    for g in groups:
        opts = dict(zip(g[::2], g[1::2], strict=False))
        key = (
            opts["--system"],
            opts["--chi"],
            int(opts["--lanes"]),
            opts["--waves"],
            opts["--k-chunk"],
            opts.get("--scheme", "verlet"),
        )
        if key in done:
            continue
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "throughput",
            "--out",
            str(args.out),
            "--device",
            args.device,
            "--n-steps",
            str(args.n_steps),
            "--chunk-steps",
            str(args.chunk_steps),
            *g,
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, check=False)
        print(f"group {key}: exit {proc.returncode} in "
              f"{time.perf_counter() - t0:.1f} s", flush=True)  # fmt: skip


# ============================================================
# DT LADDER
# ============================================================


def _window_sums(state: MDState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sum E, sum KE, n) per lane of the running sums."""
    s_ke = np.asarray(state.sum_kinetic)
    s_e = np.asarray(state.sum_phi) + s_ke
    n = np.asarray(state.n_sum).astype(np.float64)
    assert np.all(n >= 1), "every window has samples"
    assert s_e.shape == s_ke.shape == n.shape, "per lane"
    return s_e, s_ke, n


def _zero_sums(state: MDState) -> MDState:
    """Running sums zeroed; e0 and max_drift kept (unlike begin_segment)."""
    z = jnp.zeros_like(state.sum_phi)
    out = dataclasses.replace(state, sum_kinetic=z, sum_phi=z, sum_phi2=z,
                              sum_s=jnp.zeros_like(state.sum_s),
                              n_sum=jnp.zeros_like(state.n_sum))  # fmt: skip
    assert out.e0.shape == state.e0.shape, "e0 kept"
    assert out.max_drift.shape == state.max_drift.shape, "max_drift kept"
    return out


def drift_trial(
    sysd: Dict[str, Any], lanes: MDState, dt: float, n_steps: int
) -> Dict[str, Any]:
    """NVE from ``lanes`` (restarted) for n_steps: decile window means of E,
    std(E) over 256 uniform snapshots, max |E - e0| / (dof T / 2)."""
    assert n_steps % N_WINDOWS == 0 and n_steps >= 256, "n_steps"
    rec = tuple(int(x) for x in np.linspace(n_steps / 256, n_steps, 256).round())
    st = MDSettings(dt=dt, scheme="verlet", remove_com=False, drift_tol=math.inf,
                    chunk_steps=min(n_steps // N_WINDOWS, 20_000),
                    record_steps=rec)  # fmt: skip
    state = restart_lanes(sysd, lanes, st)
    e_win = np.zeros((N_WINDOWS, state.step.shape[0]))
    ke_tot, n_tot = np.zeros(state.step.shape[0]), 0.0
    t0 = time.perf_counter()
    for w in range(N_WINDOWS):
        state = md_run.simulate(
            sysd["energy"],
            sysd["waves"],
            sysd["basis_j"],
            st,
            _zero_sums(state),
            n_steps // N_WINDOWS,
        ).state
        s_e, s_ke, n = _window_sums(state)
        e_win[w], ke_tot, n_tot = s_e / n, ke_tot + s_ke, n_tot + n[0]
    wall = time.perf_counter() - t0
    ke = ke_tot / n_tot
    e_snap = np.asarray(state.snap.phi) + np.asarray(state.snap.kinetic)
    dof_half = 0.5 * sysd["d"] * (sysd["N"] - 1) * np.asarray(state.temperature)
    assert np.all(np.asarray(state.snap.count) == 256), "all snapshots taken"
    slope = np.polyfit(np.arange(N_WINDOWS), e_win, 1)[0] / ke
    return {
        "dt": dt, "n_steps": n_steps, "wall_s": wall,
        "T": np.asarray(state.temperature).tolist(),
        "drift_rel": (np.abs(e_win[-1] - e_win[0]) / ke).tolist(),
        "slope_rel_per_window": slope.tolist(),
        "window_rel": ((e_win - e_win[0]) / ke).T.tolist(),
        "std_rel": (np.std(e_snap, axis=1) / ke).tolist(),
        "max_dev_dof": (np.asarray(state.max_drift) / dof_half).tolist(),
        "ke_over_dof": (ke / dof_half).tolist(),
    }  # fmt: skip


def winding_trial(
    sysd: Dict[str, Any], lanes: MDState, dt: float, t_total: float
) -> Tuple[np.ndarray, float]:
    """(wind_abs (lanes, W), wall s) of NVE from ``lanes`` over time t_total."""
    n_steps = int(round(t_total / dt))
    assert n_steps >= 1 and abs(n_steps * dt - t_total) < 1e-9 * t_total, "steps"
    st = MDSettings(dt=dt, scheme="verlet", remove_com=False, drift_tol=math.inf,
                    chunk_steps=min(n_steps, 20_000))  # fmt: skip
    state = restart_lanes(sysd, lanes, st)
    t0 = time.perf_counter()
    out = md_run.simulate(sysd["energy"], sysd["waves"], sysd["basis_j"], st, state,
                          n_steps).state  # fmt: skip
    wind = np.asarray(out.wind_abs)
    assert wind.shape == (lanes.step.shape[0], sysd["W"]), "wind shape"
    assert np.all(np.isfinite(wind)), "finite winding"
    return wind, time.perf_counter() - t0


def bin_index(kmag_over_k: np.ndarray) -> np.ndarray:
    """|k|/K bin (0 .. N_BINS - 1) of every wave, N_BINS equal bins over (0, 2)."""
    edges = np.linspace(0.0, 2.0, N_BINS + 1)
    out = np.clip(np.digitize(kmag_over_k, edges) - 1, 0, N_BINS - 1)
    assert out.shape == kmag_over_k.shape, "one bin per wave"
    assert np.all(kmag_over_k > 0.0), "no k = 0 wave"
    return np.asarray(out)


def _bin_stats(
    a: List[np.ndarray], b: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """(Delta_b, SE_b) per bin from matched winding entries a (dt), b (dt/2).

    Delta_b = mean(a) / mean(b) - 1 (ratio of means: the mean of per-wave
    ratios is biased upward by ~ CV(b)^2 for noisy, decorrelated waves, a
    Jensen effect that does not shrink with dt); SE_b by the delta method,
    std(a - R b) / (sqrt(n) mean(b)). NaN for bins with < MIN_BIN_WAVES.
    """
    delta, se = np.full(N_BINS, np.nan), np.full(N_BINS, np.nan)
    assert len(a) == len(b) == N_BINS, "one entry list per bin"
    for k in range(N_BINS):
        x, y = a[k], b[k]
        if x.size >= MIN_BIN_WAVES and np.mean(y) > 0.0:
            ratio = float(np.mean(x) / np.mean(y))
            delta[k] = ratio - 1.0
            resid = np.std(x - ratio * y, ddof=1)
            se[k] = float(resid / (math.sqrt(x.size) * np.mean(y)))
    assert np.any(np.isfinite(delta)), "at least one judged bin"
    return delta, se


def compare_pooled(
    pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Criterion (c) per temperature, pooled over systems and lanes.

    ``pairs`` holds (W at dt, W at dt/2, |k|/K, lane T) per system. For every
    |k|/K bin, Delta_b is the relative change of the bin-mean unsigned
    winding (ratio of means over its matched (wave, lane) entries) and SE_b
    its standard error (``_bin_stats``). A bin passes if |Delta_b| <
    WIND_DT_REL when it is resolved (SE_b <= NOISE_SE), or, when chaotic
    decorrelation leaves it noise-dominated (SE_b > NOISE_SE), if Delta_b is
    not significantly above the threshold (|Delta_b| - 2 SE_b < WIND_DT_REL).
    The all-wave ratio of summed winding must also be within WIND_DT_REL.
    """
    out: Dict[str, Any] = {}
    assert pairs and all(p[0].shape == p[1].shape for p in pairs), "matched runs"
    for t in TEMPS:
        ea: List[List[np.ndarray]] = [[] for _ in range(N_BINS)]
        eb: List[List[np.ndarray]] = [[] for _ in range(N_BINS)]
        tot_a = tot_b = 0.0
        for wa, wb, kk, temps in pairs:
            sel = np.isclose(temps, t)
            idx = bin_index(kk)
            for k in range(N_BINS):
                ea[k].append(wa[sel][:, idx == k].ravel())
                eb[k].append(wb[sel][:, idx == k].ravel())
            tot_a, tot_b = tot_a + wa[sel].sum(), tot_b + wb[sel].sum()
        delta, se = _bin_stats([np.concatenate(x) for x in ea],
                               [np.concatenate(x) for x in eb])  # fmt: skip
        resolved = se <= NOISE_SE
        excess = np.where(resolved, np.abs(delta), np.abs(delta) - 2.0 * se)
        all_rel = float(abs(tot_a / tot_b - 1.0))
        ok = bool(np.nanmax(excess) < WIND_DT_REL and all_rel < WIND_DT_REL)
        out[f"{t:g}"] = {
            "pass": ok, "delta": delta.tolist(), "se": se.tolist(),
            "max_abs_delta": float(np.nanmax(np.abs(delta))),
            "max_excess": float(np.nanmax(excess)), "all_wave_rel": all_rel,
            "n_noise_dominated": int(np.sum(~resolved & np.isfinite(se))),
        }  # fmt: skip
    assert len(out) == len(TEMPS), "one entry per T"
    return out


def reevaluate_wind(
    res: Dict[str, Any], raw: Dict[str, np.ndarray], names: List[str]
) -> Dict[str, Any]:
    """Criterion (c) of every ladder dt recomputed from the raw winding."""
    assert names and all(f"{n}_kk" in raw for n in names), "raw winding per system"
    for dt in LADDER:
        pairs = [(raw[f"{n}_{dt:g}"], raw[f"{n}_{dt / 2:g}"], raw[f"{n}_kk"],
                  raw[f"{n}_T"]) for n in names]  # fmt: skip
        res["wind"][f"{dt:g}"] = compare_pooled(pairs)
    assert len(res["wind"]) >= len(LADDER), "every dt judged"
    return res


# ============================================================
# CALIBRATION (dt, drift_tol, wind_every)
# ============================================================


def calib_systems(tag: str) -> List[Dict[str, Any]]:
    """flows2d and hip2d with constrained-only and |k| < 2K wave sets."""
    out = []
    for system in CALIB_SYSTEMS:
        c = load_system(system, tag, "constrained", None)
        c["wide"] = load_system(system, tag, "2K", None)
        out.append(c)
    assert len(out) == len(CALIB_SYSTEMS), "systems"
    assert all(c["wide"]["W"] > c["W"] == c["M"] for c in out), "wave sets"
    return out


def first_lanes(state: MDState, n: int) -> MDState:
    """The first ``n`` lanes of a batched state (ground-state-major order)."""
    assert 1 <= n <= state.step.shape[0], "lane count"
    out: MDState = jax.tree_util.tree_map(lambda x: x[:n], state)
    assert out.step.shape == (n,), "lanes"
    return out


def dt_ladder(
    systems: List[Dict[str, Any]], lanes: List[MDState], args: argparse.Namespace
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Drift trials (every ladder dt, first N_DRIFT_GS ground states) and
    winding runs (every dt and dt/2, all lanes) of every system; returns the
    results with criterion (c) pooled over the systems, and the raw winding."""
    res: Dict[str, Any] = {"drift": {}, "wind": {}}
    raw: Dict[str, np.ndarray] = {}
    dts = sorted(set(LADDER) | {min(LADDER) / 2}, reverse=True)
    n_drift = N_DRIFT_GS * len(TEMPS)
    for sysd, st in zip(systems, lanes, strict=True):
        name, wide = sysd["system"], sysd["wide"]
        raw[f"{name}_kk"] = np.linalg.norm(np.asarray(wide["waves"].kvec), axis=1)
        raw[f"{name}_kk"] = raw[f"{name}_kk"] / sysd["K"]
        raw[f"{name}_T"] = np.asarray(st.temperature)
        for dt in LADDER:
            tr = drift_trial(sysd, first_lanes(st, n_drift), dt, args.n_drift)
            res["drift"][f"{name}/{dt:g}"] = tr
            print(f"drift {name} dt={dt:g}: {max(tr['drift_rel']):.3e}", flush=True)
        for dt in dts:
            raw[f"{name}_{dt:g}"], wall = winding_trial(wide, st, dt, args.t_wind)
            print(f"wind {name} dt={dt:g}: {wall:.1f} s", flush=True)
    res = reevaluate_wind(res, raw, [s["system"] for s in systems])
    assert res["drift"] and res["wind"], "trials ran"
    assert len(raw) == len(systems) * (len(dts) + 2), "raw winding per run"
    return res, raw


def choose_dt(res: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    """Largest ladder dt passing (a) drift, (b) std and (c) winding in every
    system and temperature (None if none passes); with the per-dt verdicts."""
    verdict: Dict[str, Any] = {}
    chosen = None
    for dt in LADDER:
        keys = [k for k in res["drift"] if k.endswith(f"/{dt:g}")]
        a = max(max(res["drift"][k]["drift_rel"]) for k in keys)
        b = max(max(res["drift"][k]["std_rel"]) for k in keys)
        wind = res["wind"][f"{dt:g}"].values()
        c_ok = all(v["pass"] for v in wind)
        m = max(max(res["drift"][k]["max_dev_dof"]) for k in keys)
        ok = a < DRIFT_REL and b < STD_REL and c_ok
        verdict[f"{dt:g}"] = {
            "max_drift_rel": a, "max_std_rel": b, "wind_pass": c_ok,
            "max_wind_abs_delta": max(v["max_abs_delta"] for v in wind),
            "max_wind_excess": max(v["max_excess"] for v in wind),
            "max_wind_all_rel": max(v["all_wave_rel"] for v in wind),
            "max_dev_dof": m, "pass": bool(ok),
        }  # fmt: skip
        if ok and chosen is None:
            chosen = dt
    assert chosen is None or chosen in LADDER, "chosen from the ladder"
    assert len(verdict) == len(LADDER), "a verdict per dt"
    return chosen, verdict


def offline_winding(rho: np.ndarray, rho0: np.ndarray, every: int, floor: float,
                    n_part: int) -> np.ndarray:  # fmt: skip
    """Unsigned winding (lanes, w) from rho at steps 1..n (lanes, n, w, 2) and
    step 0 (lanes, w, 2), counting only steps that are multiples of ``every``
    (run.winding_update's rule: atan2 of rho_new conj rho_old, phase floor)."""
    assert rho.ndim == 4 and rho0.shape == (rho.shape[0], *rho.shape[2:]), "shapes"
    assert every >= 1 and rho.shape[1] >= every, "cadence within the trace"
    seq = np.concatenate([rho0[:, None], rho[:, every - 1 :: every]], axis=1)
    o, nw = seq[:, :-1], seq[:, 1:]
    delta = np.arctan2(nw[..., 1] * o[..., 0] - nw[..., 0] * o[..., 1],
                       nw[..., 0] * o[..., 0] + nw[..., 1] * o[..., 1])  # fmt: skip
    s_o = (o**2).sum(-1) / n_part
    s_n = (nw**2).sum(-1) / n_part
    ok = (s_o > floor) & (s_n > floor)
    out: np.ndarray = np.where(ok, np.abs(delta), 0.0).sum(axis=1)
    return out


def wind_every_trial(
    sysd: Dict[str, Any], lanes: MDState, dt: float, n_steps: int
) -> Dict[str, Any]:
    """Unconstrained per-wave W_abs at wind_every in WIND_EVERY against 1, from
    one every-step trace of all waves (T = 1e-6 and 1e-3 lanes)."""
    wide = sysd["wide"]
    temps = np.asarray(lanes.temperature)
    sel = np.flatnonzero(np.isclose(temps, 1e-6) | np.isclose(temps, 1e-3))
    sub: MDState = jax.tree_util.tree_map(lambda x: x[sel], lanes)
    st = MDSettings(dt=dt, scheme="verlet", remove_com=False, drift_tol=math.inf,
                    chunk_steps=min(n_steps, 10_000), record_rho_every=1,
                    max_rho_records=n_steps)  # fmt: skip
    state = restart_lanes(wide, sub, st)
    rho0 = np.asarray(state.rho_prev)
    out = md_run.simulate(wide["energy"], wide["waves"], wide["basis_j"], st, state,
                          n_steps).state  # fmt: skip
    assert np.all(np.asarray(out.trace.count) == n_steps), "every step traced"
    nc = wide["waves"].n_constrained
    rho = np.asarray(out.trace.rho)[:, :, nc:]
    ref = offline_winding(rho, rho0[:, nc:], 1, st.phase_floor, wide["N"])
    dev = np.asarray(out.wind_abs)[:, nc:]
    check = float(np.max(np.abs(ref / dev - 1.0)))
    res: Dict[str, Any] = {
        "dt": dt,
        "n_steps": n_steps,
        "T": temps[sel].tolist(),
        "offline_vs_device": check,
    }
    for e in WIND_EVERY[1:]:
        w = offline_winding(rho, rho0[:, nc:], e, st.phase_floor, wide["N"])
        rel = np.abs(w / ref - 1.0)
        res[str(e)] = {
            "max_wave_rel": rel.max(axis=1).tolist(),
            "median_wave_rel": np.median(rel, axis=1).tolist(),
            "total_rel": np.abs(w.sum(1) / ref.sum(1) - 1).tolist(),
        }
    assert check < 1e-9, f"offline winding must reproduce the device: {check}"
    return res


def choose_wind_every(results: List[Dict[str, Any]]) -> int:
    """Largest wind_every whose per-wave W_abs is within WIND_EVERY_REL of 1
    for every wave, lane and system."""
    best = 1
    for e in WIND_EVERY[1:]:
        worst = max(max(r[str(e)]["max_wave_rel"]) for r in results)
        if worst >= WIND_EVERY_REL:
            break  # the cadences below must all pass
        best = e
    assert best in WIND_EVERY, "chosen from the candidates"
    assert results, "need trials"
    return best


def round_up_125(x: float) -> float:
    """Smallest value of the form {1, 2, 5} x 10^k that is >= x (x > 0)."""
    assert x > 0.0 and math.isfinite(x), "need a positive finite value"
    base = 10.0 ** math.floor(math.log10(x))
    if base > x:  # log10 rounded up to the next integer (x just below 10^k)
        base /= 10.0
    assert base <= x < 10.0 * base, "decade of x"
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * base >= x * (1.0 - 1e-12):
            return m * base
    raise AssertionError("unreachable: 10 * base >= x")


def confirm_dt(
    systems: List[Dict[str, Any]], lanes: List[MDState], verdict: Dict[str, Any],
    n_steps: int,
) -> Tuple[float, Dict[str, Any]]:  # fmt: skip
    """Full-length drift trials (n_steps) at the passing dts, largest first;
    returns the first dt passing (a) and (b) at full length, with its trials."""
    assert n_steps >= 256, "full-length trial"
    assert set(verdict) == {f"{dt:g}" for dt in LADDER}, "ladder verdicts"
    for dt in LADDER:
        if not verdict[f"{dt:g}"]["pass"]:
            continue
        n_drift = N_DRIFT_GS * len(TEMPS)
        trials = {s["system"]: drift_trial(s, first_lanes(st, n_drift), dt, n_steps)
                  for s, st in zip(systems, lanes, strict=True)}  # fmt: skip
        a = max(max(t["drift_rel"]) for t in trials.values())
        b = max(max(t["std_rel"]) for t in trials.values())
        print(f"confirm dt={dt:g} n={n_steps}: drift {a:.3e} std {b:.3e}", flush=True)
        if a < DRIFT_REL and b < STD_REL:
            return dt, trials
    raise RuntimeError("no dt passes the full-length drift trial")


def run_calib(args: argparse.Namespace) -> Dict[str, Any]:
    """dt ladder, full-length confirmation, drift_tol and wind_every for one chi."""
    out: Dict[str, Any] = {
        "tag": args.chi,
        "chi": CHI_TAGS[args.chi],
        "device": str(jax.devices()[0].device_kind),
        "n_gs_wind": N_LANE_GS,
        "n_gs_drift": N_DRIFT_GS,
    }
    systems = calib_systems(args.chi)
    t0 = time.perf_counter()
    gs = list(range(N_LANE_GS))
    lanes = [canonical_lanes(s, gs, args.n_therm, args.seed) for s in systems]
    assert len(lanes) == len(systems), "lanes per system"
    out["thermalize_s"] = time.perf_counter() - t0
    out["init_s_per_lane"] = [time_init_lane(s) for s in systems]
    if args.ladder_from is None:
        res, raw = dt_ladder(systems, lanes, args)
        wind_npz = args.out.with_name(args.out.stem + "_wind.npz")
        np.savez(wind_npz, **raw)  # type: ignore[arg-type]
    else:
        res, raw = load_ladder(args)
    out.update(ladder=res, n_drift=args.n_drift, t_wind=args.t_wind,
               ladder_from=None if args.ladder_from is None
               else str(args.ladder_from))  # fmt: skip
    dt, verdict = choose_dt(res)
    out["ladder_verdict"] = verdict
    _dump(args.out, out)
    if dt is None:
        raise RuntimeError(f"no dt in {LADDER} passes: {verdict}")
    dt, trials = confirm_dt(systems, lanes, verdict, args.n_confirm)
    max_dev = max(max(t["max_dev_dof"]) for t in trials.values())
    out.update(dt=dt, confirm=trials, n_confirm=args.n_confirm, max_dev_dof=max_dev,
               drift_tol=round_up_125(5.0 * max_dev))  # fmt: skip
    _dump(args.out, out)
    n_w = int(round(args.t_every / dt))
    sub = [first_lanes(st, N_DRIFT_GS * len(TEMPS)) for st in lanes]
    we = [wind_every_trial(s, st, dt, n_w) for s, st in zip(systems, sub, strict=True)]
    out.update(
        wind_every_trials=we,
        wind_every=choose_wind_every(we),
        t_every=args.t_every,
        wall_s=time.perf_counter() - t0,
    )
    _dump(args.out, out)
    assert out["dt"] in LADDER and out["wind_every"] in WIND_EVERY, "choices"
    return out


def load_ladder(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Ladder of an earlier calib run (drift trials kept, criterion (c)
    recomputed from its ``_wind.npz`` with the current ``compare_pooled``)."""
    path = Path(args.ladder_from)
    old = json.loads(path.read_text())
    assert old["tag"] == args.chi, "ladder of the same chi"
    with np.load(path.with_name(path.stem + "_wind.npz")) as z:
        raw = {k: np.asarray(z[k]) for k in z.files}
    args.n_drift, args.t_wind = old["n_drift"], old["t_wind"]
    res = reevaluate_wind(dict(old["ladder"]), raw, list(CALIB_SYSTEMS))
    assert set(res["drift"]) == {f"{n}/{dt:g}" for n in CALIB_SYSTEMS
                                 for dt in LADDER}, "every drift trial"  # fmt: skip
    return res, raw


def time_init_lane(sysd: Dict[str, Any]) -> float:
    """Seconds of one cli.md.init_lanes lane (Hessian, eigh, init_state)."""
    settings = MDSettings(dt=0.01)
    md_cli.init_lanes(
        sysd["energy"],
        sysd["waves"],
        sysd["basis"],
        sysd["s_final"][:1],
        [1e-6],
        0,
        settings,
    )  # warm-up / compile
    t0 = time.perf_counter()
    st = md_cli.init_lanes(sysd["energy"], sysd["waves"], sysd["basis"],
                           sysd["s_final"][1:2], [1e-6], 1, settings)  # fmt: skip
    jax.block_until_ready(st.r)  # type: ignore[no-untyped-call]
    dt = time.perf_counter() - t0
    assert dt > 0.0 and st.step.shape == (1,), "one lane"
    assert st.r.shape == (1, sysd["N"], 2), "positions of one lane"
    return dt


def _dump(path: Path, obj: Dict[str, Any]) -> None:
    """Write ``obj`` as indented JSON (atomic rename)."""
    assert path.suffix == ".json", "JSON output"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)
    assert path.is_file(), "written"


# ============================================================
# SOFTEST CONSTRAINED MODE
# ============================================================


def k_spectrum(sysd: Dict[str, Any], s: np.ndarray) -> np.ndarray:
    """Eigenvalues (ascending) of K = (2/V) J J^T, J = d(Re rho, Im rho)/dr."""
    kvec = np.asarray(sysd["waves"].kvec)[: sysd["M"]]
    r = s @ sysd["basis"]
    theta = r @ kvec.T  # (N, M)
    j_re = -np.sin(theta).T[:, :, None] * kvec[:, None, :]
    j_im = np.cos(theta).T[:, :, None] * kvec[:, None, :]
    jac = np.concatenate([j_re, j_im]).reshape(2 * sysd["M"], -1)
    vol = abs(float(np.linalg.det(sysd["basis"])))
    lam = np.linalg.eigvalsh((2.0 / vol) * (jac @ jac.T))
    assert lam.shape == (2 * sysd["M"],), "2M eigenvalues"
    assert np.all(np.isfinite(lam)), "finite spectrum"
    return np.asarray(lam)


def run_omega(n_gs: int) -> Dict[str, Any]:
    """omega_min / omega_max per (start, chi) over the first n_gs ground states,
    plus a Cartesian-Hessian cross-check on flows2d chi = 0.1."""
    out: Dict[str, Any] = {}
    for system in SYSTEMS:
        for tag in CHI_TAGS:
            sysd = load_system(system, tag, "constrained", None)
            w = np.asarray(sysd["energy"].weights)[: sysd["M"]]
            assert np.allclose(w, 1.0), "unit weights (K = (2/V) J J^T)"
            lam = np.stack([k_spectrum(sysd, sysd["s_final"][i])
                            for i in range(n_gs)])  # fmt: skip
            om_min, om_max = np.sqrt(lam[:, 0]), np.sqrt(lam[:, -1])
            out[f"{system}/{tag}"] = {
                "omega_min": om_min.tolist(), "omega_max": om_max.tolist(),
                "omega_min_min": float(om_min.min()),
                "omega_min_median": float(np.median(om_min)),
                "omega_max_max": float(om_max.max()), "M": sysd["M"],
                "N": sysd["N"], "K": sysd["K"],
            }  # fmt: skip
            print(system, tag, out[f"{system}/{tag}"]["omega_min_min"], flush=True)
    sysd = load_system("flows2d", "1-10", "constrained", None)
    s0 = sysd["s_final"][0]
    h = _cartesian_hessian(sysd["energy"], s0, sysd["basis"])
    top = np.linalg.eigvalsh(h)[-2 * sysd["M"] :]
    lam = k_spectrum(sysd, s0)
    out["hessian_check"] = {"max_rel_diff": float(np.max(np.abs(top / lam - 1.0))),
                            "case": "flows2d/1-10 ground state 0"}  # fmt: skip
    assert out["hessian_check"]["max_rel_diff"] < 1e-6, "K spectrum = Hessian's"
    return out


# ============================================================
# PROJECTION
# ============================================================


def pick_rate(
    rows: List[Dict[str, Any]], tag: str, lanes: int, waves: str, we: int,
    scheme: str = "verlet", k_chunk: Optional[str] = None,
) -> Tuple[float, str, Optional[int]]:  # fmt: skip
    """(steps/s, k_chunk, peak bytes): flows2d/hip2d mean at the best k-chunk
    (or ``k_chunk``), at the largest measured wind_every <= ``we``."""
    sel = [r for r in rows if r["tag"] == tag and r["lanes"] == lanes
           and r["waves"] == waves and r["scheme"] == scheme]  # fmt: skip
    wes = sorted({r["wind_every"] for r in sel if r["wind_every"] <= we})
    assert wes, f"no throughput row for {tag} {lanes} {waves} {scheme}"
    sel = [r for r in sel if r["wind_every"] == wes[-1]]
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in sel:
        by.setdefault(str(r["k_chunk"]).lower(), []).append(r)
    if k_chunk is not None:
        by = {k_chunk: by[k_chunk]}
    best = max(by, key=lambda k: float(np.mean([r["steps_per_s"] for r in by[k]])))
    rate = float(np.mean([r["steps_per_s"] for r in by[best]]))
    peaks = [r["peak_bytes"] for r in by[best] if r["peak_bytes"] is not None]
    assert rate > 0.0, "positive throughput"
    return rate, best, (max(peaks) if peaks else None)


def lane_rate(r10: float, r40: float, lanes: int) -> float:
    """steps/s at ``lanes`` from a time-per-step line through 10 and 40 lanes."""
    assert r10 > 0.0 and r40 > 0.0 and lanes >= 1, "rates and lanes"
    slope = (1.0 / r40 - 1.0 / r10) / 30.0
    t = max(1.0 / r10 + slope * (lanes - 10), 1.0 / r10 if lanes > 10 else 0.0)
    out = 1.0 / max(t, 1e-9)
    assert out > 0.0, "positive"
    return out


def e1_job_hours(
    rows: List[Dict[str, Any]], cal: Dict[str, Any], tag: str, n_nve: int,
    n_gs: int, n_t: int,
) -> Dict[str, Any]:  # fmt: skip
    """Hours of one E1 job (worst case: extended equilibration)."""
    we, lanes = int(cal["wind_every"]), n_gs * n_t
    r10, kc, _ = pick_rate(rows, tag, 10, "2K", we)
    r40, kc40, peak = pick_rate(rows, tag, 40, "2K", we)
    rate = lane_rate(r10, r40, lanes)
    ra40, _, _ = pick_rate(rows, tag, 40, "2K", 1, "andersen", "none")
    rv40, _, _ = pick_rate(rows, tag, 40, "2K", 1, "verlet", "none")
    r_and = rate * min(ra40 / rv40, 1.0)  # never assume Andersen is faster
    therm = 2 * N_EQUIL + CYCLES * N_ANDERSEN
    sec = (therm / r_and + CYCLES * n_nve / rate) * (1.0 + OVERHEAD_FRAC)
    sec *= HARDWARE_MARGIN
    sec += lanes * float(np.max(cal["init_s_per_lane"])) + 120.0  # init, compile
    _, every, cap = production_records(n_nve)
    w = int(rows_w(rows, tag))
    assert sec > 0.0 and lanes >= 1, "cost"
    assert peak is None or peak > 0, "measured peak"
    return {
        "hours": sec / 3600.0,
        "lanes": lanes,
        "steps_per_s": rate,
        "andersen_steps_per_s": r_and,
        "k_chunk": kc40,
        "k_chunk_10": kc,
        "peak_bytes_40": peak,
        "fits_a100_40gb": bool(peak is None or peak < 0.8 * 40 * 2**30),
        "record_rho_every": every,
        "max_rho_records": cap,
        "trace_bytes": lanes * cap * w * 16,
        "n_nve": n_nve,
        "steps_per_lane_worst": therm + CYCLES * n_nve,
    }


def rows_w(rows: List[Dict[str, Any]], tag: str) -> int:
    """Largest number of tracked waves |k| < 2K at this chi (flows2d, hip2d)."""
    ws = [r["W"] for r in rows if r["tag"] == tag and r["waves"] == "2K"]
    assert ws, "need a 2K row"
    assert all(w > 0 for w in ws), "positive wave counts"
    return int(max(ws))


def e1b_job_hours(
    rows: List[Dict[str, Any]], cal: Dict[str, Any], tag: str
) -> Dict[str, Any]:
    """Hours of one E1b spectra job (2 lanes, constrained, T = 1e-6, t = 200)."""
    n_nve = int(round(SPECTRA_T / float(cal["dt"])))
    r10, kc, _ = pick_rate(rows, tag, 10, "constrained", 1)
    steps = 2 * N_EQUIL + N_ANDERSEN + n_nve
    sec = steps / r10 * (1.0 + OVERHEAD_FRAC) * HARDWARE_MARGIN + 2 * float(
        np.max(cal["init_s_per_lane"])) + 120.0  # fmt: skip
    m = max(r["M"] for r in rows if r["tag"] == tag)
    cap = steps // SPECTRA_EVERY
    assert sec > 0.0 and cap >= 1, "cost"
    assert n_nve >= 1 and r10 > 0.0, "spectra segment"
    return {"hours": sec / 3600.0, "lanes": 2, "n_nve": n_nve, "k_chunk": kc,
            "record_rho_every": SPECTRA_EVERY, "max_rho_records": cap,
            "trace_bytes": 2 * cap * m * 16}  # fmt: skip


def e4_job_hours(
    rows: List[Dict[str, Any]], cal: Dict[str, Any], tag: str, n_nve: int
) -> Dict[str, Any]:
    """Hours of one E4 Langevin job (10 lanes: 5 ground states x 2 T, one gamma)."""
    rb, _, _ = pick_rate(rows, tag, 10, "2K", 1, "baoab", "none")
    rv, _, _ = pick_rate(rows, tag, 10, "2K", 1, "verlet", "none")
    rw, kc, _ = pick_rate(rows, tag, 10, "2K", int(cal["wind_every"]))
    rate = rw * min(rb / rv, 1.0)  # BAOAB slowdown (if any) at the chosen cadence
    sec = (2 * N_EQUIL + CYCLES * n_nve) / rate * (1.0 + OVERHEAD_FRAC)
    sec *= HARDWARE_MARGIN
    sec += 10 * float(np.max(cal["init_s_per_lane"])) + 120.0
    assert sec > 0.0 and rate > 0.0, "cost"
    assert rb > 0.0 and rv > 0.0, "measured rates"
    return {"hours": sec / 3600.0, "lanes": 10, "steps_per_s": rate, "k_chunk": kc}


def production_plan(
    rows: List[Dict[str, Any]], cals: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """E1 + E1b under the reduction order; E4 at the final n_nve."""
    steps = [
        ("none", N_NVE, 10, 4),
        ("n_nve=5e5", 500_000, 10, 4),
        ("n_nve=5e5, 6 ground states", 500_000, 6, 4),
        ("n_nve=5e5, 6 ground states, drop T=1e-5", 500_000, 6, 3),
    ]
    n_starts = len(SYSTEMS)
    e1b = {t: e1b_job_hours(rows, cals[t], t) for t in CHI_TAGS}
    e1b_total = n_starts * sum(v["hours"] for v in e1b.values())
    tried = []
    for label, n_nve, n_gs, n_t in steps:
        e1 = {t: e1_job_hours(rows, cals[t], t, n_nve, n_gs, n_t) for t in CHI_TAGS}
        e1_total = n_starts * sum(v["hours"] for v in e1.values())
        tried.append(
            {
                "reduction": label,
                "E1_hours": e1_total,
                "E1b_hours": e1b_total,
                "total": e1_total + e1b_total,
            }
        )
        if e1_total + e1b_total <= BUDGET_PROD:
            break
    e4 = {t: e4_job_hours(rows, cals[t], t, n_nve) for t in CHI_TAGS}
    e4_total = 2 * 2 * sum(v["hours"] for v in e4.values())  # 2 starts x 2 gamma
    assert tried, "at least the unreduced plan"
    assert set(e1) == set(CHI_TAGS) == set(e4), "one job type per chi"
    return {
        "reduction_applied": tried[-1]["reduction"],
        "tried": tried,
        "E1_per_job": e1,
        "E1b_per_job": e1b,
        "E4_per_job": e4,
        "E1_hours": tried[-1]["E1_hours"],
        "E1b_hours": e1b_total,
        "production_hours": tried[-1]["total"],
        "E4_hours": e4_total,
        "E4_within_budget": bool(e4_total <= BUDGET_LANGEVIN),
        "production_within_budget": bool(tried[-1]["total"] <= BUDGET_PROD),
        "n_nve": n_nve,
        "ground_states": n_gs,
        "temperatures": n_t,
    }


def run_project(args: argparse.Namespace) -> Dict[str, Any]:
    """bench.json: measurements, choices per chi, projections."""
    rows = [json.loads(x) for x in args.throughput.read_text().splitlines() if x]
    cals = {t: json.loads((args.calib_dir / f"calib_{t}.json").read_text())
            for t in CHI_TAGS}  # fmt: skip
    omega = json.loads(args.omega.read_text())
    plan = production_plan(rows, cals)
    chosen = {}
    for t, cal in cals.items():
        job = plan["E1_per_job"][t]
        om = min(v["omega_min_min"] for k, v in omega.items() if k.endswith(t)
                 and "/" in k)  # fmt: skip
        chosen[t] = {"chi": CHI_TAGS[t], "dt": cal["dt"],
                     "drift_tol": cal["drift_tol"],
                     "max_dev_dof_at_dt": cal["max_dev_dof"],
                     "wind_every": cal["wind_every"], "k_chunk": job["k_chunk"],
                     "lanes_per_job": job["lanes"], "n_nve": job["n_nve"],
                     "record_rho_every": job["record_rho_every"],
                     "max_rho_records": job["max_rho_records"],
                     "omega_min_all_starts": om}  # fmt: skip
    out = {
        "chosen": chosen,
        "plan": plan,
        "throughput": rows,
        "calib": cals,
        "omega": omega,
        "gpu_hours_used_v3": args.gpu_hours_used,
        "criteria": {
            "drift_rel": DRIFT_REL,
            "std_rel": STD_REL,
            "wind_dt_rel": WIND_DT_REL,
            "wind_every_rel": WIND_EVERY_REL,
            "n_bins": N_BINS,
        },
    }
    _dump(args.out, out)
    assert out["chosen"], "choices written"
    assert set(chosen) == set(CHI_TAGS), "a choice per chi"
    return out


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])

    # ----- mode and output -----
    parser.add_argument(
        "--mode",
        required=True,
        choices=("throughput", "grid", "calib", "omega", "project"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")

    # ----- throughput -----
    parser.add_argument("--system", choices=tuple(SYSTEMS), default="flows2d")
    parser.add_argument("--chi", choices=tuple(CHI_TAGS), default="1-10")
    parser.add_argument("--lanes", type=int, choices=(10, 40), default=10)
    parser.add_argument("--waves", choices=("constrained", "2K"), default="2K")
    parser.add_argument("--k-chunk", default="none")
    parser.add_argument("--wind-every", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--scheme", choices=("verlet", "andersen", "baoab"), default="verlet"
    )
    parser.add_argument("--n-steps", type=int, default=3000)
    parser.add_argument("--chunk-steps", type=int, default=1000)
    parser.add_argument("--n-nve", type=int, default=N_NVE)
    parser.add_argument("--seed", type=int, default=0)

    # ----- calibration -----
    parser.add_argument("--n-therm", type=int, default=20_000)
    parser.add_argument("--n-drift", type=int, default=200_000)
    parser.add_argument("--n-confirm", type=int, default=1_000_000)
    parser.add_argument("--t-wind", type=float, default=400.0)
    parser.add_argument("--t-every", type=float, default=200.0)
    parser.add_argument("--ladder-from", type=Path, default=None)

    # ----- omega / projection -----
    parser.add_argument("--n-gs", type=int, default=10)
    parser.add_argument("--throughput", type=Path, default=None)
    parser.add_argument("--calib-dir", type=Path, default=None)
    parser.add_argument("--omega", type=Path, default=None)
    parser.add_argument("--gpu-hours-used", type=float, default=None)

    args = parser.parse_args()
    args.k_chunk = None if args.k_chunk == "none" else int(args.k_chunk)

    # =========================================================
    # DEVICE
    # =========================================================
    if args.device == "gpu" and args.mode in ("throughput", "calib"):
        require_gpu()
    print(f"md_bench {args.mode} on {jax.devices()[0].device_kind}", flush=True)

    # =========================================================
    # RUN
    # =========================================================
    if args.mode == "throughput":
        rows = time_group(args)
        with open(args.out, "a") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    elif args.mode == "grid":
        run_grid(args)
    elif args.mode == "calib":
        run_calib(args)
    elif args.mode == "omega":
        _dump(args.out, run_omega(args.n_gs))
    else:
        run_project(args)
    print("Done.", flush=True)
