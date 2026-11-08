#!/usr/bin/env python3

"""
Synthetic MD outputs in the ``python -m xla_cco.cli.md`` schema, from a known model.

For tests and dry runs of batten2008_md_{analysis,stats,plot}.py; no MD is
integrated. Each (start, chi) gets ``<out>/<start>/N<N>/md_chi<tag>.npz``
and ``.json`` (the CLI's keys, lanes ground state x temperature, ground
state major; production segments from ``cli.md.protocol_phases`` with
snapshots at ``cli.md.record_schedule``), plus ``flows_chi<tag>.npz`` /
``.json`` (``s_final`` uniform random, ``basis``, K, M) as the K-spectrum
input.

Model (exact in law, sampled every step; dt is the record step):

- Constrained wave k (|k| < K): rho_k = sqrt(N T / 2) (x1 + i x2), with x1,
  x2 independent unit-variance underdamped Ornstein-Uhlenbeck oscillators of
  frequency w = f |k| and friction GAMMA_RATIO w, advanced with the exact
  discretization. So <S_k> = T, <|rho_dot|^2> = N w^2 T, and rho, rho_dot
  are independent at equal times: the Rice rate is exactly f |k| / 2 pi turns
  per unit time (f = 1 unless --inject). Net winding is a random walk for
  t >> 1 / (GAMMA_RATIO w).
- Unconstrained wave (K < |k| < 2K): rho_k = sqrt(N S0) exp(i (phi0 +
  Omega t)) + sqrt(N T / 2) (x1 + i x2) with the same oscillators at w = |k|,
  S0 ~ U(0.5, 1.5) and phi0 fixed per ground state, and a manifold drift
  Omega ~ N(0, (DRIFT_C |k|)^2 T): unsigned rate |k| sqrt(T / (pi S)) / 2 pi
  to first order, net rate proportional to sqrt(T).
- Particles: r(t) = r0 + u t with u ~ N(0, T) per component, centre of mass
  removed, so the particle MSD is d (N - 1) / N T t^2 (collapses on t sqrt T).

Winding is accumulated every step with the MD rule (atan2 increment, counted
only when S > phase_floor at both ends; batten2008_md_analysis.winding_step).
With --common-seeds every start shares the same random draws per ground
state (the starts then differ only by --inject); --langevin and --spectra
add BAOAB-labelled and dense-trace files for F9 and F10.

Usage: python -I batten2008_md_synthetic.py --out DIR [--inject hip2d=1.15]

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
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----- numerics -----
import numpy as np
from scipy.linalg import expm

# ----- project -----
from xla_cco.cli.md import protocol_phases, record_schedule
from xla_cco.kvectors import chi_from_count, enumerate_kvectors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_md_analysis import winding_step  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

STARTS: Tuple[str, ...] = ("flows2d", "gau2d", "url2d", "rsa2d", "hip2d")
CHIS: Tuple[str, ...] = ("1/10", "2/5")
TEMPS: Tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3)
GAMMA_RATIO = (
    0.2  # oscillator friction / frequency (underdamped; sampled-TV bias ~ gamma dt)
)
DRIFT_C = 0.05  # manifold drift Omega ~ DRIFT_C |k| sqrt(T) (<< jiggle rate)
NOISE_BLOCK = 64  # steps of random numbers drawn at once
PHASE_FLOOR = 1e-16  # MD default (S = |rho|^2 / N)
LANGEVIN_GAMMAS: Tuple[float, ...] = (0.003, 0.03)
LANGEVIN_STARTS: Tuple[str, ...] = ("flows2d", "hip2d")

# ============================================================
# SYSTEM
# ============================================================


def chi_tag(chi: Fraction) -> str:
    """File-name tag, e.g. '1-10'."""
    assert chi.denominator >= 1, "a fraction"
    assert chi > 0, "chi must be positive"
    return f"{chi.numerator}-{chi.denominator}"


def make_system(n: int, chi: Fraction) -> Dict[str, Any]:
    """Square cell at density 1, K between shells with M ~ chi d (N - 1), waves.

    Returns basis, K, M, kvec / n_vectors of the tracked waves (constrained
    first in enumerate_kvectors order, then K < |k| < 2K by |k|), kmag and
    the constrained mask.
    """
    side = float(np.sqrt(n))
    assert side == int(side) and n >= 4, "N must be a perfect square"
    basis = side * np.eye(2)
    target = int(round(float(chi) * 2 * (n - 1)))
    kmag = np.sqrt(enumerate_kvectors(basis, 0.0, 50.0 / side * np.pi).k_sq)
    shells = np.unique(np.round(kmag, 12))
    counts = np.searchsorted(kmag, shells + 1e-9)
    j = int(np.argmin(np.abs(counts - target)))
    k_outer = 0.5 * (shells[j] + shells[j + 1])
    con = enumerate_kvectors(basis, 0.0, k_outer)
    unc = enumerate_kvectors(basis, k_outer, 2.0 * k_outer)
    nvec = np.concatenate([con.n, unc.n]).astype(np.float64)
    kvec = np.concatenate([con.k, unc.k])
    m = con.n.shape[0]
    constrained = np.arange(kvec.shape[0]) < m
    assert m >= 1 and unc.n.shape[0] >= 1, "need both wave classes"
    return {
        "basis": basis, "K": float(k_outer), "M": m, "N": n, "d": 2,
        "kvec": kvec, "n_vectors": nvec, "kmag": np.linalg.norm(kvec, axis=1),
        "constrained": constrained, "chi": chi_from_count(m, n, 2),
    }  # fmt: skip


def oscillator_coeffs(omega: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Exact one-step map E (W, 2, 2) and noise Cholesky C (W, 2, 2) of (x, v).

    dx = v dt, dv = (-w^2 x - g v) dt + sqrt(2 g w^2) dB with g = GAMMA_RATIO
    w; stationary covariance diag(1, w^2).
    """
    assert np.all(omega > 0.0) and dt > 0.0, "need w > 0 and dt > 0"
    out_e = np.empty((omega.size, 2, 2))
    out_c = np.empty((omega.size, 2, 2))
    for i, w in enumerate(omega):
        a = np.array([[0.0, 1.0], [-(w**2), -GAMMA_RATIO * w]])
        e = expm(a * dt)
        sig = np.diag([1.0, w**2])
        cov = sig - e @ sig @ e.T
        out_e[i], out_c[i] = e, np.linalg.cholesky(0.5 * (cov + cov.T))
    assert np.all(np.isfinite(out_c)), "noise covariance not positive definite"
    return out_e, out_c


# ============================================================
# SIMULATION
# ============================================================


def _rho(st: Dict[str, Any], t: float) -> np.ndarray:
    """Complex rho (L, W) of every lane at time t."""
    out = st["amp"][:, None] * st["x"]
    u = ~st["constrained"]
    out[:, u] += st["rho0"][:, u] * np.exp(1j * st["omega_drift"][:, u] * t)
    assert out.shape == st["x"].shape, "one rho per lane and wave"
    assert np.all(np.isfinite(out)), "non-finite rho"
    return out


def _advance(st: Dict[str, Any], xi: np.ndarray) -> None:
    """One exact oscillator step of every (lane, wave); xi (2, L, W) complex."""
    e, c = st["e"], st["c"]
    assert xi.shape == (2, *st["x"].shape), "two complex noises per oscillator"
    x, v = st["x"], st["v"]
    xn = e[:, 0, 0] * x + e[:, 0, 1] * v + c[:, 0, 0] * xi[0]
    vn = e[:, 1, 0] * x + e[:, 1, 1] * v + c[:, 1, 0] * xi[0] + c[:, 1, 1] * xi[1]
    st["x"], st["v"] = xn, vn
    assert np.all(np.isfinite(xn)), "oscillator blew up"


def _record(st: Dict[str, Any], j: int, step: int, t_abs: float, rho: np.ndarray):
    """Write snapshot j of every lane."""
    snap = st["snap"]
    assert 0 <= j < snap["step"].shape[1], "snapshot index out of range"
    snap["step"][:, j], snap["time"][:, j] = step, t_abs
    snap["wind_abs"][:, j], snap["wind_net"][:, j] = st["wabs"], st["wnet"]
    snap["rho"][:, j] = np.stack([rho.real, rho.imag], axis=-1)
    snap["r"][:, j] = st["r0"] + st["u"] * t_abs
    snap["phi"][:, j] = (np.abs(rho[:, st["constrained"]]) ** 2).sum(1) / st["V"]
    snap["kinetic"][:, j] = st["ke"]
    snap["count"][:] = j + 1
    assert np.all(snap["count"] <= snap["step"].shape[1]), "count within capacity"


def _accumulate(st: Dict[str, Any], acc: Dict[str, np.ndarray], rho: np.ndarray):
    """Per-step sums (S of every wave, Phi, Phi^2) and the dense trace."""
    s_k = np.abs(rho) ** 2 / st["N"]
    phi = s_k[:, st["constrained"]].sum(1) * st["N"] / st["V"]
    acc["sum_s"] += s_k
    acc["sum_phi"] += phi
    acc["sum_phi2"] += phi**2
    tr = st.get("trace")
    if tr is not None and st["step"] % tr["every"] == 0 and tr["n"] < tr["cap"]:
        tr["rho"][:, tr["n"]] = np.stack([rho.real, rho.imag], axis=-1)
        tr["time"][:, tr["n"]] = st["step"] * st["dt"]
        tr["n"] += 1
    assert np.all(np.isfinite(phi)), "non-finite Phi"
    assert acc["sum_s"].shape == rho.shape, "one S sum per lane and wave"


def run_segment(
    st: Dict[str, Any], s0: int, n_steps: int, rec: np.ndarray, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Production segment of n_steps steps from absolute step s0.

    Resets winding and sums (begin_segment), accumulates them every step and
    records the snapshots whose absolute steps ``rec`` fall in (s0, s0 +
    n_steps]; returns the end-of-segment record fields (cli.md RECORD_FIELDS).
    """
    assert n_steps >= 1 and st["x"].ndim == 2, "need steps and lanes"
    lanes, waves = st["x"].shape
    st["wabs"], st["wnet"] = np.zeros((lanes, waves)), np.zeros((lanes, waves))
    acc = {
        "sum_s": np.zeros((lanes, waves)),
        "sum_phi": np.zeros(lanes),
        "sum_phi2": np.zeros(lanes),
        "n_valid": np.zeros((lanes, waves), np.int64),
    }
    rho = _rho(st, s0 * st["dt"])
    in_seg = np.flatnonzero((rec > s0) & (rec <= s0 + n_steps))
    nxt, xi = 0, np.empty(0)
    for i in range(n_steps):
        if i % NOISE_BLOCK == 0:
            shape = (NOISE_BLOCK, 2, lanes, waves)
            xi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        _advance(st, xi[i % NOISE_BLOCK])
        st["step"] = s0 + i + 1
        new = _rho(st, st["step"] * st["dt"])
        dth, ok = winding_step(new, rho, st["N"], PHASE_FLOOR)
        st["wabs"] += np.abs(dth)
        st["wnet"] += dth
        acc["n_valid"] += ok
        rho = new
        _accumulate(st, acc, rho)
        if nxt < in_seg.size and rec[in_seg[nxt]] == st["step"]:
            _record(st, int(in_seg[nxt]), st["step"], st["step"] * st["dt"], rho)
            nxt += 1
    assert nxt == in_seg.size, "every snapshot of the segment recorded"
    t_end = (s0 + n_steps) * st["dt"]
    return {
        "wind_abs": st["wabs"].copy(), "wind_net": st["wnet"].copy(),
        "n_valid": acc["n_valid"], "sum_kinetic": st["ke"] * n_steps,
        "sum_phi": acc["sum_phi"], "sum_phi2": acc["sum_phi2"],
        "sum_s": acc["sum_s"], "n_sum": np.full(lanes, n_steps, np.int64),
        "status": np.zeros(lanes, np.int64), "max_drift": np.zeros(lanes),
        "step": np.full(lanes, s0 + n_steps, np.int64), "time": np.full(lanes, t_end),
    }  # fmt: skip


# ============================================================
# LANES AND FILES
# ============================================================


def init_lanes(
    sysd: Dict[str, Any], s_final: np.ndarray, temps: List[float], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """State of every lane (ground state major): oscillators, rho0, drift, particles.

    ``cfg`` holds seed, start_key (0 for common seeds), chi_key, factor (rate
    injection on the constrained waves), dt, n_snapshots capacity.
    """
    g, n_t = s_final.shape[0], len(temps)
    assert g >= 1 and n_t >= 1, "need ground states and temperatures"
    rows, lane_t = np.repeat(np.arange(g), n_t), np.tile(np.asarray(temps), g)
    con, kmag, n = sysd["constrained"], sysd["kmag"], sysd["N"]
    omega = np.where(con, cfg["factor"] * kmag, kmag)
    e, c = oscillator_coeffs(omega, cfg["dt"])
    rng = np.random.default_rng([cfg["seed"], cfg["start_key"], cfg["chi_key"], 1])
    lanes, waves = rows.size, kmag.size
    x = rng.standard_normal((lanes, waves)) + 1j * rng.standard_normal((lanes, waves))
    v = omega * (rng.standard_normal((lanes, waves))
                 + 1j * rng.standard_normal((lanes, waves)))  # fmt: skip
    gs = np.random.default_rng([cfg["seed"], cfg["start_key"], cfg["chi_key"], 2])
    s0 = gs.uniform(0.5, 1.5, (g, waves))
    phi0 = gs.uniform(0.0, 2.0 * np.pi, (g, waves))
    rho0 = np.sqrt(n * s0[rows]) * np.exp(1j * phi0[rows])
    drift = (
        DRIFT_C * kmag * np.sqrt(lane_t)[:, None] * gs.standard_normal((g, waves))[rows]
    )
    u = np.sqrt(lane_t)[:, None, None] * rng.standard_normal((lanes, n, 2))
    u -= u.mean(axis=1, keepdims=True)
    cap = cfg["n_snapshots"]
    snap = {"step": np.zeros((lanes, cap), np.int64), "time": np.zeros((lanes, cap)),
            "wind_abs": np.zeros((lanes, cap, waves)),
            "wind_net": np.zeros((lanes, cap, waves)),
            "rho": np.zeros((lanes, cap, waves, 2)), "r": np.zeros((lanes, cap, n, 2)),
            "phi": np.zeros((lanes, cap)), "kinetic": np.zeros((lanes, cap)),
            "count": np.zeros(lanes, np.int64)}  # fmt: skip
    assert np.all(np.isfinite(drift)) and u.shape == (lanes, n, 2), "lane draws"
    return {
        "x": x, "v": v, "e": e, "c": c, "amp": np.sqrt(n * lane_t / 2.0),
        "rho0": np.where(con, 0.0, rho0), "omega_drift": np.where(con, 0.0, drift),
        "constrained": con, "N": n, "V": float(abs(np.linalg.det(sysd["basis"]))),
        "dt": cfg["dt"], "r0": (s_final @ sysd["basis"])[rows], "u": u,
        "ke": 0.5 * (u**2).sum(axis=(1, 2)), "snap": snap, "rows": rows,
        "lane_t": lane_t, "step": 0, "rng": rng,
    }  # fmt: skip


def _protocol(proto: Dict[str, Any]) -> Tuple[List[Any], np.ndarray]:
    """CLI phases and absolute record steps for the protocol settings."""
    ns = argparse.Namespace(
        **{k: proto[k] for k in ("n_equil", "n_andersen", "n_nve", "cycles", "scheme")}
    )
    phases = protocol_phases(ns)
    rec = np.asarray(record_schedule(phases, proto["n_snapshots"]), dtype=np.int64)
    assert sum(ph.cycle >= 0 for ph in phases) == proto["cycles"], "one per cycle"
    assert rec.size >= 2 * proto["cycles"], "snapshots in every segment"
    return phases, rec


def _run_phases(st: Dict[str, Any], phases: List[Any], rec: np.ndarray) -> List[Dict]:
    """Run the production segments (other phases only advance the step)."""
    s0, cycles = 0, []
    for ph in phases:
        if ph.cycle >= 0:
            cycles.append(run_segment(st, s0, ph.n_steps, rec, st["rng"]))
        s0 += ph.n_steps
    assert len(cycles) >= 1, "no production segment"
    assert int(st["snap"]["count"][0]) == rec.size, "all snapshots taken"
    return cycles


def output_arrays(
    st: Dict[str, Any], sysd: Dict[str, Any], cycles: List[Dict], meta: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """The arrays of ``md_chi<tag>.npz`` (cli.md.output_arrays keys)."""
    out = {f"snap_{k}": v for k, v in st["snap"].items()}
    for k, v in cycles[-1].items():
        out[k] = v
        out[f"cycle_{k}"] = np.stack([c[k] for c in cycles])
    lanes = st["rows"].size
    gate = 1.0 + 0.01 * st["rng"].standard_normal(lanes)
    t_end = float(cycles[-1]["time"][0])
    out.update(
        equil_gate_first=gate, equil_gate=gate, equil_gate_pass=np.abs(gate - 1) < 0.05,
        equil_extended=np.asarray(False), r_final=st["r0"] + st["u"] * t_end,
        v_final=st["u"], temperatures=np.asarray(meta["temps"]),
        indices=np.asarray(meta["indices"], np.int64),
        lane_index=np.asarray(meta["indices"], np.int64)[st["rows"]],
        lane_temperature=st["lane_t"], basis=sysd["basis"],
        waves_n_vectors=sysd["n_vectors"], waves_kvec=sysd["kvec"],
        waves_constrained=sysd["constrained"], record_steps=meta["rec"],
        settings=np.asarray(json.dumps(meta["settings"])),
    )  # fmt: skip
    tr = st.get("trace")
    if tr is not None:
        out.update(trace_rho=tr["rho"], trace_time=tr["time"],
                   trace_count=np.full(lanes, tr["n"], np.int64))  # fmt: skip
    assert out["lane_index"].shape == (lanes,), "one entry per lane"
    assert out["snap_wind_abs"].shape[:2] == (lanes, meta["rec"].size), "snapshots"
    return out


def summary_json(
    arrays: Dict[str, np.ndarray], sysd: Dict[str, Any], meta: Dict[str, Any],
    phases: List[Any],
) -> Dict[str, Any]:  # fmt: skip
    """JSON metadata with the keys of cli.md.summarize the analysis reads."""
    lanes = int(arrays["status"].size)
    assert lanes >= 1 and len(phases) >= 1, "need lanes and phases"
    out = {
        "N": sysd["N"], "d": sysd["d"], "M": sysd["M"], "K": sysd["K"],
        "k_track": 2.0 * sysd["K"], "chi": sysd["chi"],
        "n_waves": int(sysd["kvec"].shape[0]), "n_constrained": sysd["M"],
        "input": meta["input"], "indices": list(meta["indices"]),
        "temperatures": list(meta["temps"]), "n_lanes": lanes,
        "args": meta["args"], "settings": meta["settings"],
        "phases": [dict(ph.__dict__) for ph in phases],
        "n_record_steps": int(meta["rec"].size),
        "status_counts": {"RUNNING": lanes}, "n_running": lanes,
        "gate_tol": 0.05, "equil_extended": False,
        "npz_keys": sorted(arrays), "device": "synthetic",
        "synthetic": {"gamma_ratio": GAMMA_RATIO, "drift_c": DRIFT_C,
                      "factor": meta["factor"]},
    }  # fmt: skip
    assert out["n_waves"] == arrays["waves_kvec"].shape[0], "wave count"
    return out


def write_file(
    prefix: Path, sysd: Dict[str, Any], s_final: np.ndarray, temps: List[float],
    cfg: Dict[str, Any], proto: Dict[str, Any],
) -> Path:  # fmt: skip
    """Simulate every lane of one (start, chi) and write ``prefix``.npz / .json."""
    assert proto["cycles"] >= 1 and len(temps) >= 1, "protocol and lanes"
    phases, rec = _protocol(proto)
    cfg = {**cfg, "n_snapshots": int(rec.size)}
    st = init_lanes(sysd, s_final, temps, cfg)
    if proto.get("trace_every", 0) > 0:
        cap, lanes = int(proto["trace_cap"]), st["rows"].size
        st["trace"] = {"every": int(proto["trace_every"]), "cap": cap, "n": 0,
                       "rho": np.zeros((lanes, cap, sysd["kmag"].size, 2)),
                       "time": np.zeros((lanes, cap))}  # fmt: skip
    cycles = _run_phases(st, phases, rec)
    settings = {"dt": cfg["dt"], "wind_every": 1, "phase_floor": PHASE_FLOOR,
                "record_steps": rec.tolist(), "andersen_p": 0.01,
                "gamma": proto.get("gamma", 0.0), "remove_com": True}  # fmt: skip
    args = {**{k: proto[k] for k in proto}, "dt": cfg["dt"], "temperature": temps}
    meta = {"temps": temps, "indices": list(range(s_final.shape[0])), "rec": rec,
            "settings": settings, "args": args, "factor": cfg["factor"],
            "input": str(prefix.parent / "flows.npz")}  # fmt: skip
    arrays = output_arrays(st, sysd, cycles, meta)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(prefix) + ".npz", **arrays)
    Path(str(prefix) + ".json").write_text(
        json.dumps(summary_json(arrays, sysd, meta, phases), indent=1) + "\n"
    )
    assert Path(str(prefix) + ".npz").is_file(), "npz written"
    return Path(str(prefix) + ".npz")


def write_flows(
    root: Path, tag: str, sysd: Dict[str, Any], s_final: np.ndarray
) -> None:
    """The K-spectrum input: flows_chi<tag>.npz (s_final, basis) and .json."""
    assert s_final.ndim == 3 and s_final.shape[2] == 2, "fractional (G, N, 2)"
    assert np.all((s_final >= 0) & (s_final < 1)), "fractional coordinates"
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / f"flows_chi{tag}.npz", s_final=s_final, basis=sysd["basis"])
    (root / f"flows_chi{tag}.json").write_text(
        json.dumps({"N": sysd["N"], "d": 2, "M": sysd["M"], "K": sysd["K"]}) + "\n"
    )


def default_proto(n_nve: int, cycles: int, n_snapshots: int) -> Dict[str, Any]:
    """Protocol settings: short equilibration/Andersen phases (skipped), NVE."""
    assert n_nve >= 8 and cycles >= 1, "need production segments"
    assert 2 <= n_snapshots <= 256, "snapshot capacity as the CLI"
    return {
        "n_equil": 1000,
        "n_andersen": 500,
        "n_nve": n_nve,
        "cycles": cycles,
        "scheme": "andersen-nve",
        "gamma": 0.0,
        "n_snapshots": n_snapshots,
    }


def generate(
    out: Path, starts: List[str], chis: List[str], cfg: Dict[str, Any],
    proto: Dict[str, Any], inject: Optional[Dict[str, float]] = None,
) -> List[Path]:  # fmt: skip
    """Write md_chi<tag> (and flows_chi<tag>) for every start and chi.

    ``cfg``: n (particles), n_gs, temps, dt, seed, common (shared draws).
    """
    assert starts and chis and cfg["n_gs"] >= 1, "nothing to generate"
    inject = inject or {}
    assert set(inject) <= set(starts), f"unknown starts in --inject {inject}"
    paths: List[Path] = []
    for ci, chi in enumerate(chis):
        sysd = make_system(cfg["n"], Fraction(chi))
        tag = chi_tag(Fraction(chi))
        for si, start in enumerate(starts):
            root = out / start / f"N{cfg['n']}"
            key = 0 if cfg["common"] else si + 1
            gs = np.random.default_rng([cfg["seed"], key, ci, 3])
            s_final = gs.uniform(0.0, 1.0, (cfg["n_gs"], cfg["n"], 2))
            write_flows(root, tag, sysd, s_final)
            lane = {
                "seed": cfg["seed"],
                "start_key": key,
                "chi_key": ci,
                "factor": float(inject.get(start, 1.0)),
                "dt": cfg["dt"],
            }
            paths.append(write_file(root / f"md_chi{tag}", sysd, s_final,
                                    list(cfg["temps"]), lane, proto))  # fmt: skip
            print(f"wrote {paths[-1]}", flush=True)
    assert len(paths) == len(starts) * len(chis), "one file per (start, chi)"
    return paths


def generate_extras(out: Path, chis: List[str], cfg: Dict[str, Any],
                    proto: Dict[str, Any]) -> List[Path]:  # fmt: skip
    """Langevin-labelled files (F9, H7) and dense-trace spectra files (F10)."""
    assert chis and cfg["n_gs"] >= 2, "need chi values and two ground states"
    paths: List[Path] = []
    for ci, chi in enumerate(chis):
        sysd, tag = make_system(cfg["n"], Fraction(chi)), chi_tag(Fraction(chi))
        for si, start in enumerate(STARTS):
            root = out / start / f"N{cfg['n']}"
            with np.load(root / f"flows_chi{tag}.npz") as z:
                s_final = np.asarray(z["s_final"])
            key = 0 if cfg["common"] else si + 1
            lane = {
                "seed": cfg["seed"] + 7,
                "start_key": key,
                "chi_key": ci,
                "factor": 1.0,
                "dt": cfg["dt"],
            }
            if start in LANGEVIN_STARTS:
                for gamma in LANGEVIN_GAMMAS:
                    pro = {**proto, "scheme": "baoab", "gamma": gamma, "n_andersen": 0}
                    temps = [t for t in cfg["temps"] if t in (1e-6, 1e-4)]
                    paths.append(
                        write_file(
                            root / f"md_langevin_g{gamma:g}_chi{tag}",
                            sysd,
                            s_final[:5],
                            temps,
                            lane,
                            pro,
                        )
                    )
            pro = {**proto, "cycles": 1, "trace_every": 2, "trace_cap": 4096}
            paths.append(write_file(root / f"md_spectra_chi{tag}", sysd, s_final[:2],
                                    [1e-6], lane, pro))  # fmt: skip
    assert all(p.is_file() for p in paths), "extras written"
    return paths


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])

    # ----- required -----
    parser.add_argument("--out", type=Path, required=True)

    # ----- optional -----
    parser.add_argument("--starts", nargs="+", default=list(STARTS))
    parser.add_argument("--chi", nargs="+", default=list(CHIS))
    parser.add_argument("--n", type=int, default=64, help="particles (square)")
    parser.add_argument("--n-gs", type=int, default=10)
    parser.add_argument("--temps", type=float, nargs="+", default=list(TEMPS))
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--n-nve", type=int, default=16_000)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--n-snapshots", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--common-seeds", action="store_true")
    parser.add_argument("--inject", nargs="*", default=[], help="START=FACTOR")
    parser.add_argument("--extras", action="store_true", help="Langevin + spectra")
    args = parser.parse_args()

    # =========================================================
    # GENERATE
    # =========================================================
    base = {"n": args.n, "n_gs": args.n_gs, "temps": args.temps, "dt": args.dt,
            "seed": args.seed, "common": args.common_seeds}  # fmt: skip
    protocol = default_proto(args.n_nve, args.cycles, args.n_snapshots)
    factors = {s.split("=")[0]: float(s.split("=")[1]) for s in args.inject}
    written = generate(args.out, args.starts, args.chi, base, protocol, factors)
    if args.extras:
        written += generate_extras(args.out, args.chi, base, protocol)
    print(f"Done: {len(written)} files.", flush=True)
