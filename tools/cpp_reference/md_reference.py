#!/usr/bin/env python3

"""
C++ MD reference data for the statistical-parity test (plan task T4).

Runs the UNMODIFIED group binary ``soft_core_stealthy2.out`` in ``random ...
MD step T`` mode (``CollectiveCoordinateMD``, main.cpp:725-890) and turns its
outputs into the small committed files ``tests/data/md_cxx_reference.json`` and
``tests/data/md_cxx_samples.npz``.

Subcommands
-----------
``run --seed S``
    Write the stdin token stream, run the binary in a fresh directory under
    ``--run-root`` (one per seed) and record wall time, host, job id and the
    binary hash. Meant for ``slurm/md_cxx_reference_cpu.slurm``.
``collect --seeds 1 2 3``
    Parse each run's log and ConfigPack, recompute every sample energy at full
    precision with the C++ harness (``build/dump_reference kernel``, the
    unmodified ``ShiftedCCPotential``), recompute it with ``xla_cco``
    ``StealthyEnergy`` in the C++ box, enforce the unit gate (relative 1e-8),
    and write the JSON and npz.

Units: the box is the C++ random-mode box, side ``L = 1000^(1/d)`` (defect
D-004, density N / L^d), K is the C++ Cartesian ``k2 = Ka / a``, Phi is the C++
``ShiftedCCPotential::Energy`` = sum_{half-space k} |rho_k|^2 / V (Shift = 0,
val = 0 adds exactly 0.0). Mass 1 and k_B = 1, so KE = sum v^2 / 2 and T is
used directly as kT.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----- numerics -----
import jax.numpy as jnp
import numpy as np

# ----- local modules -----
from xla_cco.box import Box
from xla_cco.energy import make_stealthy_energy
from xla_cco.io.configpack import read_configpack
from xla_cco.kvectors import enumerate_kvectors, k_outer_from_ka, unit_ball_volume

# ============================================================
# CONSTANTS
# ============================================================

REPO: Path = Path(__file__).resolve().parents[2]
TOOL_DIR: Path = REPO / "tools" / "cpp_reference"
HARNESS: Path = TOOL_DIR / "build" / "dump_reference"
CCO_BINARY: Path = Path("/home/sd2402/ccoptimization/cco/EXC/soft_core_stealthy2.out")
RUN_ROOT: Path = Path("/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/md_cxx")
JSON_OUT: Path = REPO / "tests" / "data" / "md_cxx_reference.json"
NPZ_OUT: Path = REPO / "tests" / "data" / "md_cxx_samples.npz"

# ----- physical parameters (stdin) -----
DIM: int = 2
N_PARTICLES: int = 256
CHI_TARGET: float = 0.1
TEMPERATURE: float = 1e-5
PHI_FAKE: float = 0.05  # only sets a = (phi / V_d(1))^(1/d); Ka = K a
K1A: float = 0.0
S0: float = 0.0  # pure stealthy (ShiftedCCPotential)
VAL: float = 0.0  # soft-core off: RepulsiveCCPotential adds exactly 0.0
SIGMA: str = "0.5"  # parsed but unused with val = 0
NTHREADS: int = 1  # serial sums: harness and MD energies are bit-identical
NC: int = 200
NC_EQUI: int = 20
POTENTIAL: str = "step"
ALGO: str = "LBFGS"  # parsed; MD always pre-relaxes with RelaxStructure_NLOPT
SAVENAME: str = "cxxmd"

# ----- argv: time limit (h), tolerance, max_steps, seed (main.cpp:125-136) -----
ARGV_TOLERANCE: str = "1e-20"
ARGV_MAX_STEPS: str = "1000"

# ----- hard-coded in main.cpp / MD_System.cpp (not settable from stdin) -----
STALE_NUM: int = 1000  # main.cpp:155,171 random-mode box uses num = 1000 (D-004)
MD_DT: float = 0.01  # main.cpp:378, MDAutoTimeStep = false in random mode
STEP_PER_SAMPLE: int = 100_000  # main.cpp:379
ANDERSEN_P: float = 0.01  # main.cpp:852
PRERELAX_EVALS: int = 1000  # main.cpp:741 RelaxStructure_NLOPT(..., 1000)

# ----- unit gate -----
GATE_RTOL: float = 1e-8
SHELL_RTOL: float = 1e-9  # one |k|^2 shell: ulp-level spread, gaps >> 1e-9
N_NPZ_PER_SEED: int = 5

_FLOAT = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)"
SAMPLE_RE = re.compile(
    r"at time=(\d+),\s*2:(\d+)/(\d+),energy="
    + _FLOAT
    + r",kinetic_energy="
    + _FLOAT
    + r",dt="
    + _FLOAT
    + r",dimensionless_pressure="
    + _FLOAT
)
EQUIL_RE = re.compile(
    r"^1:(\d+)/(\d+), x0="
    + _FLOAT
    + r", Ep="
    + _FLOAT
    + r", Ek="
    + _FLOAT
    + r", dt="
    + _FLOAT
)

# ============================================================
# K-SPACE SETUP (C++ random-mode box)
# ============================================================


def cpp_box_side(dim: int) -> float:
    """Return the random-mode box side ``pow(1000, 1/d)`` (main.cpp:171, D-004)."""
    assert 1 <= dim <= 4, "dim must be 1..4"
    side = math.pow(float(STALE_NUM), 1.0 / float(dim))
    assert side > 0.0 and math.isfinite(side), "box side must be finite"
    return side


def shell_table(
    basis: np.ndarray, k_max: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lo, hi, count) of the |k|^2 shells below ``k_max``.

    Values within ``SHELL_RTOL`` of each other form one shell, because
    symmetry-equivalent k (e.g. n = (5, 0) and (3, 4)) differ by an ulp in the
    float |k|^2 (D-053). ``lo``/``hi`` are a shell's smallest/largest |k|^2 and
    ``count`` the half-space count of k up to and including it.
    """
    assert k_max > 0.0, "k_max must be positive"
    k_sq = np.sort(enumerate_kvectors(basis, 0.0, k_max).k_sq)
    assert k_sq.size > 0, "no k-vectors below k_max"
    new = np.concatenate(([True], k_sq[1:] > k_sq[:-1] * (1.0 + SHELL_RTOL)))
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], k_sq.size)
    return k_sq[starts], k_sq[ends - 1], ends


def choose_k_outer(basis: np.ndarray, m_target: int) -> Tuple[int, float]:
    """Pick the achievable M closest to ``m_target`` and a mid-gap cutoff K.

    Parameters
    ----------
    basis : np.ndarray
        (d, d) lattice basis of the C++ box.
    m_target : int
        Desired number of half-space constraints, round(chi d (N - 1)).

    Returns
    -------
    Tuple[int, float]
        (M, K): M is the cumulative shell count nearest ``m_target`` (ties go
        to the smaller M) and K = sqrt of the midpoint between that shell and
        the next, so K^2 is never on a shell.
    """
    assert m_target >= 1, "m_target must be >= 1"
    vol = abs(float(np.linalg.det(basis)))
    dim = basis.shape[0]
    # generous: a ball of radius 2 pi (4 m / V)^(1/d) holds >> m half-space points
    k_max = 2.0 * math.pi * (4.0 * m_target / vol) ** (1.0 / dim)
    lo, hi, counts = shell_table(basis, 2.0 * k_max)
    assert counts[-1] > m_target, "shell table too short for m_target"
    best = int(np.argmin(np.abs(counts[:-1] - m_target)))
    m = int(counts[best])
    k_out = math.sqrt(0.5 * (float(hi[best]) + float(lo[best + 1])))
    return m, k_out


def ka_for_m(basis: np.ndarray, m_target: int, phi: float) -> Tuple[int, float, float]:
    """Return (M, Ka, K) with K = ``k_outer_from_ka(Ka)`` holding exactly M waves.

    Ka goes to the binary as ``repr(Ka)``; K is what main.cpp computes from it.
    """
    dim = basis.shape[0]
    assert phi > 0.0 and m_target >= 1, "need phi > 0 and m_target >= 1"
    m, k_mid = choose_k_outer(basis, m_target)
    a = (phi / unit_ball_volume(dim)) ** (1.0 / dim)  # main.cpp:172
    ka = float(repr(k_mid * a))
    k_out = k_outer_from_ka(ka, phi, dim)
    got = enumerate_kvectors(basis, 0.0, k_out).n.shape[0]
    assert got == m, f"K = {k_out} holds {got} waves, expected {m}"
    return m, ka, k_out


# ============================================================
# C++ PROTOCOL TRANSCRIPTIONS
# ============================================================


def andersen_reset_steps(n_steps: int, prob: float) -> np.ndarray:
    """Return the 1-based steps of one ``AndersonEvolve`` call that reset a velocity.

    Transcribes MD_System.cpp:199-250: ``ResetCount`` starts at 0.0 in every
    call; after each Verlet step ``ResetCount += p`` and, while it is > 0, one
    uniformly chosen particle gets a fresh Maxwell-Boltzmann velocity and
    ``ResetCount -= 1``. A step can appear more than once when p > 1.
    """
    assert n_steps >= 0, "n_steps must be >= 0"
    assert 0.0 < prob <= 16.0, "prob must be in (0, 16]"
    max_resets = int(math.ceil(n_steps * prob)) + n_steps + 2
    out = np.zeros(max_resets, dtype=np.int64)
    count = 0
    acc = 0.0
    for step in range(1, n_steps + 1):
        acc += prob
        for _ in range(int(math.ceil(prob)) + 1):  # bounded "while acc > 0"
            if not acc > 0.0:
                break
            out[count] = step
            count += 1
            acc -= 1.0
    assert count <= max_resets, "reset buffer overflow"
    return out[:count]


def build_stdin(ka: float, n_samples: int, n_equil: int) -> str:
    """Return the stdin token stream of an MD run (main.cpp:162-236)."""
    assert ka > 0.0 and n_samples >= 1 and n_equil >= 0, "bad MD stdin arguments"
    toks = [
        str(DIM), repr(K1A), repr(ka), repr(S0), repr(VAL), SIGMA, repr(PHI_FAKE),
        str(NTHREADS), str(N_PARTICLES), str(n_samples), str(n_equil), "random",
        ALGO, SAVENAME, "MD", POTENTIAL, repr(TEMPERATURE),
    ]  # fmt: skip
    text = " ".join(toks) + "\n"
    assert len(toks) == 17, "MD stdin must have 17 tokens"
    return text


# ============================================================
# LOG AND HARNESS PARSING
# ============================================================


def parse_sample_lines(text: str) -> Dict[str, np.ndarray]:
    """Parse the stage-2 sample lines (main.cpp:848-877) of an MD log.

    Returns arrays ``index, wall, phi, ke, dt, pressure`` (pressure is the
    logged ``1 + virial / (N T)``); values carry the log's 6 significant digits.
    """
    assert isinstance(text, str), "text must be a string"
    rows = [m.groups() for m in SAMPLE_RE.finditer(text)]
    out = {
        "index": np.array([int(r[1]) for r in rows], dtype=np.int64),
        "wall": np.array([float(r[0]) for r in rows]),
        "phi": np.array([float(r[3]) for r in rows]),
        "ke": np.array([float(r[4]) for r in rows]),
        "dt": np.array([float(r[5]) for r in rows]),
        "pressure": np.array([float(r[6]) for r in rows]),
    }
    assert np.array_equal(out["index"], np.arange(len(rows))), (
        "sample lines out of order"
    )
    return out


def parse_equil_lines(text: str) -> Dict[str, np.ndarray]:
    """Parse the stage-1 equilibration lines (main.cpp:817) of an MD log."""
    assert isinstance(text, str), "text must be a string"
    rows = [EQUIL_RE.match(ln.strip()) for ln in text.splitlines()]
    good = [r.groups() for r in rows if r is not None]
    out = {
        "x0": np.array([float(r[2]) for r in good]),
        "phi": np.array([float(r[3]) for r in good]),
        "ke": np.array([float(r[4]) for r in good]),
        "dt": np.array([float(r[5]) for r in good]),
    }
    assert all(v.size == len(good) for v in out.values()), "ragged equil table"
    return out


def run_harness(prefix: Path, count: int, ka: float, work: Path) -> Dict[str, Any]:
    """Recompute C++ energies of a pack with ``dump_reference kernel``.

    Returns ``k_outer_hex``, ``m``, ``kvec`` (M, d) Cartesian, ``energy``
    (count,) parsed from ``%.17g``, and the command line.
    """
    assert HARNESS.is_file(), f"build the harness first: make -C {TOOL_DIR}"
    assert count >= 1, "count must be >= 1"
    out = work / "harness_energy.txt"
    cmd = [
        str(HARNESS), "kernel", "--pack", str(prefix), "--d", str(DIM),
        "--n", str(N_PARTICLES), "--rescale", "0", "--ka", repr(ka),
        "--phi", repr(PHI_FAKE), "--count", str(count), "--out", str(out),
    ]  # fmt: skip
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(work / "harness.stdout", "w") as log:
        rc = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=log).returncode
    assert rc == 0, f"harness exited {rc}"
    lines = out.read_text().splitlines()
    kvec = [[float(x) for x in ln.split()[1:-1]] for ln in lines if ln[:5] == "KVEC "]
    energy = np.array([float(ln.split()[1]) for ln in lines if ln[:2] == "E "])
    assert energy.size == count, "harness energy count mismatch"
    return {
        "k_outer_hex": next(ln.split()[1] for ln in lines if ln[:7] == "KOUTER "),
        "m": int(next(ln.split()[1] for ln in lines if ln[:2] == "M ")),
        "kvec": np.array(kvec),
        "energy": energy,
        "cmd": " ".join(cmd),
    }


# ============================================================
# XLA ENERGIES AND METADATA
# ============================================================


def xla_energies(basis: np.ndarray, k_outer: float, s: np.ndarray) -> np.ndarray:
    """Return xla_cco ``StealthyEnergy`` Phi of (B, N, d) fractional configs."""
    assert s.ndim == 3 and s.shape[2] == basis.shape[0], "s must be (B, N, d)"
    box = Box(basis=jnp.asarray(basis, dtype=jnp.float64))
    energy = make_stealthy_energy(box, enumerate_kvectors(basis, 0.0, k_outer))
    out = np.array([float(energy(jnp.asarray(cfg))) for cfg in s])
    assert np.all(np.isfinite(out)), "non-finite xla energy"
    return out


def file_meta(path: Path) -> Dict[str, Any]:
    """Return path, sha256, size and mtime (ISO) of a file."""
    assert path.is_file(), f"missing file {path}"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    st = path.stat()
    assert st.st_size > 0, f"empty file {path}"
    mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat()
    return {"path": str(path), "sha256": digest, "bytes": st.st_size, "mtime": mtime}


def rel_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return |a - b| / |b| elementwise (b must be nonzero)."""
    assert a.shape == b.shape, "shape mismatch"
    assert np.all(b != 0.0), "reference values must be nonzero"
    out: np.ndarray = np.abs(a - b) / np.abs(b)
    return out


# ============================================================
# RUN
# ============================================================


def setup_params(n_samples: int, n_equil: int) -> Dict[str, Any]:
    """Return the k-space set-up and stdin of an MD run."""
    assert n_samples >= 1 and n_equil >= 0, "need Nc >= 1 and Nc_equi >= 0"
    side = cpp_box_side(DIM)
    basis = side * np.eye(DIM)
    m_target = int(round(CHI_TARGET * DIM * (N_PARTICLES - 1)))
    m, ka, k_out = ka_for_m(basis, m_target, PHI_FAKE)
    stdin = build_stdin(ka, n_samples, n_equil)
    assert m >= 1 and k_out > 0.0, "empty k-set"
    return dict(side=side, m_target=m_target, m=m, ka=ka, k_outer=k_out, stdin=stdin)


def _run_binary(argv: List[str], stdin: str, work: Path) -> Tuple[int, float]:
    """Run ``argv`` in ``work`` with ``stdin``; return (return code, wall seconds)."""
    assert work.is_dir(), f"missing run dir {work}"
    assert stdin.endswith("\n"), "stdin must end with a newline"
    env = dict(os.environ, OMP_NUM_THREADS=str(NTHREADS))
    t0 = time.time()
    with open(work / "run.stdout", "w") as log:
        proc = subprocess.run(
            argv,
            input=stdin,
            text=True,
            cwd=work,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode, time.time() - t0


def run_seed(seed: int, hours: int, n_samples: int, n_equil: int, root: Path) -> Path:
    """Run the binary for one seed in ``root/seed<seed>``; return the run dir.

    Parameters
    ----------
    seed : int
        argv[4] seed of ``rngGod`` (>= 1).
    hours : int
        argv[1] time limit; the binary dumps .MDDump 5 min before it.
    n_samples, n_equil : int
        Nc and Nc_equi.
    root : Path
        Parent of the per-seed run directories.
    """
    assert seed >= 1 and hours >= 1, "seed and hours must be >= 1"
    work = root / f"seed{seed}"
    work.mkdir(parents=True, exist_ok=True)
    assert not any(work.glob(SAVENAME + "*")), f"{work} already holds a run"
    par = setup_params(n_samples, n_equil)
    argv = [str(CCO_BINARY), str(hours), ARGV_TOLERANCE, ARGV_MAX_STEPS, str(seed)]
    (work / "stdin.txt").write_text(par["stdin"])
    before = file_meta(CCO_BINARY)
    rc, wall = _run_binary(argv, par["stdin"], work)
    assert file_meta(CCO_BINARY)["sha256"] == before["sha256"], "binary changed"
    info = dict(
        seed=seed,
        argv=argv,
        stdin=par["stdin"],
        returncode=rc,
        wall_seconds=wall,
        host=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
        binary=before,
    )
    (work / "run.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"seed {seed}: rc={rc} wall={wall:.0f}s dir={work}", flush=True)
    assert rc == 0, f"binary exited {rc}"
    return work


# ============================================================
# COLLECT
# ============================================================


def collect_seed(work: Path, par: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray]:
    """Parse, recompute and gate one run; return (record, s) with s (B, N, d)."""
    info = json.loads((work / "run.json").read_text())
    text = (work / "run.stdout").read_text()
    log, equil = parse_sample_lines(text), parse_equil_lines(text)
    packs = read_configpack(work / SAVENAME)
    assert len(packs) == log["phi"].size >= 1, "pack and log sample counts differ"
    basis = packs[0].basis
    assert np.array_equal(basis, par["side"] * np.eye(DIM)), "unexpected C++ box"
    s = np.stack([p.relative for p in packs])
    harn = run_harness(work / SAVENAME, len(packs), par["ka"], work)
    assert harn["k_outer_hex"] == float.hex(par["k_outer"]), "C++ and xla K differ"
    kset = enumerate_kvectors(basis, 0.0, par["k_outer"])
    n_cpp = np.rint(harn["kvec"] * par["side"] / (2.0 * math.pi)).astype(np.int64)
    assert harn["m"] == par["m"] == kset.n.shape[0], "M mismatch"
    assert sorted(map(tuple, n_cpp)) == sorted(map(tuple, kset.n)), "k-set mismatch"
    phi_xla = xla_energies(basis, par["k_outer"], s)
    log_ok = [
        f"{e:g}" == f"{x:g}" for e, x in zip(harn["energy"], log["phi"], strict=True)
    ]
    rec = {
        **{k: info[k] for k in ("seed", "slurm_job_id", "host", "wall_seconds")},
        "argv": info["argv"],
        "n_samples": len(packs),
        "phi_cpp": harn["energy"].tolist(),
        "phi_xla": phi_xla.tolist(),
        **{f"{k}_log": log[k].tolist() for k in ("phi", "ke", "dt", "pressure")},
        "wall_log": log["wall"].tolist(),
        **{f"equil_{k}_log": equil[k].tolist() for k in ("phi", "ke", "dt")},
        "max_rel_diff_xla_vs_cpp": float(rel_diff(phi_xla, harn["energy"]).max()),
        "max_rel_diff_xla_vs_log": float(rel_diff(phi_xla, log["phi"]).max()),
        "log_matches_cpp_g6": int(sum(log_ok)),
        "harness_cmd": harn["cmd"],
    }
    return rec, s


def protocol_record() -> Dict[str, Any]:
    """Return the C++ protocol V2 needs to replicate (main.cpp:725-890)."""
    half = STEP_PER_SAMPLE // 2
    resets = andersen_reset_steps(half, ANDERSEN_P)
    assert resets.size > 0 and resets[0] == 1, "unexpected Andersen schedule"
    assert STEP_PER_SAMPLE % 2 == 0, "StepPerSample/2 must be exact"
    andersen = {
        "p": ANDERSEN_P,
        "mode": "count",
        "rule": "per call ResetCount = 0.0; after each full Verlet step "
        "ResetCount += p; while ResetCount > 0: one particle i = floor(U N) "
        "(with replacement) gets v_i ~ N(0, T/m) per component (Box-Muller sin "
        "branch), ResetCount -= 1 (MD_System.cpp:199-250)",
        "resets_per_call": int(resets.size),
        "first_reset_steps": resets[:4].tolist(),
        "last_reset_steps": resets[-2:].tolist(),
        "note": "fires on step 1; the float sum of 0.01 puts the 2nd reset at "
        "step 100, then every 100 steps (D-050)",
    }
    return {
        "dt": MD_DT,
        "dt_source": "main.cpp:378 MDTimeStep = 0.01, fixed (random mode passes "
        "MDAutoTimeStep = false)",
        "mass": 1.0,
        "kB": 1.0,
        "step_per_sample": STEP_PER_SAMPLE,
        "steps_per_half": half,
        "prerelax": f"RelaxStructure_NLOPT (L-BFGS, {PRERELAX_EVALS} evaluations) "
        "from a uniform random config; velocities start at 0 and the first "
        "half-kick uses a zero acceleration (D-051)",
        "stage1_equilibration": "Nc_equi times: 2 x [SetRandomSpeed(T): every "
        f"component of every v_i ~ N(0, T), no COM removal; NVE Evolve({half})]",
        "stage2_sample": f"Nc times: AndersonEvolve({half}, T, p = {ANDERSEN_P}), "
        f"then NVE Evolve({half}); then log Phi, KE and save positions",
        "integrator": "velocity Verlet: v += dt a / 2; r += dt v; a = F(r); "
        "v += dt a / 2 (MD_System.cpp:128-166); Andersen resets follow the step",
        "andersen": andersen,
        "com": "never removed: SetRandomSpeed and Andersen draw all N d "
        "components independently, so the COM momentum fluctuates and KE counts "
        "N d degrees of freedom",
        "rng": "one RandomGenerator(seed) (rngGod) drives the initial config, "
        "the velocities and the Andersen draws",
        "nthreads": NTHREADS,
    }


def units_record(par: Dict[str, Any]) -> Dict[str, str]:
    """Return the C++ -> xla units mapping."""
    vol = par["side"] ** DIM
    assert vol > 0.0, "box volume must be positive"
    assert par["k_outer"] > 0.0, "K must be positive"
    return {
        "box": f"C++ random-mode box, side L = 1000^(1/d) = {par['side']!r} (D-004: "
        f"stale num = 1000), V = {vol!r}, density N/V = {N_PARTICLES / vol!r}; "
        "use xla Box(basis = L I) and Cartesian dynamics in it",
        "K": "K = Ka / a, a = (phi_fake / V_d(1))^(1/d) (main.cpp:172, :242) = "
        "xla_cco.kvectors.k_outer_from_ka (bit-equal); same K in the same box",
        "phi": "C++ Energy = sum_{half-space k, w = 1} |rho_k|^2 / V + Shift, "
        "Shift = 0, soft core off (val = 0) = xla StealthyEnergy; no per-particle "
        "normalisation. phi_cpp: harness %.17g; phi_log: 6 digits (D-052)",
        "ke": "KE = sum_i m |v_i|^2 / 2, m = 1 (MD_System.cpp:375); "
        "<KE> = N d T / 2 (no COM removal); 6 digits only",
        "T": "kT = T directly (SetRandomSpeed sigma = sqrt(kT / m))",
        "pressure": "logged 1 + virial / (d V) / (N T) (main.cpp:875)",
    }


def reference_meta(
    par: Dict[str, Any], recs: List[Dict[str, Any]], n_samples: int, n_equil: int
) -> Dict[str, Any]:
    """Return the reference JSON content.

    Parameters
    ----------
    par : Dict[str, Any]
        Output of ``setup_params``.
    recs : List[Dict[str, Any]]
        Per-seed records from ``collect_seed``.
    n_samples, n_equil : int
        Requested Nc and Nc_equi.
    """
    assert len(recs) >= 1, "no runs"
    assert all(r["n_samples"] >= 1 for r in recs), "empty run"
    basis = par["side"] * np.eye(DIM)
    kset = enumerate_kvectors(basis, 0.0, par["k_outer"])
    gate = {
        "rtol": GATE_RTOL,
        "compared": "xla StealthyEnergy vs C++ ShiftedCCPotential (harness) on "
        "every sampled ConfigPack configuration",
        **{
            key: max(r[key] for r in recs)
            for key in ("max_rel_diff_xla_vs_cpp", "max_rel_diff_xla_vs_log")
        },
    }
    return {
        "description": "C++ CollectiveCoordinateMD reference samples (plan T4)",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/cpp_reference/md_reference.py collect",
        **dict(N=N_PARTICLES, d=DIM, M=par["m"], M_target=par["m_target"]),
        **dict(chi_target=CHI_TARGET, chi=par["m"] / (DIM * (N_PARTICLES - 1))),
        **dict(L=par["side"], basis=basis.tolist(), volume=par["side"] ** DIM),
        **dict(density=N_PARTICLES / par["side"] ** DIM, K=par["k_outer"]),
        **dict(K_hex=float.hex(par["k_outer"]), Ka=par["ka"], K1a=K1A, S0=S0),
        **dict(phi_fake=PHI_FAKE, val=VAL, sigma=SIGMA, potential_type=POTENTIAL),
        **dict(T=TEMPERATURE, Nc=n_samples, Nc_equi=n_equil, stdin=par["stdin"]),
        "seeds": [r["seed"] for r in recs],
        "n_vectors": kset.n.tolist(),
        "binary": file_meta(CCO_BINARY),
        "harness": file_meta(HARNESS),
        "units": units_record(par),
        "protocol": protocol_record(),
        "unit_gate": gate,
        "runs": recs,
    }


def write_npz(path: Path, side: float, stored: List[Tuple[int, np.ndarray]]) -> None:
    """Write the first configurations of each seed with their C++ energies."""
    assert len(stored) >= 1, "nothing to store"
    assert side > 0.0, "box side must be positive"
    np.savez_compressed(
        path,
        s=np.concatenate([s for _, s in stored]),
        basis=side * np.eye(DIM),
        seed=np.concatenate([np.full(s.shape[0], sd) for sd, s in stored]),
        sample=np.concatenate([np.arange(s.shape[0]) for _, s in stored]),
    )


def collect(seeds: List[int], root: Path, n_samples: int, n_equil: int) -> None:
    """Gate every run and write ``JSON_OUT`` and ``NPZ_OUT``."""
    assert len(seeds) >= 1, "need at least one seed"
    par = setup_params(n_samples, n_equil)
    recs: List[Dict[str, Any]] = []
    stored: List[Tuple[int, np.ndarray]] = []
    for seed in seeds:
        rec, s = collect_seed(root / f"seed{seed}", par)
        assert rec["max_rel_diff_xla_vs_cpp"] < GATE_RTOL, "unit gate failed"
        recs.append(rec)
        stored.append((seed, s[:N_NPZ_PER_SEED]))
        print(
            f"seed {seed}: n={rec['n_samples']} "
            f"gate={rec['max_rel_diff_xla_vs_cpp']:.2e} "
            f"log_g6={rec['log_matches_cpp_g6']}/{rec['n_samples']}",
            flush=True,
        )
    JSON_OUT.write_text(
        json.dumps(reference_meta(par, recs, n_samples, n_equil), indent=1) + "\n"
    )
    write_npz(NPZ_OUT, par["side"], stored)
    print(f"wrote {JSON_OUT} and {NPZ_OUT}", flush=True)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C++ MD reference data (T4)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ----- run -----
    p_run = sub.add_parser("run", help="run the binary for one seed")
    p_run.add_argument("--seed", type=int, required=True)
    p_run.add_argument("--hours", type=int, default=7, help="argv[1] time limit")

    # ----- collect -----
    p_col = sub.add_parser("collect", help="gate the runs, write JSON + npz")
    p_col.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])

    # ----- shared -----
    for p in (p_run, p_col):
        p.add_argument("--run-root", type=Path, default=RUN_ROOT)
        p.add_argument("--nc", type=int, default=NC)
        p.add_argument("--nc-equi", type=int, default=NC_EQUI)
    args = parser.parse_args()

    # =========================================================
    # DISPATCH
    # =========================================================
    print("cmd: " + " ".join(sys.argv), flush=True)
    if args.cmd == "run":
        run_seed(args.seed, args.hours, args.nc, args.nc_equi, args.run_root)
    else:
        collect(args.seeds, args.run_root, args.nc, args.nc_equi)

    print("Done.", flush=True)
