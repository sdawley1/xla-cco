#!/usr/bin/env python3

"""
Low-temperature MD production jobs of the winding experiment (plan E1, E1b, E4).

One invocation runs one cell (start, chi) through ``python -m xla_cco.cli.md``
in-process; this script only assembles the CLI arguments from the plan and
the V3 choices in ``experiments/md_winding/bench.json`` ("chosen"), and
annotates the CLI's JSON with provenance and the SLURM cost. Steps:

- ``run`` (E1): ground states 0-9 of ``flows_chi<tag>.npz`` s_final x
  T in {1e-6, 1e-5, 1e-4, 1e-3} (40 lanes, ground state major); harmonic
  start, count-mode Andersen (p = 0.01) for 2e5 steps with the <Phi>/(M T)
  gate (5 %, one extension), then 2 cycles of [Andersen 1e5, COM removal,
  NVE 1e6] with all waves |k| < 2K wound; 200 log-spaced snapshots; the
  uniform rho trace of Amendment A4 (>= 500 rows per NVE segment).
  Output ``<sys>/N<N>/md_chi<tag>``.
- ``spectra`` (E1b): ground states 0-1, T = 1e-6, constrained waves only
  (--k-track K), one NVE segment of t = 200, rho traced every 5 steps.
  Output ``md_spectra_chi<tag>``.
- ``langevin`` (E4): flows2d and hip2d, ground states 0-4, T in {1e-6,
  1e-4}, BAOAB throughout at --gamma (0.003 or 0.03), same dt, cadence
  and trace as E1. Output ``md_langevin_g<gamma>_chi<tag>``.
- ``cost``: refresh the ``cost`` entry of a finished cell's JSON from
  sacct (the job's final ElapsedRaw; no JAX import).

dt, wind_every and drift_tol per chi come from bench.json; the trace
cadence is n_nve // 500 and its capacity covers the extended protocol, as
V3 sized it. Every cell checkpoints to ``<output>.ckpt.npz`` (every 30 min
and after each phase) and resumes when rerun with the same arguments. The
SLURM jobs that worked on a cell are listed in ``<output>.jobs.json``; the
cost is GPU-h = ElapsedRaw x n_gpu / 3600 summed over them (sacct). Any
protocol option changed from the plan default is recorded under
``run_script.plan_deviations``.

Usage (from the repository root; SLURM: batten2008_md_gpu.slurm STEP=run):
  python batten2008_md_run.py --step run --system flows2d --chi 1-10 \
      --data-root /scratch/.../xla-cco/experiments/trajectories

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# CONSTANTS
# ============================================================

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO / "experiments" / "trajectories"
DEFAULT_BENCH = REPO / "experiments" / "md_winding" / "bench.json"
SYSTEMS: Dict[str, str] = {
    "flows2d": "N1024", "gau2d": "N1000", "url2d": "N1024", "rsa2d": "N999",
    "hip2d": "N1000",
}  # fmt: skip
CHI_TAGS: Tuple[str, ...] = ("1-10", "2-5")
STEPS: Tuple[str, ...] = ("run", "spectra", "langevin", "cost")
LANGEVIN_SYSTEMS: Tuple[str, ...] = ("flows2d", "hip2d")
LANGEVIN_GAMMAS: Tuple[float, ...] = (0.003, 0.03)
# ----- plan protocol (E1 / E1b / E4) -----
TEMPS_RUN: Tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3)
TEMPS_SPECTRA: Tuple[float, ...] = (1e-6,)
TEMPS_LANGEVIN: Tuple[float, ...] = (1e-6, 1e-4)
N_GS: Dict[str, int] = {"run": 10, "spectra": 2, "langevin": 5}
N_EQUIL: int = 200_000
N_ANDERSEN: int = 100_000
CYCLES: Dict[str, int] = {"run": 2, "spectra": 1, "langevin": 2}
N_NVE_RUN: int = 1_000_000
N_SNAPSHOTS: int = 200
TRACE_SAMPLES: int = 500  # Amendment A4: uniform rho rows per production segment
SPECTRA_TIME: float = 200.0  # E1b: NVE length (time units)
SPECTRA_EVERY: int = 5  # E1b: rho trace cadence (steps)
ANDERSEN_P: float = 0.01
CHOSEN_KEYS: Tuple[str, ...] = ("dt", "wind_every", "drift_tol", "k_chunk")
SACCT_FORMAT: str = "JobID,JobName,ElapsedRaw,AllocTRES,State"
MAX_JOBS: int = 1000  # bound of the per-cell job list

# ============================================================
# CELL LAYOUT AND SETTINGS
# ============================================================


def cell_dir(root: Path, system: str) -> Path:
    """``<root>/<system>/N<N>`` (ValueError on an unknown start)."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; one of {sorted(SYSTEMS)}")
    out = Path(root) / system / SYSTEMS[system]
    assert out.name.startswith("N"), "N<N> directory"
    return out


def output_stem(step: str, tag: str, gamma: Optional[float]) -> str:
    """md_chi<tag> (run), md_spectra_chi<tag>, md_langevin_g<gamma:g>_chi<tag>."""
    if tag not in CHI_TAGS:
        raise ValueError(f"unknown chi tag {tag!r}; one of {CHI_TAGS}")
    if step == "langevin":
        if gamma is None or not gamma > 0.0:
            raise ValueError("--step langevin needs --gamma > 0")
        out = f"md_langevin_g{gamma:g}_chi{tag}"
    elif step in ("run", "cost"):
        out = f"md_chi{tag}"
    elif step == "spectra":
        out = f"md_spectra_chi{tag}"
    else:
        raise ValueError(f"unknown step {step!r}; one of {STEPS}")
    assert out.startswith("md_") and out.endswith(f"_chi{tag}"), "analysis naming"
    return out


def cell_seed(step: str, system: str, tag: str, gamma: Optional[float]) -> int:
    """Distinct deterministic seed per (step, system, chi, gamma) cell."""
    assert step in STEPS and system in SYSTEMS, "known step and system"
    assert tag in CHI_TAGS, "known chi tag"
    g = 0 if gamma is None else 1 + LANGEVIN_GAMMAS.index(gamma)
    out = 1 + CHI_TAGS.index(tag) + 10 * list(SYSTEMS).index(system)
    return out + 100 * STEPS.index(step) + 1000 * g


def load_chosen(bench: Path, tag: str) -> Dict[str, Any]:
    """The V3 production settings of one chi tag (bench.json "chosen")."""
    data = json.loads(Path(bench).read_text())
    chosen = data.get("chosen", {}).get(tag)
    if not isinstance(chosen, dict):
        raise ValueError(f"{bench} has no chosen settings for chi tag {tag!r}")
    missing = [k for k in CHOSEN_KEYS if k not in chosen]
    if missing:
        raise ValueError(f"{bench} chosen[{tag!r}] lacks {missing}")
    assert float(chosen["dt"]) > 0.0, "positive dt"
    assert int(chosen["wind_every"]) >= 1, "wind_every >= 1"
    return chosen


def trace_records(
    n_equil: int, n_andersen: int, n_nve: int, cycles: int, every: int, baoab: bool
) -> int:
    """Trace rows covering the extended protocol (two equilibrations).

    The trace takes a row at every step divisible by ``every`` (global), so
    it needs total // every rows, the same rule V3 used to size memory.
    """
    assert min(n_equil, n_andersen) >= 0 and n_nve >= 1, "phase lengths"
    assert cycles >= 1 and every >= 1, "cycles and cadence"
    per_cycle = n_nve + (0 if baoab else n_andersen)
    total = 2 * n_equil + cycles * per_cycle
    return max(1, total // every)


def protocol(args: argparse.Namespace) -> Dict[str, Any]:
    """Protocol of the step: plan defaults, each overridable by an option."""
    step = args.step
    n_nve_default = N_NVE_RUN
    if step == "spectra":
        n_nve_default = int(round(SPECTRA_TIME / float(args.chosen["dt"])))
    defaults: Dict[str, Any] = {
        "n_gs": N_GS[step], "n_equil": N_EQUIL, "n_andersen": N_ANDERSEN,
        "n_nve": n_nve_default, "cycles": CYCLES[step], "n_snapshots": N_SNAPSHOTS,
        "temperature": list({"run": TEMPS_RUN, "spectra": TEMPS_SPECTRA,
                             "langevin": TEMPS_LANGEVIN}[step]),
    }  # fmt: skip
    out = {k: (v if getattr(args, k) is None else getattr(args, k))
           for k, v in defaults.items()}  # fmt: skip
    out["deviations"] = {k: {"plan": defaults[k], "used": out[k]}
                         for k in defaults if out[k] != defaults[k]}  # fmt: skip
    assert out["n_nve"] >= 1 and out["n_gs"] >= 1, "non-empty protocol"
    assert len(out["temperature"]) >= 1, "at least one temperature"
    return out


def cli_argv(args: argparse.Namespace, prefix: Path, k_outer: float) -> List[str]:
    """Arguments of ``python -m xla_cco.cli.md`` for one cell.

    Parameters
    ----------
    args : argparse.Namespace
        Run-script arguments, with ``chosen`` (bench.json settings) and
        ``proto`` (``protocol(args)``) attached.
    prefix : Path
        Output prefix (``<cell>/<output_stem>``); the checkpoint is
        ``<prefix>.ckpt.npz``.
    k_outer : float
        Constraint radius K (from the input's JSON); --k-track K for spectra.

    Returns
    -------
    List[str]
        The CLI argument vector.
    """
    ch, pr, step = args.chosen, args.proto, args.step
    baoab = step == "langevin"
    every = SPECTRA_EVERY if step == "spectra" else max(1, pr["n_nve"] // TRACE_SAMPLES)
    cap = trace_records(pr["n_equil"], pr["n_andersen"], pr["n_nve"], pr["cycles"],
                        every, baoab)  # fmt: skip
    argv = [
        "--input", str(prefix.parent / f"flows_chi{args.chi}.npz"),
        "--indices", *[str(i) for i in range(pr["n_gs"])],
        "--temperature", *[repr(float(t)) for t in pr["temperature"]],
        "--dt", repr(float(ch["dt"])), "--n-equil", str(pr["n_equil"]),
        "--n-andersen", str(pr["n_andersen"]), "--n-nve", str(pr["n_nve"]),
        "--cycles", str(pr["cycles"]), "--andersen-p", repr(ANDERSEN_P),
        "--andersen-mode", "count",
        "--wind-every", "1" if step == "spectra" else str(int(ch["wind_every"])),
        "--n-snapshots", str(pr["n_snapshots"]),
        "--record-rho-every", str(every), "--max-rho-records", str(cap),
        "--drift-tol", repr(float(ch["drift_tol"])),
        "--seed", str(cell_seed(step, args.system, args.chi, args.gamma)),
        "--save", str(prefix), "--checkpoint", str(prefix) + ".ckpt.npz",
        "--device", args.device,
    ]  # fmt: skip
    if baoab:
        argv += ["--scheme", "baoab", "--gamma", repr(float(args.gamma))]
    if step == "spectra":
        argv += ["--k-track", repr(float(k_outer))]
    if ch["k_chunk"] not in (None, "none"):
        argv += ["--k-chunk", str(int(ch["k_chunk"]))]
    assert argv.count("--save") == 1 and every >= 1, "one output"
    return argv


def check_input(cell: Path, system: str, tag: str) -> float:
    """K of the cell's flows_chi<tag>.json; ValueError if N disagrees with N<N>."""
    meta_path = cell / f"flows_chi{tag}.json"
    if not meta_path.is_file():
        raise ValueError(f"missing input metadata {meta_path}")
    meta = json.loads(meta_path.read_text())
    n_dir = int(SYSTEMS[system][1:])
    if int(meta.get("N", -1)) != n_dir or "K" not in meta:
        raise ValueError(f"{meta_path}: N = {meta.get('N')} (want {n_dir}) or no K")
    k_outer = float(meta["K"])
    assert k_outer > 0.0, "positive K"
    if not (cell / f"flows_chi{tag}.npz").is_file():
        raise ValueError(f"missing input {cell / f'flows_chi{tag}.npz'}")
    return k_outer


# ============================================================
# JOBS AND COST (SLURM)
# ============================================================


def job_record(env: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """This SLURM job (id, name, host, start time), or None outside SLURM."""
    job = env.get("SLURM_JOB_ID")
    if not job:
        return None
    out = {"job_id": job, "job_name": env.get("SLURM_JOB_NAME", ""),
           "host": socket.gethostname(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}  # fmt: skip
    assert out["job_id"].isdigit(), f"SLURM_JOB_ID {job!r} is not numeric"
    return out


def append_job(path: Path, record: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append ``record`` (if any, once) to the cell's job list; return the list."""
    jobs: List[Dict[str, Any]] = []
    if path.is_file():
        jobs = list(json.loads(path.read_text()))
    if record is not None and record["job_id"] not in {j["job_id"] for j in jobs}:
        jobs.append(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs, indent=1) + "\n")
        os.replace(tmp, path)
    assert len(jobs) <= MAX_JOBS, "unbounded job list"
    return jobs


def gpus_of(tres: str) -> int:
    """Number of GPUs in a sacct AllocTRES string (``gres/gpu=N``; 0 if absent)."""
    assert isinstance(tres, str), "AllocTRES is a string"
    count = 0
    for item in tres.split(","):
        key, _, val = item.partition("=")
        if key.strip() == "gres/gpu" and val.strip().isdigit():
            count = int(val)
    assert count >= 0, "non-negative GPU count"
    return count


def parse_sacct(text: str) -> List[Dict[str, Any]]:
    """Rows of ``sacct -X -n -P --format=JobID,JobName,ElapsedRaw,AllocTRES,State``."""
    rows: List[Dict[str, Any]] = []
    for line in text.strip().splitlines()[:MAX_JOBS]:
        parts = line.split("|")
        if len(parts) < 5 or not parts[2].strip().isdigit():
            continue
        elapsed, gpus = int(parts[2]), gpus_of(parts[3])
        rows.append({"job_id": parts[0], "job_name": parts[1], "elapsed_s": elapsed,
                     "alloc_tres": parts[3], "state": parts[4], "n_gpu": gpus,
                     "gpu_hours": elapsed * gpus / 3600.0})  # fmt: skip
    assert all(r["gpu_hours"] >= 0.0 for r in rows), "non-negative cost"
    return rows


def sacct_cost(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """GPU-h of the cell's jobs from sacct (``available: False`` if it fails)."""
    ids = [str(j["job_id"]) for j in jobs]
    assert len(ids) <= MAX_JOBS, "bounded job list"
    if not ids:
        return {"available": False, "reason": "no SLURM job recorded"}
    cmd = ["sacct", "-X", "-n", "-P", "-j", ",".join(ids), f"--format={SACCT_FORMAT}"]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                              check=True)  # fmt: skip
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"sacct failed: {exc}"}
    rows = parse_sacct(done.stdout)
    final = all(r["state"] not in ("RUNNING", "PENDING") for r in rows)
    out = {"available": True, "queried": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "jobs": rows, "gpu_hours": sum(r["gpu_hours"] for r in rows),
           "final": final}  # fmt: skip
    assert out["gpu_hours"] >= 0.0, "non-negative cost"
    return out


def annotate(json_path: Path, extra: Dict[str, Any]) -> None:
    """Merge ``extra`` into the CLI's JSON (atomic rewrite)."""
    assert extra, "nothing to add"
    meta = json.loads(Path(json_path).read_text())
    assert isinstance(meta, dict) and "npz_keys" in meta, "a cli.md summary"
    meta.update(extra)
    tmp = Path(json_path).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp, json_path)


# ============================================================
# PROVENANCE AND DRIVER
# ============================================================


def file_sha256(path: Path) -> str:
    """SHA-256 (hex) of a small file."""
    out = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert len(out) == 64, "sha256 hex digest"
    return out


def git_head() -> str:
    """Commit of the checkout this script runs from ('' if git is unavailable)."""
    try:
        done = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return ""
    out = done.stdout.strip() if done.returncode == 0 else ""
    assert out == "" or len(out) == 40, "a full commit hash"
    return out


def validate(args: argparse.Namespace) -> None:
    """Raise ValueError on a cell that the plan does not contain."""
    if args.chi not in CHI_TAGS or args.system not in SYSTEMS:
        raise ValueError(f"cell ({args.system}, {args.chi}) is not in the plan")
    if args.step == "langevin":
        if args.system not in LANGEVIN_SYSTEMS or args.gamma not in LANGEVIN_GAMMAS:
            raise ValueError(
                f"E4 runs {LANGEVIN_SYSTEMS} with gamma in {LANGEVIN_GAMMAS}, "
                f"got {args.system}, {args.gamma}"
            )
    elif args.gamma is not None:
        raise ValueError("--gamma applies to --step langevin only")
    assert args.step in STEPS, "known step"
    assert args.device in ("gpu", "cpu"), "known device"


def run_cell(args: argparse.Namespace) -> int:
    """Run one cell through cli.md; annotate its JSON; return the exit code."""
    from xla_cco.cli import md  # JAX only for the MD steps

    validate(args)
    prefix = cell_dir(args.data_root, args.system) / output_stem(
        args.step, args.chi, args.gamma)  # fmt: skip
    k_outer = check_input(prefix.parent, args.system, args.chi)
    args.chosen = load_chosen(args.bench, args.chi)
    args.proto = protocol(args)
    argv = cli_argv(args, prefix, k_outer)
    print(f"xla_cco from {Path(md.__file__).resolve().parents[1]}", flush=True)
    print("cli.md " + " ".join(argv), flush=True)
    if args.dry_run:
        return 0
    jobs = append_job(Path(str(prefix) + ".jobs.json"), job_record(dict(os.environ)))
    code = md.main(argv)
    if code != 0:
        return code
    annotate(Path(str(prefix) + ".json"), {
        "run_script": {
            "step": args.step, "system": args.system, "chi_tag": args.chi,
            "gamma": args.gamma, "bench": str(args.bench),
            "bench_sha256": file_sha256(args.bench), "chosen": args.chosen,
            "cli_argv": argv, "plan_deviations": args.proto["deviations"],
            "git_head": git_head(), "xla_cco": str(Path(md.__file__).parent),
        },
        "jobs": jobs, "cost": sacct_cost(jobs),
    })  # fmt: skip
    assert Path(str(prefix) + ".npz").is_file(), "the CLI wrote its npz"
    return 0


def refresh_cost(args: argparse.Namespace) -> int:
    """--step cost: rewrite ``cost`` of every finished output of the cell."""
    root = cell_dir(args.data_root, args.system)
    paths = sorted(root.glob(f"md*_chi{args.chi}.jobs.json"))
    assert args.chi in CHI_TAGS, "known chi tag"
    for jp in paths[:MAX_JOBS]:
        out = Path(str(jp)[: -len(".jobs.json")] + ".json")
        if not out.is_file():
            print(f"{out} not written yet; skipped", flush=True)
            continue
        cost = sacct_cost(append_job(jp, None))
        annotate(out, {"cost": cost})
        print(f"{out}: {cost.get('gpu_hours')} GPU-h", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Command line of the run script."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    # ----- required -----
    p.add_argument("--step", choices=STEPS, required=True)
    p.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    p.add_argument("--chi", choices=CHI_TAGS, required=True)
    # ----- optional -----
    p.add_argument("--gamma", type=float, default=None, help="E4 friction")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--dry-run", action="store_true", help="print the CLI call only")
    # ----- protocol overrides (recorded as plan deviations) -----
    p.add_argument("--n-gs", type=int, default=None)
    p.add_argument("--temperature", type=float, nargs="+", default=None)
    for name in ("n-equil", "n-andersen", "n-nve", "cycles", "n-snapshots"):
        p.add_argument(f"--{name}", type=int, default=None)
    assert p.get_default("device") == "gpu", "GPU unless asked otherwise"
    assert p.get_default("n_nve") is None, "plan default unless overridden"
    return p


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parsed = build_parser().parse_args()

    # =========================================================
    # RUN
    # =========================================================
    try:
        rc = refresh_cost(parsed) if parsed.step == "cost" else run_cell(parsed)
    except (ValueError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr, flush=True)
        rc = 1
    print("Done." if rc == 0 else f"exit {rc}", flush=True)
    sys.exit(rc)
