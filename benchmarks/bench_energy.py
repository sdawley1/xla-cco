#!/usr/bin/env python3

"""
Benchmark ``StealthyEnergy.value_and_grad`` and ``relax`` on CPU or GPU.

Cases: (A) d=3, N=100, B=4000 configs, chi in {0.05, 0.45}; (B) d=2, N=10^4,
B=1, chi=0.45 (chunked). Variants: ``baseline`` (per-(k, j) trig, ``n_max=None``)
and ``tables`` (per-axis phase tables), each unchunked or with an automatic
``chunk_size``. Every (case, variant) runs in a fresh subprocess so the peak
memory is its own. Writes machine-readable JSON.

    JAX_PLATFORMS=cpu python benchmarks/bench_energy.py --device cpu \
        --out benchmarks/results_cpu.json

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
import resource
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.box import cubic_box
from xla_cco.energy import StealthyEnergy, auto_chunk_size, make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors, k_outer_from_ka, ka_from_chi
from xla_cco.minimize import relax
from xla_cco.minimize.descent import DescentSettings
from xla_cco.minimize.lbfgs import LBFGSSettings

# ============================================================
# CONSTANTS
# ============================================================

PHI: float = 0.05
BUDGET_BYTES: int = 1 << 30  # work-array budget for the auto chunk (whole batch)
VARIANTS: Tuple[str, ...] = (
    "baseline",
    "baseline_autochunk",
    "tables",
    "tables_autochunk",
)
# (case, d, N, B, chi); case B is chunked for every variant.
CASES: Tuple[Tuple[str, int, int, int, float], ...] = (
    ("A", 3, 100, 4000, 0.05),
    ("A", 3, 100, 4000, 0.45),
    ("B", 2, 10_000, 1, 0.45),
)

# ============================================================
# BUILDING BLOCKS
# ============================================================


def build_case(
    d: int, n: int, chi: float, batch: int, variant: str, seed: int = 0
) -> Tuple[StealthyEnergy, jax.Array, int]:
    """Return (energy, seeded Poisson configs (B, N, d), real M) for a variant."""
    assert variant in VARIANTS, f"unknown variant {variant}"
    assert batch >= 1 and n >= 1, "sizes must be positive"
    box = cubic_box(n, d)
    k_outer = k_outer_from_ka(ka_from_chi(chi, PHI, d), PHI, d)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k_outer)
    m = int(kset.n.shape[0])
    chunk: Optional[int] = None
    if variant.endswith("autochunk") or (batch == 1 and n >= 10_000):
        chunk = auto_chunk_size(n, m, BUDGET_BYTES // batch)
    energy = make_stealthy_energy(box, kset, chunk_size=chunk)
    # force the path: the factory's own rule (TABLE_MIN_M) is not under test here
    n_max = None if variant.startswith("baseline") else int(np.abs(kset.n).max())
    energy = dataclasses.replace(energy, n_max=n_max)
    s = jnp.asarray(np.random.default_rng(seed).random((batch, n, d)))
    return energy, s, m


def peak_memory_mb() -> float:
    """Return peak device memory (GPU) or peak RSS (CPU) in MiB."""
    dev = jax.devices()[0]
    stats = dev.memory_stats() if dev.platform != "cpu" else None
    if stats:
        return float(stats["peak_bytes_in_use"]) / 2**20
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def time_call(fn: Callable[[], Any], reps: int) -> Tuple[float, float, Any]:
    """Return (first-call seconds, best of ``reps`` seconds, last output)."""
    assert reps >= 1, "reps must be positive"
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    first = time.perf_counter() - t0
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - t0)
    return first, best, out


# ============================================================
# MEASUREMENTS
# ============================================================


def bench_value_and_grad(
    d: int, n: int, chi: float, batch: int, variant: str, reps: int
) -> Dict[str, Any]:
    """Time jitted vmapped ``value_and_grad`` (configs/s), compile time, memory."""
    energy, s, m = build_case(d, n, chi, batch, variant)
    fn = eqx.filter_jit(
        eqx.filter_vmap(lambda e, x: e.value_and_grad(x), in_axes=(None, 0))
    )
    first, best, (val, grad) = time_call(lambda: fn(energy, s), reps)
    return {
        "kind": "value_and_grad",
        "m": m,
        "m_padded": int(energy.n_vectors.shape[0]),
        "n_max": int(np.max(np.abs(np.asarray(energy.n_vectors)))),
        "chunk_size": energy.chunk_size,
        "compile_s": max(first - best, 0.0),
        "run_s": best,
        "configs_per_s": batch / best,
        "peak_mem_mb": peak_memory_mb(),
        "energy_mean": float(jnp.mean(val)),
        "grad_absmax": float(jnp.max(jnp.abs(grad))),
    }


def bench_relax(
    d: int, n: int, chi: float, batch: int, variant: str, kind: str, evals: int
) -> Dict[str, Any]:
    """Time ``relax`` (cold and warm) and report evaluations per second."""
    energy, s, m = build_case(d, n, chi, batch, variant)
    if kind == "relax_lbfgs":
        cfg: Any = LBFGSSettings(max_evals_per_pass=evals)
    else:
        cfg = DescentSettings(max_evals=400)
    t0 = time.perf_counter()
    res = jax.block_until_ready(relax(energy, s, cfg))
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = jax.block_until_ready(relax(energy, s, cfg))
    warm = time.perf_counter() - t0
    n_evals = int(jnp.sum(res.n_evals))
    return {
        "kind": kind,
        "m": m,
        "relax_batch": batch,
        "cold_s": cold,
        "warm_s": warm,
        "n_evals": n_evals,
        "evals_per_s": n_evals / warm,
        "energy_mean": float(jnp.mean(res.energy)),
        "peak_mem_mb": peak_memory_mb(),
    }


def run_one(spec: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run one (case, variant, kind) measurement described by ``spec``."""
    assert spec["variant"] in VARIANTS, "unknown variant"
    d, n, chi = spec["d"], spec["n"], spec["chi"]
    if spec["kind"] == "value_and_grad":
        out = bench_value_and_grad(d, n, chi, spec["batch"], spec["variant"], args.reps)
    else:
        out = bench_relax(
            d, n, chi, args.relax_batch, spec["variant"], spec["kind"], args.relax_evals
        )
    return {**spec, **out}


# ============================================================
# DRIVER
# ============================================================


def hardware_info() -> Dict[str, Any]:
    """Return CPU model, core counts, device, jax version and date."""
    model = "unknown"
    with open("/proc/cpuinfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    dev = jax.devices()[0]
    return {
        "cpu_model": model,
        "cpu_cores_visible": len(os.sched_getaffinity(0)),
        "cpu_cores_total": os.cpu_count(),
        "device": dev.device_kind,
        "platform": dev.platform,
        "jax": jax.__version__,
        "date": datetime.date.today().isoformat(),
    }


def make_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Return the measurement list for the selected sizes."""
    specs: List[Dict[str, Any]] = []
    for case, d, n, b, chi in CASES:
        if case == "B" and args.skip_large:
            continue
        n_use = args.small_n if args.small and case == "B" else n
        b_use = args.batch if case == "A" else b
        variants = VARIANTS if case == "A" else ("baseline", "tables")
        for variant in variants:
            kinds = ["value_and_grad"]
            if case == "A" and variant in ("baseline", "tables"):
                kinds += ["relax_lbfgs", "descent_400"]
            for kind in kinds:
                specs.append(
                    dict(
                        case=case,
                        d=d,
                        n=n_use,
                        batch=b_use,
                        chi=chi,
                        variant=variant,
                        kind=kind,
                    )
                )
    return specs


def run_isolated(spec: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run ``spec`` in a fresh subprocess and parse its JSON line."""
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)]
    cmd += ["--reps", str(args.reps), "--relax-batch", str(args.relax_batch)]
    cmd += ["--relax-evals", str(args.relax_evals), "--device", args.device]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed for {spec}:\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


def check_device(device: str) -> None:
    """Fail fast: a requested GPU run never silently falls back to the CPU."""
    platform = jax.devices()[0].platform
    if device == "gpu" and platform == "cpu":
        print(
            "ERROR: --device gpu but JAX sees only the CPU.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    if device == "cpu" and platform != "cpu":
        print(
            "ERROR: --device cpu but JAX default is a GPU.", file=sys.stderr, flush=True
        )
        raise SystemExit(1)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--batch", type=int, default=4000, help="case A batch size")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--relax-batch", type=int, default=256)
    ap.add_argument("--relax-evals", type=int, default=200, help="max_evals_per_pass")
    ap.add_argument("--skip-large", action="store_true", help="skip case B")
    ap.add_argument("--small", action="store_true", help="tiny sizes (smoke test)")
    ap.add_argument("--small-n", type=int, default=200)
    ap.add_argument("--no-isolate", action="store_true", help="run in-process")
    ap.add_argument("--worker", default=None, help="internal: one JSON spec")
    cli = ap.parse_args()
    if cli.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    check_device(cli.device)
    if cli.worker is not None:
        print(json.dumps(run_one(json.loads(cli.worker), cli)), flush=True)
        raise SystemExit(0)
    results: List[Dict[str, Any]] = []
    for item in make_specs(cli):
        row = run_one(item, cli) if cli.no_isolate else run_isolated(item, cli)
        results.append(row)
        print(json.dumps(row), flush=True)
    payload = {"hardware": hardware_info(), "args": vars(cli), "results": results}
    if cli.out:
        with open(cli.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print(f"wrote {cli.out}", flush=True)
