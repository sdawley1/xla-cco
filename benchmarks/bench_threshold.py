#!/usr/bin/env python3

"""
GPU/CPU grid for the tables-vs-trig rule: L-BFGS ``relax`` over chi, d=3, N=100.

For each chi the k-set gives (M, max|n|); trig (``n_max=None``) and tables
(``n_max=max|n|``) are timed warm ``--reps`` times on the same configurations,
and per-evaluation times are reported because eval counts can differ.

    python benchmarks/bench_threshold.py --device gpu \
        --out benchmarks/results_threshold_gpu.json

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
import time
from typing import Any, Dict, List

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco._device import require_gpu
from xla_cco.box import cubic_box
from xla_cco.energy import make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors, k_outer_from_ka, ka_from_chi
from xla_cco.minimize import relax
from xla_cco.minimize.lbfgs import LBFGSSettings

# ============================================================
# MEASUREMENT
# ============================================================


def time_relax(energy: Any, s: jax.Array, evals: int, reps: int) -> Dict[str, Any]:
    """Return cold time, warm times, eval count and per-eval times of lbfgs."""
    assert reps >= 1, "reps must be positive"
    cfg = LBFGSSettings(max_evals_per_pass=evals)
    t0 = time.perf_counter()
    res = jax.block_until_ready(relax(energy, s, cfg))
    cold = time.perf_counter() - t0
    warm: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        res = jax.block_until_ready(relax(energy, s, cfg))
        warm.append(time.perf_counter() - t0)
    n_evals = int(jnp.sum(res.n_evals))
    return {
        "cold_s": cold,
        "warm_s": warm,
        "n_evals": n_evals,
        "us_per_eval": [1e6 * w / n_evals for w in warm],
    }


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--evals", type=int, default=200)
    ap.add_argument(
        "--chis", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.45]
    )
    cli = ap.parse_args()
    if cli.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    else:
        try:
            require_gpu()
        except RuntimeError as exc:
            raise SystemExit(f"ERROR: --device gpu: {exc}") from exc
    if (jax.devices()[0].platform == "cpu") != (cli.device == "cpu"):
        raise SystemExit(
            f"ERROR: --device {cli.device}, JAX default is {jax.devices()[0].platform}"
        )
    rows: List[Dict[str, Any]] = []
    s = jnp.asarray(np.random.default_rng(0).random((cli.batch, cli.n, 3)))
    box = cubic_box(cli.n, 3)
    for chi in cli.chis:
        k_outer = k_outer_from_ka(ka_from_chi(chi, 0.05, 3), 0.05, 3)
        kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k_outer)
        base = make_stealthy_energy(box, kset)
        nmax = int(np.abs(kset.n).max())
        row: Dict[str, Any] = {"chi": chi, "m": int(kset.n.shape[0]), "max_n": nmax}
        for name, e in (
            ("trig", dataclasses.replace(base, n_max=None)),
            ("tables", dataclasses.replace(base, n_max=nmax)),
        ):
            row[name] = time_relax(e, s, cli.evals, cli.reps)
        t_w, g_w = min(row["trig"]["warm_s"]), min(row["tables"]["warm_s"])
        row["speedup_wall"] = t_w / g_w
        row["speedup_per_eval"] = min(row["trig"]["us_per_eval"]) / min(
            row["tables"]["us_per_eval"]
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(cli.out, "w", encoding="utf-8") as fh:
        json.dump(
            {"device": jax.devices()[0].device_kind, "batch": cli.batch, "rows": rows},
            fh,
            indent=1,
        )
    print(f"wrote {cli.out}", flush=True)
