#!/usr/bin/env python3

"""
Command line: python -m xla_cco.cli.relax (mirrors C++ ground/input mode).

Reads a ConfigPack (``--load``), relaxes every configuration to a stealthy
ground state and writes, for prefix ``SAVE``:

- ``SAVE.ConfigPack``, ``SAVE_InitConfig.ConfigPack`` and
  ``SAVE_Success.ConfigPack`` (energy < ``--tol``), with headers written as C++
  writes them (cell size N^(1/d), basis = input basis * N^(1/d));
- ``SAVE_Sk.txt``, ``SAVE_g2.txt`` and ``SAVE_Hp.txt`` for the Success set, in
  the C++ four-column layout (``WriteFunction``);
- ``SAVE_energies.npy`` (final energy per configuration), ``SAVE_log.json``
  (the ``engine.log`` equivalent) and, if recording is on, ``SAVE_traj.npz``.

Table columns (PairCorrelation.cpp):

- g2: ``r_mid, g2(r), half-width, g2/sqrt(count)`` (l.186-205);
- Hp: ``r_mid, count/total/width, half-width, p/sqrt(count)`` (l.480-501).

DIFFERENCE FROM C++: C++ builds adaptive (data-driven) bins for g2 and Hp; they
are not reproducible, so this driver uses a FIXED uniform grid instead (g2 on
[0, 0.2 L] with ``--g2-bins`` bins; Hp on [0, max nearest-neighbour distance]
with ``--hp-bins`` bins; Hp keeps ALL bins, whereas C++ drops its last bin).
S(k) uses the C++ binning (``IsotropicStructureFactor``
with 5, 400). The input basis must be unit-volume (C++ multiplies by N^(1/d)).

Success is judged only by energy < ``--tol`` (``n_success`` in the log). The L-BFGS
status ``STATUS_LINESEARCH_FAIL`` with an energy far below tol is the normal way
L-BFGS ends at roundoff; it is not a failure.

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
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

# ----- numerics -----
import jax
import numpy as np

# ----- project -----
from xla_cco._device import require_gpu
from xla_cco.analysis.pairs import (
    g2_from_histogram,
    nearest_neighbor_distances,
    pair_distance_histogram,
)
from xla_cco.analysis.structure import isotropic_structure_factor
from xla_cco.box import Box, cubic_box, wrap_fractional
from xla_cco.energy import make_stealthy_energy
from xla_cco.io.configpack import (
    PackedConfig,
    read_configpack,
    stack_fractional,
    write_configpack,
)
from xla_cco.io.tables import write_function_table
from xla_cco.kvectors import chi_from_count, enumerate_kvectors, k_outer_from_ka
from xla_cco.minimize import relax
from xla_cco.minimize import result as result_mod
from xla_cco.minimize.descent import DescentSettings
from xla_cco.minimize.lbfgs import LBFGSSettings

Settings = Union[DescentSettings, LBFGSSettings]

# ============================================================
# CONSTANTS
# ============================================================

CPP_PARITY_TOL: float = 9.99999968e-21  # float(np.float32(1e-20)), see D-002
UNIT_VOLUME_ATOL: float = 1e-12
G2_CUTOFF_FRACTION: float = 0.2  # C++ main.cpp: 0.2 * L
SK_K_DIR_MAX: float = 5.0  # C++ main.cpp: IsotropicStructureFactor(..., 5, 400)
SK_K_MAX: float = 400.0
STATUS_NAMES: Dict[int, str] = {
    getattr(result_mod, name): name
    for name in dir(result_mod)
    if name.startswith("STATUS_")
}

# ============================================================
# ARGUMENTS
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the relaxation driver."""
    p = argparse.ArgumentParser(
        prog="python -m xla_cco.cli.relax",
        description=(
            "Relax ConfigPack configurations to stealthy ground states "
            "(C++ ground/input mode). NOTE: C++ uses adaptive bins for g2 and "
            "Hp; here both are on fixed uniform grids (--g2-bins, --hp-bins), and "
            "Hp keeps all bins while C++ drops its last bin. Success is judged "
            "only by energy < --tol (see n_success in SAVE_log.json); status "
            "STATUS_LINESEARCH_FAIL with energy far below tol is the normal "
            "end of L-BFGS at roundoff, not a failure."
        ),
    )
    # ----- physics -----
    p.add_argument("--dim", type=int, required=True, help="dimension d (1-3)")
    p.add_argument("--ka", type=float, required=True, help="K2 * a (outer radius)")
    p.add_argument("--phi", type=float, required=True, help="packing fraction")
    p.add_argument("--k1a", type=float, default=0.0, help="K1 * a (default 0)")
    # ----- files -----
    p.add_argument("--load", type=Path, required=True, help="input pack prefix")
    p.add_argument("--save", type=Path, required=True, help="output prefix")
    # ----- minimizer -----
    p.add_argument("--method", choices=("lbfgs", "descent"), default="lbfgs")
    p.add_argument(
        "--tol",
        type=float,
        default=1e-20,
        help=(
            "success threshold on the energy, used verbatim as float64 "
            f"(C++ parity value: {CPP_PARITY_TOL:.8e})"
        ),
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="evaluation budget: lbfgs max_evals_per_pass, descent max_evals",
    )
    p.add_argument("--grow", type=float, default=1.0, help="descent step growth")
    p.add_argument(
        "--stop-tol",
        type=float,
        default=0.0,
        help="early-stopping energy (0 = C++ behaviour: never stop early)",
    )
    # ----- execution -----
    p.add_argument("--batch-size", type=int, default=1024, help="configs per call")
    p.add_argument("--chunk-size", type=int, default=None, help="k-chunk of energy")
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    # ----- recording and analysis -----
    p.add_argument("--record-every", type=int, default=0, help="trajectory stride")
    p.add_argument("--max-records", type=int, default=0, help="trajectory capacity")
    p.add_argument("--g2-bins", type=int, default=200, help="fixed g2 grid bins")
    p.add_argument("--hp-bins", type=int, default=200, help="fixed Hp grid bins")
    return p


def validate_args(args: argparse.Namespace) -> None:
    """Raise ValueError with a clear message on inconsistent arguments."""
    if args.dim not in (1, 2, 3):
        raise ValueError(f"--dim must be 1, 2 or 3, got {args.dim}")
    if not args.ka > 0.0:
        raise ValueError(f"--ka must be positive, got {args.ka}")
    if not args.phi > 0.0:
        raise ValueError(f"--phi must be positive, got {args.phi}")
    if not 0.0 <= args.k1a < args.ka:
        raise ValueError(f"need 0 <= --k1a < --ka, got {args.k1a}, {args.ka}")
    if not args.tol > 0.0:
        raise ValueError(f"--tol must be positive, got {args.tol}")
    if args.stop_tol < 0.0:
        raise ValueError(f"--stop-tol must be >= 0, got {args.stop_tol}")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError(f"--max-steps must be >= 1, got {args.max_steps}")
    if args.method == "lbfgs" and args.grow != 1.0:
        raise ValueError("--grow applies to --method descent only")
    if args.grow < 1.0:
        raise ValueError(f"--grow must be >= 1, got {args.grow}")
    for name in ("batch_size", "g2_bins", "hp_bins"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.chunk_size is not None and args.chunk_size < 1:
        raise ValueError(f"--chunk-size must be >= 1, got {args.chunk_size}")
    if min(args.record_every, args.max_records) < 0:
        raise ValueError("--record-every and --max-records must be >= 0")
    if (args.record_every > 0) != (args.max_records > 0):
        raise ValueError("--record-every and --max-records must both be positive")


# ============================================================
# DEVICE
# ============================================================


def select_device(device: str) -> str:
    """Select the JAX platform explicitly and return its name.

    ``gpu`` calls ``require_gpu`` (raises RuntimeError with the cause);
    ``cpu`` pins the platform to cpu. Never falls back silently.
    """
    assert device in ("gpu", "cpu"), "device must be gpu or cpu"
    if device == "gpu":
        require_gpu()
    else:
        jax.config.update("jax_platforms", "cpu")  # type: ignore[no-untyped-call]
    platform = str(jax.devices()[0].platform)
    if (device == "cpu") != (platform == "cpu"):
        raise RuntimeError(f"requested --device {device} but JAX uses {platform}")
    return platform


# ============================================================
# INPUT AND PHYSICS
# ============================================================


def load_input(
    prefix: Path, dim: int
) -> Tuple[List[PackedConfig], np.ndarray, np.ndarray]:
    """Read a pack; return (configs, (B, N, d) fractional, unit basis).

    Raises
    ------
    ValueError
        On a dimension mismatch, mixed N/d, or a basis that is not unit volume.
    TypeError
        On non-float64 positions.
    """
    configs = read_configpack(prefix)
    if len(configs) == 0:
        raise ValueError(f"no configurations in {prefix}")
    d_pack = int(np.asarray(configs[0].relative).shape[1])
    if d_pack != dim:
        raise ValueError(f"--dim {dim} does not match pack dimension {d_pack}")
    s0, _ = stack_fractional(configs)
    s0 = np.asarray(wrap_fractional_batch(jax.numpy.asarray(s0)))
    for i, c in enumerate(configs):
        det = abs(float(np.linalg.det(np.asarray(c.basis))))
        if abs(det - 1.0) > UNIT_VOLUME_ATOL:
            raise ValueError(
                f"config {i}: input basis has |det| = {det!r}, not unit volume; "
                "C++ input mode multiplies the basis by N^(1/d), so the pack "
                "must hold a unit-volume basis (cell volume 1)"
            )
    return configs, s0, np.asarray(configs[0].basis, dtype=np.float64)


def build_physics(
    unit_basis: np.ndarray, n: int, args: argparse.Namespace
) -> Tuple[Box, Any, Dict[str, Any]]:
    """Return (box, energy, info) with K1 = k1a / a and K2 = ka / a."""
    d = args.dim
    if n < 2:
        raise ValueError(f"need at least 2 particles, got N = {n}")
    side = float(n) ** (1.0 / d)
    if np.array_equal(unit_basis, np.eye(d)):
        box = cubic_box(n, d)
    else:
        box = Box(basis=jax.numpy.asarray(unit_basis * side))
    k2 = k_outer_from_ka(args.ka, args.phi, d)
    k1 = k_outer_from_ka(args.k1a, args.phi, d) if args.k1a > 0.0 else 0.0
    kset = enumerate_kvectors(np.asarray(box.basis), k1, k2)
    m = int(kset.n.shape[0])
    if m == 0:
        raise ValueError(f"empty k-set for K1={k1}, K2={k2}")
    energy = make_stealthy_energy(box, kset, chunk_size=args.chunk_size)
    info = {
        "N": n,
        "d": d,
        "M": m,
        "chi": chi_from_count(m, n, d),
        "a": args.ka / k2,
        "K1": k1,
        "K2": k2,
    }
    return box, energy, info


def make_settings(args: argparse.Namespace) -> Settings:
    """Return the minimizer settings selected by the arguments."""
    common: Dict[str, Any] = {
        "energy_tol": args.stop_tol,
        "record_every": args.record_every,
        "max_records": args.max_records,
    }
    if args.method == "lbfgs":
        if args.max_steps is not None:
            common["max_evals_per_pass"] = args.max_steps
        return LBFGSSettings(**common)
    if args.max_steps is not None:
        common["max_evals"] = args.max_steps
    return DescentSettings(grow=args.grow, **common)


# ============================================================
# RELAXATION
# ============================================================


def relax_batches(
    energy: Any, s0: np.ndarray, settings: Settings, batch_size: int
) -> Dict[str, np.ndarray]:
    """Relax in batches; return concatenated arrays (wrapped positions)."""
    assert s0.ndim == 3 and batch_size >= 1, "bad shapes"
    parts: List[Dict[str, np.ndarray]] = []
    for lo in range(0, s0.shape[0], batch_size):
        res = relax(energy, jax.numpy.asarray(s0[lo : lo + batch_size]), settings)
        part = {
            "s": np.asarray(wrap_fractional_batch(res.s)),
            "energy": np.asarray(res.energy),
            "n_evals": np.asarray(res.n_evals),
            "n_iters": np.asarray(res.n_iters),
            "status": np.asarray(res.status),
        }
        if res.trajectory is not None:
            for name in ("s", "energy", "step", "iteration", "count"):
                part[f"traj_{name}"] = np.asarray(getattr(res.trajectory, name))
        parts.append(part)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def wrap_fractional_batch(s: Any) -> Any:
    """Wrap a (b, n, d) batch of fractional positions into [0, 1)."""
    assert s.ndim == 3, "s must have shape (b, n, d)"
    return jax.vmap(wrap_fractional)(s)


# ============================================================
# OUTPUT
# ============================================================


def pack_configs(
    template: Sequence[PackedConfig], s: np.ndarray, side: float
) -> List[PackedConfig]:
    """Build C++-style output records (cell size and basis scaled by N^(1/d))."""
    assert len(template) == s.shape[0], "one template per configuration"
    out = []
    for cfg, rel in zip(template, s, strict=True):
        out.append(
            PackedConfig(
                basis=np.asarray(cfg.basis, dtype=np.float64) * side,
                relative=np.ascontiguousarray(rel, dtype=np.float64),
                cell_size=side,
                names=cfg.names,
            )
        )
    return out


def hp_table(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the (n_bins, 4) Hp table, C++ columns (PairCorrelation.cpp:480-501).

    Columns: r_mid, count / total / width, half-width, p / sqrt(count)
    (0 for an empty bin).
    """
    counts = np.asarray(counts, dtype=np.float64)
    assert counts.shape == (edges.size - 1,), "counts must match the bins"
    total = counts.sum()
    assert total > 0.0, "need at least one nearest-neighbour distance"
    half = 0.5 * np.diff(edges)
    p = counts / total / (2.0 * half)
    err = np.divide(p, np.sqrt(counts), out=np.zeros_like(p), where=counts > 0)
    return np.stack([0.5 * (edges[1:] + edges[:-1]), p, half, err], axis=1)


def g2_table(g2: np.ndarray, counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the (n_bins, 4) g2 table, C++ columns (PairCorrelation.cpp:186-205).

    Columns: r_mid, g2, half-width, g2 / sqrt(count) (0 for an empty bin).
    """
    counts = np.asarray(counts, dtype=np.float64)
    assert g2.shape == counts.shape == (edges.size - 1,), "shapes must match"
    err = np.divide(g2, np.sqrt(counts), out=np.zeros_like(g2), where=counts > 0)
    half = 0.5 * np.diff(edges)
    return np.stack([0.5 * (edges[1:] + edges[:-1]), g2, half, err], axis=1)


def _write_table(path: Path, table: np.ndarray) -> None:
    write_function_table(path, [table[:, j] for j in range(table.shape[1])])


def write_analyses(
    prefix: str, s_succ: np.ndarray, box: Box, args: argparse.Namespace
) -> Dict[str, Any]:
    """Write S(k), g2 and Hp for the Success set; return log entries."""
    n, d = s_succ.shape[1], s_succ.shape[2]
    side = float(n) ** (1.0 / d)
    k_mod, s_k, k_err, s_err = isotropic_structure_factor(
        jax.numpy.asarray(s_succ), box, SK_K_DIR_MAX, SK_K_MAX
    )
    n_nan = int(np.isnan(s_err).sum())  # written as NaN, like C++ (D-009)
    write_function_table(Path(prefix + "_Sk.txt"), [k_mod, s_k, k_err, s_err])

    g2_edges = np.linspace(0.0, G2_CUTOFF_FRACTION * side, args.g2_bins + 1)
    g2_counts = np.zeros(args.g2_bins, dtype=np.int64)
    nn: List[np.ndarray] = []
    for lo in range(0, s_succ.shape[0], args.batch_size):
        chunk = jax.numpy.asarray(s_succ[lo : lo + args.batch_size])
        g2_counts += pair_distance_histogram(chunk, box, g2_edges)
        nn += [np.asarray(nearest_neighbor_distances(c, box)) for c in chunk]
    g2 = g2_from_histogram(g2_counts, g2_edges, n, s_succ.shape[0], float(n), d)
    _write_table(Path(prefix + "_g2.txt"), g2_table(g2, g2_counts, g2_edges))

    nn_all = np.concatenate(nn)
    hp_edges = np.linspace(0.0, float(max(nn_all.max(), 1e-12)), args.hp_bins + 1)
    hp_counts, _ = np.histogram(nn_all, bins=hp_edges)
    _write_table(Path(prefix + "_Hp.txt"), hp_table(hp_counts, hp_edges))
    return {"analysis": "written", "sk_nan_error_bins": n_nan}


# ============================================================
# DRIVER
# ============================================================


def success_mask(energy: np.ndarray, tol: float) -> np.ndarray:
    """Return the Success mask: strictly energy < tol (C++ ``E < E_max``)."""
    assert tol > 0.0, "tol must be positive"
    return np.asarray(energy) < tol


def summarize(out: Dict[str, np.ndarray], tol: float) -> Dict[str, Any]:
    """Return status counts, success count and n_evals statistics."""
    status = out["status"]
    counts = {
        STATUS_NAMES.get(int(c), f"STATUS_{int(c)}"): int((status == c).sum())
        for c in np.unique(status)
    }
    ev = out["n_evals"]
    return {
        "status_counts": counts,
        "n_success": int(success_mask(out["energy"], tol).sum()),
        "n_evals": {
            "min": int(ev.min()),
            "mean": float(ev.mean()),
            "median": float(np.median(ev)),
            "max": int(ev.max()),
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the driver; write all outputs and return the summary (the log)."""
    validate_args(args)
    t0 = time.perf_counter()
    platform = select_device(args.device)
    configs, s0, unit_basis = load_input(args.load, args.dim)
    n = int(s0.shape[1])
    box, energy, info = build_physics(unit_basis, n, args)
    settings = make_settings(args)
    t_setup = time.perf_counter()
    print(
        f"relaxing {s0.shape[0]} configs: N={n} M={info['M']} on {platform}", flush=True
    )
    out = relax_batches(energy, s0, settings, args.batch_size)
    t_relax = time.perf_counter()

    prefix = str(args.save)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    side = float(n) ** (1.0 / args.dim)
    ok = success_mask(out["energy"], args.tol)
    write_configpack(Path(prefix), pack_configs(configs, out["s"], side))
    write_configpack(Path(prefix + "_InitConfig"), pack_configs(configs, s0, side))
    np.save(prefix + "_energies.npy", out["energy"])
    summary = summarize(out, args.tol)
    if ok.any():
        succ_cfgs = [c for c, f in zip(configs, ok, strict=True) if f]
        write_configpack(
            Path(prefix + "_Success"), pack_configs(succ_cfgs, out["s"][ok], side)
        )
        summary.update(write_analyses(prefix, out["s"][ok], box, args))
    else:
        summary["analysis"] = "skipped: no Success configurations"
    if "traj_s" in out:
        traj: Dict[str, Any] = {
            k[5:]: v for k, v in out.items() if k.startswith("traj_")
        }
        np.savez(prefix + "_traj.npz", **traj)
    t_end = time.perf_counter()
    summary.update(
        {
            "args": vars(args),
            **info,
            "tol": args.tol,
            "method": args.method,
            "settings": dataclasses.asdict(settings),
            "n_configs": int(s0.shape[0]),
            "wall_seconds": {
                "setup": t_setup - t0,
                "relax": t_relax - t_setup,
                "output": t_end - t_relax,
            },
            "jax_version": jax.__version__,
            "device": platform,
        }
    )
    Path(prefix + "_log.json").write_text(
        json.dumps(summary, indent=2, default=_jsonable) + "\n"
    )
    return summary


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parsed = build_parser().parse_args()
    try:
        result = run(parsed)
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
    print(json.dumps(result, indent=2, default=_jsonable), flush=True)
