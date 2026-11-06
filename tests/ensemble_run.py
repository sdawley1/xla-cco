#!/usr/bin/env python3

"""
Data access, relaxation (with cache) and per-config analysis for T12.

Reads the C++ runs under ``CCO_RUNS``, relaxes the same initial configurations
with the production CLI (``xla_cco.cli.relax.run``), caches the JAX results by a
key (SHA-256 of every ``src/xla_cco`` source file, plus the settings), and turns
any ensemble into an ``ensemble_stats.EnsembleSample`` with OUR analysis code.

Design choices (documented for the report):

- g2 peak: per config, g2 on the fixed grid r in [0, L/2] with bin width 0.1;
  the statistic is the highest bin (a single 100-particle config has a noisy
  g2, so the global maximum over a coarse fixed grid is used; the grid is the
  same for every config and ensemble).
- RMS displacement: sqrt(mean_i |B (ds_i - round(ds_i))|^2) with ds = s - s0,
  B the box basis (minimum-image Cartesian).
- S(k): shells K2 < |k| <= 3 K2 of the cubic-cell k-lattice; per-config value
  = mean of |rho|^2/N over the shell.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
import ensemble_stats as es
from xla_cco.analysis.pairs import (
    g2_from_histogram,
    nearest_neighbor_distances,
    pair_distance_histogram,
)
from xla_cco.analysis.structure import structure_factor
from xla_cco.box import Box, cubic_box, wrap_fractional
from xla_cco.cli.relax import CPP_PARITY_TOL, build_parser, run
from xla_cco.energy import make_stealthy_energy
from xla_cco.io.configpack import read_configpack, stack_fractional, write_configpack
from xla_cco.kvectors import (
    chi_from_count,
    enumerate_kvectors,
    k_outer_from_ka,
    ka_from_chi,
)

# ============================================================
# CONSTANTS
# ============================================================

CCO_RUNS = Path("/scratch/gpfs/TORQUATO/sd2402/cco_runs")
CACHE_ROOT = Path("/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/ensemble")
SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "xla_cco"
CHIS: Tuple[float, ...] = (0.05, 0.15, 0.25, 0.35, 0.45)
PHI, DIM, N_PART = 0.05, 3, 100
REAL_TASKS: Tuple[int, ...] = (0, 1, 2)
CTRL_TASKS: Tuple[int, ...] = (10, 11, 12)
PER_TASK = 400
G2_BIN = 0.1
SK_OUTER_FACTOR = 3.0
WRONG_K_SCALE = 0.9
ENV_LIMIT = "XLA_CCO_ENSEMBLE_LIMIT"
ENV_DEVICE = "XLA_CCO_ENSEMBLE_DEVICE"


def run_dir(chi: float) -> Path:
    """Return the C++ run directory of nominal stealthiness ``chi``."""
    return CCO_RUNS / f"p3d-N100-phi0.05-chi{chi:g}-val0.0"


def limit_from_env() -> int:
    """Return configs per task (env ``XLA_CCO_ENSEMBLE_LIMIT``, default full)."""
    n = int(os.environ.get(ENV_LIMIT, PER_TASK))
    assert 2 <= n <= PER_TASK, f"{ENV_LIMIT} must be in [2, {PER_TASK}]"
    return n


def device_from_env() -> str:
    """Return the JAX device the relaxations require (default gpu)."""
    dev = os.environ.get(ENV_DEVICE, "gpu")
    assert dev in ("gpu", "cpu"), f"{ENV_DEVICE} must be gpu or cpu"
    return dev


# ============================================================
# C++ DATA AND CHECKS
# ============================================================


def task_prefix(chi: float, task: int) -> Path:
    """Return the ConfigPack prefix of the C++ final configs of ``task``."""
    return run_dir(chi) / f"task_{task:04d}" / f"ccopt-t{task:04d}"


def ic_prefix(chi: float, task: int) -> Path:
    """Return the unit-cell initial-config prefix ``ic/chunk_<task>``."""
    return run_dir(chi) / "ic" / f"chunk_{task:04d}"


def load_pack(prefix: Path, limit: int) -> np.ndarray:
    """Return the first ``limit`` configs of a pack as (B, N, d) fractional."""
    cfgs = read_configpack(prefix, list(range(limit)))
    s, _ = stack_fractional(cfgs)
    assert s.shape[1:] == (N_PART, DIM), f"unexpected shape {s.shape}"
    return np.asarray(s)


def verify_mapping(chi: float, task: int) -> Dict[str, Any]:
    """Compare ``ic/chunk_<task>`` with ``task_<task>/_InitConfig`` (all 400).

    Returns the max wrapped difference and whether they are bit-identical.
    """
    a = load_pack(ic_prefix(chi, task), PER_TASK)
    b = load_pack(Path(str(task_prefix(chi, task)) + "_InitConfig"), PER_TASK)
    d = a - b
    d -= np.round(d)
    return {
        "chi": chi,
        "task": task,
        "max_wrapped_diff": float(np.abs(d).max()),
        "bit_identical": bool(np.array_equal(a, b)),
    }


def _printed_tol(value: float) -> float:
    """Half unit of the last of 6 significant digits printed by C++."""
    return 0.5 * 10.0 ** (np.floor(np.log10(abs(value))) - 5) * (1.0 + 1e-9)


def parse_engine_log(chi: float, task: int) -> Tuple[float, float]:
    """Return (K2, chi) printed in the engine.log of ``task``."""
    text = (run_dir(chi) / f"task_{task:04d}" / "engine.log").read_text()
    k2 = re.search(r"exclusion region = the interval \[0, ([0-9.eE+-]+)\]", text)
    ch = re.search(r"chi = ([0-9.eE+-]+)", text)
    assert k2 is not None and ch is not None, "engine.log lacks window or chi"
    return float(k2.group(1)), float(ch.group(1))


def physics(chi: float, k_scale: float = 1.0) -> Dict[str, Any]:
    """Return Ka, K2, M and chi for nominal ``chi`` (K2 scaled by ``k_scale``)."""
    ka = ka_from_chi(chi, PHI, DIM) * k_scale
    k2 = k_outer_from_ka(ka, PHI, DIM)
    box = cubic_box(N_PART, DIM)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k2)
    return {
        "ka": ka,
        "k2": k2,
        "m": int(kset.n.shape[0]),
        "chi_actual": chi_from_count(int(kset.n.shape[0]), N_PART, DIM),
    }


def verify_window(chi: float, task: int) -> Dict[str, Any]:
    """Assert our window and chi match ``engine.log`` to the printed precision."""
    k2_log, chi_log = parse_engine_log(chi, task)
    ph = physics(chi)
    assert abs(ph["k2"] - k2_log) <= _printed_tol(k2_log), (ph["k2"], k2_log)
    assert abs(ph["chi_actual"] - chi_log) <= _printed_tol(chi_log), (
        ph["chi_actual"],
        chi_log,
    )
    return {**ph, "k2_log": k2_log, "chi_log": chi_log, "task": task}


# ============================================================
# JAX RELAXATION WITH CACHE
# ============================================================


def src_hash() -> str:
    """Return the SHA-256 of every Python source under ``src/xla_cco``."""
    h = hashlib.sha256()
    for p in sorted(SRC_ROOT.rglob("*.py")):
        h.update(str(p.relative_to(SRC_ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def cache_key(chi: float, task: int, k_scale: float, limit: int) -> Dict[str, Any]:
    """Return the cache key: source hash plus every relaxation setting."""
    from xla_cco.minimize.lbfgs import LBFGSSettings

    key = {
        "src_sha256": src_hash(),
        "chi": chi,
        "task": task,
        "k_scale": k_scale,
        "limit": limit,
        "tol": CPP_PARITY_TOL,
        "phi": PHI,
        "ka": ka_from_chi(chi, PHI, DIM) * k_scale,
        "settings": dataclasses.asdict(LBFGSSettings()),
        "jax_version": jax.__version__,
    }
    return dict(json.loads(json.dumps(key)))  # tuples -> lists, as read from disk


def _cache_dir(chi: float, task: int, k_scale: float) -> Path:
    label = "jax" if k_scale == 1.0 else f"wrongk{k_scale:.2f}"
    return CACHE_ROOT / f"chi{chi:g}" / label / f"task_{task:04d}"


def _run_cli(chi: float, task: int, k_scale: float, limit: int, out: Path) -> None:
    """Relax the initial configs with the production CLI; write into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    load = ic_prefix(chi, task)
    if limit < PER_TASK:  # truncated copy of the unit-cell pack
        load = out / "input"
        write_configpack(
            load, read_configpack(ic_prefix(chi, task), list(range(limit)))
        )
    ka = ka_from_chi(chi, PHI, DIM) * k_scale
    argv = [
        "--dim", str(DIM), "--ka", repr(ka), "--phi", repr(PHI),
        "--load", str(load), "--save", str(out / "relaxed"),
        "--tol", repr(CPP_PARITY_TOL), "--device", device_from_env(),
    ]  # fmt: skip
    run(build_parser().parse_args(argv))


def jax_relax(
    chi: float, task: int, k_scale: float, limit: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return (final s (B, N, d), info) of the JAX relaxation of one task.

    Recomputes unless ``key.json`` in the cache directory equals the current
    key. ``info`` holds status counts, relax wall seconds and cache status.
    """
    out = _cache_dir(chi, task, k_scale)
    key = cache_key(chi, task, k_scale, limit)
    key_file = out / "key.json"
    cached = key_file.exists() and json.loads(key_file.read_text()) == key
    if not cached:
        t0 = time.perf_counter()
        key_file.unlink(missing_ok=True)
        _run_cli(chi, task, k_scale, limit, out)
        key_file.write_text(json.dumps(key, indent=2) + "\n")
        (out / "wall_seconds.txt").write_text(f"{time.perf_counter() - t0:.3f}\n")
    log = json.loads((out / "relaxed_log.json").read_text())
    s = load_pack(out / "relaxed", limit)
    info = {
        "status_counts": log["status_counts"],
        "relax_seconds": log["wall_seconds"]["relax"],
        "wall_seconds": float((out / "wall_seconds.txt").read_text()),
        "cached": cached,
        "device": log["device"],
        "n_evals": log["n_evals"],
    }
    return s, info


# ============================================================
# PER-CONFIG ANALYSIS
# ============================================================


@dataclasses.dataclass(frozen=True)
class Analyzer:
    """Fixed analysis setup for one chi (box, energy, shells, g2 grid)."""

    box: Box
    energy: Any
    n_vectors: jax.Array
    groups: np.ndarray
    edges: np.ndarray
    k_shell: np.ndarray


def make_analyzer(chi: float) -> Analyzer:
    """Build the analysis setup: stealthy energy and S(k) shells (K2, 3 K2]."""
    box = cubic_box(N_PART, DIM)
    k2 = physics(chi)["k2"]
    basis = np.asarray(box.basis)
    energy = make_stealthy_energy(box, enumerate_kvectors(basis, 0.0, k2))
    outer = enumerate_kvectors(basis, k2, SK_OUTER_FACTOR * k2 * (1.0 + 1e-12))
    groups = es.shell_groups(outer.k_sq)
    k_shell = np.array(
        [np.sqrt(outer.k_sq[groups == g][0]) for g in range(groups.max() + 1)]
    )
    side = float(N_PART) ** (1.0 / DIM)
    edges = np.arange(0.0, side / 2.0 + 1e-9, G2_BIN)
    return Analyzer(
        box, energy, jnp.asarray(outer.n.astype(np.float64)), groups, edges, k_shell
    )


def _min_image_rms(s: np.ndarray, s0: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """RMS minimum-image Cartesian displacement per config (B,)."""
    ds = s - s0
    ds -= np.round(ds)
    cart = ds @ basis
    return np.asarray(np.sqrt((cart**2).sum(axis=2).mean(axis=1)))


def sample_from_configs(
    s: np.ndarray, s0: np.ndarray, an: Analyzer
) -> es.EnsembleSample:
    """Summarize final configs ``s`` (B, N, d) against initial ``s0``."""
    assert s.shape == s0.shape and s.ndim == 3, "s and s0 must be (B, N, d)"
    sj = jax.vmap(wrap_fractional)(jnp.asarray(s))
    en = np.asarray(jax.vmap(an.energy)(sj))
    nn = [np.asarray(nearest_neighbor_distances(c, an.box)) for c in sj]
    peaks = []
    for c in sj:
        cnt = pair_distance_histogram(c[None], an.box, an.edges)
        g2 = g2_from_histogram(cnt, an.edges, N_PART, 1, float(N_PART), DIM)
        peaks.append(float(g2.max()))
    sk = jax.jit(jax.vmap(lambda x: structure_factor(x, an.n_vectors)))
    s_k = np.concatenate(
        [np.asarray(sk(sj[i : i + 100])) for i in range(0, sj.shape[0], 100)]
    )
    return es.EnsembleSample(
        energy=en,
        success=en < CPP_PARITY_TOL,
        nn_min=np.array([x.min() for x in nn]),
        nn_mean=np.array([x.mean() for x in nn]),
        g2_peak=np.array(peaks),
        rms_disp=_min_image_rms(np.asarray(sj), s0, np.asarray(an.box.basis)),
        sk_shell=es.shell_means(s_k, an.groups),
    )


def cpp_sample(
    chi: float, tasks: Sequence[int], limit: int, an: Analyzer
) -> es.EnsembleSample:
    """Return the C++ ensemble of ``tasks`` (first ``limit`` configs of each)."""
    s = np.concatenate([load_pack(task_prefix(chi, t), limit) for t in tasks])
    s0 = np.concatenate([load_pack(ic_prefix(chi, t), limit) for t in tasks])
    return sample_from_configs(s, s0, an)


def jax_sample(
    chi: float,
    tasks: Sequence[int],
    k_scale: float,
    limit: int,
    an: Analyzer,
) -> Tuple[es.EnsembleSample, List[Dict[str, Any]]]:
    """Return the JAX ensemble of ``tasks`` and the per-task relaxation info."""
    parts, infos = [], []
    for t in tasks:
        s, info = jax_relax(chi, t, k_scale, limit)
        parts.append(s)
        infos.append(info)
    s0 = np.concatenate([load_pack(ic_prefix(chi, t), limit) for t in tasks])
    return sample_from_configs(np.concatenate(parts), s0, an), infos


def status_counts(infos: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Sum the per-task status counts of the CLI logs."""
    tot: Dict[str, int] = {}
    for inf in infos:
        for k, v in inf["status_counts"].items():
            tot[k] = tot.get(k, 0) + int(v)
    return tot
