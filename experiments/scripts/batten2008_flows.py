#!/usr/bin/env python3

"""
Gradient flows to 1D stealthy ground states from Poisson initial conditions.

Recreates the setting of the appendix of Batten, Stillinger and Torquato,
J. Appl. Phys. 104, 033504 (2008): N particles on a line (density 1, so the
cell side is L = N), stealthy target (D = 0) on the first M = round(chi N)
wavevectors k_n = 2 pi n / L, with chi = M / (d N) as in that paper. The
flow dr/dt = -grad_r Phi is integrated with xla_cco.minimize.flow until
Phi < --energy-tol, |grad_r Phi| < --grad-tol (e.g. a local minimum), or the
step budget. The default method is the L-stable Rosenbrock ROS2 with the
exact Hessian: at N = 1000 the landscape is stiff (steep and nearly flat
directions together), which caps explicit Dormand-Prince steps far below
the flow's time scale.

The integration runs in chunks of --chunk-steps accepted steps and is
checkpointed after each, so a long flow spans several SLURM jobs: rerunning
the same command resumes (the flow is autonomous; each lane restarts from
its saved positions, clocks and step size). Every accepted step is recorded
to <out>/chi<tag>/chunk_XXXXX.npz. When all flows have stopped, the paths
are thinned to at most --max-records rows each into <out>/flows_chi<tag>.npz
(+ .json metadata).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import json
import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, Tuple

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.box import cubic_box
from xla_cco.cli.relax import select_device
from xla_cco.energy import StealthyEnergy, make_stealthy_energy
from xla_cco.kvectors import chi_from_count, enumerate_kvectors
from xla_cco.minimize.flow import FlowResult, FlowSettings, gradient_flow
from xla_cco.minimize.result import STATUS_BUDGET, STATUS_RUNNING

# ============================================================
# CONSTANTS
# ============================================================

STATUS_NAMES = {
    0: "RUNNING",
    1: "ENERGY_TOL",
    2: "STEP_FLOOR",
    3: "BUDGET",
    6: "GRAD_TOL",
}

# ============================================================
# HELPERS
# ============================================================


def build_energy(n: int, chi: Fraction) -> Tuple[StealthyEnergy, np.ndarray, int]:
    """Stealthy energy on the first M = round(chi * N) wavevectors, cell L = N.

    The realized chi is M / (d N) (e.g. 333/1000 for chi = 1/3 at N = 1000).
    """
    m = round(chi * n)
    if m < 1 or m >= n:
        raise ValueError(f"M = round(chi * N) = {m} must lie in [1, N)")
    box = cubic_box(n, 1)
    basis = np.asarray(box.basis)
    side = float(basis[0, 0])
    kset = enumerate_kvectors(basis, 0.0, 2.0 * math.pi / side * (m + 0.5))
    assert kset.n.shape[0] == m, f"expected {m} wavevectors, got {kset.n.shape[0]}"
    return make_stealthy_energy(box, kset), basis, m


def chi_tag(chi: Fraction) -> str:
    """File-name tag, e.g. '1-3' for chi = 1/3."""
    return f"{chi.numerator}-{chi.denominator}"


def run_chunk(
    energy: StealthyEnergy,
    s: np.ndarray,
    dt: np.ndarray,
    basis: np.ndarray,
    fs: FlowSettings,
) -> FlowResult:
    """One vmapped chunk; each lane starts from its own step size ``dt``."""
    assert s.ndim == 3 and dt.shape == (s.shape[0],), "lane shapes"

    @eqx.filter_jit
    def batched(e: StealthyEnergy, s_b: jax.Array, dt_b: jax.Array) -> FlowResult:
        b = jnp.asarray(basis)
        return jax.vmap(lambda one, h: gradient_flow(e, one, b, fs, dt0=h))(s_b, dt_b)

    out: FlowResult = batched(energy, jnp.asarray(s), jnp.asarray(dt))
    return out


def save_atomic(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Write an npz via a temporary file so a killed job leaves no torn file."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def init_state(n_traj: int, n: int, seed: int, dt_init: float) -> Dict[str, np.ndarray]:
    """Poisson starts (i.i.d. uniform) and zeroed clocks and counters."""
    s0 = np.random.default_rng(seed).uniform(size=(n_traj, n, 1))
    return {
        "s0": s0,
        "s": s0.copy(),
        "t": np.zeros(n_traj),
        "length": np.zeros(n_traj),
        "dt": np.full(n_traj, dt_init),
        "steps": np.zeros(n_traj, dtype=np.int64),
        "status": np.full(n_traj, STATUS_RUNNING, dtype=np.int64),
        "energy": np.full(n_traj, np.nan),
        "next_chunk": np.asarray(0),
        "wall": np.asarray(0.0),
    }


def advance(
    state: Dict[str, np.ndarray], res: FlowResult, run_dir: Path, max_steps: int
) -> None:
    """Fold one chunk into the state and write its records (active lanes)."""
    active = state["status"] == STATUS_RUNNING
    tr = res.trajectory
    assert tr is not None, "chunks record every step"
    count = np.where(active, np.asarray(tr.count), 0)
    width = max(int(count.max(initial=0)), 1)
    keep = {
        "count": count,
        "s": np.asarray(tr.s)[:, :width],
        "t": np.asarray(tr.t)[:, :width] + state["t"][:, None],
        "length": np.asarray(tr.length)[:, :width] + state["length"][:, None],
        "energy": np.asarray(tr.energy)[:, :width],
    }
    k = int(state["next_chunk"])
    save_atomic(run_dir / f"chunk_{k:05d}.npz", keep)
    state["s"] = np.where(active[:, None, None], np.asarray(res.s), state["s"])
    state["dt"] = np.where(active, np.asarray(res.dt), state["dt"])
    state["energy"] = np.where(active, np.asarray(res.energy), state["energy"])
    state["t"] = np.where(active, state["t"] + np.asarray(res.t), state["t"])
    state["length"] = np.where(
        active, state["length"] + np.asarray(res.length), state["length"]
    )
    state["steps"] = np.where(
        active, state["steps"] + np.asarray(res.n_steps), state["steps"]
    )
    chunk_status = np.asarray(res.status)
    more = (chunk_status == STATUS_BUDGET) & (state["steps"] < max_steps)
    state["status"] = np.where(
        active, np.where(more, STATUS_RUNNING, chunk_status), state["status"]
    )
    state["next_chunk"] = np.asarray(k + 1)


def consolidate(
    run_dir: Path, state: Dict[str, np.ndarray], max_records: int
) -> Dict[str, np.ndarray]:
    """Join the chunk records per lane, thinned to at most max_records rows.

    Two streaming passes: the first reads only the per-chunk counts and picks
    each lane's rows (uniform in accepted-step index, first and last kept);
    the second loads every chunk array once and copies just those rows, so
    memory stays at one chunk plus the output.
    """
    n_chunks = int(state["next_chunk"])
    n_traj, n = state["s0"].shape[:2]
    counts = np.zeros((n_chunks, n_traj), dtype=np.int64)
    for k in range(n_chunks):
        with np.load(run_dir / f"chunk_{k:05d}.npz") as c:
            counts[k] = c["count"]
    # Row 0 of every chunk after the first repeats the previous chunk's end.
    usable = counts - (np.arange(n_chunks) > 0)[:, None]
    usable = np.maximum(usable, 0)
    offsets = np.vstack([np.zeros((1, n_traj), dtype=np.int64), np.cumsum(usable, 0)])
    keys = ("s", "t", "length", "energy")
    shapes = {"s": (n, 1), "t": (), "length": (), "energy": ()}
    out = {k: np.full((n_traj, max_records) + shapes[k], np.nan) for k in keys}
    picks = []
    for b in range(n_traj):
        rows = int(offsets[-1, b])
        assert rows > 0, f"lane {b} recorded nothing"
        grid = np.linspace(0, rows - 1, min(rows, max_records)).round()
        picks.append(np.unique(grid.astype(np.int64)))
    for k in range(n_chunks):
        with np.load(run_dir / f"chunk_{k:05d}.npz") as c:
            arrays = {key: c[key] for key in keys}  # one read per array
        start = 0 if k == 0 else 1
        for b in range(n_traj):
            lo, hi = offsets[k, b], offsets[k + 1, b]
            sel = np.flatnonzero((picks[b] >= lo) & (picks[b] < hi))
            if sel.size:
                rows = picks[b][sel] - lo + start
                for key in keys:
                    out[key][b, sel] = arrays[key][b, rows]
        del arrays
    return {
        "traj_s": out["s"],
        "traj_t": out["t"],
        "traj_length": out["length"],
        "traj_energy": out["energy"],
        "traj_count": np.array([p.size for p in picks], dtype=np.int64),
    }


def summarize(
    args: argparse.Namespace,
    chi: Fraction,
    m: int,
    state: Dict[str, np.ndarray],
    device: str,
) -> Dict[str, object]:
    """JSON metadata of a finished chi."""
    status = state["status"]
    return {
        "N": args.n,
        "d": 1,
        "chi_target": float(chi),
        "chi_str": str(chi),
        "M": m,
        "chi_paper": m / args.n,
        "chi_lib": chi_from_count(m, args.n, 1),
        "K": 2 * math.pi / args.n * (m + 0.5),
        "n_traj": args.n_traj,
        "seed": args.seed,
        "method": args.method,
        "rtol": args.rtol,
        "atol": args.atol,
        "energy_tol": args.energy_tol,
        "grad_tol": args.grad_tol,
        "max_steps": args.max_steps,
        "chunk_steps": args.chunk_steps,
        "device": device,
        "wall_seconds": float(state["wall"]),
        "status_counts": {
            STATUS_NAMES.get(int(c), str(c)): int(np.sum(status == c))
            for c in np.unique(status)
        },
    }


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # ----- system -----
    parser.add_argument("--n", type=int, required=True, help="particles on the line")
    parser.add_argument(
        "--chi",
        nargs="+",
        default=["1/3", "2/3"],
        help="chi = M/(dN) targets as fractions; M = round(chi N)",
    )
    parser.add_argument("--n-traj", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20081)

    # ----- integration -----
    parser.add_argument(
        "--method", choices=("rosenbrock", "dopri5"), default="rosenbrock"
    )
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument(
        "--atol", type=float, default=1e-8, help="per-step RMS Cartesian error floor"
    )
    parser.add_argument("--energy-tol", type=float, default=1e-20)
    parser.add_argument(
        "--grad-tol",
        type=float,
        default=1e-12,
        help="stop when |grad_r Phi| falls below (local minima)",
    )
    parser.add_argument("--max-steps", type=int, default=2_000_000)
    parser.add_argument("--chunk-steps", type=int, default=2000)
    parser.add_argument(
        "--wall-budget",
        type=float,
        default=2700.0,
        help="seconds of integration per invocation",
    )

    # ----- output -----
    parser.add_argument("--max-records", type=int, default=4000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")

    args = parser.parse_args()
    device = select_device(args.device)
    assert args.n_traj > 0 and args.max_records >= 2 and args.chunk_steps > 0
    started = time.time()

    # =========================================================
    # FLOWS PER CHI (resumable)
    # =========================================================
    for chi_str in args.chi:
        chi = Fraction(chi_str)
        tag = chi_tag(chi)
        energy, basis, m = build_energy(args.n, chi)
        run_dir = args.out / f"chi{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / "state.npz"
        if state_path.exists():
            with np.load(state_path) as z:
                state = {k: z[k] for k in z.files}
        else:
            state = init_state(args.n_traj, args.n, args.seed, 1e-3)
        fs = FlowSettings(
            method=args.method,
            rtol=args.rtol,
            atol=args.atol,
            energy_tol=args.energy_tol,
            grad_tol=args.grad_tol,
            max_steps=args.chunk_steps,
            record_every=1,
            max_records=args.chunk_steps + 2,
        )
        while np.any(state["status"] == STATUS_RUNNING):
            if time.time() - started > args.wall_budget:
                print(
                    f"chi={chi_str}: wall budget reached; rerun to resume", flush=True
                )
                break
            t0 = time.time()
            dt = np.where(state["status"] == STATUS_RUNNING, state["dt"], 0.0)
            res = run_chunk(energy, state["s"], dt, basis, fs)
            advance(state, res, run_dir, args.max_steps)
            state["wall"] = np.asarray(float(state["wall"]) + time.time() - t0)
            save_atomic(state_path, state)
            run = int(np.sum(state["status"] == STATUS_RUNNING))
            print(
                f"chi={chi_str} chunk {int(state['next_chunk'])}: running {run}/"
                f"{args.n_traj}; Phi median {np.nanmedian(state['energy']):.2e}; "
                f"t median {np.median(state['t']):.2e}; {time.time() - t0:.0f} s",
                flush=True,
            )
        if np.any(state["status"] == STATUS_RUNNING):
            continue

        # =====================================================
        # CONSOLIDATE
        # =====================================================
        paths = consolidate(run_dir, state, args.max_records)
        out: Dict[str, np.ndarray] = {
            "s0": state["s0"],
            "s_final": state["s"],
            "energy_final": state["energy"],
            "t_final": state["t"],
            "length_final": state["length"],
            "n_steps": state["steps"],
            "status": state["status"],
            "basis": basis,
            **paths,
        }
        np.savez(args.out / f"flows_chi{tag}.npz", **out)
        meta = summarize(args, chi, m, state, device)
        (args.out / f"flows_chi{tag}.json").write_text(json.dumps(meta, indent=2))
        print(
            f"N={args.n} chi={chi_str} M={m}: status {meta['status_counts']}; "
            f"steps {int(state['steps'].min())}-{int(state['steps'].max())}; "
            f"Phi_final max {np.max(state['energy']):.2e}; "
            f"wall {float(state['wall']):.0f} s",
            flush=True,
        )
    print("Done.", flush=True)
