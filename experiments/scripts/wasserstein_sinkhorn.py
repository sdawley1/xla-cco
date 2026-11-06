#!/usr/bin/env python3

"""
Sinkhorn Wasserstein-2 distances at every stored frame of every trajectory.

Every accepted-iteration frame r(t) is compared with the initial condition
r(0) and with the ground state r(inf) reached, using log-domain Sinkhorn on
the minimum-image cost (xla_cco.analysis.transport). Each (trajectory,
reference) pair is a vmapped GPU lane that walks its frames independently:
a frame warm-starts from the previous frame's dual potentials and iterates
until the plan's row-argmax rounding is a bijection (or the marginal error
falls below tol, or max_iters is spent); the first frame is cold-started
through an eps-annealing schedule. Per frame and reference
it records:

- w_round: rho^{1/2}[sum_i C_{i sigma(i)}]^{1/2} for the row-argmax rounding
  sigma of the plan, greedily repaired into a bijection where the rounding
  collides (a feasible matching, so >= the exact W_2; equal to it when the
  optimal permutation is recovered);
- w_ent:   rho^{1/2}[N <P, C>]^{1/2} for the entropic plan P itself;
- is_perm (rounding bijective before repair), n_re (labels matched away
  from themselves), n_iters, err.

Output: <traj>/wasserstein/sinkhorn_<ENS>.npz with per-frame arrays
concatenated over trajectories (``offsets`` delimits them; a --start/--limit
slice is written as sinkhorn_<ENS>_<start>.npz); the last axis of
each (frames, 2) array is (reference r(0), reference r(inf)).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.analysis.transport import (
    SinkhornResult,
    entropic_distance,
    matching_distance,
    repair_matching,
    round_to_matching,
    sinkhorn,
    sinkhorn_annealed,
    torus_sq_cost,
)
from xla_cco.cli.relax import select_device
from xla_cco.io.trajstore import open_trajectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wasserstein_exact import frame_times  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

OUT_KEYS = ("w_round", "w_ent", "is_perm", "n_re", "n_iters", "err")

# ============================================================
# DEVICE KERNELS
# ============================================================


def _measure(
    cost: jax.Array, res: SinkhornResult, eps: float, rho: float
) -> Dict[str, jax.Array]:
    """Distances and diagnostics of one (cost, Sinkhorn result) pair."""
    sigma, ok = round_to_matching(cost, res.g)
    sigma = repair_matching(cost, res.g, sigma)
    return {
        "w_round": matching_distance(cost, sigma, rho),
        "w_ent": entropic_distance(cost, res, eps, rho),
        "is_perm": ok,
        "n_re": jnp.sum(sigma != jnp.arange(cost.shape[0])),
        "n_iters": res.n_iters,
        "err": res.marginal_error,
    }


def _costs(refs: jax.Array, cur: jax.Array, basis: jax.Array) -> jax.Array:
    """(B, N, N) costs from each lane's reference to its current frame."""
    return jax.vmap(lambda r, c: torus_sq_cost(r, c, basis, images=False))(refs, cur)


@eqx.filter_jit
def cold_start(
    refs: jax.Array, cur: jax.Array, basis: jax.Array, cfg: Dict[str, Any]
) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
    """Annealed cold start of every lane; returns potentials and outputs."""
    cost = _costs(refs, cur, basis)
    res = jax.vmap(
        lambda c: sinkhorn_annealed(c, cfg["schedule"], cfg["cold_iters"], cfg["tol"])
    )(cost)
    out = jax.vmap(lambda c, r: _measure(c, r, cfg["eps"], cfg["rho"]))(cost, res)
    return res.f, res.g, out


@eqx.filter_jit
def tick(
    refs: jax.Array,
    cur: jax.Array,
    f: jax.Array,
    g: jax.Array,
    basis: jax.Array,
    cfg: Dict[str, Any],
) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
    """Up to ``tick_iters`` warm-started iterations on every lane's frame."""
    cost = _costs(refs, cur, basis)
    res = jax.vmap(
        lambda c, f0, g0: sinkhorn(
            c, cfg["eps"], f0, g0, cfg["tick_iters"], cfg["tol"], True
        )
    )(cost, f, g)
    out = jax.vmap(lambda c, r: _measure(c, r, cfg["eps"], cfg["rho"]))(cost, res)
    return res.f, res.g, out


# ============================================================
# HOST DRIVER
# ============================================================


def run_lanes(prefixes: List[Path], cfg: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    """Process trajectories with independent (trajectory, reference) lanes.

    Lane b = 2 l + r follows trajectory l against reference r (0: r(0),
    1: r(inf)). Every tick runs a few iterations on all lanes; a lane moves to
    its next frame once its rounding is bijective, its marginal error is below
    tol, or it has spent ``max_iters`` on the frame, so slow frames never
    stall the other lanes.
    """
    assert prefixes, "need trajectories"
    opened = [open_trajectory(p) for p in prefixes]
    frames = [o[0] for o in opened for _ in range(2)]
    n_fr = np.array([fr.shape[0] for fr in frames])
    basis = np.asarray(opened[0][2]["basis"], dtype=np.float64)
    assert all(np.allclose(o[2]["basis"], basis) for o in opened), "mixed boxes"
    assert np.allclose(basis, np.diag(np.diag(basis))), "orthogonal cell assumed"
    refs = jnp.asarray(
        np.stack([fr[0] if b % 2 == 0 else fr[-1] for b, fr in enumerate(frames)])
    )
    out = [{k: np.empty(n, dtype=np.float64) for k in OUT_KEYS} for n in n_fr]
    f, g, res = cold_start(refs, refs[0::2].repeat(2, axis=0), jnp.asarray(basis), cfg)
    res = {k: np.asarray(v) for k, v in res.items()}
    for b in range(len(frames)):
        for k in OUT_KEYS:
            out[b][k][0] = res[k][b]
    pos = np.ones(len(frames), dtype=np.int64)
    spent = np.zeros(len(frames), dtype=np.int64)
    max_ticks = int(n_fr.max()) * (cfg["max_iters"] // cfg["tick_iters"] + 1)
    for n_tick in range(max_ticks):
        if np.all(pos >= n_fr):
            break
        cur = np.stack(
            [fr[min(k, n - 1)] for fr, k, n in zip(frames, pos, n_fr, strict=True)]
        )
        f, g, res = tick(refs, jnp.asarray(cur), f, g, jnp.asarray(basis), cfg)
        res = {k: np.asarray(v) for k, v in res.items()}
        spent += res["n_iters"].astype(np.int64)
        done = (
            res["is_perm"] | (res["err"] < cfg["tol"]) | (spent >= cfg["max_iters"])
        ) & (pos < n_fr)
        for b in np.flatnonzero(done):
            for k in OUT_KEYS:
                out[b][k][pos[b]] = res[k][b]
            out[b]["n_iters"][pos[b]] = spent[b]
        pos[done] += 1
        spent[done] = 0
        if n_tick % 2000 == 0:
            print(
                f"  tick {n_tick}: frames {int(np.minimum(pos, n_fr).sum())}"
                f"/{int(n_fr.sum())}",
                flush=True,
            )
    assert np.all(pos >= n_fr), "tick budget exhausted"
    merged = []
    for lane, (_, arrays, _) in enumerate(opened):
        row = {
            k: np.stack([out[2 * lane][k], out[2 * lane + 1][k]], axis=1)
            for k in OUT_KEYS
        }
        row["u"] = frame_times(arrays)[0]
        assert row["u"].shape[0] == n_fr[2 * lane], "frame/time mismatch"
        merged.append(row)
    return merged


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # ----- inputs -----
    parser.add_argument(
        "--traj", type=Path, default=Path("experiments/trajectories/chi0.1")
    )
    parser.add_argument("--ensembles", nargs="+", default=["POS", "GAU"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)

    # ----- Sinkhorn -----
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iters", type=int, default=2_000)
    parser.add_argument("--cold-iters", type=int, default=2_000)

    # ----- execution -----
    parser.add_argument("--lanes", type=int, default=100)
    parser.add_argument("--tick-iters", type=int, default=2)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")

    args = parser.parse_args()
    select_device(args.device)
    assert args.eps > 0.0 and args.lanes > 0, "eps and lanes must be positive"
    schedule = tuple(
        e for e in (1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4) if e > args.eps
    ) + (args.eps,)
    out_dir = args.traj / "wasserstein"
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # ENSEMBLES
    # =========================================================
    for ens in args.ensembles:
        prefixes = sorted(
            p.with_suffix("") for p in (args.traj / ens).glob("traj_*.json")
        )
        sliced = args.start > 0 or args.limit is not None
        stop = None if args.limit is None else args.start + args.limit
        prefixes = prefixes[args.start : stop]
        assert prefixes, f"no trajectories in slice {args.start}:{stop}"
        rho = float(open_trajectory(prefixes[0])[2]["density"])
        cfg: Dict[str, Any] = {
            "eps": args.eps,
            "tol": args.tol,
            "max_iters": args.max_iters,
            "cold_iters": args.cold_iters,
            "schedule": schedule,
            "rho": rho,
            "tick_iters": args.tick_iters,
        }
        t0 = time.time()
        lanes: List[Dict[str, np.ndarray]] = []
        for b0 in range(0, len(prefixes), args.lanes):
            lanes += run_lanes(prefixes[b0 : b0 + args.lanes], cfg)
            print(
                f"{ens}: {len(lanes)}/{len(prefixes)} lanes, {time.time() - t0:.0f} s",
                flush=True,
            )
        offsets = np.concatenate([[0], np.cumsum([ln["u"].shape[0] for ln in lanes])])
        result = {k: np.concatenate([ln[k] for ln in lanes]) for k in lanes[0]}
        result.update(
            offsets=offsets,
            names=np.array([p.name for p in prefixes]),
            eps=np.asarray(args.eps),
            tol=np.asarray(args.tol),
            schedule=np.asarray(schedule),
        )
        # A slice of the ensemble is written as its own part file; the
        # analysis merges parts in trajectory order.
        tag = f"_{args.start:04d}" if sliced else ""
        np.savez(out_dir / f"sinkhorn_{ens}{tag}.npz", **result)
        print(
            f"{ens}: {offsets[-1]} frames; non-bijective roundings "
            f"{int(np.sum(result['is_perm'] == 0))}; max iters "
            f"{int(result['n_iters'].max())}; max err {result['err'].max():.1e}",
            flush=True,
        )
    print("Done.", flush=True)
