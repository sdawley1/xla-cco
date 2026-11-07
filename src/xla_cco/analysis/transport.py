#!/usr/bin/env python3

"""
Optimal-transport (Wasserstein-2) distance between periodic point patterns.

Two N-particle configurations are treated as uniform empirical measures on
the torus. Their squared minimum-image Cartesian distances form the cost
matrix C_ij, and the label-free analogue of the proximity metric is

    W_2 = rho^{1/2} [min over permutations sigma of sum_i C_{i sigma(i)}]^{1/2},

scaled like p_0 (Dawley & Torquato, PRE 112, 045310 (2025), Eq. (10)), so
W_2 <= p_0 (the identity is one admissible matching). The optimum is
approximated with log-domain Sinkhorn iterations (eps annealing, warm
starts); the transport plan is rounded to a matching by a row argmax, whose
cost is an exact W_2 whenever the rounded map is the optimal permutation.
Exact assignment (e.g. scipy.optimize.linear_sum_assignment) is left to the
caller, so the library has no SciPy dependency.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import itertools
from typing import Optional, Sequence, Tuple, Union

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

# ----- project -----
from xla_cco._types import Positions

# ============================================================
# COST
# ============================================================


def required_images(basis: np.ndarray) -> Tuple[int, ...]:
    """Return per-axis image radii m_k that make ``torus_sq_cost`` exact.

    After rounding, the fractional difference ds lies in [-1/2, 1/2]^d, so the
    nearest translate x = (ds + n) B satisfies |x| <= R, the largest |ds B|
    over that cube (attained at a vertex). Then |n_k| <= 1/2 + R |B^{-1} e_k|,
    so scanning n_k in [-m_k, m_k] with m_k = floor(1/2 + R |B^{-1} e_k|) is
    exact for any cell, reduced or not. A diagonal (orthogonal) basis needs
    no scan: rounding alone is exact (all m_k = 0).

    Parameters
    ----------
    basis : np.ndarray
        (d, d) concrete basis, rows = lattice vectors; finite, nonsingular.

    Returns
    -------
    Tuple[int, ...]
        The radius m_k for each fractional axis k.
    """
    b = np.asarray(basis, dtype=np.float64)
    if b.ndim != 2 or b.shape[0] != b.shape[1] or not np.all(np.isfinite(b)):
        raise ValueError(f"basis must be a finite (d, d) array, got {b.shape}")
    d = int(b.shape[0])
    if np.abs(np.linalg.det(b)) == 0.0:
        raise ValueError("basis is singular")
    if np.array_equal(b, np.diag(np.diag(b))):
        return (0,) * d
    signs = np.array(list(itertools.product((-0.5, 0.5), repeat=d)))
    reach = float(np.max(np.linalg.norm(signs @ b, axis=1)))
    col = np.linalg.norm(np.linalg.inv(b), axis=0)
    radii = np.floor(0.5 + reach * col * (1.0 + 1e-12))
    return tuple(int(m) for m in radii)


def _concrete(basis: Array) -> Optional[np.ndarray]:
    """Return ``basis`` as NumPy if it is concrete, else None (under jit)."""
    try:
        return np.asarray(basis)
    except jax.errors.TracerArrayConversionError:
        return None


def torus_sq_cost(
    a: Positions,
    b: Positions,
    basis: Float[Array, "d d"],
    images: Optional[Union[int, Sequence[int]]] = None,
) -> Float[Array, "n n"]:
    """Return C_ij, the squared minimum-image Cartesian distance a_i -> b_j.

    Parameters
    ----------
    a, b : Positions
        (N, d) fractional positions (any real values; taken mod 1).
    basis : Float[Array, "d d"]
        Rows are the lattice vectors of the periodic cell.
    images : int or sequence of int, optional
        Image radius m (or one m_k per axis): after rounding the fractional
        difference, translates n with |n_k| <= m_k are scanned. None (default)
        uses ``required_images(basis)``, which is exact for any cell but needs
        a concrete basis (pass ``images`` explicitly under jit). An explicit
        radius below ``required_images`` raises for a concrete basis (D-013).
        0 means rounding only, which is exact for orthogonal cells.

    Returns
    -------
    Float[Array, "n n"]
        Squared minimum-image Cartesian distances.
    """
    assert a.ndim == 2 and a.shape == b.shape, "a and b must be (N, d) alike"
    d = a.shape[1]
    assert basis.shape == (d, d), "basis must be (d, d)"
    radii = _image_radii(basis, images, d)
    ds = a[:, None, :] - b[None, :, :]
    ds = ds - jnp.round(ds)
    if not any(radii):
        r = ds @ basis
        return jnp.sum(r * r, axis=-1)
    shifts = jnp.asarray(
        list(itertools.product(*(range(-m, m + 1) for m in radii))), dtype=ds.dtype
    )

    def scan_image(best: Array, shift: Array) -> Tuple[Array, None]:
        r = (ds + shift) @ basis
        return jnp.minimum(best, jnp.sum(r * r, axis=-1)), None

    init = jnp.full(ds.shape[:2], jnp.inf, dtype=ds.dtype)
    best, _ = jax.lax.scan(scan_image, init, shifts)
    return best


def _image_radii(
    basis: Array, images: Optional[Union[int, Sequence[int]]], d: int
) -> Tuple[int, ...]:
    """Resolve and validate the per-axis image radii for ``torus_sq_cost``."""
    concrete = _concrete(basis)
    if images is None:
        if concrete is None:
            raise ValueError(
                "torus_sq_cost: images=None needs a concrete basis; under jit "
                "pass images=required_images(basis) computed outside"
            )
        return required_images(concrete)
    radii = (int(images),) * d if isinstance(images, int) else tuple(images)
    if len(radii) != d or min(radii) < 0:
        raise ValueError(f"images must be >= 0 per axis (d={d}), got {images}")
    if concrete is not None:
        need = required_images(concrete)
        if any(m < r for m, r in zip(radii, need, strict=True)):
            raise ValueError(
                f"images={radii} is too small for this cell; the exact minimum "
                f"image needs {need} (see required_images)"
            )
    return radii


# ============================================================
# SINKHORN
# ============================================================


class SinkhornResult(eqx.Module):
    """Dual potentials of the entropic plan P_ij = exp((f_i + g_j - C_ij)/eps)."""

    f: Float[Array, " n"]
    g: Float[Array, " n"]
    n_iters: Int[Array, ""]
    marginal_error: Float[Array, ""]


def sinkhorn(
    cost: Float[Array, "n n"],
    eps: float,
    f0: Optional[Float[Array, " n"]] = None,
    g0: Optional[Float[Array, " n"]] = None,
    max_iters: int = 1000,
    tol: float = 1e-9,
    stop_on_matching: bool = False,
) -> SinkhornResult:
    """Run log-domain Sinkhorn with uniform marginals 1/N at fixed ``eps``.

    Parameters
    ----------
    cost : Float[Array, "n n"]
        Cost matrix (finite).
    eps : float
        Entropic regularization, in the units of ``cost``.
    f0, g0 : Float[Array, " n"], optional
        Warm-start potentials (zeros when omitted).
    max_iters : int
        Hard bound on the number of (f, g) update pairs.
    tol : float
        Stop when the L1 row-marginal error sum_i |sum_j P_ij - 1/N| < tol.
    stop_on_matching : bool
        Also stop as soon as the row-argmax rounding of the plan is a
        bijection (see ``round_to_matching``), which is enough when only the
        matching is wanted. Costs one extra argmax pass per iteration.

    Returns
    -------
    SinkhornResult
        Potentials after the last update pair, with the iteration count and
        the row-marginal error of the preceding pair (columns are exact).
    """
    assert cost.ndim == 2 and cost.shape[0] == cost.shape[1], "cost must be square"
    assert eps > 0.0 and max_iters > 0 and tol > 0.0, "eps, max_iters, tol > 0"
    n = cost.shape[0]
    log_n = jnp.log(jnp.asarray(n, dtype=cost.dtype))
    f = jnp.zeros(n, cost.dtype) if f0 is None else f0
    g = jnp.zeros(n, cost.dtype) if g0 is None else g0

    def body(
        state: Tuple[Array, Array, Array, Array, Array],
    ) -> Tuple[Array, Array, Array, Array, Array]:
        f_old, g_old, it, _, _ = state
        f_new = -eps * (log_n + jax.nn.logsumexp((g_old - cost) / eps, axis=1))
        # Row sums of the (f_old, g_old) plan are exp((f_old - f_new)/eps)/N.
        err = jnp.sum(jnp.abs(jnp.expm1((f_old - f_new) / eps))) / n
        g_new = -eps * (log_n + jax.nn.logsumexp((f_new[:, None] - cost) / eps, axis=0))
        if stop_on_matching:
            matched = round_to_matching(cost, g_new)[1]
        else:
            matched = jnp.asarray(False)
        return f_new, g_new, it + 1, err, matched

    def cond(state: Tuple[Array, Array, Array, Array, Array]) -> Array:
        return (state[2] < max_iters) & (state[3] >= tol) & ~state[4]

    inf = jnp.asarray(jnp.inf, dtype=cost.dtype)
    init = (f, g, jnp.asarray(0), inf, jnp.asarray(False))
    f, g, it, err, _ = jax.lax.while_loop(cond, body, init)
    return SinkhornResult(f=f, g=g, n_iters=it, marginal_error=err)


def sinkhorn_annealed(
    cost: Float[Array, "n n"],
    eps_schedule: Tuple[float, ...],
    max_iters: int = 1000,
    tol: float = 1e-9,
) -> SinkhornResult:
    """Cold-start Sinkhorn through a decreasing ``eps_schedule`` (eps scaling).

    Each stage warm-starts from the previous one; ``n_iters`` is the total.
    """
    assert len(eps_schedule) > 0, "need at least one eps"
    assert all(
        a > b for a, b in zip(eps_schedule[:-1], eps_schedule[1:], strict=True)
    ), "decreasing"
    res = sinkhorn(cost, eps_schedule[0], max_iters=max_iters, tol=tol)
    total = res.n_iters
    for eps in eps_schedule[1:]:
        res = sinkhorn(cost, eps, res.f, res.g, max_iters=max_iters, tol=tol)
        total = total + res.n_iters
    out: SinkhornResult = eqx.tree_at(lambda r: r.n_iters, res, total)
    return out


# ============================================================
# DISTANCES
# ============================================================


def round_to_matching(
    cost: Float[Array, "n n"], g: Float[Array, " n"]
) -> Tuple[Int[Array, " n"], Array]:
    """Return the row-argmax map sigma of the plan and whether it is a bijection."""
    assert cost.ndim == 2 and g.shape == (cost.shape[1],), "shape mismatch"
    sigma = jnp.argmax(g[None, :] - cost, axis=1)
    counts = jnp.zeros(cost.shape[1], jnp.int32).at[sigma].add(1)
    return sigma, jnp.all(counts == 1)


def repair_matching(
    cost: Float[Array, "n n"], g: Float[Array, " n"], sigma: Int[Array, " n"]
) -> Int[Array, " n"]:
    """Greedily turn a rounded map ``sigma`` into a bijection.

    With scores S_ij = g_j - C_ij (the log plan up to a row constant), every
    contested column keeps its highest-scoring row (lowest index on ties); the
    other rows move to their best still-free column, and the step repeats. A
    column, once held, is never lost, and each round claims at least one new
    column, so at most N rounds are needed. A bijective ``sigma`` is returned
    unchanged. The result is a feasible matching (an upper bound on W_2), not
    necessarily the optimal one.
    """
    n = cost.shape[0]
    assert cost.shape == (n, n) and sigma.shape == (n,), "shape mismatch"
    assert g.shape == (n,), "potential/cost mismatch"
    score = g[None, :] - cost
    rows = jnp.arange(n)

    def bijective(sig: Array) -> Array:
        counts = jnp.zeros(n + 1, jnp.int32).at[jnp.where(sig >= 0, sig, n)].add(1)
        return jnp.all(counts[:n] == 1)

    def body(state: Tuple[Array, Array]) -> Tuple[Array, Array]:
        sig, it = state
        col = jnp.where(sig >= 0, sig, n)
        val = jnp.where(sig >= 0, score[rows, jnp.clip(sig, 0, n - 1)], -jnp.inf)
        best = jnp.full(n + 1, -jnp.inf, score.dtype).at[col].max(val)
        is_best = (sig >= 0) & (val == best[col])
        first = jnp.full(n + 1, n).at[jnp.where(is_best, col, n)].min(rows)
        sig = jnp.where(is_best & (first[col] == rows), sig, -1)
        used = jnp.zeros(n + 1, bool).at[jnp.where(sig >= 0, sig, n)].set(True)
        pick = jnp.argmax(jnp.where(used[None, :n], -jnp.inf, score), axis=1)
        return jnp.where(sig >= 0, sig, pick), it + 1

    def cond(state: Tuple[Array, Array]) -> Array:
        return (state[1] <= n) & ~bijective(state[0])

    out, _ = jax.lax.while_loop(cond, body, (sigma, jnp.asarray(0)))
    return out


def matching_distance(
    cost: Float[Array, "n n"], sigma: Int[Array, " n"], density: float
) -> Float[Array, ""]:
    """Return rho^{1/2} [sum_i C_{i sigma(i)}]^{1/2} for a matching ``sigma``."""
    assert density > 0.0, "density must be positive"
    assert sigma.shape == (cost.shape[0],), "one target per row"
    picked = jnp.take_along_axis(cost, sigma[:, None], axis=1)
    return jnp.sqrt(density * jnp.sum(picked))


def entropic_distance(
    cost: Float[Array, "n n"], res: SinkhornResult, eps: float, density: float
) -> Float[Array, ""]:
    """Return rho^{1/2} [N <P, C>]^{1/2} for the entropic plan of ``res``."""
    assert density > 0.0 and eps > 0.0, "density and eps must be positive"
    assert res.f.shape == (cost.shape[0],), "potential/cost mismatch"
    plan = jnp.exp((res.f[:, None] + res.g[None, :] - cost) / eps)
    return jnp.sqrt(density * cost.shape[0] * jnp.sum(plan * cost))
