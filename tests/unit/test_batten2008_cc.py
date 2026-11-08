#!/usr/bin/env python3

"""
Tests for the collective-coordinate analyses of the 2008 flows
(experiments/scripts/batten2008_{modes,hessian,floor}.py).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import sys
from fractions import Fraction
from pathlib import Path

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ----- project -----
from xla_cco.energy import collective_density
from xla_cco.minimize import relax
from xla_cco.minimize.lbfgs import LBFGSSettings

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
from batten2008_floor import (  # noqa: E402
    FiberEnergy,
    grid_starts,
    plane_offsets,
    relax_points,
)
from batten2008_flows import build_energy, load_system  # noqa: E402
from batten2008_hessian import (  # noqa: E402
    density_wave_stiffness,
    level_frames,
    qvectors,
    section_axes,
    tenfold_distance,
    wavelength_power,
)
from batten2008_modes import (  # noqa: E402
    mode_indices,
    phase_stats,
    relax_times,
    structure_factors,
)
from batten2008_offplane import (  # noqa: E402
    arclength_path,
    kl_modes,
    participation,
    per_particle,
    plane_residual,
    remove_bow,
    tangent_correlation,
)
from batten2008_slices import pair_axes  # noqa: E402
from batten2008_tail import restart_index, tail_frames  # noqa: E402
from batten2008_wasserstein import frame_subset, torus_match  # noqa: E402

# ============================================================
# HELPERS
# ============================================================

N, CHI = 12, Fraction(1, 3)  # M = 4 modes; side L = 12


def fiber(seed: int) -> FiberEnergy:
    """FiberEnergy for N = 12, chi = 1/3, on a random orthonormal plane."""
    energy, _, _ = build_energy(N, CHI)
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(N, 2)))
    return FiberEnergy(base=energy, q=jnp.asarray(q.T[:, :, None]))


def random_s(seed: int) -> jnp.ndarray:
    """Random fractional configuration (N, 1)."""
    return jnp.asarray(np.random.default_rng(seed).uniform(size=(N, 1)))


# ============================================================
# TESTS: FIBER ENERGY (valley floor)
# ============================================================


@pytest.mark.high
@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=20, deadline=None)
def test_fiber_value_is_phi_and_gradient_is_off_plane(seed: int) -> None:
    f = fiber(seed)
    s = random_s(seed + 1)
    value, grad = f.value_and_grad(s)
    base_value, base_grad = f.base.value_and_grad(s)
    q = np.asarray(f.q)[:, :, 0]
    g = np.asarray(grad)[:, 0]
    assert float(value) == pytest.approx(float(base_value), rel=1e-14)
    assert np.allclose(q @ g, 0.0, atol=1e-12 * np.linalg.norm(np.asarray(base_grad)))
    # the removed part is exactly the in-plane part of the true gradient
    gb = np.asarray(base_grad)[:, 0]
    assert np.allclose(g, gb - q.T @ (q @ gb), rtol=1e-12, atol=1e-14)


@pytest.mark.high
@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=20, deadline=None)
def test_fiber_autodiff_matches_analytic_projection(seed: int) -> None:
    f = fiber(seed)
    s = random_s(seed + 2)
    auto_value, auto_grad = jax.value_and_grad(f.__call__)(s)
    value, grad = f.value_and_grad(s)
    assert float(auto_value) == pytest.approx(float(value), rel=1e-13)
    assert np.allclose(np.asarray(auto_grad), np.asarray(grad), rtol=1e-10, atol=1e-12)


@pytest.mark.high
def test_relax_on_fiber_keeps_in_plane_coordinates_and_lowers_phi() -> None:
    f = fiber(7)
    s0 = jnp.stack([random_s(s) for s in (11, 12, 13)])
    res = relax(f, s0, LBFGSSettings(energy_tol=1e-20, max_evals_per_pass=400))
    d = np.asarray(res.s)[:, :, 0] - np.asarray(s0)[:, :, 0]
    d -= np.round(d)  # result is wrapped into [0, 1)
    q = np.asarray(f.q)[:, :, 0]
    assert np.max(np.abs(d @ q.T)) < 1e-10
    phi0 = np.asarray(jax.vmap(f.base)(s0))
    assert np.all(np.asarray(res.energy) <= phi0)


# ============================================================
# TESTS: GRID STARTS
# ============================================================


@pytest.mark.medium
def test_plane_offsets_reassemble_the_frames() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(9, N))
    e, _ = np.linalg.qr(rng.normal(size=(N, 2)))
    axes = (x.mean(axis=0), e[:, 0], e[:, 1])
    ab, off = plane_offsets(x, axes)
    back = axes[0] + ab[:, :1] * axes[1] + ab[:, 1:] * axes[2] + off
    assert np.allclose(back, x, atol=1e-12)
    assert np.allclose(off @ e, 0.0, atol=1e-12)


@pytest.mark.medium
def test_grid_point_on_a_frame_starts_at_that_frame() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, N))
    e, _ = np.linalg.qr(rng.normal(size=(N, 2)))
    axes = (np.zeros(N), e[:, 0], e[:, 1])
    ab, off = plane_offsets(x, axes)
    ra = np.sort(np.append(np.linspace(-5, 5, 4), ab[2, 0]))
    rb = np.sort(np.append(np.linspace(-5, 5, 4), ab[2, 1]))
    starts = grid_starts(ra, rb, ab, off, axes)
    i, j = (
        int(np.flatnonzero(rb == ab[2, 1])[0]),
        int(np.flatnonzero(ra == ab[2, 0])[0]),
    )
    assert np.allclose(starts[i * ra.shape[0] + j], x[2], atol=1e-12)


# ============================================================
# TESTS: MODES
# ============================================================


@pytest.mark.high
@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=20, deadline=None)
def test_mode_structure_factors_sum_to_phi(seed: int) -> None:
    energy, _, m = build_energy(N, CHI)
    nv = jnp.asarray(np.arange(1.0, m + 1)[:, None])
    rho_fn = jax.jit(jax.vmap(lambda s: collective_density(s, nv)))
    s = np.random.default_rng(seed).uniform(size=(3, N, 1))
    re, im = structure_factors(s, m, rho_fn)
    phi = np.asarray(jax.vmap(energy)(jnp.asarray(s)))
    assert np.allclose((re**2 + im**2).sum(axis=1), phi, rtol=1e-12, atol=1e-14)


@pytest.mark.medium
def test_relax_times_last_crossing_example() -> None:
    u = np.array([0.0, 0.25, 0.5, 1.0])
    sk = np.array([[1.0, 1e-3], [1e-3, 1e-3], [1.0, 1e-3], [1e-9, 1e-9]])
    out = relax_times(u, sk, (1e-2, 1e-6))
    # mode 0 is last above 1e-2 at u = 0.5; mode 1 never above 1e-2
    assert out[:, 0].tolist() == [0.5, 0.0]
    assert out[:, 1].tolist() == [0.5, 0.5]


@pytest.mark.medium
@given(m=st.integers(1, 700))
def test_mode_indices_in_range_and_include_ends(m: int) -> None:
    kmag = 2.0 * np.pi * np.arange(1, m + 1) / 1000.0  # 1D: row n is k_n
    picks = mode_indices(kmag, (0.0, 0.5, 1.0))
    assert picks[0] == 1 and picks[-1] == m
    assert np.all(np.diff(picks) > 0) and np.all((picks >= 1) & (picks <= m))


# ============================================================
# TESTS: HESSIAN SECTIONS
# ============================================================


@pytest.mark.medium
def test_level_frames_first_at_or_below_example() -> None:
    e = np.array([100.0, 10.0, 1.0, 1e-3, 1e-9])
    levels = (3.0, 2.0, 1.5, 0.0, -5.0, -12.0)
    assert level_frames(e, levels).tolist() == [0, 0, 1, 2, 4, 4]


@pytest.mark.medium
@settings(deadline=None)
@given(
    st.lists(st.floats(0.01, 3.0), min_size=1, max_size=30),
    st.lists(st.floats(-40.0, 5.0), min_size=1, max_size=10),
)
def test_level_frames_is_the_first_crossing(drops: list, levels: list) -> None:
    e = 10.0 ** (2.0 - np.cumsum([0.0] + drops))  # strictly decreasing energies
    idx = level_frames(e, tuple(levels))
    loge = np.log10(e)
    for k, lev in zip(idx, levels, strict=True):
        if loge[-1] > lev:  # never reached: the last frame
            assert k == e.shape[0] - 1
        else:
            assert loge[k] <= lev and (k == 0 or loge[k - 1] > lev)


@pytest.mark.medium
def test_restart_index_last_frame_above() -> None:
    e = np.array([1.0, 1e-4, 1e-8, 1e-10, 1e-14])
    assert restart_index(e, 1e-9) == 2
    assert restart_index(e, 10.0) == 0


@pytest.mark.medium
def test_tail_frames_splices_at_the_restart() -> None:
    flows = {
        "traj_count": np.array([4]),
        "traj_t": np.array([[0.0, 1.0, 2.0, 50.0]]),
        "traj_s": np.arange(4.0).reshape(1, 4, 1, 1),
        "traj_energy": np.array([[1.0, 1e-3, 1e-9, 1e-20]]),
    }
    tail = {
        "start_index": np.array([2]),
        "traj_count": np.array([3]),
        "traj_t": np.array([[2.0, 3.0, 4.0, np.nan]]),
        "traj_s": np.array([2.0, 2.5, 2.75, np.nan]).reshape(1, 4, 1, 1),
        "traj_energy": np.array([[1e-9, 1e-12, 1e-15, np.nan]]),
    }
    t, s, e = tail_frames(flows, tail, 0)
    assert t.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert s[:, 0, 0].tolist() == [0.0, 1.0, 2.0, 2.5, 2.75]
    assert e.tolist() == [1.0, 1e-3, 1e-9, 1e-12, 1e-15]
    t0, _, _ = tail_frames(flows, {}, 0)  # no tail: the stored rows
    assert t0.tolist() == [0.0, 1.0, 2.0, 50.0]


@pytest.mark.medium
def test_section_axes_orthonormal_and_downhill() -> None:
    rng = np.random.default_rng(5)
    grad = rng.normal(size=N)
    vecs, _ = np.linalg.qr(rng.normal(size=(N, N)))
    tau, v = section_axes(grad, vecs)
    assert tau @ grad < 0.0
    assert np.allclose([tau @ tau, v @ v, tau @ v], [1.0, 1.0, 0.0], atol=1e-12)


@pytest.mark.medium
def test_tenfold_distance_example() -> None:
    x = np.arange(5.0)[:, None] * np.ones((1, 2))
    e = np.array([100.0, 30.0, 9.0, 1.0, 0.5])
    assert tenfold_distance(x, e, 0) == pytest.approx(2.0 * np.sqrt(2.0))
    assert tenfold_distance(x, e, 2) == pytest.approx(2.0 * np.sqrt(2.0))  # 0.5 <= 0.9
    assert tenfold_distance(x, e, 3) == pytest.approx(np.sqrt(2.0))  # last frame


@pytest.mark.high
def test_density_wave_stiffness_is_k_squared_at_the_lattice() -> None:
    energy, basis, m = build_energy(N, CHI)  # N = 12, M = 4 < N - M: no aliasing
    side = float(basis[0, 0])
    x = np.arange(N, dtype=float) + 0.3  # translated integer lattice: a ground state
    h = np.asarray(energy.hessian(jnp.asarray(x[:, None] / side))) / side**2
    kvec = (2.0 * np.pi * np.arange(1, m + 1) / N)[:, None]
    stiff = density_wave_stiffness(x[:, None], h, kvec)
    k2 = (2.0 * np.pi * np.arange(1, m + 1) / N) ** 2
    assert np.allclose(stiff, k2[:, None], rtol=1e-10)


@pytest.mark.medium
def test_wavelength_power_of_a_single_cosine() -> None:
    n, mode = 16, 3
    x = np.arange(n, dtype=float)
    v = np.cos(2.0 * np.pi * mode * x / n)
    q = qvectors(n, 1, float(n))
    power = wavelength_power(x[:, None], (v / np.linalg.norm(v))[:, None], q)
    assert int(np.argmax(power)) + 1 == mode
    assert power[mode - 1] == pytest.approx(n / 2.0)
    assert np.allclose(np.delete(power[: n // 2 - 1], mode - 1), 0.0, atol=1e-20)
    assert np.all(np.isnan(wavelength_power(x[:, None], np.zeros((n, 0)), q)))


@pytest.mark.medium
def test_phase_stats_radial_decay_and_one_turn() -> None:
    t = np.linspace(0.0, 1.0, 401)
    radial = 1e-2 * np.exp(-20.0 * t) * np.exp(1j * 0.7)  # fixed phase
    turning = 1e-2 * np.exp(1j * 2.0 * np.pi * t)  # one full turn, constant S
    rho = np.stack([radial, turning], axis=1)
    out = phase_stats(rho.real, rho.imag)
    assert out["turns"][0] == pytest.approx(0.0, abs=1e-12)
    assert out["turns"][1] == pytest.approx(1.0, rel=1e-9)
    assert out["s_lock"][0] == pytest.approx(1e-4)  # locked from the start
    assert out["unresolved"].tolist() == [0.0, 0.0]


# ============================================================
# TESTS: ANY DIMENSION (2D flows)
# ============================================================


@pytest.mark.high
def test_load_system_rebuilds_the_1d_energy(tmp_path: Path) -> None:
    energy, basis, m = build_energy(N, CHI)
    k_cut = 2.0 * np.pi / N * (m + 0.5)
    (tmp_path / "flows_chi1-3.json").write_text(f'{{"K": {k_cut}, "M": {m}}}')
    np.savez(tmp_path / "flows_chi1-3.npz", basis=basis)
    rebuilt, b2, nvec, kmag = load_system(tmp_path, "1-3")
    assert np.array_equal(np.asarray(rebuilt.n_vectors), np.asarray(energy.n_vectors))
    assert np.allclose(kmag, 2.0 * np.pi * np.arange(1, m + 1) / N)
    s = random_s(3)
    assert float(rebuilt(s)) == pytest.approx(float(energy(s)), rel=1e-14)


@pytest.mark.high
@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=25, deadline=None)
def test_torus_match_is_optimal_against_all_permutations(seed: int) -> None:
    import itertools

    rng = np.random.default_rng(seed)
    side = 3.0
    a, b = rng.uniform(0, side, (5, 2)), rng.uniform(-side, 2 * side, (5, 2))
    w, disp, _ = torus_match(a, b, side)
    best = np.inf
    for perm in itertools.permutations(range(5)):
        dd = b[list(perm)] - a
        dd -= side * np.round(dd / side)
        best = min(best, float(np.sum(dd * dd)))
    assert w**2 == pytest.approx(best, rel=1e-12)
    assert np.allclose(np.sum(disp**2), w**2) and np.all(np.abs(disp) <= side / 2)


@pytest.mark.medium
def test_pair_axes_2d_moves_each_particle_along_its_net_displacement() -> None:
    rng = np.random.default_rng(6)
    n, d = 7, 2
    x0 = rng.normal(size=(n, d))
    x1 = x0.copy()
    x1[2] += [3.0, 4.0]  # largest move
    x1[5] += [0.0, -2.0]
    path = np.stack([x0, 0.5 * (x0 + x1), x1]).reshape(3, -1)
    origin, ei, ej, i, j = pair_axes(path, d)
    assert (i, j) == (2, 5)
    assert np.allclose(ei.reshape(n, d)[2], [0.6, 0.8]) and np.allclose(
        ej.reshape(n, d)[5], [0.0, -1.0]
    )
    back = origin + (path[-1] @ ei) * ei + (path[-1] @ ej) * ej
    assert np.allclose(back, path[-1])  # the end state lies in the slice


@pytest.mark.medium
def test_qvectors_2d_half_space_sorted() -> None:
    q = qvectors(16, 2, 4.0)
    ints = np.round(q * 4.0 / (2.0 * np.pi)).astype(int)
    keys = {tuple(r) for r in ints}
    assert len(keys) == ints.shape[0] == (9 * 9 - 1) // 2
    assert not any((-a, -b) in keys for a, b in keys)
    assert np.all(np.diff(np.linalg.norm(q, axis=1)) >= -1e-12)


@pytest.mark.medium
@given(count=st.integers(2, 5000), size=st.integers(0, 400))
def test_frame_subset_keeps_ends_sorted_unique(count: int, size: int) -> None:
    sel = frame_subset(count, size)
    assert sel[0] == 0 and sel[-1] == count - 1
    assert np.all(np.diff(sel) > 0)
    assert (
        sel.shape[0] == count
        if size <= 0 or size >= count
        else sel.shape[0] <= size + 2
    )


@pytest.mark.high
@pytest.mark.parametrize("d", [1, 2])
def test_relax_points_runs_and_keeps_plane_coordinates(d: int) -> None:
    n = 12
    rng = np.random.default_rng(8)
    if d == 1:
        energy, _, _ = build_energy(n, CHI)
        side = float(n)
    else:
        from xla_cco.box import Box
        from xla_cco.energy import make_stealthy_energy
        from xla_cco.kvectors import enumerate_kvectors

        side = float(np.sqrt(n))
        basis = side * np.eye(2)
        kset = enumerate_kvectors(basis, 0.0, 2.0 * np.pi / side * 1.5)
        energy = make_stealthy_energy(Box(basis=jnp.asarray(basis)), kset)
    q, _ = np.linalg.qr(rng.normal(size=(n * d, 2)))
    fib = FiberEnergy(base=energy, q=jnp.asarray(q.T.reshape(2, n, d)))
    pts = rng.uniform(0.0, side, (3, n * d))
    out = relax_points(fib, pts, side, LBFGSSettings(max_evals_per_pass=60), 2)
    assert out["phi"].shape == (3,) and out["moved"].shape == (3,)
    assert np.all(out["drift"] < 1e-8)


# ============================================================
# TESTS: OFF-PLANE SHAPE
# ============================================================


@pytest.mark.medium
def test_arclength_path_straight_line_is_uniform() -> None:
    f = np.linspace(0.0, 0.3, 7)[:, None, None] * np.ones((1, 4, 1))  # (F, N, d)
    y, length = arclength_path(f, 10.0, 5)
    assert np.isclose(length, 0.3 * 10.0 * 2.0)  # 4 particles move 3 each
    assert np.allclose(np.linalg.norm(y, axis=1), np.linspace(0.0, length, 5))


@pytest.mark.medium
@settings(deadline=None)
@given(st.integers(0, 2**31 - 1))
def test_plane_residual_pins_and_particle_rms(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(size=(60, 12)), axis=0)
    y = np.vstack([np.zeros((1, 12)), y])
    pr = plane_residual(y)
    w = pr["w"]
    star = int(pr["star"])
    for i in (0, star, y.shape[0] - 1):  # the plane holds r(0), r(s*), r(inf)
        assert np.linalg.norm(w[i]) < 1e-9 * np.linalg.norm(y).max()
    chord = float(pr["chord"])
    amp = per_particle(w, 6, chord)  # N = 6 particles, d = 2
    rms = np.sqrt(np.mean(amp**2, axis=1))
    assert np.allclose(rms, np.linalg.norm(w, axis=1) / chord)
    assert np.isclose(pr["coords"][-1, 0], 1.0)


@pytest.mark.medium
def test_kl_modes_recover_sine_mixture() -> None:
    sg = np.linspace(0.0, 1.0, 201)
    u, v = np.eye(5)[0], np.eye(5)[3]
    p = np.outer(np.sin(np.pi * sg), u) + 0.3 * np.outer(np.sin(2 * np.pi * sg), v)
    kl = kl_modes(p, 3)
    assert np.allclose(kl["frac"][:2], [1.0 / 1.09, 0.09 / 1.09], atol=1e-3)
    assert np.all(kl["overlap"][:2] > 0.999)
    left = remove_bow(p)  # only the sin(2 pi s) part survives
    assert np.allclose(left, 0.3 * np.outer(np.sin(2 * np.pi * sg), v), atol=1e-9)


@pytest.mark.medium
def test_participation_and_tangent_examples() -> None:
    assert np.allclose(participation(np.ones((3, 8))), 1.0)
    one = np.zeros((3, 8))
    one[:, 2] = 1.0
    assert np.allclose(participation(one), 1.0 / 8.0)
    line = np.outer(np.arange(10.0), np.ones(3))
    assert np.allclose(tangent_correlation(line, 4), 1.0)
