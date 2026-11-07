#!/usr/bin/env python3

"""
Tests for xla_cco.minimize.flow (adaptive Dormand-Prince gradient flow).

Oracles: the closed-form solution of a linear (quadratic-energy) flow, and
the defining properties of a gradient flow (monotone energy, the Cartesian
metric of the fractional coordinates).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Array, Float

# ----- project -----
from xla_cco.box import cubic_box
from xla_cco.energy import AbstractEnergy, make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors
from xla_cco.minimize.flow import FlowSettings, gradient_flow
from xla_cco.minimize.result import (
    STATUS_BUDGET,
    STATUS_ENERGY_TOL,
    STATUS_GRAD_TOL,
)

# ============================================================
# HELPERS
# ============================================================


class _Quadratic(AbstractEnergy):
    """Phi(s) = 1/2 sum_ij c_j (s_ij - a_ij)^2 (autodiff gradient)."""

    a: Float[Array, "n d"]
    c: Float[Array, " d"]

    def __call__(self, s: Float[Array, "n d"]) -> Float[Array, ""]:
        return 0.5 * jnp.sum(self.c * (s - self.a) ** 2)


def _stealthy(n: int, d: int, shells: int) -> Tuple[AbstractEnergy, np.ndarray]:
    """Stealthy energy constraining the first ``shells`` shells of a cubic box."""
    box = cubic_box(n, d)
    side = float(np.asarray(box.basis)[0, 0])
    k_outer = 2.0 * np.pi / side * (shells + 0.5)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k_outer)
    return make_stealthy_energy(box, kset), np.asarray(box.basis)


# ============================================================
# TESTS
# ============================================================


@pytest.mark.high
@settings(deadline=None, max_examples=15)
@given(
    side=st.floats(0.5, 3.0),
    c=st.floats(0.2, 5.0),
    t_end=st.floats(0.1, 2.0),
)
def test_linear_flow_matches_closed_form(side: float, c: float, t_end: float) -> None:
    """Quadratic Phi, cubic cell of side L: s(t) = a + (s0 - a) exp(-c t / L^2).

    The Cartesian flow dr/dt = -grad_r Phi is ds/dt = -grad_s Phi / L^2 in
    fractional coordinates, so the decay rate carries the metric.
    """
    a = jnp.asarray([[0.3, 0.6]])
    s0 = jnp.asarray([[0.9, 0.1]])
    energy = _Quadratic(a=a, c=jnp.asarray([c, c]))
    basis = jnp.eye(2) * side
    fs = FlowSettings(rtol=1e-10, atol=1e-14, t_max=t_end, max_steps=100_000)
    res = gradient_flow(energy, s0, basis, fs)
    assert int(res.status) == STATUS_BUDGET  # stopped by t_max
    t = float(res.t)
    assert t == pytest.approx(t_end, rel=1e-12)
    expected = a + (s0 - a) * np.exp(-c * t / side**2)
    np.testing.assert_allclose(np.asarray(res.s), expected, rtol=0, atol=1e-9)


@pytest.mark.high
def test_energy_monotone_along_recorded_flow() -> None:
    """Phi never increases along an accepted gradient-flow trajectory."""
    energy, basis = _stealthy(16, 1, 5)
    s0 = jnp.asarray(np.random.default_rng(1).uniform(size=(16, 1)))
    fs = FlowSettings(energy_tol=1e-20, record_every=1, max_records=50_000)
    res = gradient_flow(energy, s0, jnp.asarray(basis), fs)
    n = int(res.trajectory.count)
    e = np.asarray(res.trajectory.energy[:n])
    t = np.asarray(res.trajectory.t[:n])
    assert n > 10 and np.all(np.diff(t) > 0.0)
    assert np.all(np.diff(e) <= 1e-12 * e[:-1] + 1e-30)


@pytest.mark.high
def test_flow_reaches_stealthy_ground_state() -> None:
    """Low chi (well below 1/2): the flow ends below the energy tolerance."""
    energy, basis = _stealthy(32, 1, 4)
    s0 = jnp.asarray(np.random.default_rng(2).uniform(size=(32, 1)))
    fs = FlowSettings(energy_tol=1e-20)
    res = gradient_flow(energy, s0, jnp.asarray(basis), fs)
    assert int(res.status) == STATUS_ENERGY_TOL
    assert float(res.energy) < 1e-20


@pytest.mark.medium
def test_zero_gradient_stops_immediately() -> None:
    """A single particle has grad Phi = 0 at s = 0: stop with STATUS_GRAD_TOL."""
    energy, basis = _stealthy(1, 1, 3)
    s0 = jnp.zeros((1, 1))
    res = gradient_flow(energy, s0, jnp.asarray(basis), FlowSettings())
    assert int(res.status) == STATUS_GRAD_TOL and int(res.n_steps) == 0
    np.testing.assert_array_equal(np.asarray(res.s), np.asarray(s0))


@pytest.mark.medium
def test_recording_first_last_and_capacity() -> None:
    """Row 0 is s0, the last valid row is the final state, count <= capacity."""
    energy, basis = _stealthy(16, 1, 3)
    s0 = jnp.asarray(np.random.default_rng(3).uniform(size=(16, 1)))
    fs = FlowSettings(energy_tol=1e-20, record_every=3, max_records=4000)
    res = gradient_flow(energy, s0, jnp.asarray(basis), fs)
    tr = res.trajectory
    n = int(tr.count)
    assert 2 <= n <= 4000
    np.testing.assert_array_equal(np.asarray(tr.s[0]), np.asarray(s0))
    np.testing.assert_array_equal(np.asarray(tr.s[n - 1]), np.asarray(res.s))
    assert float(tr.t[n - 1]) == float(res.t) and float(tr.t[0]) == 0.0


@pytest.mark.high
def test_vmap_matches_loop() -> None:
    """Batched flows equal per-configuration flows."""
    energy, basis = _stealthy(8, 1, 2)
    s0 = jnp.asarray(np.random.default_rng(4).uniform(size=(3, 8, 1)))
    fs = FlowSettings(energy_tol=1e-20)
    batched = jax.vmap(lambda s: gradient_flow(energy, s, jnp.asarray(basis), fs))(s0)
    for i in range(3):
        one = gradient_flow(energy, s0[i], jnp.asarray(basis), fs)
        np.testing.assert_allclose(
            np.asarray(batched.s[i]), np.asarray(one.s), atol=1e-12
        )
        assert int(batched.n_steps[i]) == int(one.n_steps)


@pytest.mark.medium
def test_rejects_float32_and_bad_settings() -> None:
    """float32 positions and non-positive tolerances are refused."""
    energy, basis = _stealthy(4, 1, 1)
    with pytest.raises(TypeError):
        gradient_flow(
            energy, jnp.zeros((4, 1), jnp.float32), jnp.asarray(basis), FlowSettings()
        )
    with pytest.raises(ValueError):
        gradient_flow(
            energy, jnp.zeros((4, 1)), jnp.asarray(basis), FlowSettings(atol=0.0)
        )


# ============================================================
# ARCLENGTH PARAMETRIZATION
# ============================================================


@pytest.mark.high
def test_arclength_traces_the_same_streamline() -> None:
    """Time and arclength modes: same curve, same flow time, same length."""
    energy, basis = _stealthy(16, 1, 4)
    s0 = jnp.asarray(np.random.default_rng(5).uniform(size=(16, 1)))
    common = dict(energy_tol=1e-16, rtol=1e-9, record_every=1, max_records=60_000)
    a = gradient_flow(energy, s0, jnp.asarray(basis), FlowSettings(**common))
    b = gradient_flow(
        energy,
        s0,
        jnp.asarray(basis),
        FlowSettings(parametrization="arclength", **common),
    )
    assert int(a.status) == int(b.status) == STATUS_ENERGY_TOL
    assert float(b.length) == pytest.approx(float(a.length), rel=1e-6)
    na, nb = int(a.trajectory.count), int(b.trajectory.count)
    la, lb = np.asarray(a.trajectory.length[:na]), np.asarray(b.trajectory.length[:nb])
    # Clock check without interpolation: rerun the true-time flow to the
    # times recovered in arclength mode and compare positions there.
    tb = np.asarray(b.trajectory.t[:nb])
    for k in (nb // 4, nb // 2, (3 * nb) // 4):
        fs = FlowSettings(rtol=1e-9, t_max=float(tb[k]))
        at_t = gradient_flow(energy, s0, jnp.asarray(basis), fs)
        np.testing.assert_allclose(
            np.asarray(at_t.s), np.asarray(b.trajectory.s[k]), atol=1e-6
        )
    side = float(basis[0, 0])
    xa = np.asarray(a.trajectory.s[:na, :, 0]) * side
    xb = np.asarray(b.trajectory.s[:nb, :, 0]) * side
    inside = lb <= la[-1]
    for col in range(16):
        on_a = np.interp(lb[inside], la, xa[:, col])
        np.testing.assert_allclose(xb[inside, col], on_a, atol=1e-4)  # interp.


@pytest.mark.high
def test_arclength_recovers_linear_flow_time() -> None:
    """Quadratic Phi: t_end = (L^2 / c) ln|s0 - a| / |s_end - a|; length = |dx|."""
    a, c, side = 0.3, 2.0, 1.7
    energy = _Quadratic(a=jnp.asarray([[a]]), c=jnp.asarray([c]))
    s0 = jnp.asarray([[0.9]])
    fs = FlowSettings(parametrization="arclength", energy_tol=1e-14, rtol=1e-10)
    res = gradient_flow(energy, s0, jnp.eye(1) * side, fs)
    s_end = float(res.s[0, 0])
    expected_t = side**2 / c * np.log(abs(0.9 - a) / abs(s_end - a))
    assert float(res.t) == pytest.approx(expected_t, rel=1e-6)
    assert float(res.length) == pytest.approx(side * abs(0.9 - s_end), rel=1e-10)


@pytest.mark.medium
def test_arclength_rejects_t_max() -> None:
    """t_max is defined on the physical clock, which arclength mode does not step."""
    energy, basis = _stealthy(4, 1, 1)
    fs = FlowSettings(parametrization="arclength", t_max=1.0)
    with pytest.raises(ValueError, match="t_max"):
        gradient_flow(energy, jnp.zeros((4, 1)), jnp.asarray(basis), fs)


# ============================================================
# ROSENBROCK (STIFF) METHOD
# ============================================================


@pytest.mark.high
def test_rosenbrock_linear_flow_matches_closed_form() -> None:
    """ROS2 on a quadratic energy follows s(t) = a + (s0 - a) exp(-c t / L^2)."""
    a = jnp.asarray([[0.3, 0.6]])
    s0 = jnp.asarray([[0.9, 0.1]])
    energy = _Quadratic(a=a, c=jnp.asarray([1.5, 1.5]))
    fs = FlowSettings(method="rosenbrock", rtol=1e-3, atol=1e-10, t_max=1.3)
    res = gradient_flow(energy, s0, jnp.eye(2) * 1.2, fs)
    expected = a + (s0 - a) * np.exp(-1.5 * 1.3 / 1.2**2)
    np.testing.assert_allclose(np.asarray(res.s), expected, atol=1e-5)


@pytest.mark.high
@pytest.mark.parametrize("frac", [0.1, 0.5, 0.9])
def test_rosenbrock_agrees_with_dopri(frac: float) -> None:
    """Both methods integrate the same flow: equal positions at equal times."""
    energy, basis = _stealthy(16, 1, 4)
    s0 = jnp.asarray(np.random.default_rng(6).uniform(size=(16, 1)))
    full = gradient_flow(energy, s0, jnp.asarray(basis), FlowSettings(energy_tol=1e-12))
    t_stop = frac * float(full.t)
    ref = gradient_flow(
        energy, s0, jnp.asarray(basis), FlowSettings(rtol=1e-10, t_max=t_stop)
    )
    ros = gradient_flow(
        energy,
        s0,
        jnp.asarray(basis),
        FlowSettings(method="rosenbrock", rtol=1e-3, atol=1e-10, t_max=t_stop),
    )
    np.testing.assert_allclose(np.asarray(ros.s), np.asarray(ref.s), atol=1e-5)


@pytest.mark.medium
def test_rosenbrock_needs_fewer_steps_when_stiff() -> None:
    """Curvatures 1e4 and 1: explicit steps are capped near 3e-4 by stability,
    while the L-stable ROS2 step is limited only by accuracy on the slow mode."""
    energy = _Quadratic(a=jnp.asarray([[0.3, 0.6]]), c=jnp.asarray([1e4, 1.0]))
    s0 = jnp.asarray([[0.4, 0.9]])
    common = dict(energy_tol=1e-20, max_steps=400_000)
    dop = gradient_flow(energy, s0, jnp.eye(2), FlowSettings(**common))
    ros_fs = FlowSettings(method="rosenbrock", rtol=1e-3, atol=1e-8, **common)
    ros = gradient_flow(energy, s0, jnp.eye(2), ros_fs)
    assert int(dop.status) == int(ros.status) == STATUS_ENERGY_TOL
    assert int(ros.n_steps) < int(dop.n_steps) / 5


@pytest.mark.medium
def test_rosenbrock_rejects_arclength() -> None:
    """ROS2 integrates the physical-time parametrization only."""
    energy, basis = _stealthy(4, 1, 1)
    fs = FlowSettings(method="rosenbrock", parametrization="arclength")
    with pytest.raises(ValueError, match="rosenbrock"):
        gradient_flow(energy, jnp.zeros((4, 1)), jnp.asarray(basis), fs)


@pytest.mark.high
def test_resume_continues_the_same_flow() -> None:
    """Two chunks restarted from (s, dt) match one uninterrupted run."""
    energy, basis = _stealthy(16, 1, 4)
    s0 = jnp.asarray(np.random.default_rng(8).uniform(size=(16, 1)))
    b = jnp.asarray(basis)
    whole = gradient_flow(energy, s0, b, FlowSettings(energy_tol=1e-14))
    first = gradient_flow(energy, s0, b, FlowSettings(max_steps=60))
    second = gradient_flow(
        energy,
        first.s,
        b,
        FlowSettings(energy_tol=1e-14, dt_init=float(first.dt)),
    )
    assert int(second.status) == STATUS_ENERGY_TOL
    np.testing.assert_allclose(np.asarray(second.s), np.asarray(whole.s), atol=1e-6)
    total_t = float(first.t) + float(second.t)
    assert total_t == pytest.approx(float(whole.t), rel=1e-3)


@pytest.mark.medium
def test_dt0_below_floor_stops_immediately() -> None:
    """A resume with dt0 = 0 (a frozen lane) does not move."""
    from xla_cco.minimize.result import STATUS_STEP_FLOOR

    energy, basis = _stealthy(8, 1, 2)
    s0 = jnp.asarray(np.random.default_rng(9).uniform(size=(8, 1)))
    res = gradient_flow(
        energy, s0, jnp.asarray(basis), FlowSettings(), dt0=jnp.asarray(0.0)
    )
    assert int(res.status) == STATUS_STEP_FLOOR and int(res.n_steps) == 0
    np.testing.assert_array_equal(np.asarray(res.s), np.asarray(s0))
