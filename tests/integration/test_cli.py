#!/usr/bin/env python3

"""
Integration tests of the relaxation CLI (``xla_cco.cli.relax``).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tolerances import ATOL_VMAP_S, RTOL_VMAP_E
from xla_cco.analysis.structure import isotropic_structure_factor
from xla_cco.box import cubic_box

# ----- project -----
from xla_cco.cli import relax as cli
from xla_cco.io.configpack import PackedConfig, read_configpack, write_configpack
from xla_cco.kvectors import enumerate_kvectors, k_outer_from_ka, ka_from_chi

# ============================================================
# CONSTANTS
# ============================================================

LOG_KEYS = {
    "args",
    "N",
    "d",
    "M",
    "chi",
    "a",
    "K1",
    "K2",
    "tol",
    "method",
    "settings",
    "status_counts",
    "n_success",
    "n_evals",
    "wall_seconds",
    "jax_version",
    "device",
}
PHI = 0.05

# ============================================================
# HELPERS
# ============================================================


def _poisson_pack(
    path: Path, n: int, d: int, count: int, basis: Any = None
) -> List[PackedConfig]:
    rng = np.random.default_rng(7)
    b = np.eye(d) if basis is None else basis
    cfgs = [
        PackedConfig(b, rng.uniform(size=(n, d)), 1.0, (b"a\x00\x00\x00",) * n)
        for _ in range(count)
    ]
    write_configpack(path, cfgs)
    return cfgs


def _args(tmp_path: Path, *extra: str, ka: float, dim: int = 2) -> Any:
    argv = [
        "--dim",
        str(dim),
        "--ka",
        repr(ka),
        "--phi",
        str(PHI),
        "--load",
        str(tmp_path / "in"),
        "--save",
        str(tmp_path / "out"),
        "--device",
        "cpu",
        *extra,
    ]
    return cli.build_parser().parse_args(argv)


def _ka_d2() -> float:
    return ka_from_chi(0.25, PHI, 2)


def _check_outputs(tmp_path: Path, summary: dict, cfgs: List[PackedConfig]) -> None:
    for suffix in (
        ".ConfigPack",
        ".ConfigPackIndex",
        "_Success.ConfigPack",
        "_InitConfig.ConfigPack",
        "_Sk.txt",
        "_g2.txt",
        "_Hp.txt",
        "_energies.npy",
        "_log.json",
    ):
        assert Path(str(tmp_path / "out") + suffix).is_file(), suffix
    energies = np.load(tmp_path / "out_energies.npy")
    succ = read_configpack(tmp_path / "out_Success")
    assert len(succ) == int((energies < summary["tol"]).sum()) == summary["n_success"]
    side = 64**0.5
    for name in ("out", "out_InitConfig"):
        for c in read_configpack(tmp_path / name):
            assert c.cell_size == side and np.array_equal(c.basis, np.eye(2) * side)
            assert c.names == cfgs[0].names
            assert c.relative.min() >= 0.0 and c.relative.max() < 1.0
    log = json.loads((tmp_path / "out_log.json").read_text())
    assert set(log) >= LOG_KEYS


# ============================================================
# END TO END
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("method", ["lbfgs", "descent"])
def test_cli_end_to_end_small(tmp_path: Path, method: str) -> None:
    cfgs = _poisson_pack(tmp_path / "in", 64, 2, 4)
    extra = ["--method", method, "--batch-size", "3"]
    if method == "descent":
        extra += ["--max-steps", "50000", "--grow", "2.0"]
    summary = cli.run(_args(tmp_path, *extra, ka=_ka_d2()))
    assert summary["n_success"] >= 1
    _check_outputs(tmp_path, summary, cfgs)
    assert abs(summary["chi"] - 0.25) < 0.03


@pytest.mark.medium
def test_cli_records_trajectory(tmp_path: Path) -> None:
    _poisson_pack(tmp_path / "in", 64, 2, 2)
    args = _args(tmp_path, "--record-every", "5", "--max-records", "4", ka=_ka_d2())
    cli.run(args)
    traj = np.load(tmp_path / "out_traj.npz")
    assert traj["s"].shape == (2, 4, 64, 2) and traj["energy"].shape == (2, 4)


@pytest.mark.medium
def test_cli_no_success_skips_analysis(tmp_path: Path) -> None:
    _poisson_pack(tmp_path / "in", 64, 2, 2)
    summary = cli.run(_args(tmp_path, "--max-steps", "2", ka=_ka_d2()))
    assert summary["n_success"] == 0 and "skipped" in summary["analysis"]
    for suffix in ("_Success.ConfigPack", "_Sk.txt", "_g2.txt", "_Hp.txt"):
        assert not Path(str(tmp_path / "out") + suffix).exists()
    assert (tmp_path / "out.ConfigPack").is_file()
    assert (tmp_path / "out_InitConfig.ConfigPack").is_file()


def _table(path: Path) -> np.ndarray:
    rows = [r.split("\t")[:4] for r in path.read_text().splitlines()]
    return np.array([[float(x) for x in r] for r in rows])


def _mixed_run(tmp_path: Path, k1a: str = "0") -> Any:
    """Run twice so that tol splits the 4 configs 2/2; return (summary, input)."""
    cfgs = _poisson_pack(tmp_path / "in", 64, 2, 4)
    probe = cli.build_parser().parse_args(
        [*_base(tmp_path, k1a), "--save", str(tmp_path / "probe")]
    )
    cli.run(probe)
    e = np.sort(np.load(tmp_path / "probe_energies.npy"))
    tol = float(np.sqrt(e[1] * e[2]))
    assert e[1] < tol < e[2]
    args = cli.build_parser().parse_args([*_base(tmp_path, k1a), "--tol", repr(tol)])
    return cli.run(args), cfgs


def _base(tmp_path: Path, k1a: str) -> List[str]:
    return [
        "--dim", "2", "--ka", repr(_ka_d2()), "--phi", str(PHI), "--k1a", k1a,
        "--load", str(tmp_path / "in"), "--save", str(tmp_path / "out"),
        "--device", "cpu",
    ]  # fmt: skip


@pytest.mark.critical
def test_success_mask_is_strict() -> None:
    tol = 1e-20
    m = cli.success_mask(np.array([tol, np.nextafter(tol, 0.0), 2 * tol]), tol)
    assert m.tolist() == [False, True, False]


@pytest.mark.critical
def test_cli_mixed_success_selects_exact_configs(tmp_path: Path) -> None:
    summary, cfgs = _mixed_run(tmp_path)
    energies = np.load(tmp_path / "out_energies.npy")
    ok = energies < summary["tol"]
    assert ok.sum() == 2 == summary["n_success"]
    relaxed = read_configpack(tmp_path / "out")
    succ = read_configpack(tmp_path / "out_Success")
    assert len(succ) == 2
    for got, want in zip(
        succ, [c for c, f in zip(relaxed, ok, strict=True) if f], strict=True
    ):
        assert np.array_equal(got.relative, want.relative)
    # InitConfig holds the (wrapped) INPUT positions, not the relaxed ones
    for got, inp in zip(
        read_configpack(tmp_path / "out_InitConfig"), cfgs, strict=True
    ):
        assert np.array_equal(got.relative, inp.relative - np.floor(inp.relative))
    # S(k) and g2 are computed on exactly the Success configs
    s_succ = np.stack([c.relative for c in succ])
    box = cubic_box(64, 2)
    want_sk = isotropic_structure_factor(s_succ, box, 5.0, 400.0)
    sk = _table(tmp_path / "out_Sk.txt")
    for j in range(4):
        assert np.allclose(sk[:, j], want_sk[j], rtol=1e-9, atol=0.0, equal_nan=True)
    g2 = _table(tmp_path / "out_g2.txt")
    assert g2.shape[0] == 200
    assert np.isclose(g2[-1, 0] + g2[-1, 2], 0.2 * 8.0, rtol=1e-12)  # r_max = 0.2 L
    assert np.isclose(g2[0, 0] - g2[0, 2], 0.0, atol=1e-12)


@pytest.mark.high
def test_cli_init_config_is_wrapped_input(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    rel = rng.uniform(-2.0, 3.0, size=(64, 2))
    cfg = PackedConfig(np.eye(2), rel, 1.0, (b"a\x00\x00\x00",) * 64)
    write_configpack(tmp_path / "in", [cfg])
    cli.run(_args(tmp_path, "--max-steps", "2", ka=_ka_d2()))
    got = read_configpack(tmp_path / "out_InitConfig")[0].relative
    assert np.array_equal(got, rel - np.floor(rel))
    assert got.min() >= 0.0 and got.max() < 1.0


@pytest.mark.high
def test_cli_k1_uses_shared_a(tmp_path: Path) -> None:
    _poisson_pack(tmp_path / "in", 64, 2, 1)
    ka = _ka_d2()
    summary = cli.run(_args(tmp_path, "--k1a", "0.2", "--max-steps", "2", ka=ka))
    k1 = k_outer_from_ka(0.2, PHI, 2)
    k2 = k_outer_from_ka(ka, PHI, 2)
    log = json.loads((tmp_path / "out_log.json").read_text())
    assert log["K1"] == k1 and log["K2"] == k2 == summary["K2"]
    m = enumerate_kvectors(np.eye(2) * 8.0, k1, k2).n.shape[0]
    assert log["M"] == m
    assert m < enumerate_kvectors(np.eye(2) * 8.0, 0.0, k2).n.shape[0]


@pytest.mark.high
def test_cli_trajectory_multi_batch_matches_single(tmp_path: Path) -> None:
    _poisson_pack(tmp_path / "in", 64, 2, 3)
    rec = ["--record-every", "5", "--max-records", "4", "--max-steps", "60"]
    cli.run(_args(tmp_path, *rec, "--batch-size", "2", ka=_ka_d2()))
    multi = dict(np.load(tmp_path / "out_traj.npz"))
    cli.run(_args(tmp_path, *rec, "--batch-size", "8", ka=_ka_d2()))
    single = dict(np.load(tmp_path / "out_traj.npz"))
    assert multi["s"].shape == (3, 4, 64, 2) and multi["count"].shape == (3,)
    assert multi["energy"].shape == multi["step"].shape == (3, 4)
    assert multi["iteration"].shape == (3, 4)
    assert np.array_equal(multi["count"], single["count"])
    assert np.allclose(multi["s"], single["s"], rtol=0.0, atol=ATOL_VMAP_S)
    assert np.allclose(multi["energy"], single["energy"], rtol=RTOL_VMAP_E, atol=0.0)


# ============================================================
# INPUT VALIDATION
# ============================================================


@pytest.mark.medium
def test_cli_rejects_non_unit_basis(tmp_path: Path) -> None:
    """Volume 9 with N = 16 matches neither the unit nor the scaled convention."""
    _poisson_pack(tmp_path / "in", 16, 2, 2, basis=np.eye(2) * 3.0)
    with pytest.raises(ValueError, match=r"N\^\(1/d\)"):
        cli.run(_args(tmp_path, ka=_ka_d2()))


@pytest.mark.medium
@pytest.mark.parametrize(("scale", "mode"), [(1.0, "scaled"), (4.0, "unit")])
def test_cli_input_basis_mode_is_enforced(
    tmp_path: Path, scale: float, mode: str
) -> None:
    """An explicit --input-basis rejects the other convention."""
    _poisson_pack(tmp_path / "in", 16, 2, 2, basis=np.eye(2) * scale)
    with pytest.raises(ValueError, match=f"--input-basis {mode}"):
        cli.run(_args(tmp_path, "--input-basis", mode, ka=_ka_d2()))


@pytest.mark.high
def test_regression_D015_cli_reads_its_own_output(tmp_path: Path) -> None:
    """The scaled output pack (basis x N^(1/d)) is a valid input under auto."""
    _poisson_pack(tmp_path / "in", 16, 2, 3)
    first = cli.run(_args(tmp_path, ka=_ka_d2()))
    second_dir = tmp_path / "again"
    second_dir.mkdir()
    argv = _args(tmp_path, ka=_ka_d2())
    argv.load, argv.save = tmp_path / "out", second_dir / "out"
    second = cli.run(argv)
    assert second["n_configs"] == first["n_configs"] == 3
    out1 = read_configpack(tmp_path / "out")
    out2 = read_configpack(second_dir / "out")
    for a, b in zip(out1, out2, strict=True):
        np.testing.assert_allclose(np.asarray(a.basis), np.asarray(b.basis))


@pytest.mark.medium
def test_cli_rejects_dim_mismatch(tmp_path: Path) -> None:
    _poisson_pack(tmp_path / "in", 16, 2, 2)
    with pytest.raises(ValueError, match="--dim"):
        cli.run(_args(tmp_path, ka=_ka_d2(), dim=3))


@pytest.mark.medium
@pytest.mark.parametrize(
    "extra",
    [
        ["--tol", "0"],
        ["--grow", "2.0"],
        ["--k1a", "10"],
        ["--record-every", "3"],
        ["--max-steps", "0"],
        ["--batch-size", "0"],
    ],
)
def test_cli_rejects_bad_arguments(tmp_path: Path, extra: List[str]) -> None:
    with pytest.raises(ValueError, match="--"):
        cli.run(_args(tmp_path, *extra, ka=_ka_d2()))


@pytest.mark.medium
def test_cli_gpu_guard(tmp_path: Path) -> None:
    # force the guard on any host (including GPU nodes): CPU-only JAX, no GPU
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "-m",
        "xla_cco.cli.relax",
        "--dim",
        "2",
        "--ka",
        "1.0",
        "--phi",
        "0.05",
        "--load",
        str(tmp_path / "none"),
        "--save",
        str(tmp_path / "out"),
        "--device",
        "gpu",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 1
    # JAX_PLATFORMS=cpu is the actual cause here, and must be named as such.
    assert "ERROR" in proc.stderr and "JAX_PLATFORMS=cpu" in proc.stderr


@pytest.mark.low
def test_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "xla_cco.cli.relax", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "adaptive bins" in proc.stdout and "9.99999968e-21" in proc.stdout


# ============================================================
# DEVICE SELECTION
# ============================================================


class _FakeGpu:
    """Stands in for a GPU device listed first by an initialized backend."""

    platform = "gpu"


@pytest.mark.critical
def test_regression_D085_select_device_cpu_after_backend_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-085: select_device("cpu") works after the backend started on a GPU.

    jax_platforms is ignored once the backend is initialized, so the old
    check saw the GPU first in jax.devices() and raised. The fake GPU listed
    first reproduces that on CPU-only hosts; computations must land on cpu.
    """
    real = jax.devices
    cpu = real("cpu")[0]
    assert real(), "backend initialized before the device is selected"

    def devices(backend: Any = None) -> List[Any]:
        return [_FakeGpu(), *real()] if backend is None else list(real(backend))

    monkeypatch.setattr(jax, "devices", devices)
    assert cli.select_device("cpu") == "cpu"
    assert jax.config.jax_default_device == cpu
    out = jax.jit(lambda x: 2.0 * x)(jnp.ones(3))
    assert out.devices() == {cpu}


@pytest.mark.high
def test_default_device_leaks_out_of_this_test() -> None:
    """First half of a pair: set the default device and do not undo it.

    The conftest fixture ``_restore_default_device`` must reset it before the
    next test (``test_default_device_was_restored``, which runs right after
    this one in file order). Without the reset, in-process CLI tests on the
    GPU suite would move every later test to the CPU (AUDIT-V4-fixes F1).
    """
    cpu = jax.devices("cpu")[0]
    jax.config.update("jax_default_device", cpu)
    assert jax.config.jax_default_device == cpu, "set for this test only"
    assert jnp.ones(2).devices() == {cpu}, "computations follow it"


@pytest.mark.high
def test_default_device_was_restored() -> None:
    """Second half: the leaked default device of the previous test is gone."""
    assert jax.config.jax_default_device is None, "conftest restores it"
    platform = jnp.zeros(()).devices().pop().platform
    assert platform == jax.devices()[0].platform, "back on the default backend"


@pytest.mark.high
def test_select_device_gpu_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the GPU guard bypassed and no GPU device, "gpu" still raises."""
    if any(d.platform in ("gpu", "cuda") for d in jax.devices()):
        pytest.skip("a GPU is present: the no-GPU path cannot be exercised")
    monkeypatch.setattr(cli, "require_gpu", lambda: None)
    with pytest.raises(RuntimeError, match="--device gpu"):
        cli.select_device("gpu")
    assert cli.select_device("cpu") == "cpu"


# ============================================================
# TABLE HELPERS
# ============================================================


@pytest.mark.high
@given(
    st.lists(st.integers(0, 50), min_size=1, max_size=30).filter(lambda c: sum(c) > 0),
    st.floats(0.1, 5.0),
)
def test_hp_table_is_normalized_density(counts: List[int], width: float) -> None:
    edges = np.arange(len(counts) + 1) * width
    t = cli.hp_table(np.array(counts), edges)
    assert np.isclose((t[:, 1] * 2.0 * t[:, 2]).sum(), 1.0)
    assert np.allclose(t[:, 0], 0.5 * (edges[1:] + edges[:-1]))
    assert np.all(t[np.array(counts) == 0, 3] == 0.0)


@pytest.mark.medium
def test_hp_g2_table_columns_example() -> None:
    edges = np.array([0.0, 1.0, 3.0])
    h = cli.hp_table(np.array([1, 3]), edges)
    assert np.allclose(h[:, 0], [0.5, 2.0]) and np.allclose(h[:, 2], [0.5, 1.0])
    assert np.allclose(h[:, 1], [0.25, 0.375])
    assert np.allclose(h[:, 3], [0.25, 0.375 / np.sqrt(3)])
    g = cli.g2_table(np.array([2.0, 4.0]), np.array([4, 0]), edges)
    assert np.allclose(g[:, 3], [1.0, 0.0]) and np.allclose(g[:, 1], [2.0, 4.0])


@pytest.mark.medium
@given(
    st.lists(st.integers(0, 50), min_size=1, max_size=30),
    st.floats(0.5, 20.0),
)
def test_g2_table_error_column(counts: List[int], value: float) -> None:
    edges = np.linspace(0.0, 1.0, len(counts) + 1)
    c = np.array(counts)
    t = cli.g2_table(np.full(len(counts), value), c, edges)
    assert np.allclose(t[c > 0, 3], value / np.sqrt(c[c > 0]))
    assert np.all(t[c == 0, 3] == 0.0) and np.all(t[:, 1] == value)
