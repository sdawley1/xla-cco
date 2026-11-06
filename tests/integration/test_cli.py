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

CPP_RUN = Path("/scratch/gpfs/TORQUATO/sd2402/cco_runs/p3d-N100-phi0.05-chi0.05-val0.0")
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


def test_cli_records_trajectory(tmp_path: Path) -> None:
    _poisson_pack(tmp_path / "in", 64, 2, 2)
    args = _args(tmp_path, "--record-every", "5", "--max-records", "4", ka=_ka_d2())
    cli.run(args)
    traj = np.load(tmp_path / "out_traj.npz")
    assert traj["s"].shape == (2, 4, 64, 2) and traj["energy"].shape == (2, 4)


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


def test_success_mask_is_strict() -> None:
    tol = 1e-20
    m = cli.success_mask(np.array([tol, np.nextafter(tol, 0.0), 2 * tol]), tol)
    assert m.tolist() == [False, True, False]


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


def test_cli_init_config_is_wrapped_input(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    rel = rng.uniform(-2.0, 3.0, size=(64, 2))
    cfg = PackedConfig(np.eye(2), rel, 1.0, (b"a\x00\x00\x00",) * 64)
    write_configpack(tmp_path / "in", [cfg])
    cli.run(_args(tmp_path, "--max-steps", "2", ka=_ka_d2()))
    got = read_configpack(tmp_path / "out_InitConfig")[0].relative
    assert np.array_equal(got, rel - np.floor(rel))
    assert got.min() >= 0.0 and got.max() < 1.0


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


def test_cli_rejects_non_unit_basis(tmp_path: Path) -> None:
    _poisson_pack(tmp_path / "in", 16, 2, 2, basis=np.eye(2) * 4.0)
    with pytest.raises(ValueError, match=r"N\^\(1/d\)"):
        cli.run(_args(tmp_path, ka=_ka_d2()))


def test_cli_rejects_dim_mismatch(tmp_path: Path) -> None:
    _poisson_pack(tmp_path / "in", 16, 2, 2)
    with pytest.raises(ValueError, match="--dim"):
        cli.run(_args(tmp_path, ka=_ka_d2(), dim=3))


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
    assert "ERROR" in proc.stderr and "--gres=gpu" in proc.stderr


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
# TABLE HELPERS
# ============================================================


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


def test_hp_g2_table_columns_example() -> None:
    edges = np.array([0.0, 1.0, 3.0])
    h = cli.hp_table(np.array([1, 3]), edges)
    assert np.allclose(h[:, 0], [0.5, 2.0]) and np.allclose(h[:, 2], [0.5, 1.0])
    assert np.allclose(h[:, 1], [0.25, 0.375])
    assert np.allclose(h[:, 3], [0.25, 0.375 / np.sqrt(3)])
    g = cli.g2_table(np.array([2.0, 4.0]), np.array([4, 0]), edges)
    assert np.allclose(g[:, 3], [1.0, 0.0]) and np.allclose(g[:, 1], [2.0, 4.0])


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


# ============================================================
# C++ HEADER PARITY
# ============================================================


@pytest.mark.della
def test_cli_headers_match_cpp(tmp_path: Path) -> None:
    ic = read_configpack(CPP_RUN / "ic" / "chunk_0000", [0, 1, 2, 3])
    write_configpack(tmp_path / "in", ic)
    ka = 0.5963271031078479
    args = cli.build_parser().parse_args(
        [
            "--dim",
            "3",
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
            "--max-steps",
            "2000",
        ]
    )
    summary = cli.run(args)
    assert summary["M"] == 13 and abs(summary["chi"] - 0.043771) < 5e-7
    cpp_base = CPP_RUN / "task_0000" / "ccopt-t0000"
    pairs = [
        ("out", str(cpp_base)),
        ("out_InitConfig", str(cpp_base) + "_InitConfig"),
        ("out_Success", str(cpp_base) + "_Success"),
    ]
    for ours_name, cpp_name in pairs:
        ours = read_configpack(tmp_path / ours_name, [0])[0]
        ref = read_configpack(Path(cpp_name), [0])[0]
        assert ours.relative.shape[1] == ref.relative.shape[1]
        assert ours.cell_size == ref.cell_size
        assert np.array_equal(ours.basis, ref.basis)
        assert ours.names == ref.names
