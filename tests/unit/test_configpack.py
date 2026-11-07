#!/usr/bin/env python3

"""
Example tests of the ConfigPack reader, writer and fractional stacking.

Layout facts follow the C++ ConfigPack format.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path
from typing import List

# ----- numerics -----
import numpy as np
import pytest

# ----- project -----
from xla_cco.io.configpack import (
    PackedConfig,
    read_configpack,
    stack_fractional,
    write_configpack,
)

# ============================================================
# HELPERS
# ============================================================


def _cfg(
    n: int, d: int, seed: int = 0, scale: float = 1.0, cell: float = 1.5
) -> PackedConfig:
    rng = np.random.default_rng(seed)
    names = tuple(bytes([97 + (i % 26), 0, 0, 0]) for i in range(n))
    return PackedConfig(
        basis=np.eye(d) * scale,
        relative=rng.uniform(size=(n, d)),
        cell_size=cell,
        names=names,
    )


def _write(tmp: Path, cfgs: List[PackedConfig], name: str = "p") -> Path:
    prefix = tmp / name
    write_configpack(prefix, cfgs)
    return prefix


# ============================================================
# TESTS: READ / WRITE
# ============================================================


def test_round_trip_example(tmp_path: Path) -> None:
    """Fields survive write -> read exactly."""
    cfgs = [_cfg(5, 2, 1), _cfg(3, 3, 2)]
    got = read_configpack(_write(tmp_path, cfgs))
    assert len(got) == 2
    for a, b in zip(cfgs, got, strict=True):
        np.testing.assert_array_equal(a.basis, b.basis)
        np.testing.assert_array_equal(a.relative, b.relative)
        assert a.cell_size == b.cell_size
        assert a.names == b.names


def test_names_keep_trailing_nulls(tmp_path: Path) -> None:
    """Names are returned verbatim, not null-stripped."""
    cfg = _cfg(2, 1)
    got = read_configpack(_write(tmp_path, [cfg]))[0]
    assert got.names[0] == b"a\x00\x00\x00"


def test_index_offsets_consistent(tmp_path: Path) -> None:
    """Index = NConfig then offsets; 3D N=10 record is 14+72+10*28 bytes."""
    cfgs = [_cfg(10, 3, 1), _cfg(4, 2, 2), _cfg(1, 1, 3)]
    prefix = _write(tmp_path, cfgs)
    idx = np.fromfile(str(prefix) + ".ConfigPackIndex", dtype="<i8")
    assert idx[0] == 3 and idx.size == 4
    sizes = [14 + 8 * d * d + n * (8 * d + 4) for n, d in ((10, 3), (4, 2), (1, 1))]
    assert idx[1] == 0
    assert idx[2] == sizes[0]
    assert idx[3] == sizes[0] + sizes[1]
    assert Path(str(prefix) + ".ConfigPack").stat().st_size == sum(sizes)
    assert 14 + 72 + 100 * 28 == 2886


@pytest.mark.parametrize("sel", [[2, 0], [1], [1, 1, 0], []])
def test_indices_subset_order(tmp_path: Path, sel: List[int]) -> None:
    """Requested order and duplicates are honoured."""
    cfgs = [_cfg(3 + i, 2, i) for i in range(3)]
    got = read_configpack(_write(tmp_path, cfgs), sel)
    assert len(got) == len(sel)
    for i, g in zip(sel, got, strict=True):
        np.testing.assert_array_equal(g.relative, cfgs[i].relative)


@pytest.mark.parametrize("bad", [[3], [-1]])
def test_index_out_of_range_raises(tmp_path: Path, bad: List[int]) -> None:
    prefix = _write(tmp_path, [_cfg(2, 2)] * 3)
    with pytest.raises(ValueError):
        read_configpack(prefix, bad)


def test_missing_files_raise_with_both_names(tmp_path: Path) -> None:
    prefix = _write(tmp_path, [_cfg(2, 2)])
    Path(str(prefix) + ".ConfigPackIndex").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        read_configpack(prefix)
    assert "ConfigPackIndex" in str(exc.value)
    assert prefix.name + ".ConfigPack" in str(exc.value)


@pytest.mark.parametrize("cut", [1, 5, 40, 200])
def test_truncated_file_raises(tmp_path: Path, cut: int) -> None:
    """A pack cut anywhere inside a record raises ValueError."""
    prefix = _write(tmp_path, [_cfg(10, 3), _cfg(10, 3)])
    pack = Path(str(prefix) + ".ConfigPack")
    data = pack.read_bytes()
    pack.write_bytes(data[: len(data) - cut])
    with pytest.raises(ValueError):
        read_configpack(prefix)


def test_truncated_pack_header_and_empty(tmp_path: Path) -> None:
    prefix = _write(tmp_path, [_cfg(3, 3)])
    pack = Path(str(prefix) + ".ConfigPack")
    for keep in (0, 1, 2, 9, 20):
        pack.write_bytes(pack.read_bytes()[:keep] if keep else b"")
        with pytest.raises(ValueError):
            read_configpack(prefix)
        prefix = _write(tmp_path, [_cfg(3, 3)])


def test_truncated_index_raises(tmp_path: Path) -> None:
    prefix = _write(tmp_path, [_cfg(3, 2), _cfg(3, 2)])
    idx = Path(str(prefix) + ".ConfigPackIndex")
    data = idx.read_bytes()
    for keep in (0, 4, 8, 12, len(data) - 1):
        idx.write_bytes(data[:keep])
        with pytest.raises(ValueError):
            read_configpack(prefix)


def test_offset_beyond_eof_raises(tmp_path: Path) -> None:
    prefix = _write(tmp_path, [_cfg(3, 2)])
    np.array([1, 10_000], dtype="<i8").tofile(str(prefix) + ".ConfigPackIndex")
    with pytest.raises(ValueError):
        read_configpack(prefix)


def test_bad_dimension_and_negative_n_raise(tmp_path: Path) -> None:
    prefix = _write(tmp_path, [_cfg(3, 2)])
    pack = Path(str(prefix) + ".ConfigPack")
    good = pack.read_bytes()
    for dim in (0, 5):
        pack.write_bytes(np.array(dim, dtype="<u2").tobytes() + good[2:])
        with pytest.raises(ValueError):
            read_configpack(prefix)
    n_pos = 2 + 8 + 8 * 4
    pack.write_bytes(
        good[:n_pos] + np.array(-1, dtype="<i4").tobytes() + good[n_pos + 4 :]
    )
    with pytest.raises(ValueError):
        read_configpack(prefix)


def test_write_validates_shapes(tmp_path: Path) -> None:
    good = _cfg(3, 2)
    with pytest.raises(ValueError):  # basis shape
        write_configpack(
            tmp_path / "a", [PackedConfig(np.eye(3), good.relative, 1.0, good.names)]
        )
    with pytest.raises(ValueError):  # names count
        write_configpack(
            tmp_path / "b",
            [PackedConfig(good.basis, good.relative, 1.0, good.names[:2])],
        )
    with pytest.raises(ValueError):  # name length
        write_configpack(
            tmp_path / "c",
            [PackedConfig(good.basis, good.relative, 1.0, (b"ab", b"abcd", b"abcd"))],
        )
    with pytest.raises(ValueError):  # relative shape
        write_configpack(
            tmp_path / "d",
            [PackedConfig(good.basis, good.relative[:, :1], 1.0, good.names)],
        )
    with pytest.raises(TypeError):  # dtype
        write_configpack(
            tmp_path / "e",
            [
                PackedConfig(
                    good.basis, good.relative.astype(np.float32), 1.0, good.names
                )
            ],
        )


def test_write_requires_existing_parent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_configpack(tmp_path / "nope" / "p", [_cfg(2, 2)])
    assert not (tmp_path / "nope").exists()


# ============================================================
# TESTS: stack_fractional
# ============================================================


def test_stack_fractional_unit_basis() -> None:
    """Scaled basis is normalized to unit volume; s unchanged."""
    n, d = 100, 3
    side = n ** (1.0 / d)
    cfgs = [_cfg(4, d, i, scale=side) for i in range(3)]
    s, ub = stack_fractional(cfgs)
    assert s.shape == (3, 4, d) and s.dtype == np.float64
    np.testing.assert_array_equal(s[1], cfgs[1].relative)
    np.testing.assert_allclose(ub, np.eye(d), rtol=1e-14)
    assert abs(np.linalg.det(ub)) == pytest.approx(1.0, rel=1e-14)


def test_stack_fractional_rejects_mixed_shapes() -> None:
    with pytest.raises(ValueError):
        stack_fractional([_cfg(4, 2), _cfg(5, 2)])
    with pytest.raises(ValueError):
        stack_fractional([_cfg(4, 2), _cfg(4, 3)])


def test_stack_fractional_rejects_bad_bases() -> None:
    with pytest.raises(ValueError):
        stack_fractional([])
    a = _cfg(4, 2)
    sheared = PackedConfig(np.array([[1.0, 0.0], [0.3, 1.0]]), a.relative, 1.0, a.names)
    with pytest.raises(ValueError):
        stack_fractional([a, sheared])
    flat = PackedConfig(np.zeros((2, 2)), a.relative, 1.0, a.names)
    with pytest.raises(ValueError):
        stack_fractional([flat])
    neg = PackedConfig(np.diag([1.0, -1.0]), a.relative, 1.0, a.names)
    with pytest.raises(ValueError):
        stack_fractional([neg])  # non-positive determinant


def test_stack_fractional_rejects_float32() -> None:
    a = _cfg(4, 2)
    f32 = PackedConfig(a.basis, a.relative.astype(np.float32), 1.0, a.names)
    with pytest.raises(TypeError):
        stack_fractional([f32])


def test_write_rejects_wrong_length_names_total_multiple_of_four(
    tmp_path: Path,
) -> None:
    """Names (2 B, 6 B) total 8 bytes: must raise, not write a corrupt pack."""
    rel = np.zeros((2, 2))
    cfg = PackedConfig(np.eye(2), rel, 1.0, (b"ab", b"abcdef"))
    with pytest.raises(ValueError, match="exactly 4 bytes"):
        write_configpack(tmp_path / "w", [cfg])


def test_write_rejects_empty_sequence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        write_configpack(tmp_path / "e", [])
