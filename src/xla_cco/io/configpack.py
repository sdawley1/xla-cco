#!/usr/bin/env python3

"""
ConfigPack reader/writer (C++ binary format) and fractional stacking.

Positions in a file are relative (fractional) to a unit-volume basis; the
box is the unit basis times N^(1/d).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ----- numerics -----
import numpy as np

# ============================================================
# CONTAINERS
# ============================================================


@dataclasses.dataclass(frozen=True)
class PackedConfig:
    """One configuration as stored in a ConfigPack file.

    Attributes
    ----------
    basis : np.ndarray
        (d, d) float64 unit-volume lattice basis, rows = lattice vectors.
    relative : np.ndarray
        (N, d) float64 fractional positions.
    cell_size : float
        Cell size stored in the file.
    names : Tuple[bytes, ...]
        Per-particle name tags.
    """

    basis: np.ndarray
    relative: np.ndarray
    cell_size: float
    names: Tuple[bytes, ...]


# ============================================================
# CONSTANTS
# ============================================================

PACK_SUFFIX: str = ".ConfigPack"
INDEX_SUFFIX: str = ".ConfigPackIndex"
MAX_DIMENSION: int = 4  # C++ MaxDimension
NAME_BYTES: int = 4
_FIXED_HEADER_BYTES: int = 2  # uint16 dim
_UNIT_RTOL: float = 1e-12  # basis agreement after unit-volume normalization

# ============================================================
# HELPERS
# ============================================================


def _paths(prefix: Path) -> Tuple[Path, Path]:
    """Return (pack, index) paths for a prefix without extension."""
    return Path(str(prefix) + PACK_SUFFIX), Path(str(prefix) + INDEX_SUFFIX)


def _read_index(index_path: Path) -> np.ndarray:
    """Return the int64 offsets from an index file, validating its size."""
    raw = np.fromfile(index_path, dtype="<i8")
    if raw.size < 1:
        raise ValueError(f"truncated index (no count): {index_path}")
    n_config = int(raw[0])
    if n_config < 0 or raw.size != n_config + 1:
        raise ValueError(
            f"index {index_path} declares {n_config} configs "
            f"but holds {raw.size - 1} offsets"
        )
    return raw[1:]


def _parse_config(buf: np.ndarray, offset: int) -> PackedConfig:
    """Parse one record from the uint8 buffer ``buf`` at byte ``offset``."""
    size = int(buf.size)
    if not 0 <= offset <= size - _FIXED_HEADER_BYTES:
        raise ValueError(f"offset {offset} outside pack ({size} B)")
    dim = int(np.frombuffer(buf, dtype="<u2", count=1, offset=offset)[0])
    if not 1 <= dim <= MAX_DIMENSION:
        raise ValueError(f"bad dimension {dim} at offset {offset}")
    header = np.dtype([("cell", "<f8"), ("basis", "<f8", (dim, dim)), ("n", "<i4")])
    start = offset + _FIXED_HEADER_BYTES
    if start + header.itemsize > size:
        raise ValueError(f"truncated header at offset {offset}")
    hdr = np.frombuffer(buf, dtype=header, count=1, offset=start)[0]
    n = int(hdr["n"])
    if n < 0:
        raise ValueError(f"negative particle count {n} at offset {offset}")
    record = np.dtype([("x", "<f8", (dim,)), ("name", "u1", (NAME_BYTES,))])
    start += header.itemsize
    if start + n * record.itemsize > size:
        raise ValueError(
            f"config at offset {offset} overruns pack "
            f"({start + n * record.itemsize} > {size} B)"
        )
    recs = np.frombuffer(buf, dtype=record, count=n, offset=start)
    names = recs["name"].copy().view(f"V{NAME_BYTES}").ravel().tolist()
    return PackedConfig(
        basis=np.array(hdr["basis"], dtype=np.float64),
        relative=np.array(recs["x"], dtype=np.float64),
        cell_size=float(hdr["cell"]),
        names=tuple(bytes(x) for x in names),
    )


def _config_bytes(cfg: PackedConfig) -> bytes:
    """Validate and serialize one configuration to its byte record."""
    basis, rel = np.asarray(cfg.basis), np.asarray(cfg.relative)
    for label, arr in (("basis", basis), ("relative", rel)):
        if arr.dtype != np.float64:
            raise TypeError(f"{label} must be float64, got {arr.dtype}")
    if rel.ndim != 2 or not 1 <= rel.shape[1] <= MAX_DIMENSION:
        raise ValueError(f"relative must be (N, d), d in 1..4; got {rel.shape}")
    n, d = rel.shape
    if basis.shape != (d, d):
        raise ValueError(f"basis shape {basis.shape} != ({d}, {d})")
    if len(cfg.names) != n:
        raise ValueError(f"{len(cfg.names)} names for {n} particles")
    if any(len(nm) != NAME_BYTES for nm in cfg.names):
        raise ValueError(f"each name must be exactly {NAME_BYTES} bytes")
    record = np.dtype([("x", "<f8", (d,)), ("name", "u1", (NAME_BYTES,))])
    recs = np.empty(n, dtype=record)
    recs["x"] = rel
    recs["name"] = np.frombuffer(b"".join(cfg.names), dtype=np.uint8).reshape(
        n, NAME_BYTES
    )
    return b"".join(
        (
            np.array(d, dtype="<u2").tobytes(),
            np.array(cfg.cell_size, dtype="<f8").tobytes(),
            np.ascontiguousarray(basis, dtype="<f8").tobytes(),
            np.array(n, dtype="<i4").tobytes(),
            recs.tobytes(),
        )
    )


# ============================================================
# I/O
# ============================================================


def read_configpack(
    prefix: Path, indices: Optional[Sequence[int]] = None
) -> List[PackedConfig]:
    """Read configurations from a ConfigPack file set.

    Parameters
    ----------
    prefix : Path
        File prefix of the pack (no extension).
    indices : Optional[Sequence[int]]
        Configurations to read, in this order, duplicates allowed
        (None = all).

    Raises
    ------
    FileNotFoundError
        If ``prefix.ConfigPack`` or ``prefix.ConfigPackIndex`` is missing.
    ValueError
        On truncated or malformed files, or an index out of range.
    """
    pack_path, index_path = _paths(Path(prefix))
    if not (pack_path.is_file() and index_path.is_file()):
        raise FileNotFoundError(
            f"need both {pack_path} and {index_path} "
            f"(pack exists: {pack_path.is_file()}, "
            f"index exists: {index_path.is_file()})"
        )
    offsets = _read_index(index_path)
    sel = list(range(offsets.size)) if indices is None else list(indices)
    if any(not 0 <= i < offsets.size for i in sel):
        raise ValueError(f"config index outside [0, {offsets.size})")
    if not sel:
        return []
    if os.path.getsize(pack_path) == 0:
        raise ValueError(f"empty pack for {offsets.size} configs: {pack_path}")
    buf = np.memmap(pack_path, dtype=np.uint8, mode="r")
    try:
        out = [_parse_config(buf, int(offsets[i])) for i in sel]
    finally:
        del buf
    assert len(out) == len(sel), "read count mismatch"
    return out


def write_configpack(prefix: Path, configs: Sequence[PackedConfig]) -> None:
    """Write configurations as a ConfigPack file set.

    Raises
    ------
    FileNotFoundError
        If the parent directory of ``prefix`` does not exist.
    ValueError, TypeError
        On inconsistent shapes, bad name lengths or non-float64 arrays.
    """
    prefix = Path(prefix)
    if not prefix.parent.is_dir():
        raise FileNotFoundError(f"missing parent directory: {prefix.parent}")
    blobs = [_config_bytes(c) for c in configs]
    sizes = np.array([len(b) for b in blobs], dtype="<i8")
    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype("<i8")
    assert offsets.size == len(blobs), "offset/config count mismatch"
    pack_path, index_path = _paths(prefix)
    pack_path.write_bytes(b"".join(blobs))
    index = np.concatenate(([len(blobs)], offsets[: len(blobs)])).astype("<i8")
    index.tofile(index_path)


def stack_fractional(
    configs: Sequence[PackedConfig],
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack configurations into ((B, N, d) fractional, (d, d) unit basis).

    The unit basis is ``basis / |det(basis)|**(1/d)``; the relative
    coordinates are returned unchanged.

    Raises
    ------
    ValueError
        On an empty sequence, mixed N or d, a non-positive determinant, or
        bases that differ after unit normalization.
    TypeError
        On non-float64 positions.
    """
    if len(configs) == 0:
        raise ValueError("no configurations to stack")
    shape = np.asarray(configs[0].relative).shape
    bases = []
    for c in configs:
        rel, basis = np.asarray(c.relative), np.asarray(c.basis)
        if rel.dtype != np.float64 or basis.dtype != np.float64:
            raise TypeError("positions and basis must be float64")
        if rel.shape != shape:
            raise ValueError(f"mixed shapes {rel.shape} and {shape}")
        d = shape[1]
        if basis.shape != (d, d):
            raise ValueError(f"basis shape {basis.shape} != ({d}, {d})")
        det = float(np.linalg.det(basis))
        if not det > 0.0:
            raise ValueError(f"non-positive basis determinant {det}")
        bases.append(basis / det ** (1.0 / d))
    for ub in bases[1:]:
        if not np.allclose(ub, bases[0], rtol=_UNIT_RTOL, atol=_UNIT_RTOL):
            raise ValueError("bases differ across configurations")
    return np.stack([np.asarray(c.relative) for c in configs]), bases[0]
