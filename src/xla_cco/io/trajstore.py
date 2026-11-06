#!/usr/bin/env python3

"""
Append-only on-disk trajectory store with memory-mapped reads.

A trajectory with prefix P is three files:

- ``P.positions.f64``: raw little-endian float64 frames, shape (F, N, d),
  appended chunk by chunk (wrapped fractional coordinates);
- ``P.npz``: per-iteration arrays (written on close);
- ``P.json``: metadata including the frame shape (written on close).

``open_trajectory`` returns the frames as a read-only ``np.memmap`` so
analyses never load a whole trajectory into memory.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json
from pathlib import Path
from typing import Any, Dict, Tuple

# ----- numerics -----
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================

FRAME_DTYPE = np.dtype("<f8")

# ============================================================
# WRITER
# ============================================================


class TrajectoryWriter:
    """Append frames for one trajectory, then finalize with ``close``."""

    def __init__(self, prefix: Path, n_particles: int, dim: int) -> None:
        """Create (truncate) the frame file at ``prefix``."""
        if n_particles < 1 or dim < 1:
            raise ValueError("n_particles and dim must be >= 1")
        if not prefix.parent.is_dir():
            raise FileNotFoundError(f"directory {prefix.parent} does not exist")
        self.prefix = prefix
        self.frame_shape: Tuple[int, int] = (n_particles, dim)
        self.n_frames = 0
        self._fh = open(f"{prefix}.positions.f64", "wb")  # noqa: SIM115

    def append(self, frames: np.ndarray) -> None:
        """Append (k, N, d) float64 frames (k may be 0)."""
        if frames.dtype != np.float64:
            raise TypeError(f"frames must be float64, got {frames.dtype}")
        if frames.ndim != 3 or frames.shape[1:] != self.frame_shape:
            raise ValueError(f"frames shape {frames.shape} != (k, *{self.frame_shape})")
        self._fh.write(np.ascontiguousarray(frames, dtype=FRAME_DTYPE).tobytes())
        self.n_frames += int(frames.shape[0])

    def close(self, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
        """Write per-iteration arrays and metadata, then close the frame file."""
        if "frame_shape" in meta or "n_frames" in meta:
            raise ValueError("meta must not set frame_shape or n_frames")
        self._fh.close()
        with open(f"{self.prefix}.npz", "wb") as fh:
            np.savez(fh, **arrays)  # type: ignore[arg-type]
        full = dict(meta, frame_shape=list(self.frame_shape), n_frames=self.n_frames)
        Path(f"{self.prefix}.json").write_text(json.dumps(full, indent=1))


# ============================================================
# READER
# ============================================================


def open_trajectory(
    prefix: Path,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Return (frames memmap (F, N, d), per-iteration arrays, metadata).

    Raises
    ------
    FileNotFoundError
        If any of the three files is missing.
    ValueError
        If the frame file size disagrees with the metadata.
    """
    paths = [Path(f"{prefix}{ext}") for ext in (".positions.f64", ".npz", ".json")]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing trajectory files: {missing}")
    meta: Dict[str, Any] = json.loads(paths[2].read_text())
    n, d = meta["frame_shape"]
    count = int(meta["n_frames"])
    if paths[0].stat().st_size != count * n * d * FRAME_DTYPE.itemsize:
        raise ValueError(f"{paths[0]} size does not match {count} frames of {(n, d)}")
    if count == 0:  # np.memmap cannot map an empty file
        frames = np.empty((0, n, d), dtype=FRAME_DTYPE)
    else:
        frames = np.memmap(paths[0], dtype=FRAME_DTYPE, mode="r", shape=(count, n, d))
    with np.load(paths[1]) as npz:
        arrays = {k: npz[k] for k in npz.files}
    return frames, arrays, meta
