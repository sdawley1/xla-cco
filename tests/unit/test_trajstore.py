#!/usr/bin/env python3

"""
Tests for xla_cco.io.trajstore (append-only writer, memmap reader).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import tempfile
from pathlib import Path
from typing import List

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# ----- project -----
from xla_cco.io.trajstore import TrajectoryWriter, open_trajectory

# ============================================================
# TESTS
# ============================================================


@settings(deadline=None)
@given(
    st.sampled_from((1, 4)),
    st.sampled_from((1, 2, 3)),
    st.lists(st.integers(0, 4), min_size=0, max_size=4),
    st.data(),
)
def test_round_trip_chunks(
    n: int, d: int, sizes: List[int], data: st.DataObject
) -> None:
    chunks = [
        data.draw(hnp.arrays(np.float64, (k, n, d), elements=st.floats(0.0, 1.0)))
        for k in sizes
    ]
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "traj"
        w = TrajectoryWriter(prefix, n, d)
        for c in chunks:
            w.append(c)
        w.close({"energy": np.arange(3.0)}, {"chi": 0.1})
        frames, arrays, meta = open_trajectory(prefix)
        want = np.concatenate(chunks) if chunks else np.empty((0, n, d))
        np.testing.assert_array_equal(np.asarray(frames), want)
        np.testing.assert_array_equal(arrays["energy"], np.arange(3.0))
        assert meta["chi"] == 0.1 and meta["n_frames"] == sum(sizes)


def test_reader_detects_truncation(tmp_path: Path) -> None:
    w = TrajectoryWriter(tmp_path / "t", 2, 2)
    w.append(np.zeros((3, 2, 2)))
    w.close({}, {})
    raw = tmp_path / "t.positions.f64"
    raw.write_bytes(raw.read_bytes()[:-8])
    with pytest.raises(ValueError, match="size"):
        open_trajectory(tmp_path / "t")


def test_writer_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TrajectoryWriter(tmp_path / "missing" / "t", 2, 2)
    w = TrajectoryWriter(tmp_path / "t", 2, 2)
    with pytest.raises(TypeError):
        w.append(np.zeros((1, 2, 2), np.float32))
    with pytest.raises(ValueError):
        w.append(np.zeros((1, 3, 2)))
    with pytest.raises(ValueError):
        w.close({}, {"n_frames": 1})


def test_reader_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_trajectory(tmp_path / "nope")
