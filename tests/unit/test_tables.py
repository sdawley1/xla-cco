#!/usr/bin/env python3

"""
Tests of the C++-format function-table writer.

Format source: GeometryVector.cpp:146-172 (``WriteFunction``).

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
from hypothesis import given
from hypothesis import strategies as st

# ----- project -----
from xla_cco.io.tables import write_function_table

# ============================================================
# TESTS
# ============================================================


def test_write_function_table_format(tmp_path: Path) -> None:
    """Exact bytes: 4 tab-terminated %1.10e fields, zero-padded columns."""
    out = tmp_path / "f.txt"
    cols = [np.array([1.5, -2.0]), np.array([0.0, 1e-22])]
    write_function_table(out, cols)
    expected = (
        "1.5000000000e+00\t0.0000000000e+00\t0.0000000000e+00\t0.0000000000e+00\t\n"
        "-2.0000000000e+00\t1.0000000000e-22\t0.0000000000e+00\t0.0000000000e+00\t\n"
    )
    assert out.read_bytes() == expected.encode("ascii")


def test_write_function_table_four_columns(tmp_path: Path) -> None:
    """Four columns reproduce a real C++ line."""
    out = tmp_path / "f.txt"
    vals = [1.3536712390e00, 2.4540768640e-22, 0.0, 1.1926785651e-23]
    write_function_table(out, [np.array([v]) for v in vals])
    line = "1.3536712390e+00\t2.4540768640e-22\t0.0000000000e+00\t1.1926785651e-23\t\n"
    assert out.read_text() == line


def test_write_function_table_rejects_bad_input(tmp_path: Path) -> None:
    """Ragged, too many, empty-column-list and infinite inputs raise."""
    out = tmp_path / "f.txt"
    with pytest.raises(ValueError):
        write_function_table(out, [np.zeros(2), np.zeros(3)])
    with pytest.raises(ValueError):
        write_function_table(out, [np.zeros(2)] * 5)
    with pytest.raises(ValueError):
        write_function_table(out, [])
    with pytest.raises(ValueError):
        write_function_table(out, [np.array([np.inf])])
    with pytest.raises(ValueError):
        write_function_table(out, [np.array([-np.inf])])
    with pytest.raises(ValueError):
        write_function_table(out, [np.zeros((2, 2))])


@given(
    st.lists(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, width=64),
            min_size=3,
            max_size=3,
        ),
        min_size=1,
        max_size=4,
    ),
    st.integers(min_value=1, max_value=4),
)
def test_write_function_table_property(cols: List[List[float]], n_cols: int) -> None:
    """Each line has 4 tab-terminated fields that parse back to the input."""
    import tempfile

    arrays = [np.array(c) for c in cols[:n_cols]]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "f.txt"
        write_function_table(out, arrays)
        raw = out.read_bytes()
    lines = raw.decode("ascii").split("\n")
    assert lines[-1] == ""
    rows = lines[:-1]
    assert len(rows) == 3
    for i, row in enumerate(rows):
        assert row.endswith("\t")
        fields = row[:-1].split("\t")
        assert len(fields) == 4
        for j, f in enumerate(fields):
            assert f == "%1.10e" % (float(arrays[j][i]) if j < len(arrays) else 0.0)


def test_regression_D009_function_table_writes_nan_like_cpp(tmp_path: Path) -> None:
    """NaN is written as glibc printf %1.10e does ("nan"/"-nan"); inf raises."""
    out = tmp_path / "f.txt"
    neg_nan = np.copysign(np.nan, -1.0)
    write_function_table(out, [np.array([np.nan, neg_nan]), np.array([1.0, 2.0])])
    assert out.read_text() == (
        "nan\t1.0000000000e+00\t0.0000000000e+00\t0.0000000000e+00\t\n"
        "-nan\t2.0000000000e+00\t0.0000000000e+00\t0.0000000000e+00\t\n"
    )
    with pytest.raises(ValueError):
        write_function_table(out, [np.array([np.inf])])
