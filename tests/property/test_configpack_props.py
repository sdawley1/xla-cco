#!/usr/bin/env python3

"""
Property tests of the ConfigPack reader/writer and stack_fractional.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import tempfile
from pathlib import Path
from typing import Any, List

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# ----- project -----
from xla_cco.io.configpack import (
    PackedConfig,
    read_configpack,
    stack_fractional,
    write_configpack,
)

# ============================================================
# STRATEGIES
# ============================================================

FINITE = st.floats(allow_nan=False, allow_infinity=False, width=64)


@st.composite
def packed_configs(draw: Any, dim: Any = None, n: Any = None) -> PackedConfig:
    """Draw one PackedConfig with arbitrary finite floats and 4-byte names."""
    d = draw(st.integers(1, 3)) if dim is None else dim
    npart = draw(st.integers(1, 50)) if n is None else n
    return PackedConfig(
        basis=draw(hnp.arrays(np.float64, (d, d), elements=FINITE)),
        relative=draw(hnp.arrays(np.float64, (npart, d), elements=FINITE)),
        cell_size=draw(FINITE),
        names=tuple(
            draw(
                st.lists(
                    st.binary(min_size=4, max_size=4), min_size=npart, max_size=npart
                )
            )
        ),
    )


def _roundtrip(cfgs: List[PackedConfig], sel: Any = None) -> List[PackedConfig]:
    with tempfile.TemporaryDirectory() as tmp:
        write_configpack(Path(tmp) / "p", cfgs)
        return read_configpack(Path(tmp) / "p", sel)


# ============================================================
# PROPERTIES
# ============================================================


@given(st.lists(packed_configs(), min_size=1, max_size=5))
def test_round_trip_bytes_identical(cfgs: List[PackedConfig]) -> None:
    """write -> read -> write gives identical bytes in both files."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        write_configpack(a, cfgs)
        write_configpack(b, read_configpack(a))
        for ext in (".ConfigPack", ".ConfigPackIndex"):
            assert Path(str(a) + ext).read_bytes() == Path(str(b) + ext).read_bytes()


@given(st.lists(packed_configs(), min_size=1, max_size=5))
def test_round_trip_preserves_fields(cfgs: List[PackedConfig]) -> None:
    got = _roundtrip(cfgs)
    assert len(got) == len(cfgs)
    for a, b in zip(cfgs, got, strict=True):
        np.testing.assert_array_equal(a.basis, b.basis)
        np.testing.assert_array_equal(a.relative, b.relative)
        assert a.cell_size == b.cell_size
        assert a.names == b.names


@given(
    st.lists(packed_configs(), min_size=1, max_size=5),
    st.data(),
)
def test_subset_order_and_duplicates(cfgs: List[PackedConfig], data: Any) -> None:
    """Any index list (order, duplicates) returns exactly those configs."""
    sel = data.draw(st.lists(st.integers(0, len(cfgs) - 1), min_size=0, max_size=8))
    got = _roundtrip(cfgs, sel)
    assert len(got) == len(sel)
    for i, g in zip(sel, got, strict=True):
        np.testing.assert_array_equal(g.relative, cfgs[i].relative)
        assert g.names == cfgs[i].names


@given(
    st.integers(1, 3),
    st.floats(0.1, 10.0),
    st.integers(1, 4),
    st.integers(1, 20),
    st.data(),
)
def test_stack_fractional_normalizes(
    d: int, scale: float, b: int, n: int, data: Any
) -> None:
    """Scaled bases give unit volume; s is returned unchanged."""
    cfgs = [
        PackedConfig(
            np.eye(d) * scale,
            data.draw(
                hnp.arrays(
                    np.float64, (n, d), elements=st.floats(0, 1, exclude_max=True)
                )
            ),
            scale,
            (b"a\x00\x00\x00",) * n,
        )
        for _ in range(b)
    ]
    s, ub = stack_fractional(cfgs)
    assert s.shape == (b, n, d)
    np.testing.assert_array_equal(s, np.stack([c.relative for c in cfgs]))
    np.testing.assert_allclose(abs(np.linalg.det(ub)), 1.0, rtol=1e-12)
    np.testing.assert_allclose(ub, np.eye(d), rtol=1e-12)


@given(st.data())
def test_wrong_length_names_raise(data: Any) -> None:
    """Names not all 4 bytes long raise, even if the total is a multiple of 4."""
    n = data.draw(st.integers(1, 6))
    lens = data.draw(
        st.lists(st.integers(0, 9), min_size=n, max_size=n).filter(
            lambda ls: sum(ls) % 4 == 0 and any(x != 4 for x in ls)
        )
    )
    names = tuple(b"x" * k for k in lens)
    cfg = PackedConfig(np.eye(1), np.zeros((n, 1)), 1.0, names)
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(ValueError, match="exactly 4 bytes"),
    ):
        write_configpack(Path(tmp) / "p", [cfg])
