#!/usr/bin/env python3

"""
Tests of the float64 dtype guard.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

# ----- project -----
from xla_cco._types import require_float64

# ============================================================
# TESTS
# ============================================================


@pytest.mark.medium
def test_float64_accepted() -> None:
    require_float64(jnp.zeros((2, 3), dtype=jnp.float64), "s")
    require_float64(np.zeros(4), "s")


@pytest.mark.medium
def test_float32_rejected_names_argument_and_dtype() -> None:
    with pytest.raises(TypeError, match=r"my_arg.*float32"):
        require_float64(jnp.zeros(3, dtype=jnp.float32), "my_arg")


@pytest.mark.medium
def test_integer_rejected() -> None:
    with pytest.raises(TypeError, match="int64"):
        require_float64(np.arange(3), "n")


@pytest.mark.medium
@given(
    dtype=st.sampled_from(
        (np.float16, np.float32, np.float64, np.int32, np.int64, np.complex128)
    ),
    shape=st.sampled_from(((), (3,), (2, 4))),
)
def test_only_float64_passes(dtype: type, shape: tuple[int, ...]) -> None:
    x = np.zeros(shape, dtype=dtype)
    if dtype is np.float64:
        require_float64(x, "x")
    else:
        with pytest.raises(TypeError, match="x"):
            require_float64(x, "x")
