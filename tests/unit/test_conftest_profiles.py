#!/usr/bin/env python3

"""
Regression test: Hypothesis profiles carry no wall-clock deadline (D-007).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- testing -----
import pytest
from hypothesis import settings

# ============================================================
# TESTS
# ============================================================


@pytest.mark.low
@pytest.mark.parametrize("name", ["dev", "ci", "thorough"])
def test_regression_D007_profiles_have_no_deadline(name: str) -> None:
    assert settings.get_profile(name).deadline is None
