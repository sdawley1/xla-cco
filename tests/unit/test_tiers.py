#!/usr/bin/env python3

"""
Tests for the priority-tier plugin (tests/tiers.py).

These are tier critical although they test tooling: every reduced CI run
(--priority critical/high/medium) depends on the selection being right, and
only a critical test runs in all of them.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- testing -----
import pytest

# ----- project -----
from tiers import TIERS

# ============================================================
# HELPERS
# ============================================================

ONE_PER_TIER = "\n".join(
    f"import pytest\n\n@pytest.mark.{t}\ndef test_{t}():\n    pass\n" for t in TIERS
)

# ============================================================
# TESTS
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("cut", range(len(TIERS)))
def test_priority_runs_its_tier_and_all_higher(
    pytester: pytest.Pytester, cut: int
) -> None:
    pytester.makepyfile(ONE_PER_TIER)
    res = pytester.runpytest("-p", "tiers", "--priority", TIERS[cut])
    res.assert_outcomes(passed=cut + 1, deselected=len(TIERS) - cut - 1)
    assert res.ret == pytest.ExitCode.OK


@pytest.mark.critical
def test_default_runs_every_tier(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(ONE_PER_TIER)
    pytester.runpytest("-p", "tiers").assert_outcomes(passed=len(TIERS))


@pytest.mark.critical
def test_marker_expression_selects_tiers(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(ONE_PER_TIER)
    res = pytester.runpytest("-p", "tiers", "-m", "critical or high")
    res.assert_outcomes(passed=2, deselected=2)


@pytest.mark.critical
@pytest.mark.parametrize(
    "decorators", ["", "@pytest.mark.high\n@pytest.mark.low\n"], ids=["none", "two"]
)
def test_test_without_exactly_one_tier_is_a_usage_error(
    pytester: pytest.Pytester, decorators: str
) -> None:
    pytester.makepyfile(f"import pytest\n\n{decorators}def test_x():\n    pass\n")
    res = pytester.runpytest("-p", "tiers", "--priority", "critical")
    assert res.ret == pytest.ExitCode.USAGE_ERROR
    res.stderr.fnmatch_lines(["*exactly one tier marker*", "*test_x*"])


@pytest.mark.critical
def test_module_level_tier_counts_once(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        "import pytest\n\npytestmark = pytest.mark.medium\n\ndef test_x():\n    pass\n"
    )
    res = pytester.runpytest("-p", "tiers", "--priority", "high")
    res.assert_outcomes(deselected=1)
