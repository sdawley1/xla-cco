#!/usr/bin/env python3

"""
Pytest plugin: priority tiers for every test.

Every test carries exactly one tier marker, set on the test function:

- critical: the numbers the library produces. A failure means energies,
  gradients, k-sets, relaxed configurations or the files holding them can be
  silently wrong. Run on every push.
- high: secondary numerical paths and invariants (chunking, tables,
  Hessian, gradient flow, transport, structure analysis) and regressions of
  defects that changed results.
- medium: error contracts, input validation, bookkeeping (dtypes,
  recording, status codes), file-format edge cases and the test oracles.
- low: tooling and messages (help text, GPU diagnostics, Hypothesis
  profiles, benchmark smoke, export lists).

``--priority=<tier>`` runs that tier and every tier above it, e.g.
``--priority=high`` runs critical and high. The default, ``low``, runs all.
Plain ``-m "critical or high"`` works too. A test with no tier, or more than
one, is a usage error, so new tests must be classified.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import List

# ----- testing -----
import pytest

# ============================================================
# CONSTANTS
# ============================================================

TIERS = ("critical", "high", "medium", "low")  # highest priority first
DESCRIPTIONS = {
    "critical": "core numerical results; a failure means wrong physics output",
    "high": "secondary numerical paths, invariants and result-changing defects",
    "medium": "error contracts, validation, bookkeeping, format edge cases, oracles",
    "low": "tooling and messages: help text, GPU diagnostics, profiles, smoke",
}

# ============================================================
# HOOKS
# ============================================================


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--priority``: the lowest tier to run (cumulative from critical)."""
    parser.addoption(
        "--priority",
        choices=TIERS,
        default="low",
        help="run this tier and all higher ones: critical > high > medium > low "
        "(default: low, i.e. every tier)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the tier markers (works with --strict-markers)."""
    for tier in TIERS:
        config.addinivalue_line("markers", f"{tier}: {DESCRIPTIONS[tier]}")


def pytest_report_header(config: pytest.Config) -> str:
    """Show which tiers this session runs."""
    cut = TIERS.index(config.getoption("priority"))
    return "priority tiers: " + ", ".join(TIERS[: cut + 1])


def tiers_of(item: pytest.Item) -> List[str]:
    """Tier markers that apply to a collected test (function, class, module)."""
    return [m.name for m in item.iter_markers() if m.name in TIERS]


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Reject unclassified tests, then drop tiers below ``--priority``."""
    bad = [
        f"{it.nodeid} {tiers_of(it) or '(none)'}"
        for it in items
        if len(tiers_of(it)) != 1
    ]
    if bad:
        raise pytest.UsageError(
            "every test needs exactly one tier marker "
            f"({', '.join(TIERS)}); offending tests:\n  " + "\n  ".join(bad)
        )
    cut = TIERS.index(config.getoption("priority"))
    keep = [it for it in items if TIERS.index(tiers_of(it)[0]) <= cut]
    drop = [it for it in items if TIERS.index(tiers_of(it)[0]) > cut]
    assert len(keep) + len(drop) == len(items), "partition lost tests"
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
