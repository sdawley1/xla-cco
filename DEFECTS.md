# Defects log

Every defect gets an entry here and a `test_regression_D###_<slug>`.
Upstream (C++ reference at `/home/sd2402/ccoptimization/cco/`) defects are
`wontfix-upstream`: the reference is read-only; we document and, where
needed for parity, reproduce the behaviour.

| ID | Status | Location | Summary |
|----|--------|----------|---------|
| D-001 | wontfix-upstream | `RepulsiveCCPotential.h:103-118` | Soft-core `AllForce` double-counts the CC force. |
| D-002 | wontfix-upstream | `main.cpp:119-136` | Tolerance parsed with `std::stof`; `1e-20` becomes the float32 value 9.99999968e-21. Parity runs use `float(np.float32(1e-20))`. |
| D-003 | wontfix-upstream | `StructureOptimization.cpp:724` | `MaxStep -= 2` on a `size_t` underflows for odd budgets. |
| D-004 | wontfix-upstream | `main.cpp:171` | Random-mode box is built from the stale default `num = 1000`. |
| D-005 | wontfix-upstream | `StructureOptimization.cpp:841` | NLopt L-BFGS stopval is `-MaxEnergy` with no ftol/xtol, so it runs to maxeval or roundoff. |

## D-007 — Hypothesis wall-clock deadline flakes on loaded nodes
- **Found:** 2026-11-05, while integrating T6 (CI)
- **Status:** fixed
- **Where:** tests/conftest.py (`dev` and `ci` Hypothesis profiles)
- **Symptom:** property tests intermittently fail with `DeadlineExceeded` although the code is correct
- **Repro:** tests/property/test_configpack_props.py::test_wrong_length_names_raise hit DeadlineExceeded at 332 ms against the 200 ms default deadline on the Della login node; it passed on re-run
- **Impact:** spurious red runs locally and in CI; no effect on computed results
- **Regression test:** tests/unit/test_conftest_profiles.py::test_regression_D007_profiles_have_no_deadline
- **Fix:** `deadline=None` set in the dev, ci and thorough profiles in tests/conftest.py (example counts unchanged)
