#!/usr/bin/env python3

"""
Single home of every numerical tolerance used by the test suite.

Samuel J. Dawley
11/2026
"""

# ============================================================
# CONSTANTS
# ============================================================

# Energy is a sum of M positive terms; summation-order differences stay ~1e3 ulp.
RTOL_KERNEL_E: float = 1e-12
# Forces add a trig-weighted sum of M vectors, so cancellation costs ~2 digits.
RTOL_KERNEL_F: float = 1e-10
# Central finite differences with h ~ 1e-6 are accurate to ~h^2 + eps/h ~ 1e-6.
ATOL_FD: float = 1e-6
# Descent trajectories amplify roundoff mildly over O(1e3) steps (energy level).
TRAJ_RTOL_E: float = 1e-9
# Fractional-position drift of a trajectory vs C++ over the compared horizon.
TRAJ_ATOL_S: float = 1e-10
# Two-sample KS p-value floor for ensemble statistics; below it parity fails.
KS_PMIN: float = 1e-3
# Max |z| between binned S(k) of ensembles (about 4 sigma: false alarms ~6e-5).
SK_ZMAX: float = 4.0
# Binned S(k) vs C++ golden: relative tolerance; sums are re-ordered (JAX scan vs
# sequential C++), which costs ~1e2 ulp.
RTOL_SK: float = 1e-8
# Below the stealthy window S ~ 1e-22 is pure cancellation noise of rho (|rho| ~
# 1e-10 from O(1) terms, phases differing at 1 ulp in Cartesian vs fractional form).
ATOL_SK_S: float = 1e-26
# SEM of |k| is sqrt of a difference of equal sums (all k in a bin share |k|), i.e.
# sqrt(eps) * |k| / sqrt(count) noise up to ~1e-5 at |k| = 400; true value is 0.
ATOL_SK_KERR: float = 1e-5
