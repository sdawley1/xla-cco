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
# FD roundoff: energy and phase rounding (eps*theta) give C*eps*(E+theta*dE/dtheta),
# divided by h in the stencil; the factor 8 covers summation and the two evaluations.
FD_ROUNDOFF_C: float = 8.0
# Central-difference truncation is (h^2/6)|E_sss|; |E_sss| <= (2 pi n_max)^3 sum(w)
# N^2 / V (loose: only the 2(N-1) pair terms containing particle i depend on s_i).
FD_TRUNC_C: float = 1.0 / 6.0

# Descent iterates vs the transcribed C++ loop: both use the same arithmetic up to
# summation order of the energy sums (~1e-16 relative), times step 1e-3 per move.
ATOL_DESCENT_ITERATE: float = 1e-14
# First 10 descent iterations vs the C++ golden (min-image |ds|): C++ evaluates
# Cartesian forces, we fractional gradients; roundoff ~1e2 ulp per step.
ATOL_DESCENT_GOLDEN_S: float = 1e-12
# L-BFGS pass-schedule equivalence (3 passes vs 3 restarts): the only difference is
# the s -> x -> s round trip, ~1 ulp per pass amplified over a short run.
ATOL_PASS_RESTART_S: float = 1e-8
# vmap vs unbatched execution: XLA fuses/reduces batched dots in a different order,
# so lanes differ at ~1 ulp per step (measured 1e-15 on s after 150 descent steps).
ATOL_VMAP_S: float = 1e-12
# Same effect on energies and gradient norms (cancellation in |g| costs ~2 digits).
RTOL_VMAP_E: float = 1e-9

# Per-config absolute floor for kernel energy parity: C_E eps sqrt(N M E_i), the
# cancellation scale of M unit-weight terms; measured max ratio 1.0 across all 60
# golden configs; 4x headroom.
C_E: float = 4.0
