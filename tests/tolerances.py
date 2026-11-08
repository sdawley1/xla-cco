#!/usr/bin/env python3

"""
Single home of every numerical tolerance used by the test suite.

Oracle tolerances must come from this file. Test-local literals are
permitted only for analytic or round-trip checks.

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
# FD roundoff: energy and phase rounding (eps*theta) give C*eps*(E+theta*dE/dtheta),
# divided by h in the stencil; the factor 8 covers summation and the two evaluations.
FD_ROUNDOFF_C: float = 8.0
# Central-difference truncation is (h^2/6)|E_sss|; |E_sss| <= (2 pi n_max)^3 sum(w)
# N^2 / V (loose: only the 2(N-1) pair terms containing particle i depend on s_i).
FD_TRUNC_C: float = 1.0 / 6.0

# Descent iterates vs the transcribed C++ loop: both use the same arithmetic up to
# summation order of the energy sums (~1e-16 relative), times step 1e-3 per move.
ATOL_DESCENT_ITERATE: float = 1e-14
# L-BFGS pass-schedule equivalence (3 passes vs 3 restarts): the only difference is
# the s -> x -> s round trip, ~1 ulp per pass amplified over a short run.
ATOL_PASS_RESTART_S: float = 1e-8
# vmap vs unbatched execution: XLA fuses/reduces batched dots in a different order,
# so lanes differ at ~1 ulp per step (measured 1e-15 on s after 150 descent steps).
ATOL_VMAP_S: float = 1e-12
# Same effect on energies and gradient norms (cancellation in |g| costs ~2 digits).
RTOL_VMAP_E: float = 1e-9

# Per-config absolute floor for kernel energy checks: C_E eps sqrt(N M E_i), the
# cancellation scale of M unit-weight terms; measured max ratio 1.0 over the
# kernel test configs; 4x headroom.
C_E: float = 4.0
# Grad-sum roundoff: each particle gradient is a sum whose terms reach
# (4 pi n_max / V) sum(w) N before cancellation (|rho| <= N); N such terms add
# with error ~ C eps each, and the factor 8 covers summation order and trig error.
GRAD_SUM_C: float = 8.0
# Absolute floor for the grad-sum check when every term is tiny.
GRAD_SUM_FLOOR: float = 1e-14

# ----- molecular dynamics (xla_cco.dynamics) -----
# P1 placeholders: each MD task replaces a value only with a derivation here.
# Verlet on a harmonic oscillator follows the exact discrete map; only roundoff
# (~1e-16 per step, grown linearly over <= 1e4 steps) separates them.
ATOL_MD_VERLET_EXACT: float = 1e-12
# Global error ratio for dt -> dt/2 is 4 for a second-order method; +-0.3 allows
# the O(dt^4) correction at the dt used in the order test.
MD_ORDER_RATIO_TOL: float = 0.3
# Family-wise false-alarm rate of every statistical MD test (Bonferroni over the
# m comparisons of one test: z_crit = Phi^-1(1 - MD_FAMILY_ALPHA / (2 m))).
MD_FAMILY_ALPHA: float = 1e-3
# Smallest relative bias the statistical validations must detect with z >= 6.
MD_BIAS_DETECT: float = 0.05

# ----- C++ MD reference (tests/unit/test_md_cxx_reference.py, plan T4) -----
# Unit gate: xla Phi vs the full-precision C++ ShiftedCCPotential energy of the same
# configuration. Both are a sum of M = 50 positive terms |rho_k|^2 / V with rho_k a
# sum of N = 256 phases; summation order and libm differences give ~N eps / |rho|^2
# relative error, ~1e-12 at the sampled |rho_k|^2 ~ V T; 1e-8 is the plan's gate.
RTOL_MD_CXX_PHI: float = 1e-8
# The C++ MD log uses the 6-significant-digit stream default (D-052): a value
# m 10^e with 1 <= m < 10 is rounded to 0.5e-5 10^e, i.e. relative <= 5e-6. The
# extra 1e-3 relative slack covers the ~1e-12 xla vs C++ difference.
RTOL_CXX_LOG_6SIG: float = 5e-6 * (1.0 + 1e-3)
# Units check of the pooled C++ samples: <Phi>/(M T) and <KE>/(N d T/2) must be 1.
# A unit error is a factor >= 2 (half vs full k-space, KE without 1/2) or V/N =
# 3.9 (D-004 box); the statistical error of the pooled mean of 600 samples is
# sqrt(1/M)/sqrt(600) = 0.6 % for Phi and sqrt(2/(N d))/sqrt(600) = 0.3 % for KE,
# x3 for sample correlation, so 0.1 is > 5 sigma and far below any unit error.
MD_CXX_UNITS_TOL: float = 0.1
