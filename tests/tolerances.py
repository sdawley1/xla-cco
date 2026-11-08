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
# ----- molecular dynamics: T1 (integrators and thermal sampling) -----
# Time reversal of Verlet (fixed-seed unit test only, |r| <= 10; for any |r| see
# MD_REVERSE_ULPS_PER_STEP): forward and backward maps cancel up to ~1 ulp of
# |r| <= 10 per step and coordinate (1e-15), grown at most linearly by the smooth
# stealthy force over <= 200 steps: 200 * 2e-15 ~ 4e-13 < 1e-12.
ATOL_MD_REVERSE: float = 1e-12
# Verlet vs a NumPy Cartesian oracle in a 32 x 16 sqrt(3) cell: same products up to
# summation order (~1 ulp of |r| <= 60, i.e. 7e-15 per step) on a stable linear map,
# random-walk growth over 500 steps: sqrt(500) * 7e-15 ~ 2e-13 < 1e-12.
ATOL_MD_CART_ORACLE: float = 1e-12
# Verlet energy on a harmonic oscillator: the map conserves exactly
# v^2 + (1 - (w h)^2 / 4) w^2 x^2, so E oscillates about its time mean <E> with
# max |E - <E>| / <E> = (w h)^2 / (8 - (w h)^2) <= 1.1 (w h)^2 / 8 for w h <= 0.85.
MD_VERLET_ENERGY_FACTOR: float = 1.1
# COM removal: each v_i - mean rounds by <= eps |v_i| / 2 and the check's own sum
# adds as much again, so |sum_i v_i| <= n eps max|v| (factor 1, both bounds summed).
MD_COM_ROUNDOFF_C: float = 1.0
# Spec form for the reference draws (n = 16, T = 1): the n eps max|v| bound is
# 16 * 2.2e-16 * 5 ~ 1.8e-14 worst case, but the roundoff is a random walk of 16
# terms (~4 * 2.2e-16 * 1 ~ 1e-15); measured max over 2000 draws is ~4e-15.
ATOL_MD_COM: float = 1e-14
# Blocked (Flyvbjerg-Petersen) z-score bound for BAOAB <x^2> = T / w^2: one
# comparison per configuration, P(|z| > 4) = 6e-5 under the null (< MD_FAMILY_ALPHA).
MD_BLOCK_Z: float = 4.0
# Kolmogorov-Smirnov acceptance threshold (spec: p > 1e-3 per component; the draws
# are seeded, so a pass is deterministic).
MD_KS_PVALUE: float = 1e-3
# Harmonic draws have no component along zero modes: the null-space projection is
# the eigh rounding eps * lambda_max / lambda_min(nonzero) <~ 1e-12 (condition
# <= 1e4 in the test systems) times |displacement|; 1e-9 leaves 1000x headroom.
RTOL_MD_ZERO_MODE: float = 1e-9
# Cartesian Hessian vs jax.hessian of r -> Phi(r inv(B)): the same sums in another
# order, cancellation in the diagonal blocks costs ~3 digits (relative to max|H|).
RTOL_MD_HESSIAN: float = 1e-12
# Quadratic form dr K dr^T of a harmonic draw vs |xi|^2: eigh of K (condition
# <= (40 / 0.5)^2 * cond(C) * (1 + |shear|)^2 ~ 7e4 in the property test) gives a
# relative error ~ eps * cond ~ 1.5e-11; 1e-8 leaves ~1000x headroom.
RTOL_MD_QUADFORM: float = 1e-8
# Blocked-error self-test: the returned maximum over levels with >= 128 blocks has
# per-level relative sd <= sqrt(2 / 127) = 0.125, so it stays within ~2.8 sd = 0.35
# of the true standard error (one-sided upward bias by construction).
MD_BLOCK_SE_RTOL: float = 0.35
# One Verlet step on O(1) values (|x| <= 1.5): <= 4 roundings of eps/2 |x| each,
# ~7e-16 < 1e-15 (the NumPy check repeats the same operations).
ATOL_MD_ONE_STEP: float = 1e-15
# Sums of n d <= 33 squares in different orders (JAX vs NumPy): <= n d eps ~ 7e-15.
RTOL_MD_KE: float = 1e-14
# sqrt(T) x vs sqrt(T) x from another (correctly rounded) sqrt: identical on CPU;
# 4 ulps allow one extra rounding in sqrt and in the product on other backends.
MD_SQRT_T_ULPS: float = 4.0
# COM shift vs x - mean(x): mean summation order differs by <= eps max|x| and the
# subtraction rounds by <= eps max|x|; factor 4 (times eps max|x|) leaves 2x room.
MD_COM_SHIFT_C: float = 4.0
# Free-particle drift r + dt/2 v + dt/2 v' vs r + dt/2 (v + v'): <= 4 roundings of
# eps/2 times (|r| + dt |v|) each; factor 8 (times eps * scale) leaves 4x room.
MD_DRIFT_IDENTITY_C: float = 8.0
# Time reversal at any |r| (property test, D-020): roundoff of ~1 ulp of max(1, |r|)
# per step, amplified by the chaotic flow over the 2 n forward/backward steps. The
# bound C * 2 n * eps * max(1, |r|) with C = 16 (= 1280 eps |r| at n = 40) covers the
# measured worst: 17 eps |r| over 8000 runs at dt <= 0.02 (the property domain) and
# 538 eps |r| at the dt = 0.05 Hypothesis repro (the largest of 8000 dt = 0.05 runs).
MD_REVERSE_ULPS_PER_STEP: float = 16.0
