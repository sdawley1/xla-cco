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

# ----- MD observables and the energy extension (T2) -----
# Winding vs batten2008_winding.cumulative_winding and vs analytic rotations: each
# step is atan2 of the product rho_new conj(rho_old), rounded to <= ~2 ulp(pi)
# ~ 1e-15 (the reference takes np.angle of the same product, the analytic path
# rounds its phase by eps * omega t <= 1e-14); over the <= 400 steps of those
# tests the sums drift by <= 4e-13, so 1e-12 leaves 2.5x headroom.
ATOL_MD_WINDING: float = 1e-12
# The accumulators also round each addition by <= eps W / 2 at magnitude W (and the
# reference sums turns, then multiplies by 2 pi): over F <= 400 steps that is
# <= F eps / 2 ~ 4.4e-14 relative (measured 3.8e-15 at W ~ 470 rad); 1e-13.
RTOL_MD_WINDING: float = 1e-13
# wave_densities (k . r) vs collective_density (2 pi n . s, s = r inv(B)): the two
# phases differ by the rounding of r inv(B) and of n 2 pi inv(B)^T, a few eps
# |theta| with |theta| <= 2 pi max|n| d max|s| <= 2 pi * 4 * 3 * 4 ~ 300 in the
# tests (unwrapped s up to 4), i.e. <= ~2e-13 per particle; the check scales the
# bound by N (atol = ATOL_MD_WAVE_PER_PARTICLE * N), 5x headroom.
ATOL_MD_WAVE_PER_PARTICLE: float = 1e-12
# Phase-table rho vs trig rho: a table factor z^p (cumulative product) carries
# ~p eps relative error and z itself eps 2 pi |s| (as does the trig phase), so one
# (k, j) term is off by <= (1 + 2 pi |s|) n_max d eps; N terms add. The check uses
# atol = C eps N (1 + 2 pi) n_max d, i.e. the |s| <= 1 form, with C = 4 covering
# |s| <= 2 of the tests (measured error <= 1.83 such units even at |s| <= 20).
TABLE_RHO_C: float = 4.0
# Rice formula <|theta'|> = sigma_rho' / sigma_rho, and zero mean net winding of
# the same paths: the plan's 3 sigma criterion per fixed-seed z-test (false alarm
# 0.27% per seed; measured z SD 1.00 over 12 seeds).
MD_RICE_Z: float = 3.0
# center_of_mass vs np.mean: two summation orders of n = 9 values |r| <= 12 differ
# by <= (n - 1) eps max|r| = 8 * 2.2e-16 * 12 = 2.1e-14 (measured 5.6e-17).
ATOL_MD_COM_MEAN: float = 3e-14
# com_correct vs NumPy rho exp(-i k . R_cm): rounding <= c eps |rho| (1 + |theta|),
# |rho| <= N = 9, |theta| <= 60 in the tests, c = 4: 4 * 2.2e-16 * 9 * 61 = 4.8e-13.
ATOL_MD_COM_RHO: float = 1e-12
# One winding step from 0.25: atan2 (1 ulp) of a product whose angle rounds by
# <= 2 eps, plus the addition (1 ulp), at magnitude <= pi + 0.25 (ulp 4.4e-16):
# <= 4 * 4.4e-16 = 1.8e-15.
ATOL_MD_WIND_ONE_STEP: float = 2e-15
# COM-corrected winding of rigidly translated paths: the corrected rho differ by
# roundoff N eps |theta| <= 40 * 2.2e-16 * 600 ~ 5e-12 (translations <= 8 add to
# |theta|); with the test floor S > 1e-4 (|rho| >= 1e-2) each step's angle moves
# by <= 5e-10 and 8 steps by <= 4e-9; 1e-8 leaves 2.5x headroom.
ATOL_MD_COM_WIND: float = 1e-8

# ----- MD winding analysis and statistics (E2: batten2008_md_*.py) -----
# Least-squares slope of exactly linear W(t) = w t (+ c) on <= 200 log-spaced times
# in [1e-3, 1e4]: the centred sums carry ~R eps relative rounding (R = 200: 4.4e-14),
# amplified by t_max / sd(t) <~ 5 from centring; 1e-11 leaves > 40x headroom.
RTOL_MD_RATE_EXACT: float = 1e-11
# Windowed rates by linear interpolation at snapshot times that include t/4, t/2
# and t exactly: one subtraction and one division of O(1e3) values, ~4 eps relative;
# 1e-12 is ample.
RTOL_MD_WINDOW_EXACT: float = 1e-12
# Exact power laws msd = c t^a through fit_msd: polyfit of exact log-log data
# (R <= 100 points, |log t| <= 25) returns a to ~R eps |log t| ~ 5e-13; 1e-9.
ATOL_MD_ALPHA_EXACT: float = 1e-9
# Exponent of the analytic Ornstein-Uhlenbeck MSD 2 D tau^2 (t/tau - 1 + e^-t/tau)
# fitted over t in [10 tau, 1e4 tau]: the local slope 1 + tau / t - ... decays
# from 1.11 at 10 tau to 1.0001 at 1e4 tau; the log-spaced least-squares average
# is 1.009 (computed, 120 points in [1e-2, 1e4]); 0.04 bounds it with 4x headroom,
# while the all-lag fit (ballistic t^2 included) gives 1.34, far outside.
ATOL_MD_ALPHA_OU: float = 0.04
# Exponent and D of the pooled MSD of 4000 Gaussian random walks at 60 log-spaced
# times in [1, 1e3]: the MSD at each time has relative sd sqrt(2 / 4000) = 2.2 %
# and the times are correlated; the log-log slope over 3 decades has sd 0.0035
# (measured over 20 seeds), so 0.05 is a > 10 sd bound that still rejects any
# exponent outside [0.95, 1.05]. D from the 2 D t fit is dominated by the last
# points: relative sd 1.9 % (measured over 20 seeds); 0.12 = 6 sd.
ATOL_MD_ALPHA_RW: float = 0.05
RTOL_MD_D_RW: float = 0.12
# Synthetic MD (batten2008_md_synthetic) recovered by the analysis, pooled over
# all lanes and waves of the end-to-end set (6 ground states x 3 T, 12
# constrained and ~40 unconstrained waves, 2 segments of t = 400; measured with
# seed 3: Rice ratio 0.994 constrained / 1.000 unconstrained, S/T 0.991, Phi/(MT)
# 0.991, median alpha 1.012, nu 0.999):
# - unsigned rate / Rice: the sampled total variation of the Ornstein-Uhlenbeck
#   oscillators undercounts by O(gamma dt) (0.3 % constrained, 0.5 % unconstrained
#   at dt = 0.025, measured against dt = 0.0125); the pooled statistical sd is
#   ~0.7 %; 3 % bounds bias + 3 sd.
RTOL_MD_SYN_RATE: float = 0.03
# - <S_k> / T and <Phi> / (M T): per-wave relative sd ~ sqrt(tau_S / t) ~ 9 %
#   (tau_S ~ 1 / gamma), pooled over ~200 wave series ~0.7 %; 3 % = 4 sd.
RTOL_MD_SYN_EQUIPARTITION: float = 0.03
# - median MSD exponent over the 18 lanes: lane sd 0.18 at t = 400 (window
#   [~38, 400], 24 walks per lane), median of 18 -> ~0.05; 0.15 = 3 sd (the
#   pre-registered band is 0.1; this set is far shorter than production runs).
ATOL_MD_SYN_ALPHA: float = 0.15
