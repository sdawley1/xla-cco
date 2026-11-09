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

# ----- MD driver (T5) -----
# make_waves kvec = n @ 2 pi inv(B)^T vs the same product in another summation order
# (KSet.k, NumPy matmul): <= d roundings of eps |k| with |k| <= 60 in the tests,
# i.e. <= 3 * 1.1e-16 * 60 = 2e-14; 1e-12 (absolute; scaled by max|k| against KSet).
RTOL_MD_KVEC: float = 1e-12
# Force/Phi/rho of make_force_fn vs jax.grad of the direct Cartesian sum: the phases
# 2 pi n . (r inv(B)) and k . r differ by a few eps |theta| (|theta| <= ~200 for the
# unwrapped test positions), so each of the M N terms (|term| <= |k|) is off by
# <= ~1e-13 relative; sums of M N <= 600 terms: <= 6e-11 relative to the largest
# term, below 1e-10 (times max(1, max|F|), or N for rho).
RTOL_MD_FORCE: float = 1e-10
# Galilean boost V = (0.3, -0.17) over 50 Verlet steps (t = 0.5): the forces at r and
# r + V t agree to the phase rounding above (~1e-13 relative); the positions then
# differ by <= 50 steps * dt^2 * 1e-13 * |F| plus the rounding of r + V t (~1e-15),
# and each winding step differs by the rounding of two atan2 of O(1) rho (~1e-14);
# 50 steps: <= 1e-12. 1e-9 bounds both with 1000x headroom and stays 1e3 below the
# smallest boost phase |k . V| t (>= 0.07) asserted by the test.
ATOL_MD_GALILEAN: float = 1e-9
# Linearized dynamics (N = 32 ground state, T = 1e-10, t <= 50): the reference is
# the exact discrete Verlet solution of the Hessian modes mapped through J, so the
# only differences are (i) the second-order term of rho, -1/2 sum_j (k . dr_j)^2
# exp(i k . r0_j), relative to the first-order term ~ K |dr| sqrt(N) / 2 with
# |dr| ~ sqrt(T / lambda_min) = 1.1e-5 (lambda_min = 0.86), i.e. ~1e-4, and its
# effect on the mode frequencies (relative ~ (K |dr|)^2, negligible over t = 50),
# and (ii) roundoff (|rho| ~ 2e-4 vs N eps ~ 1e-14). Measured 1.0e-4; the plan's
# criterion 1e-3 leaves 10x headroom.
RTOL_MD_LINEAR: float = 1e-3
# Power spectrum: a Hann-windowed peak of a pure tone lies within half a bin of the
# tone; two modes closer than ~2 bins merge into one peak between them, so a peak
# must lie within 2 bins (2 pi / t_total each) of some sqrt(eig K).
MD_SPECTRUM_BINS: float = 2.0
# Peaks considered: local maxima above 1e-3 of the largest; the Hann sidelobes of
# the strongest line fall below 1e-3 (-31 dB first sidelobe, then 18 dB/octave)
# beyond ~2 bins, where they would count as (wrongly placed) peaks.
MD_SPECTRUM_PEAK_FRAC: float = 1e-3

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
# ----- MD statistical validation (V1, tests/integration/test_md_equipartition.py) -----
# Power requirement of the plan: a relative bias of MD_BIAS_DETECT in <Phi>/(M T) must
# stand MD_POWER_Z standard errors from 1, i.e. SE <= 0.05 / 6 = 8.3e-3. At z = 6 a
# real 5 % bias is missed (|t| < t_crit = 3.87 for m = 8 at 511 dof) with
# probability Phi(3.87 - 6) ~ 1.7 % (AUDIT-V1 F2).
MD_POWER_Z: float = 6.0
# Effective sample size behind that power: Phi / T ~ Gamma(M, 1), so one sample of
# Phi / (M T) has variance 1/M and SE = 1 / sqrt(M n_eff); z = MD_BIAS_DETECT
# sqrt(M n_eff) >= 6 needs n_eff >= 36 / (0.05^2 M) = 600 at M = 24 (the plan's
# n_eff >= 600; asserted with the measured variance, so the two checks agree).
MD_MIN_NEFF: float = 600.0
# Negative control: the same data reported at a 10 % wrong temperature must fail the
# <Phi>/(M T) = 1 test (expected |t| ~ 0.09 / SE >= 11 at the required power).
MD_CONTROL_T_FACTOR: float = 1.1
# Velocity Verlet samples the shadow-canonical measure of the harmonic modes, biasing
# <Phi>/(M T) by (1/2M) sum lam dt^2 / (4 - lam dt^2) (5e-4 at dt = 0.02 for the
# N = 64 system, mean lam = 4.9). Comparing against exactly 1 is valid while this
# predicted bias stays below a quarter of the SE: it then shifts t by < 0.25 (and the
# family false-alarm rate from 1e-3 to < 2e-3).
MD_SHADOW_BIAS_FRAC: float = 0.25

# ----- MD C++ parity (V2, tests/parity/test_md_cxx_parity.py) -----
# Two-sample KS acceptance of JAX vs pooled C++ samples of Phi/(M T) and
# KE/(N d T/2) (the plan's V2 criterion); also applied to C++ seed A vs seed B.
MD_PARITY_KS_PVALUE: float = 1e-3
# Mean agreement: z = (m_jax - m_cxx) / sqrt(se_jax^2 + se_cxx^2) with both SEs from
# (nearly) independent groups (JAX lanes, 40; C++ batch means, 60 pooled / 20 per
# seed). Under the null |z| > 4 has probability 6e-5 (normal) to 1e-3 (Student t
# with 19 dof, the per-seed control), so the ~10 comparisons of the file keep a
# family false-alarm rate <~ 5e-3. Resolution (pooled): 4 sqrt(0.0057^2 + 0.0050^2)
# ~ 3 % in <Phi>/(M T) and ~ 1.6 % in <KE>/(N d T/2).
MD_PARITY_Z: float = 4.0
# Negative control: JAX run at 1.1 T against C++ must fail the KS test with p below
# this. Expected: a 10 % shift is 0.7 sd of Phi/(M T) (sd = 1/sqrt(M) = 0.14) and
# 1.6 sd of KE/(N d T/2) (sd = sqrt(2/(N d)) = 0.0625), i.e. KS distances D ~ 0.28
# and ~ 0.58; with 400 vs 300 thinned samples (sqrt(n m/(n + m)) = 13.1) the
# asymptotic p ~ 2 exp(-2 (13.1 D)^2) ~ 3e-12 and < 1e-100.
MD_PARITY_CONTROL_P: float = 1e-6
# Thinning before every KS test: consecutive C++ samples (and JAX samples of one
# lane) are correlated through the ~(1 - 1/N)^501 = 14 % of velocities that one
# Andersen call leaves untouched and through the positions; measured lag-1
# autocorrelation 0.17-0.21 (KE) and -0.02-0.21 (Phi). Keeping every 2nd sample
# leaves lag ~ 0.2^2 = 0.04, which changes the KS null distribution (effective n
# factor (1 - rho) / (1 + rho)) by < 8 %, against the 1e-3 threshold negligible.
MD_PARITY_KS_THIN: int = 2
# Batch-mean length for the C++ standard errors (one series of 200 per seed): for
# an AR(1) series with rho = 0.21 the correlation of adjacent means of B = 10
# samples is rho (1 - rho^B)^2 / (B (1 - rho^2)) = 0.022, so the 20 means per seed
# are nearly independent and the SE is underestimated by ~ 2 %.
MD_PARITY_BLOCK: int = 10
