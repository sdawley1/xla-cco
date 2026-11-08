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
# ~p eps relative error and z itself eps 2 pi |s|, so one (k, j) term is off by
# <= (1 + 2 pi) n_max d eps; N terms add: atol = C eps N (1 + 2 pi) n_max d with
# C = 4 for the final sum and the trig path's own rounding.
TABLE_RHO_C: float = 4.0
# Rice formula <|theta'|> = sigma_rho' / sigma_rho: the plan's 3 sigma criterion
# for one fixed-seed comparison (false alarm 0.27% per seed, fixed seed).
MD_RICE_Z: float = 3.0
# COM-corrected winding of rigidly translated paths: the corrected rho differ by
# roundoff N eps |theta| <= 40 * 2.2e-16 * 600 ~ 5e-12 (translations <= 8 add to
# |theta|); with the test floor S > 1e-4 (|rho| >= 1e-2) each step's angle moves
# by <= 5e-10 and 8 steps by <= 4e-9; 1e-8 leaves 2.5x headroom.
ATOL_MD_COM_WIND: float = 1e-8
