# Experiments

## Dawley & Torquato (2025) trajectories, χ = 0.1 and χ = 0.4

This experiment reproduces part of *Evolution of various initial many-particle configurations to disordered stealthy hyperuniform ground states*, PRE **112**, 045310 (2025). It runs collective-coordinate optimization from Poisson (POS) and Gaussian-pair-statistics (GAU, hyperuniform) initial conditions to χ = 0.1 stealthy ground states with `xla_cco`'s adaptive gradient descent. Each run is stored at full resolution.

Only scripts are tracked. All generated data lives in git-ignored directories:

| Directory | Contents |
|---|---|
| `initial_conditions/` | `POS-2D-N1024` (100 Poisson configs, seed 20251106); `GAU-2D-N1000` (50 configs copied from `CONFIGURATIONS/N1000_configs_GaussianPairStat`, basis rescaled to unit volume); `provenance.json` |
| `trajectories/chi<χ>/<ENS>/` | `traj_XXXX.{positions.f64,npz,json}` for each trajectory, plus `manifest.json`; `trajectories/chi<χ>/summary.npz` (ensemble statistics); SLURM logs |
| `figures/` | `chi<χ>_pos_gau.{png,pdf}` (S(k) of initial conditions; distance and energy vs t/T, log-log) and `chi<χ>_ground_state_sk.{png,pdf}` (S(k) of the converged ground states) |

### Pipeline (run from the repository root)

```bash
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"  # any directory you can write
uv sync --dev --extra plot --locked            # once: adds matplotlib for the plot scripts
uv run --no-sync python -I experiments/scripts/make_initial_conditions.py \
    --gau-source <prefix of the GAUSSIAN-2D-N_1000 ConfigPack>   # also creates experiments/trajectories/
sbatch experiments/slurm/run_trajectories_gpu.slurm      # chi = 0.1 (default); CPU: run_trajectories.slurm
sbatch --export=ALL,CHI=0.4 experiments/slurm/run_trajectories_gpu.slurm
# for chi = 0.4 pass --traj experiments/trajectories/chi0.4 to the analysis and
# --summary .../chi0.4/summary.npz --out experiments/figures/chi0.4_<name> to the plot scripts
JAX_PLATFORMS=cpu uv run --no-sync python -I experiments/scripts/analyze_trajectories.py
uv run --no-sync python -I experiments/scripts/plot_figure.py
uv run --no-sync python -I experiments/scripts/plot_ground_state_sk.py
```

### Conventions

- **Descent rule** (paper, Sec. IV). The descent uses the normalized gradient. The step doubles after an accepted downhill move and halves otherwise; acceptance is as in the C++ `LocalGradientDescent`, with rel_tol = 0.01. The initial step is γρ^{1/2} = 1e-3 in Cartesian units, which is a fractional step of 1e-3/L.
- **χ = 0.4** uses K = 4.483993. That gives M = 820 and χ = 0.40078 for POS, and M = 802 and χ = 0.40140 for GAU. The GPU job was 17000661.
- **χ = 0.1** uses K = 2.241996. That gives M = 210 and χ = 0.10264 for POS (N = 1024), and M = 200 and χ = 0.10010 for GAU (N = 1000).
- **Ground state.** Each run continues until Φ < 1e-22 (the paper's tolerance) or until the step floor. The step floor is where float64 can no longer resolve energy changes, at Φ ≈ 1e-22. Any terminal Φ < 1e-20 counts as a ground state.
  - All 150 trajectories reach a ground state.
  - 49/100 POS and 29/50 GAU runs reach Φ < 1e-22.
  - Each run's `status` and `final_energy` are in its JSON file.
  - At χ = 0.4 the float64 floor is higher: all 150 runs stop at the step floor with Φ between 4.9e-20 and 1.4e-18 (median about 2.5e-19). The floor rises with M, which is about 4x larger here, so the manifests' `n_ground_states` (Φ < 1e-20) reads 0.
    - These endpoints are still ground states to within about 1e-9. Polishing POS and GAU `traj_0000` with the library's L-BFGS takes Φ from about 1e-19 to 6e-26 and moves the whole configuration by only 3-4e-9 in total, against p_0(∞) values of 26 and 11.
    - In ground-state S(k), S ≤ 1.5e-19 for k < K − 0.1. The 0.1-wide bin straddling K (mean |k| = 4.45) mixes vectors inside and outside the window, so do not read it as part of the window.
- **Time.** An iteration contributes t_m = γρ^{1/2} (its Cartesian step) if accepted and 0 otherwise. T is the total, and the figures use t/T.
- **Proximity metric** (Eq. 10). p_0(t) = ρ^{1/2} [Σ_j |r_j(t) − r_j(0)|²]^{1/2}, computed on unwrapped paths. The plotted distance to the manifold is p_0(∞) − p_0(t).

### Loading a trajectory for further analysis

```python
from pathlib import Path
from xla_cco.io.trajstore import open_trajectory
from xla_cco.analysis.trajectory import unwrap_fractional, proximity_metric
import numpy as np

frames, arrays, meta = open_trajectory(
    Path("experiments/trajectories/chi0.1/POS/traj_0000")
)
# frames: memmap (F, N, 2), wrapped fractional positions; frame 0 is the initial condition,
#         followed by the state after every ACCEPTED iteration
# arrays: per-iteration 'iteration', 'energy', 'step_used', 'step_used_cartesian',
#         'accepted', 'frame_index' (frame holding the state after that iteration), 'time', 'energy0'
# meta:   N, d, L, basis, chi_actual, M, K, settings, status, final_energy, git_commit, ...
paths = unwrap_fractional(np.asarray(frames))
p0 = proximity_metric(paths, paths[0], np.asarray(meta["basis"]), meta["density"])
positions_at_iteration_m = frames[arrays["frame_index"][m - 1]]  # m >= 1
```

Rejected iterations leave positions unchanged, so storing frames only at accepted iterations is lossless. Every iteration is still recoverable through `frame_index`.

## Wasserstein-2 distance along the trajectories

This section compares the proximity metric with a label-free distance: the periodic Wasserstein-2 (optimal-transport) distance between N-particle configurations. It uses the uniform empirical measures on the torus and the minimum-image cost.

**Definition.** W(a, b) = ρ^{1/2} [min over permutations σ of Σ_i |a_i − b_σ(i)|²]^{1/2}. It is scaled like p_0, so W ≤ p_0, because the identity labelling is one of the admissible matchings. Two quantities are tracked per trajectory:

- W_0(t) = W(r(0), r(t)), the analogue of p_0(t). It is plotted as W_0(∞) − W_0(t).
- W_∞(t) = W(r(t), r(∞)), the distance to the ground state actually reached. Its labelled counterpart is q_∞(t) = ρ^{1/2} |r(t) − r(∞)|.

The library code is in `xla_cco.analysis.transport`:

- the cost matrix;
- log-domain Sinkhorn with eps annealing, warm starts and an optional stop on a bijective rounding;
- rounding with greedy repair.

### Two computations

- **Exact** (`wasserstein_exact.py`, CPU): Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on 201 frames per trajectory, at t = 0 and 200 times log-spaced in t/T ∈ [1e-6, 1]. The labelled p_0 and q_∞ are stored on the same frames.
  - Output: `wasserstein/exact_<ENS>.npz`, with arrays of shape (trajectory, frame).
- **Sinkhorn** (`wasserstein_sinkhorn.py`, GPU): runs at every stored frame of every trajectory.
  - Parameters: eps = 1e-2 (Cartesian² units, ρ = 1).
  - Lanes: each (trajectory, reference) pair is an independent lane that warm-starts from its previous frame.
  - Stopping: a frame stops once the plan's row-argmax rounding is a bijection, once its marginal error is below tol = 1e-4, or after 100 iterations.
  - Repair: a rounding that is still not a bijection is repaired greedily and flagged (`is_perm = 0`). Such frames are not certified; the exact fallback below replaces them.
  - Reported value: `w_round`, the cost of that matching. It is a feasible matching, so `w_round` ≥ exact W for finite inputs.
  - Diagnostic only: `w_ent`, the entropic plan's own cost. It carries a bias of order eps, which swamps W near t → T.
  - Output: `wasserstein/sinkhorn_<ENS>[_<start>].npz`, with per-frame arrays concatenated over trajectories and delimited by `offsets`. The last axis is (r(0), r(∞)). Slices from `--start/--limit` are merged by the analysis.

### Accuracy of Sinkhorn against exact, and the exact fallback

- **Certified frames.** A frame is trusted only if the plan's row-argmax rounding is a bijection. All 76k compared bijective frames equalled the exact optimum (worst relative error 3e-16).
- **Uncertified frames.** Where the rounding collides, the greedy repair can be suboptimal, even when the plan has converged (D-014). This happens at genuine near-ties, where two close particles' partner swap costs less than ~eps, and at plans still unconverged at the 100-iteration cap, at t/T ≈ 0.7–0.9. The repair's error reached 2.8e-3 relative on converged frames and 100% on unconverged ones.
- **Exact fallback.** `wasserstein_fallback.py` solves the exact assignment at every uncertified (frame, reference) entry, about 3–5% of POS r(0) entries and essentially no GAU ones. It writes `wasserstein/fallback_<ENS>.npz`. `analyze_wasserstein.py` substitutes these values and stops with an error if any are missing, so every curve value is exact.
- **Diagnostics.** `summary.npz` stores the worst errors per class (`<ENS>/disc_rel_certified`, `disc_rel_repaired`, `disc_rel_final`, `disc_rel_ent`) and the fallback fraction (`disc_frac_fallback`).
- **Why not a higher cap.** A 2000-iteration cap removes most collisions but stalls for hours on persistent near-ties, because GPU lanes wait on a few hard frames.

### Results

- **χ = 0.1:** the optimal matching is the identity at every frame, for both ensembles, so W ≡ p_0 and W_∞ ≡ q_∞ exactly.
- **χ = 0.4, POS:** on average 30 of 1024 particles end up matched to a different initial particle. ⟨W_0(∞)⟩ = 25.540 against ⟨p_0(∞)⟩ = 25.605, a gap of ≤ 0.6%.
- **χ = 0.4, GAU:** about 1 particle is re-matched. ⟨W_0(∞)⟩ = 11.646 against ⟨p_0(∞)⟩ = 11.648.
- **Re-matching over time:** relative to r(∞), re-matching vanishes over the last decade of t/T. The labelled q_∞ therefore equals W_∞ there.

### Running

```bash
sbatch --export=ALL,CHI=0.1 experiments/slurm/wasserstein_gpu.slurm      # exact (CPU cores) + Sinkhorn (GPU)
# chi = 0.4 needs slices to fit gputest's 59 min:
sbatch --export=ALL,CHI=0.4,ENSEMBLES=POS,START=0,LIMIT=20,EXACT=1 experiments/slurm/wasserstein_gpu.slurm
sbatch --export=ALL,CHI=0.4 experiments/slurm/wasserstein_fallback.slurm  # after all Sinkhorn slices; CPU only
JAX_PLATFORMS=cpu uv run --no-sync python -I experiments/scripts/analyze_wasserstein.py --traj experiments/trajectories/chi0.4
uv run --no-sync python -I experiments/scripts/plot_wasserstein.py \
    --summary experiments/trajectories/chi0.4/wasserstein/summary.npz --out experiments/figures/chi0.4
```

### Figures

The figures are `figures/chi<χ>_w2.{png,pdf}` (the Wasserstein mirror of `chi<χ>_pos_gau`) and `figures/chi<χ>_w2_vs_p0.{png,pdf}` (exact W against labelled p on the same frames). The docstring of `plot_wasserstein.py` lists the panels.

## Batten, Stillinger & Torquato (2008) appendix: landscapes and gradient flows

This section recreates the energy-landscape picture in the appendix of *Classical disordered ground states: super-ideal gases and stealth and equi-luminous materials*, J. Appl. Phys. **104**, 033504 (2008). The setting is the stealth case (D = 0) for N particles on a line at density 1, so the cell side is L = N. The constraint covers the first M = round(χN) wavevectors k_n = 2πn/L, with χ = M/(dN) as in that paper.

- **N = 3 (Fig. 15 setting).** Contours of Φ(x₂, x₃) with x₁ = 0, at χ = 1/3 (M = 1) and χ = 2/3 (M = 2). The figure overlays 50 gradient flows from Poisson starts, drawn in relative coordinates (x₂ − x₁, x₃ − x₁) mod 1. The landscape is exact for any x₁ because Φ is translation invariant.
- **N = 1000.** 20 Poisson starts at each of χ = 1/3 (M = 333) and χ = 2/3 (M = 667). The landscape is 1000-dimensional, so each flow is drawn on three 2D slices (`batten2008_slices.py`):
  - **Start-end plane.** It passes through r(0) and is spanned by the net displacement r(∞) − r(0) and the flow's largest perpendicular excursion. Both ends lie in the plane, and each recorded point's distance from the plane is stored.
  - **PCA plane.** It passes through the path mean and is spanned by the path's first two principal axes. Their variance fractions are stored.
  - **Two-particle slice.** All particles are frozen at r(∞) except the two that moved most, as in the 3-particle figures. The overlay shows those two particles' coordinates along the flow, but the other particles are not frozen during the flow.

**Energy.** Φ is xla_cco's stealthy energy (1/V) Σ_k |ρ(k)|². The paper's Eq. (A1) with D = 0 is (|ρ|²/2)², a monotone function of Φ. The contours therefore coincide, with relabelled levels, and the flow lines are the same curves up to time reparametrization.

**Dynamics: the true gradient flow dr/dt = −∇Φ.** It is integrated with `xla_cco.minimize.flow.gradient_flow`.

- **Why an implicit method.** At N = 1000 the landscape is stiff: steep and nearly flat directions coexist. Explicit Dormand-Prince steps are then capped near Δt ≈ 0.5 by stability, while the flow needs a time of about 10⁶. A GPU probe of 10⁵ explicit steps reached only Φ ≈ 0.02 at χ = 1/3. Integrating in arclength traces the same curve but does not remove the stiffness.
- **The method used.** The flows use the L-stable Rosenbrock ROS2 method with the exact closed-form Hessian (`StealthyEnergy.hessian`) and a filtered error estimate. Settings: `rtol = 1e-2` and `atol = 1e-8` per step (RMS Cartesian), stopping at Φ < 1e-20, or |∇Φ| < 1e-12 for local minima.
- **Checkpointing.** Integration runs in checkpointed chunks (`--chunk-steps`), so a flow can span several 59-minute jobs.
- **Accuracy.** Rerunning the first 3 χ = 1/3 flows with `rtol = 1e-3` changed the paths by at most 8×10⁻³ RMS per coordinate at equal arclength, against displacements of 6–13. The path lengths agree to 10⁻⁵.
- **The total time T is not robust.** It differs by up to 1.7× between the two tolerances, because the last stretch below Φ ≈ 10⁻¹⁶ is an exponentially slow approach to the manifold. Plots against t/T inherit this.

**Results.**

- **Ground states reached.** Every flow reached a stealthy ground state (Φ < 1e-20), with no local minima: 50/50 at N = 3 for both χ, and 20/20 at N = 1000 for both χ.
- **N = 3 endpoints.** The flows end at the two crystalline arrangements (x₂, x₃) = (1/3, 2/3) and (2/3, 1/3), as in Fig. 15.
- **N = 1000 cost.** Flows took 3,400–3,800 ROS2 steps at χ = 1/3, about 3 min on an A100 for all 20, and 9,900–18,000 steps at χ = 2/3, about 15 min.
- **N = 1000 is dominated by long-range transport.** Particles move 8.7 spacings RMS from the Poisson start (median over flows; 5.8–14.8, with single particles up to about 21). A 1D Poisson pattern has large long-wavelength density fluctuations, and the smallest constrained wavevectors force them out by moving particles over many spacings.
- **χ = 2/3 ends on the crystal; χ = 1/3 ends disordered.** In 1D, χ > 1/2 leaves the integer lattice as the only ground state, and every χ = 2/3 flow ends on it: the deviation from a translated lattice is ≤ 1e-9 and every spacing equals 1. At χ = 1/3 the ground-state manifold contains disordered states, and every flow ends in one. Particles sit 0.24–0.25 spacing RMS from the nearest lattice site (max 0.5), with spacings from 0.001 to 2.1 (std 0.47).
- **The two χ values share their large-scale path.** Both constrain the same longest wavelengths, so from the same start the paths agree at large scales and look alike in the start-end plane. The endpoints differ by 0.25 spacing RMS, which is exactly the χ = 1/3 state's distance from the lattice. At χ = 2/3 the extra short wavelengths remove that local disorder and lengthen the path by about 3% (median arclength 410 vs 423).
- **Shape of the flows.** In the start-end and PCA planes each flow first rises out of the plane's straight line into a long, narrow low-Φ valley, then follows the valley floor to the ground state. In the two-particle slices, the two most-moved particles drift together across 10–25 spacings, and the low-Φ lines mark either particle at its ground-state position.
- **The contours do not give the energy along the flow** (`batten2008_N1000_mismatch`, computed by `batten2008_mismatch.py`). A slice shows Φ at the projected point, not at r(t). Panels (d–f) give log₁₀(Φ_plane/Φ) against t/T over all flows:
  - *Start-end:* exact at both ends, since both lie in the plane, and 1–2 orders too high in between.
  - *PCA:* within about an order until t/T ≈ 0.1. After that the plane misses the ground state: r(∞) lies 2–10 spacings off it, and the contour reads Φ ≈ 0.1–6 (median 0.3 at χ = 1/3, 2.1 at χ = 2/3) where the true value is below 1e-20.
  - *Two-particle:* about 2.4 orders too low at the start, because the other 998 particles already sit at r(∞). Late in the flow it is about 2 orders too high, and it matches only at the end.

  The slices show the shape of the valley, not the energy on the path; Φ(t) is in `batten2008_energy`.

```bash
uv run --no-sync python -I experiments/scripts/batten2008_flows.py --n 3 --n-traj 50 \
    --chunk-steps 400 --out experiments/trajectories/batten2008/N3 --device cpu
uv run --no-sync python -I experiments/scripts/batten2008_slices.py \
    --dir experiments/trajectories/batten2008/N3 --device cpu
sbatch --export=ALL,CHI=1/3 experiments/slurm/batten2008_flows_gpu.slurm   # N = 1000; resubmit to resume
sbatch --export=ALL,CHI=1/3 experiments/slurm/batten2008_slices_gpu.slurm  # after the flows finish
sbatch experiments/slurm/batten2008_mismatch_gpu.slurm   # contour vs true energy, N = 1000
uv run --no-sync python -I experiments/scripts/batten2008_plot.py
```

**Data.**

- `trajectories/batten2008/N<N>/flows_chi<a-b>.{npz,json}` hold the thinned paths: up to 4000 rows each, unwrapped fractional positions, with t, arclength and Φ.
- `N<N>/chi<a-b>/chunk_*.npz` hold every accepted step.
- `views_chi<a-b>.npz` and `landscape_chi<a-b>.npz` hold the slices.
- `mismatch_chi<a-b>.npz` holds Φ at the projected point and the off-plane distance, per view and frame.

**Figures.** `figures/batten2008_N3`, `figures/batten2008_N1000_chi<a-b>_{se,pca,pair}`, `figures/batten2008_N1000_mismatch` and `figures/batten2008_energy`.

### Wasserstein-2 along the 2008 flows

`batten2008_wasserstein.py` computes the exact W₂ at every stored frame of every flow (N = 3: 50 flows per χ; N = 1000: 20), against both r(0) and r(∞). It uses `xla_cco.analysis.transport.periodic_assignment_1d`, an exact O(N log N) solver for squared cost on a circle: the optimal matching is a cyclic shift of the sorted orders, found by FFT and checked directly (Delon, Salomon & Sobolevski 2010). It is tested against the Hungarian algorithm.

The overlays (`batten2008_w2_plot.py`) add three curves to each landscape figure:

- the flow **relabelled** by its optimal matching to r(0);
- the flow relabelled by its optimal matching to r(∞);
- the **W₂ geodesic** r(0) + s·d, s ∈ [0, 1], where d is the optimal displacement from r(0) to r(∞). This is the shortest transport path between the two ends.

For N = 1000 the curves are projected onto the same start-end, PCA and two-particle planes, and `batten2008_w2_N1000_offplane` reports each point's distance from its plane.

**Results.**

- **The optimal matching is the identity at every frame of every flow,** for both references, both χ and both N. So W₀(t) = p₀(t) and W∞(t) = q∞(t) to round-off (|W₀ − p₀| ≤ 1e-13), and the relabelled curves lie exactly on the flows. This holds for any 1D collective-coordinate energy:
  - ∂Φ/∂x_i depends on i only through x_i, so all particles move in one velocity field G(x, t);
  - trajectories of a 1D ODE in a common field cannot cross, so the cyclic order is preserved;
  - with the centre of mass also conserved, the identity stays optimal.
  `tests/unit/test_periodic_assignment.py::test_1d_stealthy_flow_keeps_the_identity_matching` checks it. In 2D, particles pass around one another, which is why the PRE trajectories re-match (≈30 particles for POS at χ = 0.4).
- **The geodesic is the straight chord r(0) → r(∞),** and the flow is longer than it:
  - N = 3: the flow arclength is 1.06× W₀(∞) (median; up to 1.5×);
  - N = 1000: it is 1.5× (1.3–1.6×).
- **The geodesic crosses high-energy ground** (`batten2008_w2_geodesic_energy`). Halfway along at N = 1000, Φ on the geodesic is 87 (χ = 1/3) and 197 (χ = 2/3), against 1.2–1.4 on the flow, starting from Φ(0) ≈ M. In the PCA planes the chord cuts straight across the low-Φ valley that the flow follows. At N = 3 the two energy profiles nearly coincide.

```bash
sbatch experiments/slurm/batten2008_wasserstein_gpu.slurm      # exceeds login-node limits at N = 1000
uv run --no-sync python -I experiments/scripts/batten2008_w2_plot.py
uv run --no-sync python -I experiments/scripts/plot_wasserstein.py --exact --ensembles chi1-3 chi2-3 \
    --summary experiments/trajectories/batten2008/N1000/w2_summary.npz --out experiments/figures/batten2008_N1000
```

**Figures.**

- `figures/batten2008_w2_N3`
- `figures/batten2008_w2_N1000_chi<a-b>_{se,pca,pair}`
- `figures/batten2008_w2_N1000_offplane`
- `figures/batten2008_w2_geodesic_energy`
- `figures/batten2008_N{3,1000}_w2{,_vs_p0}`: the PRE-style summaries.

### Collective coordinates, valley floors and Hessians along the 2008 flows

Flat 2D slices give the path's shape but not its energy (`batten2008_N1000_mismatch`), and no single plane does much better: the best plane through an arclength-resampled path still leaves it up to 12% of |r(∞) − r(0)| away; 3, 5 and 10 principal axes leave 7.5%, 3.6% and under 1%. Three analyses of the N = 1000 flows avoid projecting Φ:

1. **Collective coordinates** (`batten2008_modes.py`). Φ = Σₙ S(kₙ) with S(kₙ) = |ρ(kₙ)|²/N over the M constrained modes, so the energy along the flow is exact mode by mode (Σₙ S matches the recorded Φ to 1e-15).
2. **Valley floors** (`batten2008_floor.py`). On the start-end and PCA planes, Φ is minimized over the N − 2 directions orthogonal to the plane at each grid point (L-BFGS on an energy whose gradient has its in-plane part removed). Each grid point starts from the nearest flow frame's off-plane offset, and sampled flow frames are relaxed from r(t) itself.
3. **Hessians** (`batten2008_hessian.py`). Eigenvalues of the exact Hessian at 15 times on every flow (about three per decade of t/T), the curvature along each density wave, the wavelength content of the negative-curvature eigenvectors, and cross-sections of Φ through r(t) spanned by the flow direction τ = −∇Φ/|∇Φ| and the stiffest eigenvector.

**Results.**

- **Short wavelengths relax first; the energy ends up in the longest ones** (`batten2008_N1000_modes`). The last time S(kₙ) exceeds 10⁻² scales as n⁻²: over n = 1–300 at χ = 1/3, and up to n ≈ 30 at χ = 2/3. By t/T ≈ 10⁻³ the lowest 10% of modes carry essentially all of Φ.
- **The longest wavelength obeys the linearized flow exactly.** Linearizing gives dρ(k)/dt = −k²ρ(k). Mode n = 1 decays as S(k₁, t) = S(k₁, 0) exp(−2k₁²t) from the first step: the fitted rate is 0.499 × 4k₁² at both χ, and its phase changes by at most 0.02 rad over a whole flow (`batten2008_N1000_mode_planes`: straight radial lines). Its time constant 1/(2k₁²) = L²/(8π²) ≈ 1.3 × 10⁴ sets the length of the final approach. Because it ignores the other modes, S(k₁, t) from a given start is the same at χ = 1/3 and 2/3 (the curves overlap in panel (d)). Higher modes decay more slowly than their own linear rate 2kₙ², and their phases wind before collapsing: they are driven by the slow modes.
- **χ = 2/3 orders in one event.** Every mode above n ≈ 30 falls below 10⁻⁸ together at t/T ≈ 2.5 × 10⁻⁴; the lower modes then follow the n⁻² law. At χ = 1/3 the modes relax one by one.
- **Ground-state manifold dimension** (`batten2008_N1000_hessian`). At the end of every χ = 1/3 flow the Hessian has exactly 334 = N − 2M zero eigenvalues: the disordered ground states form a locally 334-dimensional manifold, one dimension per unconstrained degree of freedom. The near-zero count grows along the flow: 12 at t/T = 10⁻³, 60 at 10⁻², and all 334 by 0.1. At χ = 2/3 only the translation mode is zero: the lattice is an isolated minimum.
- **Curvature along the path.** At the Poisson start about 55% of the eigenvalues are negative. At χ = 2/3 they are gone by t/T = 10⁻³, after which the flow is inside the lattice's convex basin. At χ = 1/3 about 170 weakly negative directions (λ ≈ −10⁻³ to −10⁻⁴) persist until the flow reaches the manifold. The stiffest eigenvalue settles at 6.3 (χ = 1/3) and 22 (χ = 2/3).
- **Phase planes for every scale** (`batten2008_N1000_mode_planes_full`, `batten2008_N1000_mode_phases`). A radial move in these plots changes that mode's energy; an angular move slides its density wave at fixed energy. Results:
  - *Winding.* Mode 1 turns by 0.001 turns, i.e. not at all. Mode 2 turns 0.4 (χ = 1/3) or 0.3 (χ = 2/3). Modes n ≥ 3 wind 1–10 turns at χ = 1/3 but stay under 1 turn up to n ≈ 30 at χ = 2/3.
  - *Lock-in.* For n ≥ 3 the phase keeps moving until S(kₙ) is down to 10⁻¹¹–10⁻¹⁵, so these modes spiral almost all the way in. Mode 2 locks at about 10⁻¹² (χ = 1/3) or 10⁻⁸ (χ = 2/3). Frame spacing resolves the phases: at most 0.2% of steps exceed π/2.
  - *Late-time slaving, S_n ∝ S₁^{p_n}.* At χ = 1/3, p_n = 1.00 for n = 2–10: the final approach is linear relaxation along one Hessian direction, which carries every density wave in fixed proportion. At χ = 2/3, p₂ = 2.00, p₃ = 3.01, p₄ = 4.2, i.e. p_n = n: near the lattice the last phonon is long-wavelength, and density wave n is its n-th harmonic, with amplitude ∝ Jₙ(kₙa) ∝ aⁿ. Independent decay would give n².
- **Harmonic stiffness near the ground states** (`batten2008_N1000_hessian_detail`, 15 times per flow). At the end of every flow the positive eigenvalues come in pairs at k_p² (cosine and sine density waves): the ratios are 0.999 at χ = 1/3 and exact at χ = 2/3. At χ = 2/3 even λ_max = 21.94 = k₃₃₃² + k₆₆₇², the two density waves that alias onto the same lattice phonon.
  - *χ = 2/3 is harmonic early.* Along every density wave the curvature equals k_n² to 1% from t/T ≈ 3 × 10⁻⁴, while Φ is still about 5. For n ≥ 334 it equals the aliased sums (k_n² + k_{N−n}²).
  - *χ = 1/3 is not.* Along pure density waves the end state is stiffer than k_n²: 47× for n = 1, 24× for n = 2, 7.5× for n = 10, 1.2× for n = M. Coupling through ρ(k > K) mixes each long wave with short ones, so the normal modes of a disordered ground state are mixtures, while their eigenvalues stay at k_p².
  - *Negative curvature.* At χ = 2/3 the last negative directions are long-wavelength (72–83% of their power below 0.1K), and they vanish by t/T = 10⁻⁴. At χ = 1/3 about 170 persist until t/T ≈ 0.1. By t/T = 10⁻³ more than 99% of their motion lies at the unconstrained wavelengths q > K, the directions that become the 334 zero modes. Their strength falls as √Φ (|λ_min|/√Φ ≈ 1–2 × 10⁻⁴). The flow therefore approaches the manifold through a nearly flat, slightly saddle-shaped region whose curvature changes sign across directions tangent to the manifold.
- **The valley is narrow and the path bends** (`batten2008_N1000_sections`). At t/T = 0.1, Φ grows 100-fold within 0.53 (χ = 1/3) or 0.24 (χ = 2/3) along the stiffest direction, while the flow travels 12–15 for a tenfold drop. Straight down the local gradient, Φ starts to rise again within a few units, so the descent direction turns continually.
- **Valley floors** (`batten2008_N1000_floor`; flow 0, 7 × 7 grid at χ = 1/3, 9 × 9 at χ = 2/3, plus 20 sampled frames):
  - *χ = 1/3: the floor is zero everywhere.* Every grid point on both planes, and every sampled frame including the Poisson start, relaxes to a ground state (Φ ≤ 5 × 10⁻¹⁹) without leaving its fiber. As dimension counting predicts, the 334-dimensional ground-state manifold meets every (N − 2)-dimensional fiber, so the flat slices' barriers disappear once the other 998 directions can move. Getting there takes a median 1,200–1,300 units of motion, 4.5× the flow's net displacement.
  - *χ = 2/3: a broad funnel to the lattice.* The floor stays at Φ ≈ 0.003–20 (median 4), 15–50× below the flat slice, and falls toward r(∞), unlike the flat slices' narrow groove. Beneath the path the floor is 12–16 at the start, where the true Φ is 629. From t/T ≈ 10⁻² on it equals Φ(r(t)) to 0.1%, so from then on the flow runs along its valley floor.
  - *Caveats.* At χ = 2/3, L-BFGS (up to 3 × 10⁵ evaluations per point) ended on stagnation, line-search failure or the budget. Those floors are therefore upper bounds, local minima on each fiber. The in-plane drift has median 10⁻⁹ but reaches 2.3 at χ = 2/3, which is 12% of the 18.7 grid spacing. The cause is not yet confirmed; a likely one is that L-BFGS wraps positions into the cell between iterations. Dense grids are not affordable at N = 1000 without a preconditioner for the diffusive (λ ∝ k²) stiffness: one plane takes 14–28 min on an A100.

```bash
sbatch --export=ALL,STEP=modes experiments/slurm/batten2008_cc_gpu.slurm
sbatch --export=ALL,STEP=hessian experiments/slurm/batten2008_cc_gpu.slurm
# valley floors: one job per chi and plane (each under an hour)
sbatch --export=ALL,STEP=floor,STEP_ARGS="--chi 1/3 --views se --grid 7 --path-points 20 --max-evals 100000 --batch 72" \
    experiments/slurm/batten2008_cc_gpu.slurm
sbatch --export=ALL,STEP=floor,STEP_ARGS="--chi 2/3 --views se --grid 9 --path-points 20 --max-evals 100000 --batch 104" \
    experiments/slurm/batten2008_cc_gpu.slurm      # and --views pca for both
uv run --no-sync python -I experiments/scripts/batten2008_cc_plot.py
```

**Data.** `trajectories/batten2008/N1000/{modes,hessian}_chi<a-b>.npz` and `floor_chi<a-b>_{se,pca}.npz`.

**Figures.** `figures/batten2008_N1000_{modes,mode_planes,mode_planes_full,mode_phases,hessian,hessian_detail,sections,floor}`.

### The same analyses in 2D: Poisson starts to stealthy ground states

`flows2d.py` integrates the true gradient flow dr/dt = −∇Φ (ROS2 with the exact Hessian, as for the 1D flows) from the first 20 Poisson starts of `initial_conditions/POS-2D-N1024` (N = 1024, density 1, L = 32), at χ = 0.1 (M = 210) and χ = 0.4 (M = 820). The energies are the same as in `run_trajectories.py`. All 40 flows reach Φ < 10⁻²⁰: about 2,000 steps (530 s for 20 flows) at χ = 0.1 and 2,670–2,920 steps (790 s) at χ = 0.4. Every analysis script takes `--root experiments/trajectories/flows2d`. The energy is rebuilt from each run's stored basis and K, the Wasserstein matching runs on the torus (Hungarian assignment per frame), and the two-particle slice moves each particle along its own net displacement.

**Results, compared with 1D.**

- **The 2D flows are local.** Particles move 0.70 (χ = 0.1) and 0.81 (χ = 0.4) mean spacings RMS, at most 2.7. In 1D they moved up to 8.7. At χ = 0.1 the optimal matching to r(0) is the identity at every frame, so W₀ = p₀ to 7 × 10⁻¹⁵. At χ = 0.4, 51% of frames relabel some particles (at most 56 of 1024) and |W₀ − p₀| ≤ 0.18 out of 22–28 (`flows2d_N1024_w2`). The W₂ geodesic between the endpoints climbs to a median Φ of 207 (χ = 0.1) and 808 (χ = 0.4) (`flows2d_w2_geodesic_energy`).
- **Mode relaxation follows the 1D picture** (`flows2d_N1024_modes`). The last time S(k) exceeds 10⁻² scales as |k|⁻² across all shells at both χ. The longest wavelength decays at 0.978 × 2k₁² (χ = 0.1) and 0.85 × 2k₁² (χ = 0.4), close to the linearized rate but less exact than in 1D (0.998), because in 2D the |k| = k₁ shell holds two independent waves.
- **Phases** (`flows2d_N1024_mode_planes_full`, `flows2d_N1024_mode_phases`). The two k₁ waves, (0, 1) and (1, 0), turn by 0.006 (χ = 0.1) and 0.03–0.04 (χ = 0.4) turns, i.e. straight radial lines. Higher modes wind at most about 1 (χ = 0.1) or 2 (χ = 0.4) turns, far less than the 1–10 turns at 1D χ = 1/3. Phases lock at S ≈ 10⁻⁸ to 10⁻¹⁰. Late-time slaving gives p ≈ 1 for every sampled mode at both χ, as at 1D χ = 1/3: the final approach is linear relaxation along one Hessian direction. No harmonic (p = n) regime exists, because neither χ is crystalline.
- **Ground-state manifold dimension** (`flows2d_N1024_hessian`). Every end state has exactly Nd − 2M zero eigenvalues: 1628 at χ = 0.1 and 408 at χ = 0.4, so both are disordered, locally (Nd − 2M)-dimensional manifolds, like 1D χ = 1/3. The zero count jumps to its final value between t/T = 10⁻² and 0.1. The stiffest eigenvalue settles at 6.4 (χ = 0.1) and 33 (χ = 0.4).
- **Negative curvature lasts longer than in 1D** (`flows2d_N1024_hessian_detail`). About 1,160 (χ = 0.1) and 1,130 (χ = 0.4) of the 2048 eigenvalues are negative at the Poisson start. At χ = 0.1 about 820 remain at t/T = 10⁻², 83 at 0.1 and none at 0.3. At χ = 0.4 the count falls faster, to about 200 by 10⁻², and reaches zero at 0.3. Unlike 1D χ = 1/3 (>99% of the negative directions' motion at q > K), the 2D negative directions favor the unconstrained wavelengths only 2–3× (χ = 0.1) or 1.3× (χ = 0.4) per eigenvector.
- **Harmonic stiffness.** End-state eigenvalues pair at |k_p|² less exactly than in 1D. λ/|k|² is 0.97–0.98 for the first pairs at χ = 0.1 (median 0.94), and 0.84–0.87 for the first pairs at χ = 0.4 (median 0.64). Along pure density waves the end state is stiffer than |k|²: 1.6–2.6× for the longest waves at χ = 0.1 and 9–15× at χ = 0.4.
- **Flat slices misreport the energy as in 1D** (`flows2d_N1024_mismatch`). The median log₁₀(Φ_plane/Φ) is +0.34 (χ = 0.1) and +0.84 (χ = 0.4) on the start-end plane, +2.8 on the PCA plane over the second half of the flow (reaching 19–20 at the end), and −1.7 and −0.7 for the two-particle slice.
- **The valley floor is zero everywhere** (`flows2d_N1024_floor`; flow 0, 21 × 21 grid on each plane plus 40 sampled frames). All 441 grid points per plane and every sampled frame relax to a ground state (Φ ≤ 10⁻²⁰, all with the energy-tolerance status) without leaving their fiber, at both χ. The in-plane drift is at most 7 × 10⁻¹¹. Reaching the floor takes a median 44–63 units of motion on the grid and 58–77 from r(t). On the grid that is 1.7–2.8× flow 0's net displacement, against 4.5× in 1D. In 2D, unlike 1D χ = 2/3, dense floors are cheap: a 21 × 21 plane takes 34–38 s at χ = 0.1 and about 250 s at χ = 0.4 on an A100.

```bash
sbatch --export=ALL,CHI=0.1 experiments/slurm/flows2d_gpu.slurm   # resubmit to resume
sbatch --export=ALL,CHI=0.4 experiments/slurm/flows2d_gpu.slurm
R=experiments/trajectories/flows2d
sbatch --export=ALL,STEP=slices,STEP_ARGS="--dir $R/N1024 --chi 0.1 0.4" experiments/slurm/batten2008_cc_gpu.slurm
for s in mismatch modes hessian; do
  sbatch --export=ALL,STEP=$s,STEP_ARGS="--root $R --sizes 1024 --chi 0.1 0.4" experiments/slurm/batten2008_cc_gpu.slurm
done
sbatch --export=ALL,STEP=wasserstein,STEP_ARGS="--root $R --sizes 1024 --chi 0.1 0.4 --max-frames 300" \
    experiments/slurm/batten2008_cc_gpu.slurm
for c in 0.1 0.4; do for v in se pca; do       # one floor job per chi and plane
  sbatch --export=ALL,STEP=floor,STEP_ARGS="--root $R --n 1024 --chi $c --views $v --grid 21 --path-points 40 --max-evals 100000 --batch 256" \
      experiments/slurm/batten2008_cc_gpu.slurm
done; done
C="--root $R --chi 0.1 0.4 --prefix flows2d --dim 2"
uv run --no-sync python -I experiments/scripts/batten2008_plot.py $C
uv run --no-sync python -I experiments/scripts/batten2008_cc_plot.py $C --n 1024
uv run --no-sync python -I experiments/scripts/batten2008_w2_plot.py $C
uv run --no-sync python -I experiments/scripts/plot_wasserstein.py --exact --ensembles chi1-10 chi2-5 \
    --summary $R/N1024/w2_summary.npz --out experiments/figures/flows2d_N1024
```

**Data.** `trajectories/flows2d/N1024/`: `flows_chi{1-10,2-5}.{npz,json}` and the same analysis files as in 1D.

**Figures.** `figures/flows2d_N1024_*` and `figures/flows2d_w2_*`, with the same panels as the corresponding `batten2008_*` figures.
