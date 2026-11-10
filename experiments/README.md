# Experiments

## Dawley & Torquato (2025) trajectories, χ = 0.1 and χ = 0.4

This experiment reproduces part of *Evolution of various initial many-particle configurations to disordered stealthy hyperuniform ground states*, PRE **112**, 045310 (2025). It runs collective-coordinate optimization from Poisson (POS) and Gaussian-pair-statistics (GAU, hyperuniform) initial conditions to χ = 0.1 stealthy ground states with `xla_cco`'s adaptive gradient descent. Each run is stored at full resolution.

Only scripts are tracked. All generated data lives in git-ignored directories:

| Directory | Contents |
|---|---|
| `initial_conditions/` | `POS-2D-N1024` (100 Poisson configs, seed 20251106); `GAU-2D-N1000` (50 configs copied from `CONFIGURATIONS/N1000_configs_GaussianPairStat`, basis rescaled to unit volume); `URL-2D-N1024` (100 uniformly randomized triangular lattices copied from `CONFIGURATIONS/TRIANGULAR_URL`, rectangular cell); `RSA-2D-N999` (20 saturated RSA packings, `saturated_rsa.py`); `provenance.json` |
| `trajectories/chi<χ>/<ENS>/` | `traj_XXXX.{positions.f64,npz,json}` for each trajectory, plus `manifest.json`; `trajectories/chi<χ>/summary.npz` (ensemble statistics); SLURM logs |
| `figures/` | `chi<χ>_pos_gau.png` (S(k) of initial conditions; distance and energy vs t/T, log-log) and `chi<χ>_ground_state_sk.png` (S(k) of the converged ground states) |

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

The figures are `figures/chi<χ>_w2.png` (the Wasserstein mirror of `chi<χ>_pos_gau`) and `figures/chi<χ>_w2_vs_p0.png` (exact W against labelled p on the same frames). The docstring of `plot_wasserstein.py` lists the panels.

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
3. **Hessians** (`batten2008_hessian.py`). Eigenvalues of the exact Hessian at 93 energies on every flow (every quarter decade of Φ from 10³ to 10⁻²⁰), the curvature along each density wave, the wavelength content of the negative-curvature eigenvectors, and cross-sections of Φ through r(t) spanned by the flow direction τ = −∇Φ/|∇Φ| and the stiffest eigenvector.
   - *The clock is Φ, not t/T.* Φ decreases monotonically along a gradient flow, so every flow is compared at equal energy. t/T is a poor clock here: once the remaining motion is below `atol`, the L-stable steps grow without bound. The last three steps make up 40–55% of T in 1D and 87% in 2D, and below Φ ≈ 10⁻¹³ each stored flow has only 4–29 frames.
   - *The tail is re-integrated* (`batten2008_tail.py`) to resolve the endgame. Each flow restarts from its last stored frame with Φ ≥ 10⁻⁹ and runs with the same method, tolerances and stops, but with the step capped at τ₁/8, where τ₁ = 1/(2k₁²) is the relaxation time of S(k₁). This takes 360 steps per flow in 1D (30 s for 20 flows) and 210 in 2D (75 s). Every level then falls within 0.05 decades of a frame.

**Results.**

- **Short wavelengths relax first; the energy ends up in the longest ones** (`batten2008_N1000_modes`). The last time S(kₙ) exceeds 10⁻² scales as n⁻²: over n = 1–300 at χ = 1/3, and up to n ≈ 30 at χ = 2/3. By t/T ≈ 10⁻³ the lowest 10% of modes carry essentially all of Φ.
- **The longest wavelength obeys the linearized flow exactly.** Linearizing gives dρ(k)/dt = −k²ρ(k). Mode n = 1 decays as S(k₁, t) = S(k₁, 0) exp(−2k₁²t) from the first step: the fitted rate is 0.499 × 4k₁² at both χ, and its phase changes by at most 0.02 rad over a whole flow (`batten2008_N1000_mode_planes`: straight radial lines). Its time constant 1/(2k₁²) = L²/(8π²) ≈ 1.3 × 10⁴ sets the length of the final approach. Because it ignores the other modes, S(k₁, t) from a given start is the same at χ = 1/3 and 2/3 (the curves overlap in panel (d)). Higher modes decay more slowly than their own linear rate 2kₙ², and their phases wind before collapsing: they are driven by the slow modes.
- **χ = 2/3 orders in one event.** Every mode above n ≈ 30 falls below 10⁻⁸ together at t/T ≈ 2.5 × 10⁻⁴; the lower modes then follow the n⁻² law. At χ = 1/3 the modes relax one by one.
- **Ground-state manifold dimension** (`batten2008_N1000_hessian`). At the end of every χ = 1/3 flow the Hessian has exactly 334 = N − 2M zero eigenvalues: the disordered ground states form a locally 334-dimensional manifold, one dimension per unconstrained degree of freedom. The near-zero count grows along the flow, at a rate set by the cut: all 334 have |λ| < 10⁻⁶ λ_max by Φ ≈ 10⁻²·⁶, and |λ| < 10⁻⁹ λ_max by Φ ≈ 10⁻⁸·⁶. At χ = 2/3 only the translation mode is zero: the lattice is an isolated minimum.
- **Curvature along the path.** At the Poisson start about 55% of the eigenvalues are negative. At χ = 2/3 they are gone by Φ ≈ 10 (a median of Φ = 10⁰·⁷⁵), after which the flow is inside the lattice's convex basin. At χ = 1/3 about 165 weakly negative directions persist from Φ ≈ 10 to the manifold (see *Negative curvature* below). The stiffest eigenvalue settles at 6.3 (χ = 1/3) and 22 (χ = 2/3).
- **Phase planes for every scale** (`batten2008_N1000_mode_planes_full`, `batten2008_N1000_mode_phases`). A radial move in these plots changes that mode's energy; an angular move slides its density wave at fixed energy. Results:
  - *Winding.* Mode 1 turns by 0.001 turns, i.e. not at all. Mode 2 turns 0.4 (χ = 1/3) or 0.3 (χ = 2/3). Modes n ≥ 3 wind 1–10 turns at χ = 1/3 but stay under 1 turn up to n ≈ 30 at χ = 2/3.
  - *Lock-in.* For n ≥ 3 the phase keeps moving until S(kₙ) is down to 10⁻¹¹–10⁻¹⁵, so these modes spiral almost all the way in. Mode 2 locks at about 10⁻¹² (χ = 1/3) or 10⁻⁸ (χ = 2/3). Frame spacing resolves the phases: at most 0.2% of steps exceed π/2.
  - *Late-time slaving, S_n ∝ S₁^{p_n}.* At χ = 1/3, p_n = 1.00 for n = 2–10: the final approach is linear relaxation along one Hessian direction, which carries every density wave in fixed proportion. At χ = 2/3, p₂ = 2.00, p₃ = 3.01, p₄ = 4.2, i.e. p_n = n: near the lattice the last phonon is long-wavelength, and density wave n is its n-th harmonic, with amplitude ∝ Jₙ(kₙa) ∝ aⁿ. Independent decay would give n².
- **Harmonic stiffness near the ground states** (`batten2008_N1000_hessian_detail`). At the end of every flow the positive eigenvalues come in pairs at k_p² (cosine and sine density waves): the ratios are 0.999 at χ = 1/3 and exact at χ = 2/3. At χ = 2/3 even λ_max = 21.94 = k₃₃₃² + k₆₆₇², the two density waves that alias onto the same lattice phonon.
  - *χ = 2/3 is harmonic early.* Along every density wave the curvature equals k_n² to 1% once Φ is below about 5. For n ≥ 334 it equals the aliased sums (k_n² + k_{N−n}²).
  - *χ = 1/3 is not.* Along pure density waves the end state is stiffer than k_n²: 47× for n = 1, 24× for n = 2, 7.5× for n = 10, 1.2× for n = M. Coupling through ρ(k > K) mixes each long wave with short ones, so the normal modes of a disordered ground state are mixtures, while their eigenvalues stay at k_p².
  - *Negative curvature.* At χ = 2/3 the last negative directions are long-wavelength (72–83% of their power below 0.1K), and they vanish by Φ ≈ 10. At χ = 1/3 about 165 persist, which is half of the 334 directions that become zero modes. By Φ = 1, more than 99% of their motion lies at the unconstrained wavelengths q > K.
  - *They never switch off; they fade as √Φ.* A fit of log |λ_min| against log Φ over 10⁻¹² < Φ < 10⁻³ gives slope 0.499 (10–90%: 0.495–0.508), with |λ_min| ≈ 1.2 × 10⁻⁴ √Φ. Where the count reaches zero is therefore set by the zero cut. With |λ| > 10⁻⁹ λ_max it ends at Φ ≈ 10⁻⁸·⁶, exactly where the 10⁻⁹ zero count completes. With 10⁻¹³ λ_max, the same 165 last to Φ ≈ 10⁻¹⁴ and fade by 10⁻¹⁶.
  - *Why √Φ.* The Hessian of Φ = Σ_k |ρ_k|²/N is (2/N) Σ_k [Re(∇ρ_k ∇ρ_k*) + Re(ρ_k* ∇∇ρ_k)]. Directions tangent to the ground-state manifold see only the second term, which is linear in the residual amplitudes |ρ_k| ∝ √Φ and indefinite. So close to the manifold its soft directions curve up and down at strength √Φ, about half of them negative. The flow approaches the manifold through a nearly flat, saddle-shaped region whose negative curvature vanishes only on the manifold itself.
- **The valley is narrow and the path bends** (`batten2008_N1000_sections`, at Φ = 10, 0.1 and 10⁻⁸). At Φ = 0.1, Φ grows 100-fold within 1.8 (χ = 1/3) or 0.93 (χ = 2/3) along the stiffest direction, while the flow travels 49 for a tenfold drop. At Φ = 10⁻⁸ these are 5.6 × 10⁻⁴ and 3.0 × 10⁻⁴ against 0.015. At χ = 1/3 the projected path leaves the τ axis within a few units (panel b), so the descent direction turns continually.
- **Valley floors** (`batten2008_N1000_floor`; flow 0, 7 × 7 grid at χ = 1/3, 9 × 9 at χ = 2/3, plus 20 sampled frames):
  - *χ = 1/3: the floor is zero everywhere.* Every grid point on both planes, and every sampled frame including the Poisson start, relaxes to a ground state (Φ ≤ 5 × 10⁻¹⁹) without leaving its fiber. As dimension counting predicts, the 334-dimensional ground-state manifold meets every (N − 2)-dimensional fiber, so the flat slices' barriers disappear once the other 998 directions can move. Getting there takes a median 1,200–1,300 units of motion, 4.5× the flow's net displacement.
  - *χ = 2/3: a broad funnel to the lattice.* The floor stays at Φ ≈ 0.003–20 (median 4), 15–50× below the flat slice, and falls toward r(∞), unlike the flat slices' narrow groove. Beneath the path the floor is 12–16 at the start, where the true Φ is 629. From t/T ≈ 10⁻² on it equals Φ(r(t)) to 0.1%, so from then on the flow runs along its valley floor.
  - *Caveats.* At χ = 2/3, L-BFGS (up to 3 × 10⁵ evaluations per point) ended on stagnation, line-search failure or the budget. Those floors are therefore upper bounds, local minima on each fiber. The in-plane drift has median 10⁻⁹ but reaches 2.3 at χ = 2/3, which is 12% of the 18.7 grid spacing. The cause is not yet confirmed; a likely one is that L-BFGS wraps positions into the cell between iterations. Dense grids are not affordable at N = 1000 without a preconditioner for the diffusive (λ ∝ k²) stiffness: one plane takes 14–28 min on an A100.

```bash
sbatch --export=ALL,STEP=modes experiments/slurm/batten2008_cc_gpu.slurm
sbatch --export=ALL,STEP=tail experiments/slurm/batten2008_cc_gpu.slurm      # before hessian
sbatch --export=ALL,STEP=hessian,STEP_ARGS="--chi 1/3" experiments/slurm/batten2008_cc_gpu.slurm  # and 2/3
# valley floors: one job per chi and plane (each under an hour)
sbatch --export=ALL,STEP=floor,STEP_ARGS="--chi 1/3 --views se --grid 7 --path-points 20 --max-evals 100000 --batch 72" \
    experiments/slurm/batten2008_cc_gpu.slurm
sbatch --export=ALL,STEP=floor,STEP_ARGS="--chi 2/3 --views se --grid 9 --path-points 20 --max-evals 100000 --batch 104" \
    experiments/slurm/batten2008_cc_gpu.slurm      # and --views pca for both
uv run --no-sync python -I experiments/scripts/batten2008_cc_plot.py
```

**Data.** `trajectories/batten2008/N1000/{modes,tail,hessian}_chi<a-b>.npz` and `floor_chi<a-b>_{se,pca}.npz`.

**Figures.** `figures/batten2008_N1000_{modes,mode_planes,mode_planes_full,mode_phases,hessian,hessian_detail,sections,floor}`.

### The same analyses in 2D: Poisson starts to stealthy ground states

`flows2d.py` integrates the true gradient flow dr/dt = −∇Φ (ROS2 with the exact Hessian, as for the 1D flows) from the first 100 Poisson starts of `initial_conditions/POS-2D-N1024` (N = 1024, density 1, L = 32), at χ = 0.1 (M = 210) and χ = 0.4 (M = 820). The energies are the same as in `run_trajectories.py`. All 200 flows reach Φ < 10⁻²⁰: 1,986–2,024 steps at χ = 0.1 and 2,605–3,027 at χ = 0.4 (1.0 and 1.6 h for 100 flows on one GPU). The figures and the counts below use all 100 flows; the mode decay rates, phase statistics and harmonic-stiffness ratios were measured on the first 20. Every analysis script takes `--root experiments/trajectories/flows2d`. The energy is rebuilt from each run's stored basis and K, the Wasserstein matching runs on the torus (Hungarian assignment per frame), and the two-particle slice moves each particle along its own net displacement.

**Results, compared with 1D.**

- **The 2D flows are local.** Particles move 0.69 (χ = 0.1) and 0.80 (χ = 0.4) mean spacings RMS (median). In 1D they moved up to 8.7. At χ = 0.1 the optimal matching to r(0) is the identity at every frame, so W₀ = p₀ to 7 × 10⁻¹⁵. At χ = 0.4 some frames relabel particles (at most 56 of 1024) and |W₀ − p₀| ≤ 0.18 out of 22–28 (`flows2d_N1024_w2`). The W₂ geodesic between the endpoints climbs to a median Φ of 211 (χ = 0.1) and 814 (χ = 0.4) (`flows2d_w2_geodesic_energy`).
- **Mode relaxation follows the 1D picture** (`flows2d_N1024_modes`). The last time S(k) exceeds 10⁻² scales as |k|⁻² across all shells at both χ. The longest wavelength decays at 0.978 × 2k₁² (χ = 0.1) and 0.85 × 2k₁² (χ = 0.4), close to the linearized rate but less exact than in 1D (0.998), because in 2D the |k| = k₁ shell holds two independent waves.
- **Phases** (`flows2d_N1024_mode_planes_full`, `flows2d_N1024_mode_phases`). The two k₁ waves, (0, 1) and (1, 0), turn by 0.006 (χ = 0.1) and 0.03–0.04 (χ = 0.4) turns, i.e. straight radial lines. Higher modes wind at most about 1 (χ = 0.1) or 2 (χ = 0.4) turns, far less than the 1–10 turns at 1D χ = 1/3. Phases lock at S ≈ 10⁻⁸ to 10⁻¹⁰. Late-time slaving gives p ≈ 1 for every sampled mode at both χ, as at 1D χ = 1/3: the final approach is linear relaxation along one Hessian direction. No harmonic (p = n) regime exists, because neither χ is crystalline.
- **Ground-state manifold dimension** (`flows2d_N1024_hessian`). Every end state has exactly Nd − 2M zero eigenvalues: 1628 at χ = 0.1 and 408 at χ = 0.4, so both are disordered, locally (Nd − 2M)-dimensional manifolds, like 1D χ = 1/3. The zero count completes at Φ ≈ 10⁻⁶·⁵ (|λ| < 10⁻⁶ λ_max) or 10⁻¹²·⁵ (10⁻⁹ λ_max), later than in 1D because the soft eigenvalues are larger at a given Φ (below). The stiffest eigenvalue settles at 6.4 (χ = 0.1) and 33 (χ = 0.4).
- **Negative curvature: half the tangent directions, fading as √Φ** (`flows2d_N1024_hessian`, `flows2d_N1024_hessian_detail`). About 1,160 (χ = 0.1) and 1,130 (χ = 0.4) of the 2048 eigenvalues are negative at the Poisson start. The count falls to a plateau by Φ ≈ 1 and stays there to Φ ≈ 10⁻⁸: 812 at χ = 0.1 and 201 at χ = 0.4 (median), half of Nd − 2M (1628 and 408), as at 1D χ = 1/3 (165 of 334). |λ_min| scales as Φ^0.500, about 0.0076 √Φ (χ = 0.1) and 0.063 √Φ (χ = 0.4). The plateau ends at Φ ≈ 10⁻¹²·⁵ only because |λ_min| crosses the 10⁻⁹ λ_max cut; with a 10⁻¹³ cut it lasts to Φ ≈ 10⁻¹⁸. (The earlier t/T sampling showed a gradual fall, 820 → 83 → 0, because its samples happened to straddle this cut-off.) Unlike 1D χ = 1/3 (>99% of the negative directions' motion at q > K), the 2D negative directions favor the unconstrained wavelengths only 2–3× (χ = 0.1) or 1.3× (χ = 0.4) per eigenvector.
- **Harmonic stiffness.** End-state eigenvalues pair at |k_p|² less exactly than in 1D. λ/|k|² is 0.97–0.98 for the first pairs at χ = 0.1 (median 0.94), and 0.84–0.87 for the first pairs at χ = 0.4 (median 0.64). Along pure density waves the end state is stiffer than |k|²: 1.6–2.6× for the longest waves at χ = 0.1 and 9–15× at χ = 0.4.
- **Flat slices misreport the energy as in 1D** (`flows2d_N1024_mismatch`). The median log₁₀(Φ_plane/Φ) is +0.31 (χ = 0.1) and +0.75 (χ = 0.4) on the start-end plane, +2.7 on the PCA plane over the second half of the flow (reaching 19–20 at the end), and −1.6 and −0.7 for the two-particle slice.
- **The valley floor is zero everywhere** (`flows2d_N1024_floor`; flow 0, 21 × 21 grid on each plane plus 40 sampled frames). All 441 grid points per plane and every sampled frame relax to a ground state (Φ ≤ 10⁻²⁰, all with the energy-tolerance status) without leaving their fiber, at both χ. The in-plane drift is at most 2 × 10⁻¹¹. Reaching the floor takes a median 44–63 units of motion on the grid and 58–77 from r(t). On the grid that is 1.7–2.8× flow 0's net displacement, against 4.5× in 1D. In 2D, unlike 1D χ = 2/3, dense floors are cheap: a 21 × 21 plane takes 34–38 s at χ = 0.1 and about 250 s at χ = 0.4 on an A100.

```bash
sbatch --export=ALL,CHI=0.1 experiments/slurm/flows2d_gpu.slurm   # resubmit to resume
sbatch --export=ALL,CHI=0.4 experiments/slurm/flows2d_gpu.slurm
R=experiments/trajectories/flows2d
sbatch --export=ALL,STEP=slices,STEP_ARGS="--dir $R/N1024 --chi 0.1 0.4" experiments/slurm/batten2008_cc_gpu.slurm
for s in mismatch modes tail; do
  sbatch --export=ALL,STEP=$s,STEP_ARGS="--root $R --sizes 1024 --chi 0.1 0.4" experiments/slurm/batten2008_cc_gpu.slurm
done
for c in 0.1 0.4; do   # after tail; about 10 min each
  sbatch --export=ALL,STEP=hessian,STEP_ARGS="--root $R --sizes 1024 --chi $c" experiments/slurm/batten2008_cc_gpu.slurm
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

### The two-hump off-plane shape, per particle and per flow

Each flow's distance from its start-end plane (`*_w2_N*_offplane`) rises and falls twice, in 1D and in 2D. `batten2008_offplane.py` (NumPy, a few seconds per χ on a CPU) resolves that curve. Each flow is resampled at 400 points uniform in arclength, s = ℓ/ℓ(∞). The off-plane residual W(s) is split into its signed principal direction e₃ and into each particle's share W_j. A plane-free version removes the sin(πs) bow from every coordinate instead of projecting on a plane. The script also computes the Karhunen–Loève (principal-component) modes of the path off the chord.

**Results.**

- **The plane pins three zeros; the humps are the path's second sine mode.** The plane contains r(0), r(∞) and the frame farthest from the chord (s* ≈ 0.5), so W vanishes there, and so does every W_j. The path off the chord is a bow sin(πs) plus an S-bend sin(2πs): the KL eigenfunctions overlap sin(kπs) at 0.99/0.97/0.80 (1D χ = 1/3) and 1.00/1.00/0.99 (2D), and the first two hold 94–98% of the variance. The plane absorbs the bow, and |sin 2πs| leaves humps at s ≈ 1/4 and 3/4 (`*_offplane_modes`).
- **The two humps are one signed S.** Along e₃ every flow (20/20 at each χ) has a positive lobe and then a negative one, crossing the plane at s*. The magnitude of that S is the two-hump curve. e₃ carries 74–79% (2D χ = 0.1) and 61–63% (χ = 0.4) of |W|² at the humps. In 1D it carries 87% at the first hump and 56–59% at the second, which turns through other directions (`*_offplane_3d`, panels a and b).
- **The shape is not a construction artefact.** With the bow removed and no plane, the residual still has two humps, with a dip to 0.37–0.41 (1D) and 0.41–0.53 (2D) of the hump height near s = 1/2, rather than a pinned zero.
- **Every particle takes part, and the average of individual particles gives the same shape** (`*_offplane_particles`). Particle shares, each scaled to its own maximum, average to a profile that correlates with the configuration's off-plane curve at 0.989 (1D) and 0.998–0.999 (2D). Their root-mean-square is the curve exactly. The participation ratio at the first hump, (Σ|W_j|²)²/(N Σ|W_j|⁴), is 0.32–0.33 in 1D and 0.48–0.49 in 2D. Those are the values for Gaussian-distributed displacements with one or two components (1/3 and 1/2), so the bend is a collective, Gaussian displacement field, not a few particles moving far. Averaged over the 20 flows, the curve has the same shape as each flow's.
- **1D is lopsided, 2D symmetric.** The mean humps are 0.145/0.084 in 1D (ratio 1.7) and 0.125/0.131 (χ = 0.1) and 0.189/0.183 (χ = 0.4) in 2D. In 1D, 96% of particles peak in the first half, against 62–65% in 2D, and the 1D KL eigenfunctions k = 2, 3 are compressed toward early s. The 1D path bends early and then runs straight, because the k₁ wave, which relaxes last along a fixed phase, carries most of the net displacement (61% in the linear estimate 2/k² per mode). In 2D the displacement is spread over the |k| shells, so the direction turns at a steady rate.
- **The path is smooth.** The KL variances fall faster than a random-walk bridge's 1/k² (about 1/k⁴ for k = 2–5 in 1D and 2D χ = 0.1, between 1/k² and 1/k⁴ at χ = 0.4), and the tangent keeps 0.30–0.58 of its direction over a quarter of the path.

```bash
uv run --no-sync python -I experiments/scripts/batten2008_offplane.py
uv run --no-sync python -I experiments/scripts/batten2008_offplane.py --root experiments/trajectories/flows2d --sizes 1024 --chi 0.1 0.4
uv run --no-sync python -I experiments/scripts/batten2008_offplane_plot.py
uv run --no-sync python -I experiments/scripts/batten2008_offplane_plot.py --root experiments/trajectories/flows2d --n 1024 --chi 0.1 0.4 --prefix flows2d --dim 2
```

**Data.** `trajectories/{batten2008/N1000,flows2d/N1024}/offplane_chi<a-b>.npz`.

**Figures.** `figures/{batten2008_N1000,flows2d_N1024,gau2d_N1000,url2d_N1024,rsa2d_N999}_offplane_{particles,modes,3d}`.

### Why the humps are lopsided: linear relaxation of density waves

`figures/offplane_lopsided` compares the ten 2D ensembles: Poisson, Gaussian, URL, saturated RSA and HIP, each at χ = 0.1 (solid) and 0.4 (dashed), with 100 flows per start (50 for Gaussian).

**The linear relaxation model.** `batten2008_offplane.py` also stores a model with no fitted parameter.
- In the linearized flow, density wave n relaxes as ρ(kₙ) ∝ exp(−|kₙ|²t).
- Cancelling the wave from the start takes squared displacement 2S(kₙ, 0)/|kₙ|², the least motion that zeroes ρ(kₙ).
- The model path is y(t) = Σₙ √(2S(kₙ,0))/|kₙ| (1 − e^{−|kₙ|²t}) êₙ, along orthogonal directions êₙ, and it is analysed exactly like the flows.

The 1D flows (`batten2008_N1000_offplane_*`) follow the same picture: hump ratio 1.74–1.75, against 1.75–1.76 from the model.

**(a) Measured off-plane curve.** This is the mean over flows of |W(s)|/|Δr|, scaled to its maximum.
- The humps sit at s ≈ 0.21–0.33 and 0.74–0.84, around a pinned zero at s* ≈ 0.47–0.58.
- The hump ratio, first over second, orders the starts:

  | Start | χ = 0.1 | χ = 0.4 |
  |---|---|---|
  | HIP | 1.20 | 1.16 |
  | Poisson | 0.98 | 1.04 |
  | RSA | 0.80 | 0.97 |
  | Gaussian | 0.52 | 0.76 |
  | URL | 0.51 | 0.65 |

- For the hyperuniform starts the humps and the zero shift late (zero at s ≈ 0.57, second hump at 0.83).

**(b) The model, no fit.** It reproduces every χ = 0.1 shape, including the late shift.

  | Start | Measured / model hump ratio, χ = 0.1 |
  |---|---|
  | Poisson | 0.98 / 0.84 |
  | Gaussian | 0.52 / 0.48 |
  | URL | 0.51 / 0.47 |
  | RSA | 0.80 / 0.79 |
  | HIP | 1.20 / 1.12 |

  At χ = 0.4 the hyperuniform flows are less lopsided than the model: 0.76 and 0.65 against 0.48. HIP is less front-heavy than the model: 1.16 against 1.28.

**(c) Turning done by s. This panel is the mechanism.**

*What Θ(s) measures.* Θ(s)/Θ(1) is the fraction of the total angle through which the path's unit tangent has rotated by arclength s. The tangent of a gradient flow is the force direction, −∇Φ/|∇Φ|. So Θ(s) records where along the path the force changes direction. The diagonal is uniform turning, with constant curvature per unit length.

*Why the force turns.* In the linear picture the velocity is Σₙ |kₙ|² aₙ e^{−|kₙ|²t} êₙ. Its direction changes only when the dominant share of the motion still to come passes from one relaxation rate to a slower one.
- Waves at nearly the same |k| relax together and push in a fixed combined direction, so the path runs straight.
- Waves at very different |k| hand the motion over in sequence, so the path turns.
- How much path length each handover costs is set by the displacement weight 2S₀(k)/k² at those scales.

*The three regimes.*
- **Front-loaded (HIP; IR-heavy).** Long waves carry most of the displacement, so most of the length is spent at the end on slow, nearly single-rate k₁ motion, and the handovers happen early on little length. At χ = 0.1, 35% of the turning is done by s = 1/4 and 63% by s = 1/2 (model 28% and 59%).
- **Uniform (Poisson, RSA; flat in log k).** In 2D the number of waves per shell grows as k. A flat S₀ therefore gives a weight per shell ∝ k·(1/k²) = 1/k, the same per octave. Each octave of handover costs equal length and equal angle, so Θ(s) is the diagonal: 45% by s = 1/2 for Poisson at both χ (model 42–44%), and 40% for RSA χ = 0.1 (model 39%).
- **Back-loaded (Gaussian, URL; UV-heavy).** S₀ ∝ k² gives weight ∝ k per shell, concentrated near K. The bulk of the length is spent early on short waves that relax at nearly one rate, so the path is straight. The long-wave handovers, at widely different rates, crowd into the last stretch.

  | Start, χ = 0.1 | Θ(1/4) | Θ(1/2) | Θ(0.9) | Model Θ(1/2) | Model Θ(0.9) |
  |---|---|---|---|---|---|
  | Gaussian | 0.05 | 0.14 | 0.63 | 0.12 | 0.63 |
  | URL | 0.04 | 0.12 | 0.57 | 0.10 | 0.57 |

  More than 35% of all turning happens in the last 10% of the path.

*Why this sets the humps.* The off-plane residual is largest where the path has bent most relative to the plane through r(0), r(s*) and r(∞). Turning concentrated in the first half makes the first hump larger; turning concentrated in the second half makes the second hump larger. Across the ten ensembles the measured hump ratio rises with Θ(1/2): from 0.51 (Θ = 0.12) to 1.20 (Θ = 0.63).

*χ = 0.4 departs from the model.* Every start's measured Θ(s) is closer to the diagonal than the model's: |Θ(1/2) − 1/2| is 0.22 against 0.39 (Gaussian), 0.24 against 0.40 (URL), 0.08 against 0.12 (RSA), and 0.05 against 0.13 (HIP). The coupled flow spreads its turning more evenly than separable relaxation would, for back-loaded and front-loaded starts alike. The KL analysis below traces this to slaving of the short waves.

**(d) Share of the remaining motion on the k₁ shell.**
- Poisson and RSA χ = 0.1 rise from 0.18–0.19 at the start to 0.44 at s = 1/2 and 0.92–0.95 at s = 0.9: the end of the path is pure k₁ relaxation.
- HIP starts at 0.36–0.38 and reaches 0.79–0.85.
- Gaussian and URL stay ≤ 0.05 until s = 1/2 and reach only 0.13–0.55 by s = 0.9 (0.03 for URL χ = 0.4). Even at the end their remaining motion is spread over several long waves, because k₁ carries 0.1–1.5% of |Δr|².

**(e) Net displacement on the long waves.** This is the cumulative share of |Δr|² on the density waves up to |k|, measured (solid), with the linear prediction 2S₀(k)/k² (dotted).
- The k₁ share agrees: measured / predicted is 0.187 / 0.194 (Poisson), 0.015 / 0.016 (Gaussian), 0.003 / 0.003 (URL), 0.180 / 0.180 (RSA) and 0.376 / 0.399 (HIP) at χ = 0.1. At χ = 0.4, RSA moves less on k₁ than predicted (0.079 against 0.122).
- The curves pass 1 for HIP because the wave directions at r(0) are not mutually orthogonal: projections onto them overlap. The panel stops at 12 k₁ because larger moves scramble the phases defined at r(0).

**(f) Hump ratio per flow, flow against model.** The Spearman correlation is 0.91 over all 900 flows. Within each ensemble it is:

  | Start | χ = 0.1 | χ = 0.4 |
  |---|---|---|
  | RSA | 0.99 | 0.61 |
  | Poisson | 0.97 | 0.79 |
  | URL | 0.93 | 0.00 |
  | Gaussian | 0.90 | −0.18 |
  | HIP | 0.72 | 0.49 |

  The model predicts each flow's own lopsidedness from its own S₀(k), except for the hyperuniform starts at χ = 0.4, where it fails.

**(g) Hump ratio against the measured k₁ share of |Δr|².** The Spearman correlation is 0.80 over all flows. Ratios above 1 occur only when k₁ carries ≳ 10% of the displacement.

**(h) Linear theory.** This is Σ S₀(k)/k² up to k, for d = 1, 2, 3, with S₀ = 1 (solid) and S₀ ∝ k² (dotted).
- d = 1 is dominated by the first shell (61% at k₁).
- d = 2 with S₀ = 1 is linear in log k, the marginal case.
- d = 3, or any S₀ ∝ k², is dominated by the shell at K.

A 2D hyperuniform start therefore behaves like a Poisson start in a higher dimension, and HIP (S₀ diverging at small k) like the 1D Poisson start.

### KL eigenfunctions: where they come from, and why they leave the linear model

**What the eigenfunctions are.** `batten2008_kl_plot.py` writes `figures/offplane_kl_compare` (every 2D start, both χ) and `figures/offplane_kl_origin`.
- φ_k(s) are the Karhunen–Loève (principal-component) eigenfunctions of the path off its chord: the eigenvectors over s of the Gram matrix P(s)·P(s′).
- They are the temporal shapes of the path's independent bends: φ₁ is the bow, φ₂ the S-bend, and so on. The variance λ_k measures how much bending each carries.
- For a path whose direction decorrelates at a constant rate in arclength, the kernel depends only on s − s′ (with pinned ends), and the eigenfunctions are sin(kπs). Sines are the signature of uniform turning (panel c above). Where the path turns faster, the eigenfunctions oscillate faster.

**Measured against model (`offplane_kl_compare` (a–j); (o) summarizes).** The median overlap |⟨φ_k, φ_k^model⟩| over flows, for k = 1–6:

  | Start | χ = 0.1 | χ = 0.4 |
  |---|---|---|
  | Poisson | 1.00 1.00 0.99 0.96 0.90 0.80 | 1.00 1.00 0.98 0.92 0.81 0.62 |
  | Gaussian | 1.00 1.00 0.99 0.97 0.94 0.86 | 0.99 0.90 0.63 0.23 0.12 0.25 |
  | URL | 1.00 1.00 0.99 0.97 0.94 0.85 | 0.99 0.92 0.67 0.28 0.09 0.26 |
  | RSA | 1.00 1.00 1.00 1.00 1.00 0.99 | 1.00 0.99 0.96 0.92 0.85 0.75 |
  | HIP | 1.00 1.00 0.99 0.97 0.91 0.83 | 1.00 1.00 1.00 0.99 0.97 0.95 |

- *χ = 0.1.* The model predicts the eigenfunctions of every start, including the late-compressed shapes of the hyperuniform starts.
- *χ = 0.4.* It fails for Gaussian and URL from k = 3 on, degrades for Poisson and RSA at k ≥ 5, and still works for HIP.

**Nodes (k, l).** The interior zeros of φ_k are plotted against the sine nodes j/k.
- *χ = 0.1.* Gaussian and URL nodes sit 0.05–0.2 late, and the model (open markers) puts them in the same place. HIP nodes sit early, again as in the model.
- *χ = 0.4.* The model still compresses the hyperuniform eigenfunctions toward the end (first node at 0.35 for j/k = 1/6), but the measured nodes lie within about 0.05 of j/k. The flow's eigenfunctions are nearly sines where the model's are strongly distorted.

**Spectra (m, n).** The measured spectra fall more slowly than the model's.
- *Above the S-bend.* At χ = 0.4 the variance beyond the S-bend (k ≥ 3) is 2–3× the model's: 0.055 against 0.024 (Poisson), 0.043 against 0.014 (Gaussian), 0.035 against 0.013 (URL), 0.044 against 0.023 (RSA) and 0.042 against 0.021 (HIP).
- *At χ = 0.1.* The ratio is 1.0–1.4, except HIP at 2.1.
- *Fine wiggles.* The measured paths carry small-scale bends that separable relaxation cannot make. At k ≈ 10 the measured fraction is 10–10³ times the model's, whose high modes are nearly empty.

**Origin of the difference (`offplane_kl_origin`).**
- **(a) The flow's clock lags the model's.** The time to cover half the path, flow over model, is:

  | Start | χ = 0.1 | χ = 0.4 |
  |---|---|---|
  | Poisson | 1.04 | 1.36 |
  | Gaussian | 1.05 | 2.81 |
  | URL | 1.05 | 2.31 |
  | RSA | 1.00 | 1.91 |
  | HIP | 1.08 | 1.33 |

  The model spends the hyperuniform displacement within t ≈ 0.01 τ₁, at the short waves' own rates. The flows take 2–3× longer.
- **(b, c) The path is longer than the model's.** Path length, flow over model, is ≤ 1.03 at χ = 0.1 and, at χ = 0.4:

  | Start | Path length / model | Net displacement / model |
  |---|---|---|
  | HIP | 1.06 | 0.99 |
  | Poisson | 1.20 | 1.06 |
  | RSA | 1.30 | 1.25 |
  | URL | 1.34 | 1.11 |
  | Gaussian | 1.48 | 1.20 |

  The net displacement column is compared with the least motion that cancels every wave. Per flow, the overlap deficit 1 − ⟨overlap⟩_{k=3..6} has Spearman 0.76 with the clock lag and 0.78 with the length excess, over all 900 flows.
- **(d) Slaved short waves.** Each system's deficit ranks with the share of the model's displacement on short waves (|k| > 0.6K) times their measured slaving at Φ = 10, 1 − Γ_k/|k|², where Γ_k = −d ln|ρ_k|/dt (Spearman 0.92 over the ten systems).
  - Gaussian and URL at χ = 0.4 score 0.51–0.53 (deficit 0.67–0.69).
  - RSA χ = 0.4 scores 0.24 and Poisson 0.14 (deficits 0.13–0.17).
  - Every other ensemble scores ≤ 0.06 (deficit ≤ 0.09).

**Mechanism.**
- *The model's assumption.* The model spends each wave's displacement at that wave's own rate |k|².
- *What the flows do instead.* At χ = 0.4 the short waves decay at only 0–10% of that rate (the slaving table in "The triad identity in detail"): their decay is held up by triad coupling to the slower waves.
- *Consequences.* The motion the model spends early, fast and in a nearly fixed direction is spread over the slow waves' time scale. So the clock lags and the turning is spread along the path (panel c above), and the eigenfunctions relax toward sines. The coupling also adds sideways sliding (winding), so the path is 20–50% longer and its spectrum carries extra small-scale bends.
- *When the model holds.* It works where little displacement sits on slaved short waves: every start at χ = 0.1, and HIP at both χ (95% of its model displacement is on |k| < K/2).

**What it says about the landscape.**
- **χ = 0.1: a separable funnel.** Around the path, Φ behaves as a sum of independent quadratic wells in the collective coordinates ρ_k, with stiffness |k|². The path's geometry, including its lopsidedness, its KL eigenfunctions and its length, follows from the starting spectrum S₀(k) alone, with no fitted parameter.
- **χ = 0.4: a curved valley.** The constrained waves are coupled through the free fluctuations (the off-diagonal G_kq of the triad section).
  - The short-wave directions form a valley whose floor moves as the long waves relax, so the flow follows that slow manifold rather than relaxing each wave separately.
  - This costs a longer path and a lagging clock, and it erases the start's imprint on the path shape: the eigenfunctions become generic sines, whatever S₀(k) was.
  - How strongly the start's memory is erased depends on how much of the start's own displacement lives on the slaved short waves. HIP's, on the long waves, is barely affected.
- **Consistent with the Hessian.** The same picture shows in the Hessian: along the flow, curvature is |k|² only for decoupled waves (`*_hessian_detail` (b, f)), and the valley floor is curved (next section).

### The landscape along the 2D flows: independent wells, a moving floor, a slow curved valley

**The exact frame.** Let u = (Re ρ_q, Im ρ_q) over the constrained set, 2M numbers, with gradients J (2M × Nd). Then Φ = |u|²/V and ∇Φ = (2/V) Jᵀu, so along the flow

  du/dt = −K u,   K = (2/V) J Jᵀ   (exact; checked to 10⁻¹⁶).

- *Wells.* K_qq ≈ |q|² is the stiffness of wave q's own well, and the off-diagonal K_pq couples the waves through the free density fluctuations. Independent quadratic wells means K diagonal.
- *The script.* `batten2008_landscape.py` (CPU, 2–3 min per start) evaluates K at every half decade of Φ for flows 0–9 of every 2D start.
- *The figures.* `batten2008_landscape_plot.py` writes three figures, one per claim. Numbers below are medians.

**1. χ = 0.1: independent wells; χ = 0.4: coupled waves (`figures/landscape_wells`).**
- *Every wave starts in its own well at χ = 0.1* (c, e). In an independent well, S(k, t)/S(k, 0) = e^{−2|k|²t}. At χ = 0.1 every |k| band follows this down to S/S₀ ≈ 10⁻²–10⁻³. The time |k|² t at which S has fallen e²-fold is:

  | Start | χ = 0.1 | χ = 0.4 |
  |---|---|---|
  | RSA | 1.02–1.03 | 1.15–1.73 |
  | Gaussian | 1.04–1.07 | 1.43–1.85 |
  | URL | 1.04–1.07 | 1.13–1.41 |
  | Poisson | 1.06–1.12 | 1.29–1.58 |
  | HIP | 0.85–1.04 | 0.73–1.09 |

  For an independent well this time is 1. At χ = 0.4 the waves leave their wells before two e-folds (d). In HIP, coupling to the large long waves speeds the short ones up (< 1).
- *The coupling is set by χ* (f, g). C = D^{-1/2} K D^{-1/2} (D = diag K) is the identity for independent wells. For 2M random wave gradients in Nd dimensions, C follows the Marchenko–Pastur law with ratio c = 2M/(Nd) ≈ 2χ, and its soft edge (1 − √c)² closes at χ = 1/2.

  | χ | c | Marchenko–Pastur range | Measured eigenvalues of C |
  |---|---|---|---|
  | 0.1 | 0.20 | 0.30–2.1 | 0.24–0.80 to 1.2–1.7 (narrower: the flows end more decoupled than random) |
  | 0.4 | 0.80 | 0.011–3.6 | soft edge 0.006–0.011, on the random soft edge for every start |

  Along the χ = 0.4 flows, the waves combine into collective modes 100× softer than their own wells.
- *Soft modes are single waves only at χ = 0.1* (h). The softest collective modes (the 10% lowest-|k| coordinates) lie on the longest waves' own axes at 0.96–1.00 at χ = 0.1, but only 0.36–0.47 at χ = 0.4, where they are mixtures of many waves.

**2. The floor moves while the particles are carried (`figures/landscape_floor`).**
- *The own-well ratio* (a, b). Γ_k/|k|², with Γ_k = −d ln|ρ_k|/dt, is 1 while a wave relaxes in its own well. It is ≪ 1 once the wave sits on a floor set by the other waves (slaved: ρ_k ≈ Σ_{q≠k} K_kq ρ_q / |k|²). The white contour marks S(k) = 10⁻².
- *When each wave becomes slaved* (c). At χ = 0.1 the short waves (|k| > 0.6K) are slaved only after S has fallen 10²–10⁵-fold, once their displacement is spent: S/S₀ = 10⁻⁵–10⁻² at onset (HIP 0.04). At χ = 0.4 they are slaved at S/S₀ = 0.04–0.7, before relaxing.
  - Exception: the long waves of the hyperuniform starts begin with S ∝ k² ≈ 0, so they ride the floor from the start at both χ. They carry ≤ 1.5% of the displacement.
- *The floor moves* (d, e). The soft subspace of K rotates from one half decade to the next. Its overlap per half decade falls to:

  | Start | χ = 0.1 | χ = 0.4 |
  |---|---|---|
  | Poisson | 0.97 | 0.57 |
  | Gaussian, URL, RSA | ≥ 0.99 | 0.76–0.80 |
  | HIP | 0.79 (it moves particles 2.2 spacings) | 0.41 |

  The rotation happens while the particles are carried (Φ ≳ 0.1). Halfway along the path the soft subspace overlaps its final orientation at 0.93–1.00 (χ = 0.1; HIP 0.67), against 0.20–0.50 (χ = 0.4). The floor reaches its final place only as the path ends (rank correlation 1 with s). After that, the long waves finish relaxing along a fixed direction: the straight radial plunge of `*_mode_planes_mean`.
- *The waves wind on the floor* (f). 86–98% of the short waves' winding accrues while they are slaved: the moving floor drags their phases. At χ = 0.1 this happens only after the wave is spent, so it does not shape the path.

**3. The flow follows a slow, curved valley (`figures/landscape_valley`).**
- *The residual decays slowly for its content* (a). u^T K u/u^T D u is the decay rate of the residual u relative to the rate independent wells would give for the same wave content.
  - At χ = 0.1 it is 0.88–1.00 after Φ ≈ 1 (HIP 0.90).
  - At χ = 0.4 it falls to 0.01–0.10 and settles at 0.03–0.19: the residual lives in soft collective modes.
- *Slower than the slowest single well* (b). −½ d ln Φ/dt = u^T K u/|u|² exactly. Late in the flow it equals k₁² at χ = 0.1 (0.96–1.01; URL 1.30). At χ = 0.4 it is 0.86–0.91 k₁² (URL 1.24), slower than the slowest independent well, which only a coupled valley allows. Panel (c) shows the residual's weight on K's eigenmodes (2D Gaussian): at χ = 0.4 it sits below k₁² by Φ = 10⁻⁴, on modes softer than any single wave.
- *The valley floor is curved* (d). The negative tangent curvature |λ_min|/√Φ ∝ (normal stiffness) × (manifold curvature) is 0.003–0.013 at χ = 0.1 and 0.06–0.18 at χ = 0.4, about 10× more curved. In both cases half the floor directions curve down (previous section).
- *Path consequences* (e, f). Following the soft, curved valley makes the flows slower and longer than the separable funnel: clock lag 1.33–2.81 and path length 1.06–1.48 at χ = 0.4, against 1.00–1.08 and ≤ 1.03 at χ = 0.1. Across the ten ensembles the clock lag ranks with the valley softness 1 − u^T K u/u^T D u at Φ = 1 (Spearman 0.92).

**Synthesis.**
- *χ = 0.1.* The landscape near the path is a set of nearly independent quadratic wells, one per constrained wave, stiffness |k|². Waves relax in their own wells, the soft directions are the long waves themselves and stay put, the residual decays at the independent-well rate, and the linear model predicts the path.
- *χ = 0.4.* The 2M constraints fill 80% of the Nd degrees of freedom, and the wave gradients are as non-orthogonal as random vectors (Marchenko–Pastur soft edge 0.011). The waves form collective modes up to 100× softer than their own wells, the short waves are slaved before relaxing, and the soft valley floor rotates while the particles are carried. The flow follows this soft, curved valley, slower (×1.3–2.8) and longer (×1.06–1.48) than the separable funnel, and the start's spectral imprint on the path is erased (`offplane_kl_compare`, `offplane_kl_origin`).
- *Toward χ = 1/2.* The soft edge (1 − √(2χ))² → 0, the coupled valley becomes flat in some directions, and the separable picture fails entirely.

### Negative curvature along the 2D flows: why half the soft directions curve down

**The figures.**
- `figures/hessian_negative_compare` puts panels (c) and (g) of `*_hessian_detail` for every 2D start on shared axes:
  - (a, b) the number of negative eigenvalues against Φ, all 100 flows (|λ| above the 10⁻⁹ λ_max cut);
  - (c, d) the most negative eigenvalue |λ_min|.
- `figures/hessian_plateau_mechanism` explains the plateau. Its data come from `batten2008_plateau.py` (NumPy, CPU): flows 0–9 of every system, at log₁₀Φ = 2, 1, 0, −1, −2, −3, −4, −6, −8 and −10.

**The plateau holds for every start.** After an initial excess, every start settles on half of the Nd − 2M soft directions:

  | Start | χ | Nd − 2M | (Nd − 2M)/2 | Negative at start | Negative at Φ = 10⁻⁴ | λ(PRP) < 0 at Φ = 10⁻⁶ | \|λ_min\|/√Φ |
  |---|---|---|---|---|---|---|---|
  | Poisson | 0.1 · 0.4 | 1628 · 408 | 814 · 204 | 1160 · 1130 | 813 · 201 | 814 · 200 | 0.008 · 0.060 |
  | Gaussian | 0.1 · 0.4 | 1600 · 396 | 800 · 198 | 958 · 942 | 796 · 196 | 795 · 196 | 0.006 · 0.073 |
  | URL | 0.1 · 0.4 | 1632 · 406 | 816 · 203 | 969 · 949 | 814 · 199 | 814 · 200 | 0.005 · 0.180 |
  | RSA | 0.1 · 0.4 | 1598 · 402 | 799 · 201 | 886 · 598 | 799 · 199 | 800 · 199 | 0.003 · 0.063 |
  | HIP | 0.1 · 0.4 | 1600 · 396 | 800 · 198 | 1342 · 989 | 800 · 194 | 800 · 194 | 0.013 · 0.061 |

- *When it is reached.* The count comes within 2% of the plateau by Φ ≈ 10⁻⁰·⁵–10⁻¹·⁷⁵.
- *How large it is.* The plateau is (Nd − 2M)/2 to within 2%. Excluding the d exact translation modes, the maximum is (Nd − 2M − 2)/2.
- *How the negative curvature scales.* |λ_min| ∝ √Φ for every start, with prefactors 0.003–0.013 (χ = 0.1) and 0.06–0.18 (χ = 0.4).
- *Where it ends.* In the cut-based count the plateau ends at Φ ≈ 10⁻¹¹, only because |λ_min| crosses the 10⁻⁹ λ_max cut. The cut-free count of P R P (below) stays at half down to Φ = 10⁻¹⁰, the deepest level computed. In exact arithmetic the plateau lasts until the state reaches the manifold.

**The exact split behind it.** The Hessian of Φ = (1/V) Σ_{q∈C} |ρ_q|² splits exactly into H = G + R.
- *The Gauss–Newton part.* G = (2/V) JᵀJ, where J stacks the gradients of Re ρ_q and Im ρ_q. G is positive semi-definite with rank exactly 2M: it is the stiffness of the constrained waves.
- *The residual-curvature part.* R is block-diagonal, one d × d block per particle:

  R_j = −(2/V) Σ_q w_jq q qᵀ,  with w_jq = Re(ρ_q* e^{iq·r_j}).

  R is the curvature carried by the residual waves, of size |ρ| ∝ √Φ.
- *The tangent directions.* On the Nd − 2M directions that G does not see (the null space P of J, tangent to the ground-state manifold), the soft eigenvalues are those of P R P to first order in √Φ. Panel (c) confirms this flow by flow: at Φ ≤ 10⁻³ the soft spectrum of H equals that of P R P to 2 × 10⁻² of the soft scale. The error falls as √Φ (10⁻⁴ at Φ = 10⁻⁶), and the signs agree for ≥ 99.5% of directions.
- *A known scale.* The P R P spectrum is one fixed shape scaled by √Φ (panel d: the distributions at Φ = 10⁻¹ to 10⁻¹⁰ coincide). That is why |λ_min| ∝ √Φ and the count is constant.
- *The gradient never has a tangent component.* ∇Φ = (2/V) Jᵀ(Re ρ, Im ρ) lies entirely in the normal space, exactly. The flow never moves along the negative directions at linear order. They describe the shape of the valley, not an instability, and they do not slow the descent.

**Why exactly half: a sum rule.**
- *Balanced weights.* Summed over particles, each wave's weights give Σ_j w_jq = Re(ρ_q* ρ_q) = |ρ_q|², which is ≈ 0 near any ground state. So the per-particle curvature weights of each residual wave are balanced in sign: half the particles sit where that wave curves the energy down, half where it curves it up. Panel (e) shows |Σ_j w| / Σ_j |w| falling ∝ √Φ to 10⁻⁹–10⁻¹⁰.
- *Half the particle curvatures are negative.* R is a sum of such balanced blocks, so half of its 2 × 2 block eigenvalues are negative: 0.498–0.503 for every start (panel f). The trace of R averaged over particles is 10⁻⁵ of its mean magnitude at Φ = 10⁻⁶.
- *Projection keeps the balance.* Projecting on the tangent space removes 2M directions but keeps the balance, giving (Nd − 2M)/2 negative directions.

**Geometric picture (panel a).**
- *Distance to a curved manifold.* Near the manifold, Φ ≈ Σ_normal λ_a d_a², where d_a is the distance along the a-th normal direction. Moving tangentially, the distance to a curved manifold changes at second order. It falls, and with it Φ, when the state is on the concave side of that curvature, and rises on the convex side.
- *The tangent block.* The tangent block of H is −2 Σ_a λ_a d_a 𝐈𝐈_a, where 𝐈𝐈_a is the manifold's second fundamental form along normal a. Its size is ∝ d ∝ √Φ.
- *What the plateau says.* The signature of 𝐈𝐈 along the approach direction is (half, half). The ground-state manifold bends toward the descending state in as many directions as it bends away: it has nearly zero mean curvature along that normal. The sum rule is the algebraic form of this statement.

**Why it does not depend on the initial condition.**
- *What the argument uses.* Only that the constrained ρ_q are small, which holds near every ground state, and the number Nd − 2M of tangent directions, which is set by N, d and χ.
- *Where the start enters.* Only above Φ ≈ 1, where R is not small compared with the stiffness of G. There the count exceeds half, by more for starts with larger residual waves: from 598 (RSA, χ = 0.4) to 1342 (HIP, χ = 0.1). The extra negative directions disappear once √Φ is small.
- *What the plateau means physically.* The flow descends in a narrow valley (2M stiff directions, curvature ~|k|²) whose floor is the (Nd − 2M)-dimensional ground-state manifold, and the floor is curved with no preferred side. A state above the floor can always lower Φ by sliding along half the floor directions. Lower energy is always available by sliding along the floor, but the gradient never points that way, so each flow ends wherever its descent first meets the floor.

#### Negative and positive counts together

`batten2008_hessian_sign_plot.py` writes two figures.
- **`figures/hessian_sign_compare`:** panels (a, b) show the negative (solid) and positive (dashed) eigenvalue counts against Φ for every 2D start and both χ, with the cut-free P R P counts as points. Panels (c, d) show the soft-sector counts n₋ and n₊ − 2M over (Nd − 2M)/2.
- **`figures/hessian_sign_mechanism`:**
  - (a, b) the signed spectrum against Φ;
  - (c) how the Nd directions split;
  - (d) the soft magnitudes;
  - (e) the cut-free sign fractions;
  - (f) the stiff band.

`figures/hessian_sign_starts` shows, for every 2D start, the signed spectrum at χ = 0.1 and 0.4 and how the Nd directions split (one row per start). The structure is the same for every start. The soft ± branches are symmetric and shrink as √Φ, the stiff band of 2M stays fixed, and both branches cross the cut at Φ ≈ 10⁻¹¹. The starts differ only above Φ ≈ 1, where R is not yet small:
- *HIP* starts with the largest negative excess (61% negative at χ = 0.1).
- *RSA* χ = 0.4 starts with its stiff band already separated (0.74 of the Nd directions positive and stiff at the start, against 0.53–0.61 for the others).

**The counts at Φ = 10⁻⁴** (median over 100 flows) are:

  | Start | χ = 0.1: n₊ / 2M / n₋ | χ = 0.4: n₊ / 2M / n₋ |
  |---|---|---|
  | Poisson | 1232 / 420 / 813 | 1845 / 1640 / 201 |
  | Gaussian | 1202 / 400 / 796 | 1802 / 1604 / 196 |
  | URL | 1231 / 416 / 814 | 1847 / 1642 / 199 |
  | RSA | 1197 / 400 / 799 | 1797 / 1596 / 199 |
  | HIP | 1198 / 400 / 800 | 1804 / 1604 / 194 |

In every case n₊ = 2M + n₋ to within 1–3, and the remaining ≈ 3 eigenvalues are the zero modes (2 of them translations).

**What the figures show.**
- **The stiff band is fixed.** It holds the 2M wave directions, with eigenvalues between k₁² and 10³ k₁² independent of Φ below Φ ≈ 1 (panel f). It is the "positive count" floor 2M that remains at the end.
- **The soft branches are symmetric.** The other Nd − 2M eigenvalues form a positive and a negative branch of equal weight (signed spectrum, panels a and b). Their median magnitudes coincide and shrink together as √Φ (panel d). The cut-free P R P spectrum is 50 ± 1% negative from Φ ≈ 0.1 down (panel e).
- **Why the counts drop.** The negative count and the soft part of the positive count fall together at Φ ≈ 10⁻¹¹. This happens only because both branches cross the cut c = 10⁻⁹ λ_max (red dashed line); they then join the near-zero count. So n₊ falls from 2M + (Nd − 2M)/2 to 2M at the same Φ at which n₋ falls to 0.
- **At high Φ the count is not symmetric.** Above Φ ≈ 1, R is not small compared with G, and the counts favor negative curvature (n₋ > (Nd − 2M)/2 > n₊ − 2M). At χ = 0.4 the stiff band also has not separated: its 2M-th eigenvalue is still negative at the start.
- **The mechanism is the sum rule of the previous section.** The residual waves curve the energy down at half the particles and up at the other half, so the floor directions split evenly into positive and negative curvature, both of size √Φ. The positive half is the counterpart of the negative half: the valley floor bends toward the state in (Nd − 2M)/2 directions and away from it in as many.

### Flow-averaged mode planes

`batten2008_planes_mean_plot.py` writes `figures/<start>_mode_planes_mean`, one per 2D start. It uses the same layout as `*_mode_planes_full`: angle arg ρ(kₙ), radius log₁₀ S + 20, rows 1–5, 10 and |k| = K/4 to K, at both χ.

**How the flows are aligned.** Flows start at random phases and turn either way (38–64% counterclockwise per mode), so a plain average of ρ cancels.
- Each flow is rotated to start at angle 0.
- Flows with a negative net turn are mirrored, so every flow turns counterclockwise.
- The unwrapped phase is followed while S > 10⁻¹⁶ and held after.
- Phase and log₁₀ S are averaged over flows at equal Φ.

The mean curve is coloured by log₁₀ Φ, and 20 aligned flows are drawn in grey.

**What the averages show.**
- **One shape for every start: a turning arc, then a radial plunge.**
  - While S falls from its start value to about 10⁻⁸ (Φ from the start to about 10⁻⁵), the mean phase turns: the waves turn while they still hold amplitude.
  - Below that, the phase locks and the curve runs radially to S = 10⁻²⁰. This is the late linear relaxation along a fixed Hessian direction.
- **The longest waves do not turn.** Mean net turn for k₁ = (0, 1), (1, 0): 0.001–0.02 at χ = 0.1 and 0.03–0.14 at χ = 0.4. These are straight radial lines.
- **The turn grows with |k|.** Mean net turns:

  | Start | χ | Next shells | n = 10 | \|k\| ≈ K/4–K |
  |---|---|---|---|---|
  | Poisson | 0.1 · 0.4 | 0.24–0.32 | 0.30 | 0.43–0.44 · 0.44–0.65 |
  | Gaussian | 0.1 · 0.4 | 0.18–0.32 | 0.29–0.31 | 0.33–0.43 · 0.33–0.52 |
  | URL | 0.1 · 0.4 | 0.18–0.32 | 0.27–0.33 | 0.31–0.41 · 0.39–0.50 |
  | RSA | 0.1 · 0.4 | 0.15–0.28 | 0.25–0.31 | 0.33–0.39 · 0.37–0.45 |
  | HIP | 0.1 · 0.4 | 0.27–0.40 | 0.34–0.45 | 0.48–0.55 · 0.53–0.87 |

- **Starts differ in amplitude, not shape.** The long waves of hyperuniform starts begin at S ≈ 10⁻²–10⁻¹ (radius 18–19) and wind less. HIP's long waves start beyond S = 1 and wind most.
- **Signed turns are smaller than unsigned winding.** The coherent turn (0.3–0.9 turns) is smaller than the unsigned winding of `winding_compare` (0.75–3.9 turns), because each wave also turns back and forth.

### Windiness of the mode-plane trajectories

`batten2008_winding_plot.py` (`figures/winding_compare`) quantifies how much each density wave's trajectory winds in its mode plane (`*_mode_planes_full`). The metric is the total winding T_n = (1/2π) Σ_t |Δ arg ρ(kₙ)|, counted while S(kₙ) > 10⁻¹⁶. ρ(kₙ) sums over all particles, so T_n already averages over the particles of a flow. Also stored per mode (`batten2008_modes.py`): the net winding |Σ Δ arg ρ|/2π, and T_n split by log₁₀ S at each step.

- **At χ = 0.4 the difference is physical, not statistical.** For |k| > 0.25K the Poisson waves wind 1.1–1.9 turns (rising with |k|), against 0.9–1.2 for Gaussian and URL, 0.85–1.0 for RSA, and 2–4 for HIP. The short-wave windings (|k| > 0.6K) of Poisson and Gaussian flows separate completely (Mann–Whitney p = 2 × 10⁻²³, 100 against 50 flows), and Poisson against URL, RSA or HIP gives p ≤ 3 × 10⁻³⁴. The net (coherent) winding differs the same way.
- **What sets it.** The short-wave winding tracks how far the particles move, |Δr|/√N: Spearman 0.95 over all 450 flows and 0.69 within the 100 Poisson flows alone. The within-ensemble correlation cannot come from comparing different kinds of start. The extra Poisson winding happens while S(k) of the short wave is already 10⁻²–10⁻⁶ (panel g): the wave has nearly cancelled, but its particles are still being carried by the slow, long-wave relaxation. That transport slides the residual short-wave pattern, rotating arg ρ(k) without changing |ρ(k)| (the near-circular arcs in the Poisson mode planes). Hyperuniform and RSA starts need little long-wave motion (per-particle displacement 0.25–0.37, against 0.80), so their short waves slide less. HIP needs much more (2.2) and winds most (3.9 turns at |k| > 0.6K). The winding is set by the triad coupling to the unconstrained waves, and grows with the slip K·ℓ/2π (next section).
- **At χ = 0.1 the starts differ less.** Poisson, Gaussian, URL and RSA wind 0.75–0.87 turns at large |k|; HIP winds 1.42. The differences between the first four are statistically real (Poisson against Gaussian p = 3 × 10⁻¹⁶), but within Poisson winding is not correlated with displacement (Spearman −0.08), so displacement sets it only when the particles move far.

### Winding as a collective variable, and its relation to the potential

`batten2008_winding.py` (GPU; one job per start) follows every constrained wave's phase θ_k = arg ρ_k along every flow. It stores the cumulative winding W_k on the Φ clock and at every frame, and the exact instantaneous rates from the gradient. `batten2008_windcv_plot.py` draws three kinds of figure:
- **`*_winding_kymograph`:** W_k(t) for every wave, in time and on the Φ clock, plus the spiral pitch and a 3D surface.
- **`winding_total`:** Σ_k W_k over the whole exclusion region |k| ≤ K, against log₁₀ Φ, path fraction s and t/τ₁.
- **`winding_cv`:** the collective-variable diagnostics below.

`winding_total` and `winding_cv` show the ten 2D ensembles only. The 1D numbers quoted below come from the same `winding_chi*.npz` files for `batten2008/N1000`; the 1D flows appear only in `batten2008_N1000_winding_kymograph`.

- **An exact relation to the potential.** Φ = Σ_k |ρ_k|²/N depends only on the moduli, so turning one wave's phase costs that wave no energy. Along the flow ṙ = −∇Φ,

  dρ_k/dt = (1/N) Σ_q (k·q) [ρ_q* ρ_{k+q} − ρ_q ρ_{k−q}]   (q over the constrained set; the q = k term with ρ₀ = N is the linear decay −|k|²ρ_k),

  so the winding rate θ̇_k = Im(ρ_k* ρ̇_k)/|ρ_k|² comes entirely from three-wave (triad) terms. Equivalently, θ̇_k = −∇θ_k · ∇Φ = −|∇θ_k| |∇Φ| cos α_k. A wave winds when the force has a component along its own phase direction, and that component is pure mode coupling: it vanishes for separable, harmonic relaxation. The identity holds to 10⁻⁹–10⁻¹⁴ along the flows (`winding_cv` (d), 2D Poisson χ = 0.1, flow 0).
- **The coupling runs through the unconstrained waves.** Below Φ ≈ 0.1, essentially all of each wave's phase torque (median share 0.95–1.00) comes from triads whose partner k ± q lies outside the exclusion region, |k ± q| > K. Those density fluctuations are free (S ~ 1), so they act as a bath that pushes the constrained waves around. Panel (e) shows the share per |k|.

#### Why every start leaves the plateau at the same energy

The x-axis of `hessian_sign_starts` is Φ, not time. Within one χ the starts change their sign counts at the same energies, but they reach those energies at times up to 100× apart: Φ = 3 is reached at t/τ₁ ≈ 0.03 (URL, RSA) and at 2.6 (HIP) for χ = 0.1. `batten2008_sign_universal_plot.py` writes `figures/hessian_sign_universal`, which explains the shared energies with three scales. None of them depends on the start.

**1. The plateau is reached as √Φ (panels a–d).** Write H = G + R, where R is block-diagonal with weights w_jq = Re(ρ_q* e^{iq·r_j}).
- The trace of R is set by the sum rule Σ_j w_jq = |ρ_q|²: tr R = −(2/V) Σ_q |q|² |ρ_q|² ∝ Φ.
- The per-particle spread of R is ∝ √Φ, because each w_jq has a random phase.
- The soft spectrum is therefore symmetric to leading order, with a negative bias relative to its width of

  μ = −⟨tr R_j⟩ / σ(tr R_j) = f √(2Φ/N),   f = Σ|q|²S / √(Σ|q|⁴S · ΣS) ≤ 1.

  The shape factor f reflects how the residual is spread over |q|. It equals 1 when one shell holds the residual (Cauchy–Schwarz).
- Panel (c) confirms the law with no fitted parameter over seven decades of μ, for every start and both χ.
- The bias sits in the (1 − c)Nd soft directions, with c = 2M/(Nd). The negative excess n₋/[(Nd−2M)/2] − 1 then follows μ/(1 − c):
  - the measured ratio is 0.7–1.5 for every start and χ at 10⁻² ≤ Φ ≤ 1;
  - without the (1 − c) factor it would be 0.9–1.9 at χ = 0.1 and 3.8–7.2 at χ = 0.4;
  - panel (d) shows the collapse.
- **What this sets.** The excess falls below 5% at Φ ≈ 2–3 for χ = 0.1 and at Φ ≈ 0.1–0.6 for χ = 0.4. The starts enter only through f:
  - χ = 0.1: f = 0.4 (HIP) to 0.8 (Gaussian, URL) at Φ = 1;
  - χ = 0.4: f = 0.14 (HIP) to 0.78 (URL).
  - Since μ ∝ f√Φ, this spread moves the onset by at most one decade in Φ.
- The early rise of soft + and fall of soft − are the same √Φ approach to symmetry, seen from the two sides.

**2. The bands separate (panel e).** The largest soft |λ| (∝ √Φ) falls below the smallest stiff eigenvalue λ₂ₘ ≈ k₁² near Φ ≈ 1–10.
- Before that point soft and stiff directions mix, and the counts are not yet those of two separate bands.
- The crossing again depends only on the soft prefactor a and on λ₂ₘ:
  - a = (1.0–2.3)·10⁻³ at χ = 0.1 and (1.2–1.8)·10⁻² at χ = 0.4;
  - λ₂ₘ/k₁² = 0.95–1.0 at χ = 0.1 and 0.83–0.91 at χ = 0.4.

**3. The final drop is the counting cut (panel f).** The soft branches go to zero as a√Φ and are counted only while a√Φ > c·λ_max, with c = 10⁻⁹.
- With Φ_× = (c λ_max/a)², the drops of every start fall onto one curve centred at Φ/Φ_× = 1.
- Φ_× = (0.9–2.9)·10⁻¹¹ at χ = 0.1 and (0.3–0.8)·10⁻¹¹ at χ = 0.4.
- In exact arithmetic the plateau n₋ = (Nd−2M)/2 holds for every Φ > 0.

**The starts are not funnelled to one place.** The shared energies come from self-averaging over N ≈ 1000 particles and M waves, which makes μ, a and λ₂ₘ functions of (N, d, χ, Φ) and of the shape factor f. They do not show that the flows converge.
- *χ = 0.4:* the ground states share one local landscape. The coupling spectrum C = D^{-1/2} K D^{-1/2} is the same Marchenko–Pastur-like distribution for every start (10–90%: 0.08–0.12 to 1.87–1.90; panel g).
- *χ = 0.1:* the local landscapes differ. The 10–90% widths of C are 0.88–1.12 (RSA), 0.71–1.29 (Gaussian, URL), 0.60–1.40 (Poisson) and 0.50–1.50 (HIP).
- *End states (panel h):* they keep their origin outside the exclusion region. At 1.05–1.5 K and χ = 0.1, S(k) is 0.08 (RSA), 0.46–0.49 (URL, Gaussian), 1.02 (Poisson) and 1.80 (HIP). At χ = 0.4 these values are 0.96–1.22, and URL keeps a Bragg-like peak S ≈ 7 at |k| ≈ 1.5 K.
- *Summary:* the flows reach different points of the ground-state manifold. Near any of them the Hessian looks statistically the same because it is a sum over many random-phase waves.

#### The wave coupling against the Marchenko–Pastur law

`batten2008_mp_plot.py` writes `figures/coupling_mp_compare`. It compares the coupling spectrum of the 2D Poisson flows at χ = 0.1, 0.25, 0.4 and 0.45 with the Marchenko–Pastur (MP) law.
- **What is measured:** the spectrum of C = D^{-1/2} K D^{-1/2}, with K = (2/V) J Jᵀ and D = diag K, so that C has a unit diagonal.
- **The MP prediction:** if the 2M wave gradients were independent random vectors in Nd dimensions, C would follow MP with ratio c = 2M/(Nd). The support would be [(1 − √c)², (1 + √c)²], with mean 1, variance c and ⟨1/λ⟩ = 1/(1 − c).
- **Data:** the spectra come from the end states (ground states) and the Poisson starts of flows 0–19. They are cached in `<root>/mp_chi<tag>.npz`.

**The Poisson starts follow MP in the bulk.**
- Var λ/c = 0.99–1.00, and the Kolmogorov distance from MP is 0.03–0.08.
- The soft edge is the exception at χ ≥ 0.25. λ_min is 0.07, 10⁻³ and 10⁻⁴ of the MP edge at χ = 0.25, 0.4 and 0.45.
- The softest mode there is a void mode. Its field peaks 1.45–1.59 from the nearest particle, inside the largest hole of the configuration; the largest hole is 1.75, and 99.9% of points lie within 1.46 of a particle. A band-limited combination of waves can sit in a hole of a Poisson configuration and barely move any particle.

**The ground states are not MP.** Every entry of C is set by the density at the pair wave vectors:

  C(Re q, Re p) ∝ (q̂·p̂) Re[ρ(q − p) − ρ(q + p)].

- At a ground state, ρ = 0 inside the exclusion region, so every pair with |q ± p| < K is exactly uncoupled.
- With random phases elsewhere this gives

  Var λ / c = ⟨(q̂·p̂)² [S(q − p) + S(q + p)]⟩ / ⟨2 (q̂·p̂)²⟩

  averaged over pairs. With S = 1 everywhere this is MP. With S = 0 inside K and 1 outside, it is the geometric factor 0.407, close to the share 1 − (1 − 3√3/4π) = 0.414 of point pairs in a disc that lie more than its radius apart.
- **Panel (f):** with the measured end-state S this pair prediction is 0.405, 0.429, 0.507 and 0.564. The measured values are 0.411, 0.425, 0.504 and 0.564.
- The variance rises above 0.41 at large χ because S just beyond K grows with χ.

The spectrum is therefore narrower than MP at every χ: λ_max is 0.76 → 0.57 of the MP edge. Its shape is closest to MP with the measured variance, c_eff = Var λ.
- **χ = 0.1:** the Kolmogorov distance is 0.037.
- **χ = 0.25 to 0.45:** the distance grows to 0.07, 0.12 and 0.15, as two features appear.
  - **A soft tail.** λ_min is 0.010 and 0.001 at χ = 0.4 and 0.45, close to the MP(c) edge rather than the MP(c_eff) edge. These soft modes are collective, not void modes:
    - each spans about 320–360 waves;
    - each peaks within 0.1 of a particle;
    - the ground states have no hole larger than 0.8.
  - **A step at λ ≈ 1.** 3.5–4.5% of the eigenvalues lie within 0.02 of 1, about 5× the MP density. They are long waves (median |q|/K = 0.31–0.38 at χ = 0.4 and 0.45). A long wave couples only to waves near the rim of the exclusion region, so its row of C is half as strong (√Σ_p C²_qp = 0.36–0.40 for |q| < 0.3K, against 0.72–0.78 for |q| > 0.7K).

**What this means for the landscape.** At a ground state the waves are less coupled than random gradients would make them, because S vanishes inside the exclusion region. This is why χ = 0.1 behaves like nearly independent wells. As χ → 1/2 the coupling, and with it the soft tail and the slow valley, returns through the waves just outside K.

#### The shoulder at λ(C) = 1 is physical

At χ = 0.4 and 0.45 the ground-state cumulative curves of `coupling_mp_compare` (c, d) have a shoulder at λ(C) ≈ 1, and the F − F_MP curves of panel (e) have a matching kink. `batten2008_mp_shoulder_plot.py` (`figures/coupling_mp_shoulder`) shows that this is a physical feature of stealthy ground states, not a numerical artifact.

**Mechanism: the long waves cannot couple to one another.** The coupling is fixed exactly by the density at the pair wave vectors:
- K(Re q, Re p) = (q·p)/V Re[ρ(q−p) − ρ(q+p)];
- K(Re q, Im p) = −(q·p)/V Im[ρ(q+p) + ρ(q−p)];
- K(Im q, Im p) = (q·p)/V Re[ρ(q−p) + ρ(q+p)].

Rebuilding C this way from ρ on the 2K lattice reproduces the spectrum from the gradients to ≤ 7·10⁻¹⁴.

Consequences at a ground state, where ρ = 0 inside K:
- Every pair with |q ± p| < K is uncoupled. In particular, all waves with |q| < K/2 are uncoupled from one another (panel b): they make up 24–26% of the coordinates at every χ, and |C| inside their block is ≤ 5·10⁻¹³.
- They couple only to the outer waves, through ρ just beyond K. Their eigenvalues therefore stay near 1. Every eigenvector within 0.02 of 1 is a long wave: median |q|/K = 0.51, 0.48, 0.38 and 0.31 at χ = 0.1, 0.25, 0.4 and 0.45, against about 0.70 for all eigenvectors (panel c).
- At large χ the outer waves are strongly coupled and split into a soft pile near 0 and a stiff hump near 2 (panel a). The long-wave cluster stays behind at 1 as a sharp peak.

**Peak ratio** ρ(1)/½[ρ(0.8) + ρ(1.2)] (panel f):

| | χ = 0.1 | 0.25 | 0.4 | 0.45 |
|---|---|---|---|---|
| ground states | 1.18 | 1.30 | 2.54 | 3.91 |
| outer waves only (\|q\| ≥ K/2) | 0.86 | 1.00 | 0.83 | 1.01 |
| ideal-gas surrogate (S = 1) | 1.04 | 1.00 | 0.98 | 1.01 |
| step-S surrogate, N = 1024 | 1.30 | 1.20 | 1.13 | 1.21 |
| step-S surrogate, N = 4096 | 1.28 | 1.25 | 1.30 | 1.21 |
| phases randomized, \|ρ\| kept | 1.27 | 1.26 | 1.40 | 1.45 |

**Why it is not a numerical artifact:**
- the identity holds to machine precision;
- no eigenvalue is exactly 1, so it is not a degeneracy;
- removing the long waves removes the peak and the kink in F − F_MP (panel e, dotted);
- an ideal gas has no peak;
- a stealthy random-phase surrogate shows the same peak at N = 1024 and 4096, so it is not a finite-size effect.

**Why the random-phase surrogates fall short at large χ.** A real C is the Gram matrix of 2M vectors in Nd dimensions, so it is positive.
- The surrogates violate this at χ ≥ 0.25: 0.1–9% of their eigenvalues are negative at χ = 0.25–0.45, and 0.5–17% for the ideal gas.
- The ρ(k) in the coupling annulus K < |k| < 2K carry about 3·2M real numbers: 5580 at χ = 0.45, set by only Nd = 2048 coordinates. They cannot be independent.
- At χ = 0.1 (1260 < 2048) the surrogates are valid and match the ground state (1.27–1.30 against 1.18).
- The surrogates' long-to-outer coupling block has the same singular values as the ground state's (median 0.47 at χ = 0.45). The sharper ground-state peak therefore comes from how the outer waves are organized (positivity), not from weaker long-wave coupling.

**Why panel (e) shows a shoulder only at large χ.** The cluster adds a near-step of about 4% to F at λ = 1, while F_MP rises smoothly; F − F_MP inherits the step as the kink.
- At χ = 0.1 the whole ground-state spectrum lies in 0.4–1.6, around 1. The cluster raises the local density by only 18%, so it merges into the bulk.
- At χ = 0.45 the bulk has moved away from 1, toward 0 and 2, and the cluster stands out 3.9× above its surroundings.
- A comparison with MP free of this feature uses the outer block alone (panel e, dotted).

#### A vibrational band gap near χ → 1/2, and what it says about composites

`batten2008_phonon_gap_plot.py` writes `figures/coupling_phonon_gap`. It uses 2D Poisson ground states at six χ, 6 per χ.

**What K is.** At a ground state the nonzero normal modes (mass 1) have ω² = λ(K). K is also the plane-wave (Galerkin) form of the scalar wave operator −∇·η∇ with η a sum of point weights at the particles: ⟨q|−∇·η∇|p⟩ = (q·p) η̃(q − p), with η̃ ∝ ρ, restricted to |q|, |p| < K.
- For a two-phase composite (TE photonic, anti-plane elastic or acoustic waves), the same matrix appears with ρ(k) replaced by m̃(k) ρ(k) (inclusion form factor) and with all plane waves kept.
- K is therefore the point-scatterer, truncated limit of the composite's plane-wave Hamiltonian.
- λ(C) divides each wave by its own |q|². That is a congruence, which keeps only the signs of the eigenvalues, not the gaps. The frequency variable is ω = √λ(K).

**Result (N = 1024).** The largest spacing of ω/K in 0.3–1, compared with the largest spacing of the uncoupled spectrum ω = |q| (the k-shells):

| χ | largest spacing (ω/K) | Δω/ω_mid | uncoupled shell spacing | gap? | modes below |
|---|---|---|---|---|---|
| 0.10 | 0.308–0.332 | 0.076 | 0.046 | no | 0.10 |
| 0.25 | 0.716–0.719 | 0.004 | 0.020 | no | 0.57 |
| 0.35 | 0.861–0.863 | 0.003 | 0.017 | no | 0.71 |
| 0.40 | 0.753–0.760 | 0.010 | 0.016 | no (pseudo-gap) | 0.62 |
| 0.455 | **0.607–0.683** | **0.118** | 0.015 | **yes, in every state** | 0.55 |
| 0.491 | **0.517–0.605** | **0.156** | 0.014 | **yes, in every state** | 0.51 |

**Where the gap sits.**
- The effective dispersion (panel c) shows modes below the gap softened (ω < |q|) and modes above stiffened (ω > |q|): a level repulsion.
- The gap lies 8–16% below the Bragg-like estimate ½k_peak/K = 0.705 and 0.670, set by the first S(k) peak beyond K (S ≈ 2.2–2.5).

**The gap is not a two-point property.** Keeping every |ρ(k)|, and so S(k) at every wave vector, while randomizing the phases fills the gap completely:
- the density in the ground-state gap window rises from 0–0.01 to 0.69–0.79;
- the largest spacing falls to 0.002;
- 9–11% of the surrogate eigenvalues are negative, so no point configuration can produce them.

The gap is therefore enforced by the constraint that the ρ(k) come from N points, that is, by correlations beyond S(k).

**Finite size** (`figures/coupling_mp_finite_size`). Six 2D Poisson ground states per χ at N = 4096 (`flows2d/N4096`, starts `initial_conditions/n4096`, seed 20261108; 2.1 GPU-h), compared with N = 1024:

| χ (N = 1024 / 4096) | ω gap at 1024 | ω gap at 4096 | shell spacing at 4096 | λ(C) dip near 0.5 | λ(C) dip near 1.1 |
|---|---|---|---|---|---|
| 0.455 / 0.451 | 0.609–0.682 (0.074 ± 0.007) | 0.626–0.667 (0.040 ± 0.012), in-gap states near 0.65 | 0.0048 | 0.219 → 0.231 | 0.102 → 0.069 |
| 0.491 / 0.490 | 0.517–0.606 (0.089 ± 0.012) | **0.525–0.606 (0.082 ± 0.006)** | 0.0046 | 0.178 → 0.182 | 0.194 → 0.189 |

- **χ = 0.49:** the frequency gap is N-independent within error and 18× the shell spacing.
- **χ = 0.45:** the gap narrows as isolated in-gap (band-tail) states appear. Whether it survives as N → ∞ is open.
- **λ(C) near 0.5:** the dip does not deepen with N, so it is a pseudo-gap.
- **λ_min:** follows the MP hard edge at both N.

**Not yet tested:** composites with finite inclusions and all plane waves.

#### What the gap needs: phases, winding and the transparency interval

These checks were run ad hoc, not by a committed script.

**(a) Partial phase randomization** (χ = 0.491, one ground state). Each phase θ_k is shifted by ε·φ_k, with φ_k uniform in [−π, π]; every |ρ(k)| is kept.

| ε | 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1 |
|---|---|---|---|---|---|---|---|
| gap width (ω/K) | 0.092 | 0.089 | 0.075 | 0.038 | 0.006 | 0.003 | 0.002 |
| density in the ground-state gap | 0 | 0 | 0 | 0.15 | 0.40 | 0.61 | 0.67 |
| negative ω² | 0 | 0.062 | 0.081 | 0.099 | 0.109 | 0.110 | 0.106 |
| Var λ(C) | 0.5735 at every ε | | | | | | |

- Realizability is lost at the smallest noise (6% negative at ε = 0.05), while the gap survives to ε ≈ 0.1 and closes by 0.3. The gap is therefore a property of the phase arrangement, robust to phase errors of about ±0.1π, and not simply of positivity.
- Var λ(C) depends only on |ρ| (an exact identity), so it is unchanged at every ε.

**(b) Moments.** The third moment ⟨(λ − 1)³⟩ of λ(C) is a sum over closed triangles of pair vectors, ρ(a − b) ρ(b − c) ρ(c − a), so it carries the triad phases.
- At χ = 0.455 it is −0.0005 ± 0.0010 in the ground states, indistinguishable from the phase-randomized value of +0.0001.
- At χ = 0.491 it is −0.0053, against −0.0011 at ε = 1.
- The gap therefore needs loops of four or more pair vectors, i.e. higher-order phase correlations.

**(c) Winding does not set the gap.** Over 40 flows at χ = 0.455:
- the gap width is 0.071 ± 0.011 (coefficient of variation 0.16) and the total winding is 1888 ± 119 (0.06);
- Spearman correlation with total winding is −0.07 (p = 0.66); with winding of waves above 0.5K, −0.08; with path length, −0.10; with flow time, −0.01;
- the third moment against winding gives +0.02.

**(d) Leakage out of the transparent sector.** For each long wave (|q| < K/2) the leakage is r_q = Σ_p C²_qp over outer waves p (|p| ≥ K/2). Medians by |q|/K:

| χ | 0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | r_q/(\|q\|/K) |
|---|---|---|---|---|---|---|
| 0.10 | 0.014 | 0.027 | 0.040 | 0.054 | 0.068 | 0.15 |
| 0.25 | 0.030 | 0.067 | 0.108 | 0.143 | 0.175 | 0.40 |
| 0.35 | 0.040 | 0.095 | 0.148 | 0.193 | 0.233 | 0.55 |
| 0.40 | 0.057 | 0.107 | 0.158 | 0.216 | 0.288 | 0.62 |
| 0.455 | 0.052 | 0.109 | 0.195 | 0.292 | 0.404 | 0.79 |
| 0.491 | 0.056 | 0.132 | 0.238 | 0.378 | 0.518 | 1.03 |

The leakage is close to linear in |q|: only partners with |p| > K − |q| couple, a rim of area ∝ 2πK|q|. Its slope grows sevenfold from χ = 0.1 to 0.49.

**Effective dispersion** (`figures/coupling_phonon_dispersion`): one panel per χ, showing the median ω/K per bin of eigenvector mean |q|/K, with 16–84% and 2.5–97.5% bands. Bins with fewer than 10 modes are left out.
- At every χ the curve crosses the uncoupled line ω = |q| near |q|/K ≈ 0.7–0.8: modes below the crossing are softened and modes above it stiffened.
- At χ ≥ 0.45 the gap opens at this crossing.
- At χ ≥ 0.40 no frequency mode has eigenvector mean |q|/K below about 0.3–0.45. The long waves are not normal modes in frequency: they hybridize with the soft outer modes. In λ(C) they still form a near-identity block.

#### Loop moments and the thermal stability of the gap

These checks were run ad hoc, not by a committed script. Three ground states per χ, against one phase-randomized surrogate each (every |ρ(k)| kept).

**Moments are sums over closed polygons of momentum transfers.** In the complex wave basis K_qp = (q·p) ρ(q − p)/V with ρ(0) = N, so

  tr K^n = V^(−n) Σ Π_i (q_i·q_(i+1)) ρ(k_i),   k_i = q_i − q_(i+1),   Σ k_i = 0.

Each term is an n-point Fourier product whose phase is the translation-invariant loop phase Σ arg ρ(k_i). Terms in which the k_i cancel in pairs reduce to products of S(k) and survive phase randomization. Irreducible polygons (n = 3: bispectrum, g₃; n = 4: trispectrum, g₄; …) average to zero under it.

Central moments of λ(K)/K², ground state against surrogate (relative difference):

| χ | m₂ | m₃ | m₄ | m₆ | m₈ |
|---|---|---|---|---|---|
| 0.103 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.401 | 0.00 | +0.02 | −0.07 | −0.13 | −0.19 |
| 0.455 | 0.00 | +0.02 | −0.10 | −0.16 | −0.22 |
| 0.491 | 0.00 | +0.01 | −0.11 | −0.18 | −0.23 |

- m₂ is exactly S-only.
- The triad (three-point) term changes m₃ by only 1–2%.
- The first substantial phase-dependent contribution is at four-wave polygons, and it accumulates at higher orders.
- At χ = 0.1 phases carry no spectral information at any order.
- In λ(C) the same pattern appears, but larger (m₄ −23 to −32%), because the surrogates' negative eigenvalues inflate their tails.
- Moments cannot by themselves certify a gap.

**Thermal phase noise (harmonic).** For displacements u ~ N(0, T·H⁺) about a ground state, the waves outside K (K < |k| < 2K) acquire phase noise linear in √T. The rms phase per √T is 7.0 (|ρ|²-weighted) to 8.5 (median) at χ = 0.455, and 21 to 26 at χ = 0.491; the near-zero soft modes dominate H⁺.

Mapped onto the partial-randomization curve, where uniform noise of half-width επ has rms επ/√3 (gap halves at ε ≈ 0.2, closes at ε ≈ 0.3), this gives:
- χ = 0.455: halving at T ≈ 2–3·10⁻³, closing at 4–6·10⁻³;
- χ = 0.491: halving at T ≈ 2–3·10⁻⁴, closing at 5–7·10⁻⁴.

**Direct harmonic sampling** (gap of K at r + u; 2 states × 2 samples per T):

| T | 0 | 10⁻⁵ | 10⁻⁴ | 3·10⁻⁴ | 10⁻³ | 3·10⁻³ | 10⁻² |
|---|---|---|---|---|---|---|---|
| χ = 0.455: gap (Φ/MT) | 0.075 | 0.075 (1.00) | 0.077 (1.03) | 0.066 (1.03) | 0.065 (1.16) | 0.042 (1.42) | 0.007 (2.31) |
| χ = 0.491: gap (Φ/MT) | 0.092 | 0.091 (1.13) | 0.101 (2.31) | 0.078 (3.41) | 0.057 (10.3) | 0.007 (16.4) | 0.002 (30.4) |

- At χ = 0.455 the linear estimate and the sampling agree, and the harmonic picture holds to T ≈ 10⁻³.
- At χ = 0.491 Φ/(MT) departs from 1 already at T ≥ 10⁻⁴, so harmonic theory fails there and MD is needed.

#### Isolating the loop (beyond-S(k)) contribution

`batten2008_loop_moments.py` computes and caches `loops_chi<tag>.npz`; `batten2008_loop_moments_plot.py` writes `figures/loop_moments`. Data:
- six χ at N = 1024 (6 ground states, 8 random-phase surrogates each);
- χ = 0.45 and 0.49 at N = 4096 (4 states, 2 surrogates each).

**Method.**
- In the complex basis {±q}, H = D + W with H_ab = (Q_a·Q_b) ρ(Q_a − Q_b)/V. Its spectrum equals that of K: the Re/Im basis agrees to ≤ 10⁻¹².
- w_n = (1/2M) tr (W/K²)ⁿ splits exactly into three parts:
  - **non-crossing pairings:** the SCBA recursion with variance profile s_ab = |W_ab/K²|²;
  - **crossing pairings:** surrogate mean minus SCBA;
  - **irreducible loops:** ground state minus surrogate mean. Odd n are entirely irreducible.
- The S(k)-only theory is the SCBA vector Dyson equation, G_a = 1/(z − D_a − Σ_b s_ab G_b).
- **Checks:** a flat profile gives the Catalan numbers; a sampled Wigner matrix with a variance profile gives m₄ = 1.99 and m₆ = 4.94, against 2.00 and 5.01 non-crossing; the semicircle density at 0 is 0.318 (1/π).

**Results:**
- **The SCBA is the S(k)-only theory.** Crossing pairings are ≤ 1% of w₄ at every χ, so the SCBA reproduces the surrogates' paired part. Crossings grow at high order (36% of the non-crossing moment at n = 12, χ = 0.49).
- **Irreducible share of the S(k)-only moment, I_n/P_n** (N = 4096 in brackets):

  | χ | n = 4 | n = 6 | n = 8 | n = 12 |
  |---|---|---|---|---|
  | 0.10 | −0.03 | −0.07 | −0.11 | −0.19 |
  | 0.40 | −0.20 | −0.41 | −0.57 | −0.79 |
  | 0.455 | −0.236 (−0.235) | −0.452 (−0.452) | −0.618 (−0.620) | −0.826 (−0.832) |
  | 0.491 | −0.246 (−0.248) | −0.466 (−0.471) | −0.633 (−0.640) | −0.838 (−0.844) |

  The loops always reduce the moments relative to S(k) alone: the true spectrum has lighter tails. The effect grows with order and with χ, and is the same at N = 1024 and 4096, so it is not a finite-size effect.
- **Triads are small.** The standardized third moment I₃/w₂^1.5 is 0.004–0.026.
- **S(k) alone gives no gap.** In the ground-state gap window the density is 0.001–0.004 in the ground states, against 0.54–0.57 for the surrogates and 0.56–0.59 for the SCBA. The SCBA also puts weight at negative x, which a Gram matrix cannot have.
- **How much S(k) misses.** The total-variation distance between the ground-state spectrum and the S(k)-only spectra rises from 0.03–0.05 (χ = 0.10) to 0.23–0.25 (χ = 0.49).
- **Panel (e) shows where the loops act:**
  - they remove weight from the soft shoulder below the gap (x ≈ 0.25–0.5) and from negative x;
  - they move it into the soft-mode peak at x ≈ 0 and to just above the gap (x ≈ 0.6–0.9).

**Cost:** SCBA 11–241 s per χ at N = 1024 and about 15 min at N = 4096, on CPU.

#### Same two-point statistics, different loops

`batten2008_loop_statistics_plot.py` writes `figures/loop_statistics` and caches `loopstats_chi<tag>.npz`. It uses 4 ground states per χ at N = 1024, with 4 surrogates each, and compares the three models of the vibrational spectrum:
- ground states;
- random-phase surrogates (same |ρ(k)| at every k);
- the SCBA (a Gaussian field with the ground-state S(k)).

**Two-point statistics are identical by construction:**
- the surrogates match S(k) wave vector by wave vector, |ΔS| ≤ 5·10⁻¹⁶;
- the SCBA input is the state-averaged S(k), which differs from a single state only by the about 10% state-to-state scatter.

**Amplitude statistics also agree.** Beyond K the intensities |ρ|²/⟨|ρ|²⟩ are exponential, as for a Gaussian field: ⟨I²⟩ = 1.96–2.03 against 2, at every χ. So the SCBA's Gaussian-amplitude assumption holds.

**The difference is in loop sums, not in single loop phases.**
- Each closed walk's phase is nearly uniform: the |T|-weighted coherence per loop is ±0.01 in both ground states and surrogates.
- There are about 2M times more irreducible walks than paired ones, so a coherence of order 1/(2M) suffices.
- The difference is therefore measured exactly, from the diagonals of W³ and W⁴: the irreducible share of each wave's return amplitude.

**Results:**
- **Whole-spectrum quadrilateral share I₄/P₄:** −0.017, −0.083, −0.146, −0.209, −0.235 and −0.250 for χ = 0.10 → 0.49 (matching `loop_moments`).
- **Triad share:** +0.003 to +0.011.
- **Per wave (panel d):** the effect is concentrated on waves with |q|/K ≈ 0.35–0.5, just inside the K/2 edge of the uncoupled long-wave block. It reaches −0.42 at χ = 0.49 and is −0.16 to −0.07 at the outer edge.
- **Per transfer length (panel e):** loops built from transfers in a single |k|/K band carry little (|share| ≤ 0.12, mixed signs; the largest is in the band 1.25–1.5, where the S(k) peak lies at χ ≥ 0.45). So the coherence that sets the gap comes mostly from loops that combine different transfer lengths.

**Phase distributions** (panels g–i; cache `loopphase_chi<tag>.npz`). Three models, compared at the largest χ:
- the ground states;
- the random-phase model: the same |ρ(k)| with uniform phases;
- Gaussian-field (SCBA) realizations: complex Gaussian ρ(k) with E|ρ|² = N·S̄(k), the state-averaged S.

What each panel shows:
- **(g) Single-wave phases** arg ρ(k), K < |k| < 2K: uniform in all three models. Absolute phases change under translation, so they carry no information.
- **(h) Four-wave loop phases** arg T, |T|-weighted, from 5·10⁵ sampled walks per realization with no cancelling pair: flat within sampling noise (±0.1 per bin) in all three. The exact ground-state first harmonic, 2c₁ cos φ, has amplitude 0.004, far below that noise. Loop by loop, the ground states cannot be told apart from either random model.
- **(i) The exact first harmonic** c₁ = Σ Re T / Σ |T| over irreducible loops, computed from traces with no sampling. For triads it is c₁ = tr W³ / tr |W|³; for quadrilaterals it is (tr W⁴ − P₄)/(tr |W|⁴ − P₄), with P₄ the pairing sum averaged over phase randomizations.

  2M·c₁ by model (4 realizations each):

  | χ | 0.10 | 0.25 | 0.35 | 0.40 | 0.45 | 0.49 |
  |---|---|---|---|---|---|---|
  | quadrilaterals, ground states | −0.51 | −1.25 | −2.34 | −3.25 | −3.35 | −3.47 |
  | quadrilaterals, random-phase | −0.26 | −0.07 | +0.07 | −0.01 | −0.07 | −0.08 |
  | quadrilaterals, Gaussian | −0.07 | +0.02 | −0.06 | +0.10 | +0.14 | +0.08 |
  | triads, ground states | +3.1 | +9.2 | +11.2 | +23.6 | +19.2 | +10.4 |
  | triads, random models | −2.6 to +3.1 (realization noise) | | | | | |

- **Interpretation.** The ground states carry a weak, systematic loop-phase bias of order 1/2M per loop: positive for triads, negative for quadrilaterals. Summed over the roughly 2M times more irreducible loops than paired ones, it becomes the 25% reduction of the fourth moment.
- **Triads are coherent too,** but they enter only odd moments, which barely move the spectrum (m₃ changes by 1–2%). This is why the scorecard (f) shows a small triad effect.

#### The coupling and loop analyses for every start

The coupling and loop figures above (`coupling_mp_compare`, `coupling_mp_shoulder`, `coupling_phonon_gap`, `coupling_phonon_dispersion`, `loop_moments`, `loop_statistics`) run for every 2D start: Gaussian, URL, saturated RSA and HIP besides Poisson. With `--prefix <start>` (gau2d, url2d, rsa2d, hip2d) the figures are named `<start>_<figure>` and carry the start as a title; without it the names and the Poisson figures are unchanged (re-rendered pixel-identical). Every script takes the χ tags present in its root unless `--tags` is given.
- **Rectangular cells.** URL starts live in a 34.39 × 29.78 cell. The wave lattice is k = 2π(n_x/L_x, n_y/L_y), V = L_x L_y, with the integer reach ⌈K L_i/2π⌉ per axis (`batten2008_lattice.py`, NumPy only; sheared cells are rejected). Checks on URL ground states: ρ on the lattice matches the direct sum Σ_j exp(ik·r_j) to 6·10⁻¹³; S(k) ≤ 4·10⁻²³ for every lattice point inside K; the complex and Re/Im coupling spectra agree to 4·10⁻¹⁵ (relative); diag H = |k|²N/V to 10⁻¹⁵; C from ρ matches C from the gradients J to 2·10⁻¹⁴.
- **Square cells are unchanged.** Recomputing every Poisson χ = 0.1 cache (`mp`, `mp_shoulder`, `phonon`, `loops`, `loopstats`, `loopphase`) reproduces the stored values bitwise, or to ≤ 1.3·10⁻¹⁴ where LAPACK threading enters.
- **One SLURM CPU job per start** (`slurm/coupling_loops_cpu.slurm`) computes all caches and draws the six figures with the Poisson settings (20 flows for `mp`, 3 states for the shoulder, 6 for the phonon gap and the loop moments with 8 surrogates each, 4 for the loop statistics). With χ = 0.1 and 0.4 it takes 6–7 min on 16 cores, of which `mp` (20 starts and ends) takes 3 min and `loop_moments` about 2 min.

`batten2008_starts_compare_plot.py` writes `figures/coupling_starts_compare` from these caches, one color per start (as in `hessian_sign_starts`):
- (a) the relative width of the largest ω spacing in 0.3–1 K (median and 16–84% band over ground states; filled where it exceeds two uncoupled shell spacings);
- (b) the gap edges, drawn only where (a) finds a gap;
- (c) the irreducible share I_n/P_n at n = 4 and 12;
- (d, e) the four-wave and triad loop-phase bias 2M·c₁ of the ground states, against the envelope of the random-phase and Gaussian models of every start;
- (f) the Kolmogorov distance of λ(C) from MP with the measured variance, ground states and starts.

**First results (χ = 0.1 and 0.4).**
- **The loop share is universal.** I₄/P₄ = −0.009 to −0.052 at χ = 0.1 and −0.20 to −0.25 at χ = 0.4 for every start (Poisson −0.20); I₁₂/P₁₂ = −0.79 to −0.87 at χ = 0.4.
- **URL and saturated RSA open a frequency gap already at χ = 0.4.** The largest spacing is 0.732–0.790 (URL, Δω/ω = 0.072) and 0.740–0.781 (RSA, 0.056): median widths of 5.2 and 2.6 uncoupled shell spacings (0.011 and 0.016; per state 0.034–0.074 and 0.028–0.058); Poisson, Gaussian and HIP have none at χ = 0.4 (0.009–0.011). In the URL ground states the density in that window is 0.012, against 0.36 for the random-phase surrogates and 0.43 for the SCBA.
- **The four-wave bias tracks the gap.** 2M·c₁ (quadrilaterals) at χ = 0.4 is −6.5 (URL), −4.0 (RSA) and −3.1 to −3.3 (Poisson, Gaussian, HIP); the random models stay within ±0.25. The triad bias changes sign with the start: +24 to +28 (Poisson, Gaussian, HIP), −13 (RSA) and −43 (URL). URL keeps its Bragg-like S(k) peak at |k| ≈ 1.5 K, so its amplitudes are not Gaussian there (⟨I²⟩ = 2.48 against 2).
- **The starts differ, the ground states less.** At χ = 0.1 the λ(C) spectrum of saturated-RSA ground states lies within 0.85–1.15 (5–95%), so its dip window near 0.5 is empty and the peak ratio at 1 is meaningless. HIP starts are not random gradients: at χ = 0.4 their C has 42–134 zero modes and Var λ = 2.96 c, so MP(c_eff) is undefined there (D-018, D-019).

```bash
# per start (16 CPUs; SCRIPTS points at the checkout whose scripts to run; TAGS optional)
env -u JAX_PLATFORMS sbatch --export=ALL,ROOT=experiments/trajectories/hip2d/N1000,PREFIX=hip2d \
    experiments/slurm/coupling_loops_cpu.slurm      # likewise gau2d/N1000, url2d/N1024, rsa2d/N999
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/batten2008_starts_compare_plot.py   # coupling_starts_compare
```

**Data.** `trajectories/{gau2d/N1000,url2d/N1024,rsa2d/N999,hip2d/N1000}/{mp,mp_shoulder,phonon,loops,loopstats,loopphase}_chi<tag>.npz`.

**Figures.** `figures/{gau2d,url2d,rsa2d,hip2d}_{coupling_mp_compare,coupling_mp_shoulder,coupling_phonon_gap,coupling_phonon_dispersion,loop_moments,loop_statistics}` and `figures/coupling_starts_compare`.

#### All four χ for 2D Poisson

`hessian_sign_mechanism`, `hessian_negative_compare` and `hessian_plateau_mechanism` now include 2D Poisson flows at χ = 0.25 (tag 1-4) and χ = 0.45 (tag 9-20), with 100 flows each. Pass the extra tags with `--extra-chi 1-4 9-20`; the other starts have χ = 0.1 and 0.4 only.
- **Same structure at every χ:**
  - the stiff band holds 2M directions;
  - the soft ± branches shrink as √Φ;
  - the plateau is n₋ = (Nd − 2M)/2;
  - the cut crossing is near Φ ≈ 10⁻¹¹.
- **The stiff band's lower edge λ₂ₘ:** 0.95–1.0 k₁² at χ = 0.1, 0.25 and 0.4, but 0.22 k₁² at χ = 0.45. This is the soft edge of the coupling spectrum above.
- **Landscape figures (`--extra-chi 1-4 9-20`):** `landscape_wells`, `landscape_floor` and `landscape_valley` now include both new χ. The modes, winding and off-plane steps were run for them.
  - **χ = 0.45 has the slowest flow.** The residual's decay rate falls to 0.22 k₁², against 0.85–1.0 k₁² at χ ≤ 0.4 (valley b), and u^T K u / u^T D u falls to 0.005 (valley a).
  - **This is the soft edge of K setting the clock.** The same 0.22 k₁² appears as the stiff band's lower edge. The flow slows down as χ → 1/2.
  - **χ = 0.25 sits between χ = 0.1 and 0.4** on every measure.
- **The first-order agreement H ≈ PRP** still holds and falls as √Φ at every χ. Its prefactor grows with χ as the stiff gap closes: the largest deviation at Φ = 10⁻³ is 10⁻⁵, 8·10⁻⁵, 2·10⁻³ and 5·10⁻² of max |λ| for χ = 0.1, 0.25, 0.4 and 0.45.
- **Open: a small, systematic sign asymmetry at large χ.** The cut-free soft spectrum (P R P) has slightly more positive than negative eigenvalues in every flow, at every Φ:
  - (neg, pos) ≈ (200, 208) of 408 at χ = 0.4;
  - ≈ (88, 100) of 188 at χ = 0.45, where tr(PRP)/Σ|λ| = +0.13;
  - balanced at χ = 0.1 and 0.25.

  The Φ-independent bias suggests that the stiff part of R, tr((1 − P) R), is correlated with the residual. This has not been examined further.

#### The triad identity in detail

**Derivation.** With ρ_k = Σ_j exp(ik·r_j) and Φ = (1/N) Σ_{q∈C} |ρ_q|² (C the constrained set), the force on particle j is −∇_jΦ = −(1/N) Σ_q [iq ρ_q* e^{iq·r_j} − iq ρ_q e^{−iq·r_j}]. Along the flow, dρ_k/dt = Σ_j ik·ṙ_j e^{ik·r_j}. Substituting the force and using Σ_j e^{i(k±q)·r_j} = ρ_{k±q} gives

  dρ_k/dt = (1/N) Σ_{q∈C} (k·q) [ρ_q* ρ_{k+q} − ρ_q ρ_{k−q}].

No approximation enters. Φ is a sum of |ρ|², and the product of two plane waves is another plane wave, so every term couples three waves: k, q and k ± q.

**What it says.**
- *Linear decay.* The q = k term contains ρ₀ = N and gives −|k|²ρ_k. On its own, each wave would decay at rate |k|² (|ρ_k|² at 2|k|², hence τ₁ = 1/(2k₁²)). This term is a real multiple of ρ_k, so it never turns the phase.
- *Phase.* The phase rate θ̇_k = Im(ρ_k* ρ̇_k)/|ρ_k|² is made only of triad terms. Each is proportional to sin(θ_{k+q} − θ_k − θ_q) or sin(θ_k − θ_q − θ_{k−q}), the triad phase of three-wave interactions in fluids and nonlinear optics. Equivalently, θ̇_k = −∇θ_k·∇Φ. Wave k's own energy does not depend on its phase, so the phase moves only when the forces of the other waves push along it.
- *Near the ground-state manifold.* Constrained amplitudes are small and unconstrained ones are O(√N). The leading terms are those with one constrained factor ρ_q and one unconstrained partner ρ_{k±q}. The flow is then linear in the constrained amplitudes, ρ̇_k ≈ −Σ_q G_kq ρ_q (plus conjugate terms), with:
  - G_kk = |k|²;
  - G_kq = (k·q) ρ_{k−q}/N, which couples waves through the free density fluctuations.

  G is the Gram matrix of the gradients ∇ρ_k, the Hessian of Φ as seen by the constrained waves. Plane waves are not its eigenvectors, so they mix and their phases turn. G also changes as the particles move, which keeps the phases turning.
- *1D and 2D.* In 1D, k and q lie in 1..M, so k − q is always constrained (or zero); only k + q partners are free (50% of pairs at χ = 1/3). In 2D, 61% of k + q partners and 22% of k − q partners lie outside the exclusion region, so the free fluctuations couple through both.

**It holds in 2D for every start, to round-off.** The identity was checked on flow 0 of every ensemble at log₁₀Φ = 1, −1, −3, −6 and −9, against dρ_k/dt computed directly from −∇Φ. The table gives the worst relative error over the five levels, and the median share of the phase torque carried by triads with an unconstrained partner.

  | Ensemble | χ | max \|triad − exact\| / max \|exact\| | Unconstrained share at Φ = 10 / 0.1 / 10⁻³ |
  |---|---|---|---|
  | 1D Poisson | 1/3 · 2/3 | 3 × 10⁻¹² · 1 × 10⁻¹¹ | 0.89 / 1.00 / 1.00 · 0.93 / 0.99 / 0.99 |
  | 2D Poisson | 0.1 · 0.4 | 2 × 10⁻⁹ · 7 × 10⁻¹⁴ | 0.75 / 0.99 / 1.00 · 0.90 / 1.00 / 1.00 |
  | 2D Gaussian | 0.1 · 0.4 | 5 × 10⁻¹⁵ · 2 × 10⁻¹³ | 0.77 / 0.96 / 1.00 · 0.88 / 0.99 / 1.00 |
  | 2D URL | 0.1 · 0.4 | 7 × 10⁻¹⁵ · 3 × 10⁻¹³ | 0.78 / 0.95 / 1.00 · 0.87 / 0.99 / 1.00 |
  | 2D RSA | 0.1 · 0.4 | 6 × 10⁻¹⁵ · 2 × 10⁻¹³ | 0.55 / 0.96 / 1.00 · 0.89 / 0.99 / 1.00 |
  | 2D HIP | 0.1 · 0.4 | 3 × 10⁻⁹ · 6 × 10⁻¹⁴ | 0.93 / 0.99 / 1.00 · 0.95 / 1.00 / 1.00 |

  The two 10⁻⁹ entries occur at Φ = 10⁻⁹. There dρ/dt has itself shrunk about 10⁹-fold, and the float64 sum loses digits to cancellation. At Φ ≥ 0.1, every 2D entry is ≤ 3 × 10⁻¹³. By Φ = 10⁻⁶, even the least-coupled 2D wave has a share ≥ 0.96.

**Slaving: the coupling also sets the amplitudes.** Without coupling, every wave would decay at its own rate, −d ln|ρ_k|/dt = |k|². The table gives the measured ratio (−d ln|ρ_k|/dt)/|k|². Each entry is a median over all flows, for waves with S > 10⁻⁸, written long waves (|k| < 0.2K) / short waves (|k| > 0.6K).

  | Ensemble | χ = 0.1, Φ = 10 | χ = 0.1, Φ = 0.1 | χ = 0.4, Φ = 10 | χ = 0.4, Φ = 0.1 |
  |---|---|---|---|---|
  | Poisson | 0.96 / 0.21 | 0.95 / 0.01 | 0.72 / 0.02 | 0.11 / 0.00 |
  | Gaussian | 1.01 / 0.94 | 0.99 / 0.20 | 0.61 / 0.06 | 0.49 / 0.01 |
  | URL | 0.86 / 0.95 | 0.99 / 0.24 | 0.74 / 0.09 | 0.54 / 0.01 |
  | RSA | 0.99 / 1.00 | 1.00 / 0.05 | 0.95 / 0.10 | 0.41 / 0.01 |
  | HIP | 0.92 / 0.02 | 0.91 / 0.01 | 0.34 / 0.00 | 0.08 / 0.00 |

- *Long and short waves.* At χ = 0.1, long waves decay at their own rate. Short waves soon decay at only 0–24% of it, because the triad feed from slower waves nearly balances their decay: ρ_k ≈ (1/|k|²) Σ_{q≠k} G_kq ρ_q.
- *Slaving is the winding.* A slaved wave's phase is set by the phases of the slow waves and of the free fluctuations ρ_{k−q}. As the slow waves carry particles, those phases change and the slaved wave turns.
- *By start, at χ = 0.1.* The hyperuniform and RSA starts begin almost separable (short waves at 0.94–1.00 of their own rate at Φ = 10) and wind little. Poisson and especially HIP short waves are slaved from the start, and wind more.
- *At χ = 0.4.* Each wave has four times as many triad partners. Slaving reaches the long waves too (0.08–0.54 at Φ = 0.1), and every start winds more.
- **Why relaxed waves wind fast, and the numerical caveat.** |∇θ_k| = |k|/√(2S_k) to within 1% (measured), so a wave's phase becomes ever easier to turn as it relaxes. The alignment |cos α_k| is tiny, 10⁻⁴–10⁻³ (median), against 1/√(Nd) ≈ 0.02 for a random direction; the 1/√S amplification turns it into finite winding. The same amplification makes the instantaneous rate of a nearly cancelled wave sensitive to any small displacement. At a stored state, waves with S ≲ 10⁻¹⁰ show rates set by the integrator's error: following the exact flow from such a state, they fall 25–80-fold within 2 time units at fixed Φ. Instantaneous rates are therefore shown only for S > 10⁻⁸. Cumulative winding is robust. Rerunning flow 0 at atol = 10⁻¹¹ (from 10⁻⁸) and rtol = 10⁻³ (from 10⁻²), with 16–20× more steps, gives the same per-wave winding: 9.51 against 9.52 turns per wave (1D χ = 1/3), and 1.595 against 1.596 (2D Poisson χ = 0.4, per-wave correlation 1.000).
- **As a collective variable.** W_k is a path functional, not a function of the configuration: the energy is blind to each wave's phase, and two configurations at the same Φ can carry any winding. So winding cannot replace Φ as a coordinate on the landscape. It records how the flow got there:
  - It is monotone in time, like arclength.
  - It is distinct for different starts at the same Φ (`winding_total`).
  - Its short-wave value tracks the slip K·ℓ/2π, the phase a wave at K gains when particles move ℓ (`winding_cv` (g)).
  - The spiral number σ = Ω/(−½ d ln Φ/dt) is an instantaneous state function. Ω is the energy-weighted RMS winding rate, and σ is the ratio of iso-energetic (phase) motion to descent in collective-coordinate space. It marks the nonlinear stage. In 2D, σ ~ 0.1–1 above Φ ≈ 1 and falls to 10⁻³–10⁻² once Φ < 0.1, where the flow becomes nearly separable radial relaxation and winding stops. In 1D χ = 1/3 (not plotted), σ reaches about 14 near Φ = 0.1: the waves spiral more than they decay while the slow waves carry them.
- **How winding builds up** (`*_winding_kymograph`, `winding_total`). Each wave winds between its own relaxation fronts (the last times its S exceeds 10⁻² and 10⁻⁸) and somewhat after. Short waves wind first, while long waves are still carrying their particles; the total saturates once Φ ≲ 10⁻². Against path fraction (not plotted), 1D χ = 1/3 winds at a nearly constant rate per unit arclength all the way to s = 1. 1D χ = 2/3 stops at s ≈ 0.3, where the lattice locks in. Every start finishes winding by Φ ≈ 10⁻²–10⁻³. Against path fraction the 2D starts split as their turning does: Poisson and HIP wind steadily along the path, while Gaussian, URL and RSA at χ = 0.1 gain much of their winding in the last 10% of it. Against time all 2D starts have the same shape, with HIP leading by about a decade.

#### Why Ω, σ and |cos α| behave as they do (`winding_cv` (a), (b), (f))

`batten2008_windcv_explain_plot.py` writes `figures/winding_cv_explained`.

**The exact partition.** Write l_k = ln|ρ_k|, so that |ρ̇_k|² = |ρ_k|²(l̇_k² + θ̇_k²). The S-weighted mean square speed of the residual waves then splits exactly into three parts:

  ⟨l̇² + θ̇²⟩_S = ⟨l̇⟩_S² + Var_S(l̇) + Ω².

- **Coherent decay, ⟨l̇⟩_S²:** the waves shrink together. −⟨l̇⟩_S is the decay rate in the denominator of σ.
- **Differential decay, Var_S(l̇):** different waves shrink at different rates.
- **Rotation, Ω²:** the waves turn.

Over all constrained waves, the left side equals ⟨K²⟩_u and −⟨l̇⟩_S equals ⟨K⟩_u = −½ d ln Φ/dt. Here K = (2/V)JJᵀ is the wave-coupling matrix of the landscape section, and the identities hold to 4 digits in the data.

**Panels (a) and (b), Ω and σ.**
- **Rotation is a small share.** At Φ = 10 it is 2–3% of the residual speed at χ = 0.1 (HIP 1.5%, RSA 0.7%) and 21–28% at χ = 0.4. By Φ = 10⁻³ it has fallen to at most 4 × 10⁻⁵ (χ = 0.1) and to 2 × 10⁻⁴–3 × 10⁻² (χ = 0.4) (`winding_cv_explained` (a–c)). Differential decay takes 13–43% at Φ = 10 and vanishes with it.
- **σ is bounded by the stiffness spread.** Since Ω² ≤ ⟨K²⟩_u − ⟨K⟩_u², the spiral number obeys

  σ = Ω/⟨K⟩_u ≤ CV_u(K),

  the coefficient of variation of the collective stiffness over the residual (panel d; the grey region is forbidden).
- **Why σ falls.** Every flow and level lies on or below the bound. Its largest σ/CV_u(K) is 0.90–1.00 for every start except URL at χ = 0.1 (0.40). σ falls because the residual concentrates on the softest collective modes of K, so CV_u(K) → 0.
- **How close σ stays to the bound.** The median efficiency σ/CV_u(K) is 0.46–0.67 at χ = 0.4 and 0.02–0.55 at χ = 0.1. With independent wells, the stiffness spread shows up as differential decay rather than rotation (a diagonal K turns nothing).
- **The late plateau of Ω and σ.** At χ = 0.4 (and for HIP at χ = 0.1), σ settles near 0.6–0.85 of its bound (URL χ = 0.4: 0.27). This is beating between near-degenerate soft collective modes: the last spread left in the residual is mostly rotation.

**Panel (f), |cos α|.** It factorizes exactly as

  |cos α_k| = (|θ̇_k|/|k|) · √(S_k/Φ) / √⟨K⟩_u,

using |∇Φ|² = 2⟨K⟩_u Φ and |∇θ_k| = |k|/√(2S_k). The product matches the measured median to 3 digits (panel e).
- **Why it is tiny.** Early on, the factors are a normalized winding rate of order 10⁻¹ and an energy share √(S_k/Φ) of order 10⁻², since one wave holds about 1/M of Φ. 1/√⟨K⟩_u grows toward 1/k₁ as the residual moves to the slow modes.
- **Why it dips.** Both the rate and the share fall, to a minimum of about 10⁻⁵ near Φ ≈ 10⁻³.
- **Why it rises late, which is a selection effect.** winding_cv keeps only waves with S_k > 10⁻⁸, and that set empties: half the constrained waves are gone by Φ ≈ 10⁻¹–10⁻⁴·², and 99% by Φ ≈ 10⁻⁴·²⁵–10⁻⁷·⁵ (panel f). The survivors are the last-relaxing waves, which hold most of the remaining energy (√(S_k/Φ) → 0.7). So their |cos α| rises to 10⁻³–10⁻². Below Φ ≈ 10⁻⁴, the three panels describe only a handful of slow waves, not the constrained set.

#### Can single waves inside the exclusion region be excited independently?

`batten2008_exclusion.py` analyses the coupling matrix K = (2/V)JJᵀ at 5 ground states of every 2D start (NumPy; seconds). `batten2008_exclusion_plot.py` writes `figures/exclusion_mode_control`.

At a ground state, a displacement δr changes the constrained waves by δu = J δr. The energy is Φ = |u|²/V, gradient flow gives u̇ = −K u, and MD (mass 1) gives ü = −K u.

**The question.** The coherent decay of `winding_cv_explained` suggests that, near the manifold, waves in the exclusion region can be excited one at a time, so that S(k) could be raised selectively at chosen |k|. The analysis checks this in three parts.

**1. Statically, yes; it costs extra displacement at large χ (panel c).**
- *Why it is possible.* For χ < ½, J has full row rank 2M. So any target δu, including one wave alone, is reachable with the least displacement |δr|² = (2/V) δuᵀK⁻¹δu.
- *The energy cost is the same for every k.* Raising S(k) by s costs ΔΦ = s.
- *The displacement cost relative to an independent well is |k|²(K⁻¹)_kk.* Its median is:

  | χ | All waves | Long waves (\|k\| < 0.3K) | Marchenko–Pastur mean 1/(1 − 2M/Nd) |
  |---|---|---|---|
  | 0.1 | 1.01–1.20 | — | 1.25 |
  | 0.4 | 4.3–6.3 | 1.6–1.7 | 5.0 |

  At χ = 0.4 the cost rises with |k|.

**2. Coherent decay is not independence (panels a, b).** In the gradient flow, coherent decay means the residual's waves all shrink at one rate.
- *One stiffness band.* Below a start-dependent Φ (≈ 10 for HIP, 10⁻¹–10⁻³ for Poisson, RSA and Gaussian, ≈ 10⁻⁷ for URL), more than 95% of |u|² sits in collective modes within 10% of the residual's mean stiffness ⟨K⟩_u ≈ k₁².
- *The longest waves carry it.* The longest-wave shell |k| = k₁ holds at least 97% of Σ S(k).
- *Many modes, not one.* The single softest mode of K spans only 1–2 waves and holds 10–40% of the residual. The narrow band is made of the nearly degenerate long-wave modes.
- *Everything else is slaved.* All other waves decay at the same rate because they are slaved to the long waves (`landscape_floor`). This is the opposite of independent excitation.

**3. At fixed temperature every wave is equally excited, and excitations do not stay put (panels d–f).**
- *Equipartition.* In the harmonic canonical ensemble, ⟨u uᵀ⟩ = (V T/2) I. So ⟨S(k)⟩ = (2/V)(J H⁺ Jᵀ)_kk · T = T for every constrained k, exactly (to 10⁻¹² at every ground state and start). The coupling K changes the dynamics, not this static distribution. Making S(k) selective at fixed T needs a weighted potential, ⟨S(k)⟩ = T/v(k), or a non-equilibrium drive.
- *A one-wave excitation leaks.* Under the flow, the share of it still in its own wave after a time |k|²t = 1 is 0.90–0.99 at χ = 0.1 and 0.73–0.74 at χ = 0.4. By |k|²t = 3 it is 0.44–0.93 at χ = 0.1 and 0.37–0.39 at χ = 0.4 (panel e).
- *In MD, one wave rings at many frequencies.* A single wave's fluctuation is spread over the collective frequencies √λ(K). At χ = 0.1 it is a narrow line at ω ≈ |k|: 9–56 modes, spread 0.11–0.20 of the mean. At χ = 0.4 it is a broad Marchenko–Pastur band: about 370 modes, spread 0.42, mean 0.92|k| (panel f).
- *What this means for the MD winding experiment.* At χ = 0.4 one wave's phase is driven by hundreds of collective modes; at χ = 0.1 it is close to a single oscillator.

**Verdict.**
- *Correct:* the constrained waves are independent coordinates, so S(k) can be prepared selectively at small Φ, at a displacement cost that grows as 1/(1 − 2χ).
- *Not correct, (i):* the coherent decay does not show independent excitation. It shows that everything left is slaved to the longest waves.
- *Not correct, (ii):* at fixed low T such excitations are not stable or selective. Equipartition sets S(k) = T for every k, and the coupling K moves a single-wave excitation into other waves on its own relaxation time.

**Drives that select one wave (`figures/exclusion_drive`; `batten2008_exclusion_spectra_plot.py`).** These are computed from the eigensystem of K at ground state 0 of every start, in the harmonic regime. Selectivity is the share of the response in the driven wave.

| Drive | Physics | χ = 0.1 | χ = 0.4 |
|---|---|---|---|
| Static field −ε cos(k·r − φ) | an equilibrium tilt: minimizing \|u\|²/V − ε(cos φ a_k + sin φ b_k) gives u* = (Vε/2)(cos φ, sin φ) on wave k only (J has full rank), so ⟨S(k)⟩ = T + (Vε/2)²/N and every other wave stays at T, to first order | 1 (exact) | 1 (exact) |
| Travelling field ε cos(k·r − Ωt), Langevin friction γ = 0.05\|k\|, Ω = \|k\|/2 | steady response û ∝ (K − Ω² − iγΩ)⁻¹ K (e_a + i e_b) | 0.72–0.99 | 0.25–0.28 |
| The same, on resonance Ω = \|k\| | — | 0.23–0.80 | 0.11–0.17 |
| The same, γ = 0.5\|k\|, Ω = \|k\| (2D Poisson) | — | ≈ 0.8 | ≈ 0.5 |
| Heat bath acting only on wave k (overdamped) | excess covariance from K Σ + Σ K = 2D | 0.93–0.996 | 0.75–0.76 |

**What the table means.**
- *The static field is perfectly selective* to first order. It is not a drive: it changes the potential, shifting the mean ⟨ρ_k⟩ while the fluctuations stay at T.
- *Periodic fields trade energy transfer against selectivity.* Selectivity is 1 at Ω → 0 and lowest near resonance, Ω ≈ |k|. There the field pumps every collective mode of that frequency, and those modes span many waves at χ = 0.4.
- *Friction helps.* Strong friction suppresses those resonances (thin lines in panel a), at the cost of continuous dissipation.
- *A heated wave shares its excess heat.* About 25% goes to other waves at χ = 0.4.

**How one excited wave spreads over modes and waves (`figures/exclusion_mode_spectra`).** This extends panel (f) of `exclusion_mode_control` to every constrained wave.
- *χ = 0.1.* Long waves sit on a single line ω = |k|: their excitation stays in one collective mode and its energy stays in that wave (panels a–c). From |k| ≈ 0.4K up, the frequency band broadens to about 50 modes, and the energy spreads to waves of similar |q|.
- *χ = 0.4.* Every wave's weight spreads across the whole Marchenko–Pastur band, from ω ≈ 0.1K to 1.3K, over 10–450 modes (panel d). Over long times its energy spreads over essentially all waves: the median self-share is 0.004 at χ = 0.4, against 0.03 at χ = 0.1.
- *The frequency moments.* The mean frequency stays at 0.9–1.0|k|, and the spread is 0.05–0.24|k| at χ = 0.1 against 0.25–0.5|k| at χ = 0.4 (panel h).

### Other initial conditions: Gaussian, URL, saturated RSA and HIP

The full 2D analysis set (flows, slices, mismatch, modes, tail, Hessians, Wasserstein, valley floors, off-plane, winding and all figures) was repeated from four more starts at χ = 0.1 and 0.4. Every 2D start has 100 flows per χ except Gaussian, which uses all 50 configurations that exist. All 900 2D flows reach Φ < 10⁻²⁰. The starts:

- **Gaussian pair statistics** (`GAU-2D-N1000`; hyperuniform, S ≈ k²/4π at small k);
- **uniformly randomized triangular lattice** (`URL-2D-N1024`; the group's patterns, rectangular cell 32 × 16√3, rescaled to density 1);
- **saturated RSA** (`RSA-2D-N999`). `saturated_rsa.py` is a NumPy port of the group's voxel algorithm (`CONFIGURATIONS/SATURATED_RSA/saturated_rsa.py`). One correction: a voxel is dropped only when it lies inside an exclusion disk (radius D), not the particle radius D/2, so refinement runs to true saturation. Saturation fixes the packing fraction, not N. A pool of 2,080 patterns (N = 969–1026) was therefore generated, and the first 100 of the 124 with the most common N (999) kept; φ = 0.5467. A unit test checks no overlap and no room for another disk.
- **hyperplane (line) intersection process** (`HIP-2D-N1000`; antihyperuniform, S(k) → ∞ as k → 0). `hip.py` is a vectorized port of the group's `CONFIGURATIONS/HYPERPLANE_INTERSECTION/hpp.py` (Klatt et al., Nat. Commun. 10, 811 (2019)). It draws a Poisson number of isotropic lines through the window and keeps their pairwise intersections, at unit intensity in a √1000 window. N varies by ±260, so samples were drawn until 100 had N = 1000 exactly (67,184 drawn). The window is used as a periodic cell (lines end at its edge), as in the group's HPP-2D pack. HIP flows are the slowest: 3,485–4,884 steps at χ = 0.4.

Rectangular cells required one generalization: every script now converts positions with per-axis cell lengths (`cell_lengths`). The valley-floor fiber holds the Cartesian plane fixed in fractional coordinates (`fractional_plane`). Square-cell results are unchanged.

Results that differ between starts (medians over flows):

| | Poisson | Gaussian | URL | RSA | HIP |
|---|---|---|---|---|---|
| per-particle displacement, χ = 0.1 / 0.4 | 0.69 / 0.80 | 0.18 / 0.37 | 0.16 / 0.32 | 0.16 / 0.25 | 2.15 / 2.20 |
| k₁ share of \|Δr\|², χ = 0.1 | 0.19 | 0.015 | 0.003 | 0.18 | 0.38 |
| hump ratio, χ = 0.1 / 0.4 | 0.98 / 1.04 | 0.52 / 0.76 | 0.51 / 0.65 | 0.80 / 0.97 | 1.20 / 1.16 |
| short-wave winding, χ = 0.1 / 0.4 | 0.87 / 1.83 | 0.79 / 1.13 | 0.85 / 1.12 | 0.75 / 0.97 | 1.42 / 3.88 |
| W₂ geodesic max Φ, χ = 0.1 / 0.4 | 211 / 814 | 38 / 413 | 33 / 397 | 13 / 89 | 958 / 2,370 |

Shared by every start:
- Exactly Nd − 2M zero modes at the end.
- About (Nd − 2M)/2 negative directions near the manifold (812/201, 795/196, 811/201, 796/199, 798/194 for Poisson, Gaussian, URL, RSA, HIP), fading as √Φ.
- Zero valley floors on both planes.
- KL spectra with 82–92% in the bow.
- A per-particle participation ratio of 0.43–0.51 (a collective Gaussian field).

```bash
# initial conditions
uv run --no-sync python -I experiments/scripts/make_initial_conditions.py --ensembles url \
    --url-source <CONFIGURATIONS>/TRIANGULAR_URL/TRIANGULAR_URL-N_1024-rho_1.0/TRIANGULAR_URL-N_1024-rho_1.0
for k in "480 20261107 000" "1600 20261587 001"; do set -- $k   # 2,080 saturated RSA patterns
  sbatch --cpus-per-task=32 --time=01:00:00 --wrap "uv run --no-sync python -I experiments/scripts/saturated_rsa.py \
      --n-configs $1 --seed $2 --workers 32 --shard experiments/initial_conditions/rsa_pool/pool_$3.npz"
done
uv run --no-sync python -I experiments/scripts/saturated_rsa.py --keep 100 \
    --assemble experiments/initial_conditions/rsa_pool/pool_000.npz experiments/initial_conditions/rsa_pool/pool_001.npz
uv run --no-sync python -I experiments/scripts/hip.py --keep 100
# flows (100 per chi; --n-traj 50 for GAU-2D-N1000), then per start the batten2008_cc_gpu.slurm steps:
# slices, modes, tail, mismatch, wasserstein (one job per chi), hessian (one per chi), winding, floor
for c in 0.1 0.4; do
  sbatch --time=06:00:00 --export=ALL,CHI=$c,FLOW_ARGS="--ic experiments/initial_conditions/HIP-2D-N1000 \
      --out experiments/trajectories/hip2d/N1000 --n-traj 100 --wall-budget 20000" experiments/slurm/flows2d_gpu.slurm
done   # likewise POS-2D-N1024 -> flows2d/N1024, GAU-2D-N1000 -> gau2d/N1000, URL -> url2d/N1024, RSA -> rsa2d/N999
sbatch --time=04:00:00 --export=ALL,STEP=winding,STEP_ARGS="--root experiments/trajectories/hip2d --sizes 1000 --chi 0.1 0.4" \
    experiments/slurm/batten2008_cc_gpu.slurm
uv run --no-sync python -I experiments/scripts/batten2008_offplane.py --root experiments/trajectories/hip2d --sizes 1000 --chi 0.1 0.4
# figures per start (--prefix flows2d / gau2d / url2d / rsa2d / hip2d), then the comparisons
uv run --no-sync python -I experiments/scripts/batten2008_windcv_plot.py --root experiments/trajectories/hip2d --n 1000 \
    --chi 0.1 0.4 --prefix hip2d --dim 2                                    # winding kymograph
uv run --no-sync python -I experiments/scripts/batten2008_offplane_plot.py --compare <offplane_chi*.npz, two per system> --labels ...
uv run --no-sync python -I experiments/scripts/batten2008_kl_plot.py --roots experiments/trajectories/{flows2d/N1024,gau2d/N1000,url2d/N1024,rsa2d/N999,hip2d/N1000} \
    --labels "2D Poisson" "2D Gaussian" "2D URL" "2D RSA" "2D HIP"             # offplane_kl_compare, offplane_kl_origin
uv run --no-sync python -I experiments/scripts/batten2008_plateau.py --roots experiments/trajectories/{flows2d/N1024,...} --flows 10   # CPU, ~2 min per chi
uv run --no-sync python -I experiments/scripts/batten2008_plateau_plot.py --roots ... --labels ...   # hessian_negative_compare, hessian_plateau_mechanism
uv run --no-sync python -I experiments/scripts/batten2008_planes_mean_plot.py --root experiments/trajectories/flows2d/N1024 --prefix flows2d_N1024   # per start
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/batten2008_hessian_sign_plot.py --roots experiments/trajectories/{flows2d/N1024,...} --labels ...   # hessian_sign_compare, hessian_sign_mechanism, hessian_sign_starts
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/batten2008_sign_universal_plot.py --roots experiments/trajectories/{flows2d/N1024,...} --labels ...   # hessian_sign_universal
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/batten2008_mp_plot.py   # coupling_mp_compare (computes and caches mp_chi<tag>.npz; ~10 CPU-min, use a SLURM CPU job)
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/batten2008_mp_shoulder_plot.py   # coupling_mp_shoulder (caches mp_shoulder_chi<tag>.npz; ~2 CPU-min)
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/batten2008_phonon_gap_plot.py   # coupling_phonon_gap (caches phonon_chi<tag>.npz)
sbatch --cpus-per-task=16 --mem=48G --wrap ".../python -I experiments/scripts/batten2008_loop_moments.py --roots experiments/trajectories/flows2d/N1024 --tags 1-10 1-4 7-20 2-5 9-20 49-100 --states 6 --surrogates 8"   # and N4096: --tags 9-20 49-100 --states 4 --surrogates 2 (32 CPUs, 96G)
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/batten2008_loop_moments_plot.py --roots experiments/trajectories/flows2d/N1024 experiments/trajectories/flows2d/N4096   # loop_moments
# new chi for 2D Poisson (flows, tail, hessian, plateau, landscape, modes, winding, offplane): pass --chi 0.25 0.45 to each step; plots take --extra-chi 1-4 9-20
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/batten2008_windcv_explain_plot.py --roots ... --labels ...   # winding_cv_explained (needs landscape_chi*.npz)
uv run --no-sync python -I experiments/scripts/batten2008_exclusion.py --roots experiments/trajectories/{flows2d/N1024,...} --flows 5   # exclusion_chi*.npz
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/batten2008_exclusion_plot.py --roots ... --labels ...   # exclusion_mode_control
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/batten2008_exclusion_spectra_plot.py --roots ... --labels ...   # exclusion_mode_spectra, exclusion_drive
uv run --no-sync python -I experiments/scripts/batten2008_landscape.py --roots experiments/trajectories/{flows2d/N1024,...} --flows 10   # CPU, 2-3 min per start
uv run --no-sync python -I experiments/scripts/batten2008_landscape_plot.py --roots ... --labels ...   # landscape_wells, landscape_floor, landscape_valley
uv run --no-sync python -I experiments/scripts/batten2008_winding_plot.py --roots experiments/trajectories/{flows2d/N1024,gau2d/N1000,url2d/N1024,rsa2d/N999,hip2d/N1000} \
    --labels "2D Poisson" "2D Gaussian" "2D URL" "2D RSA" "2D HIP"
uv run --no-sync python -I experiments/scripts/batten2008_windcv_plot.py --compare <winding_chi*.npz, two per system> --labels ...
uv run --no-sync python -I experiments/scripts/batten2008_windcv_plot.py --total --compare-name winding_total --compare ... --labels ...
```

**Data.** `trajectories/{gau2d/N1000,url2d/N1024,rsa2d/N999,hip2d/N1000}/` with the same files as `flows2d/N1024`, plus `winding_chi<tag>.npz`, `plateau_chi<tag>.npz` and `landscape_chi<tag>.npz` everywhere. The 20-flow runs are kept in `trajectories/<start>/N<N>_n20/`. The tolerance checks are in `trajectories/{batten2008_tight/N1000,flows2d_tight/N1024}/` (flow 0, atol 10⁻¹¹, rtol 10⁻³).

**Figures.** `figures/{flows2d_N1024,gau2d_N1000,url2d_N1024,rsa2d_N999,hip2d_N1000}_*` (including `*_winding_kymograph`), `figures/batten2008_N1000_winding_kymograph`, `figures/*_mode_planes_mean`, and the comparisons `figures/offplane_lopsided`, `offplane_kl_compare`, `offplane_kl_origin`, `hessian_negative_compare`, `hessian_plateau_mechanism`, `hessian_sign_compare`, `hessian_sign_mechanism`, `hessian_sign_starts`, `hessian_sign_universal`, `coupling_mp_compare`, `coupling_mp_shoulder`, `coupling_phonon_gap`, `coupling_phonon_dispersion`, `coupling_mp_finite_size`, `coupling_starts_compare`, `loop_moments`, `loop_statistics`, `winding_cv_explained`, `exclusion_mode_control`, `exclusion_mode_spectra`, `exclusion_drive`, `landscape_wells`, `landscape_floor`, `landscape_valley`, `winding_compare`, `winding_cv` and `winding_total`. All figures are PNG only.

### The 1D landscape against χ, in 3D

`batten2008_landscape3d.py` (`figures/batten2008_N6_landscape3d`) stacks the 1D landscape Φ(x₂, x₃) at heights z = χ, with Blues filled contours on a shared scale and no trajectories. N = 3 allows only χ = 1/3 and 2/3, so the figure uses N = 6 (χ = 1/6 to 5/6, M = 1–5) and slices its five-dimensional landscape at x₁ = 0 with x₄, x₅, x₆ on their lattice sites. The landscape's corrugation and depth grow with χ: Φ ≤ 2.3 at χ = 1/6 and up to 6.5 at χ = 5/6.

## Low-temperature MD around stealthy ground states: does arg ρ_k keep winding?

We ran molecular dynamics at fixed low temperature around the 2D stealthy ground states of the five starts above (Poisson, Gaussian, URL, saturated RSA, HIP), at χ = 0.1 and 0.4. The question was whether the phase of each collective density ρ_k keeps winding at fixed T or becomes stationary, and whether its rate depends on the start.

**Setup.** The MD library is `xla_cco.dynamics` (`python -m xla_cco.cli.md`). We took 10 ground states per (start, χ) and ran each at T ∈ {1e-6, 1e-5, 1e-4, 1e-3}. Each lane goes through:
1. a harmonic start;
2. count-mode Andersen thermalization (p = 0.01);
3. two cycles of [Andersen 10⁵ steps, COM removal, NVE 10⁶ steps], with the winding of every wave with |k| < 2K accumulated on the device.

There are two additional sets:
- **E1b:** ρ_k traces of 2 ground states at T = 1e-6, for spectra.
- **E4:** a BAOAB Langevin subset (flows2d and hip2d, γ = 0.003 and 0.03).

The hypotheses H1–H8 were pre-registered in `md_winding/HYPOTHESES.md` (Amendments A1–A4). `md_winding/REPORT.md` holds the verdicts, the answers and every number; `md_winding/BUDGET.md` is the GPU ledger (24.48 of 100 GPU-h, of which 4.71 for the INCONCLUSIVE-remedy reruns).

**Results.**
- **Winding persists.** A constrained wave (|k| < K) winds at the Rice rate |k|/2π turns per unit time, whatever T, χ and the start. The normalized rate has per-start medians 0.997–1.0025, i.e. about 9.6 × 10³ turns per NVE segment. The rate over [t/2, t] matches the rate over [t/4, t/2] within 0.44%.
- **No start dependence.** The unsigned rate does not depend on the start (mean η²_H = −0.010, medians within 0.53%).
- **Unconstrained waves** (K < |k| < 2K) wind at |k|√(T/S)/2π, the circular-Gaussian Rice rate. Drift along the ground-state manifold randomizes their ρ_k. The drift is diffusive with MSD ∝ √T t, so their net-winding rate scales as T^{1/4}.
- **Net winding** has zero mean. Its RMS is 0.9% (χ = 0.4) to 3.5% (χ = 0.1) of the unsigned winding. It is diffusive at χ = 0.4, but superdiffusive under NVE at χ = 0.1 (α = 1.18–1.28), and weak Langevin friction removes that (α = 1.01–1.02 at γ = 0.03).
- **Registered verdicts:**
  - PASS: H5, H6b.
  - INCONCLUSIVE: H3, H7 (|z| > 3 at deviations ≤ 0.44%). Both remain INCONCLUSIVE after the decision-table remedy: one cell each rerun at 2× cycles (`batten2008_md_remedy.py`, deviations ≤ 0.32%).
  - FAIL: H1, H2, H8 (few-wave |k| bins), H4 (NVE superdiffusion) and H6 (diffusive rather than ballistic drift).

  A post hoc pooled-bin analysis and descriptive diagnostics are kept separate from the verdicts.

```bash
# MD production (GPU; one job per cell, job names mdw_*); STEP = run (E1), spectra (E1b), langevin (E4, with --gamma)
env -u JAX_PLATFORMS sbatch -J mdw_e1_flows2d_1-10 --time=03:00:00 \
    --export=ALL,STEP=run,STEP_ARGS="--system flows2d --chi 1-10" experiments/slurm/batten2008_md_gpu.slurm
# final sacct cost into each output JSON (no GPU)
uv run --no-sync python experiments/scripts/batten2008_md_run.py --step cost --system flows2d --chi 1-10
# E2 + E3 on a CPU node: per-file analysis, H1-H8 verdicts, post hoc diagnostics, figures F1-F10 and X1
mkdir -p experiments/trajectories
env -u JAX_PLATFORMS sbatch --export=ALL,PAR=4 experiments/slurm/batten2008_md_analysis_cpu.slurm
# or step by step
uv run --no-sync python experiments/scripts/batten2008_md_analysis.py \
    --md $(ls experiments/trajectories/*/N*/md_*chi*.npz | grep -v -e ckpt -e md_analysis) \
    --out-dir experiments/trajectories/md_results          # -> md_results/<start>/N<N>/md_analysis*_chi<tag>.npz
uv run --no-sync python experiments/scripts/batten2008_md_stats.py --analysis experiments/trajectories/md_results \
    --out-json experiments/trajectories/md_results/verdicts.json
uv run --no-sync python -I experiments/scripts/batten2008_md_diagnostics.py --results experiments/trajectories/md_results
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/batten2008_md_plot.py \
    --root experiments/trajectories/md_results --out experiments/figures --exploratory
```

**Data.**
- `trajectories/<start>/N<N>/`, per χ tag:
  - `md_chi<tag>.{npz,json}` (E1, 40 lanes);
  - `md_spectra_chi<tag>.{npz,json}` (E1b);
  - `md_langevin_g<γ>_chi<tag>.{npz,json}` (E4; flows2d and hip2d only);
  - `*.jobs.json` (SLURM jobs) and `*.ckpt.npz` (checkpoints).
- `trajectories/md_results/`:
  - per-cell analyses `<start>/N<N>/md_analysis*_chi<tag>.npz`;
  - `verdicts.{json,md}`, with the exploratory pooled bins as a separate section;
  - `diagnostics.{json,md}`.
- `trajectories/md_rerun/` (INCONCLUSIVE remedy; `md_winding/REPORT.md` §3 has the commands):
  - `flows2d/N1024/md_chi2-5.*`, `md_langevin_g<γ>_chi1-10.*`: the 2×-cycles reruns (`--cycles 4`);
  - `results/`: their analyses and `remedy.{json,md}` (`batten2008_md_remedy.py`).

**Figures.**
- `figures/md_F1_kymograph_chi<tag>`: winding kymographs.
- `md_F2_rates`: rates against the Rice lines.
- `md_F3_stationarity`: windowed-rate ratios.
- `md_F4_net_msd`: net-winding MSD and exponent α.
- `md_F5_starts`: start dependence.
- `md_F6_temperature`: T dependence.
- `md_F7_equipartition`: equipartition.
- `md_F8_drift`: manifold drift.
- `md_F9_thermostat`: BAOAB against Andersen + NVE.
- `md_F10_spectra`: ρ_k spectra against √eig K.
- `md_X1_pooled_posthoc_chi<tag>`: post hoc pooled bins.

All figures are PNG only. The docstring of `batten2008_md_plot.py` lists the panels.
