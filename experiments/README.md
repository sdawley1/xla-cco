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
