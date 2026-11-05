# Library evaluation for a JAX collective-coordinate optimization (CCO) library

Status as of 2026-11-05 (JAX 0.12). Sources: the C++ reference at `~/ccoptimization/cco/`, Dawley & Torquato, PRE 112, 045310 (2025), the Equinox README, and the lockwo/awesome-jax README.

## Scope assumptions

- **Precision:** float64 only (`jax_enable_x64`). The target is the stealthy energy Φ < 1e-22, as in Dawley & Torquato. Quad-double is out of scope.
- **Sizes:** d = 1–3, N ≤ ~10⁴. There are M ≈ χ·d·(N−1) independent half-space k-vectors. Most workloads are many small systems at once, e.g. d = 3 with N = 100, or d = 2 with N ~ 10³.
- **Physics:** pure stealthy first, written so a soft-core repulsion term can be added. The C++ soft-core term is the harmonic overlap ε(1 − r/σ)² for r < σ.
- **ML:** equivariant GNNs, generative models and sampling, and manifold learning or optimal transport (OT) between configurations.
- **Hardware:** Della A100/H100 GPUs, which have full-rate FP64.

## C++ feature → JAX approach

| C++ (`cco/`) | JAX approach |
|---|---|
| Φ = (1/V) Σ_k v(k)\|ρ(k)\|², ρ(k) recomputed in O(MN) each call (`CollectiveCoordinatePotential.h:99`) | Direct sum: an `exp(i k·r)` matrix reduced with `jnp.einsum`, chunked with `lax.map` when N is large. Batch many configurations with `vmap` + `jit`. |
| `GetKs`: reciprocal-lattice half-space with \|k\| < K, k = 0 dropped (`StructureFactor.cpp:49`) | Plain NumPy, computed once and passed in as a static array. It must handle shells at \|k\| = K explicitly; the C++ strict `<` makes χ depend on rounding. |
| NLopt L-BFGS (`m = 100`, no ftol/xtol, three rescale passes); GSL steepest descent and CG; MINOP trust region | Optimistix `LBFGS`, `BFGS`, `NonlinearCG`, `GradientDescent`, `Dogleg`, `LevenbergMarquardt`; Optax `lbfgs` as an alternative. |
| Dawley & Torquato: normalized steepest descent, step doubled on success and halved on failure, recorded as a trajectory | A small custom `lax.while_loop` (an Optimistix `AbstractMinimiser` subclass). Record states with `lax.scan` or a fixed-size buffer. |
| Soft-core repulsion through cell lists, OpenMP | jax-md `space.periodic(_general)` + `partition.neighbor_list`. Do **not** port the C++ force, which double-counts the CC force (`RepulsiveCCPotential.h:110-118`); compute gradients with `jax.grad` instead. |
| MD: velocity Verlet with an Andersen thermostat; unused MC, Wang–Landau, parallel tempering, SGLD, basin hopping | jax-md `simulate` (NVE, NVT Nosé–Hoover, Langevin, FIRE); BlackJAX for MCMC. |
| No Hessian, saddle, or basin analysis | `jax.hessian` or `jax.jacfwd` (dense), and HVPs from `jax.jvp(jax.grad)` with matfree Lanczos (large N). See the notes below. |
| `.ConfigPack` binary I/O; g₂, S(k), H_p analysis | A NumPy reader and writer for ConfigPack. Analysis in pure JAX. |

## Recommended stack

| Role | Library | Verdict |
|---|---|---|
| Core | **jax** (x64), **equinox** | Equinox Modules are PyTrees, so a potential's parameters (K, v(k), σ, ε) can cross `jit`/`grad`/`vmap` boundaries. Use it as the base for every potential and model. |
| Minimization | **optimistix** | Provides L-BFGS, BFGS, nonlinear CG, trust-region (dogleg), Gauss–Newton, and Levenberg–Marquardt. Built on Equinox, works under `vmap`, and supports implicit differentiation through the solution. It is the primary minimizer library. |
| Minimization | **optax** | Its first-order and L-BFGS transforms are good for custom loops. Optimistix can wrap them (`optx.OptaxMinimiser`). |
| Linear algebra | **lineax** | The linear solves inside Gauss–Newton/LM (QR, SVD, CG on normal equations), with operator abstractions for Hessian/Jacobian operators. |
| Gradient flow | **diffrax** | Solves the steepest-descent ODE dr/dt = −∇Φ with adaptive or stiff solvers and `SaveAt`. This gives true steepest-descent paths for studying landscape topography. Also handles Langevin SDEs. |
| Hessian spectra | **matfree** | Lanczos/Arnoldi eigenproblems, stochastic Lanczos quadrature and trace estimation from matrix-free HVPs. Needed once dN ≳ 5×10³, where dense `eigh` gets slow. |
| Particles | **jax-md** | Periodic displacement functions, neighbor lists, FIRE, and MD integrators. Actively maintained. Use it for the soft-core term and dynamics, not for the stealthy k-space term. |
| Sampling | **blackjax** | HMC, MALA and NUTS, plus SMC/tempering, to sample the landscape at low temperature or near the ground-state manifold. |
| Generative | **flowjax** (+ distreqx) | Normalizing flows built on Equinox, for learning densities over manifold samples. |
| Equivariant GNN | **e3nn-jax**; nequip-eqx as a reference | Provides O(3) irreps and tensor products. For 2D, use E(2)-invariant edge features with plain Equinox message passing (`jax.ops.segment_sum`). Build edges from jax-md neighbor lists (periodic images). |
| OT / distances | **ott-jax** | Sinkhorn/Wasserstein between configurations with a custom periodic-image cost. Can serve as a differentiable version of the Dawley & Torquato proximity metric. Use exact assignment (`scipy.optimize.linear_sum_assignment`) for evaluation. |
| Manifold learning | **scikit-learn**, **umap-learn** (CPU, non-JAX) | For embedding manifold samples or descriptors; small enough to stay off the GPU. |
| Infra | **jaxtyping** (+ beartype), **chex**, **pytest** + **hypothesis**, **orbax** or `h5py` | Shape/dtype checking, JAX-aware test assertions, property-based tests, and checkpointing (orbax for models; HDF5/NPZ for configuration ensembles). |

## Evaluated, not adopted

- **jaxopt**: its README says it is no longer maintained; its L-BFGS was ported to Optax.
- **jraph**: archived in 2024. Write message passing directly in Equinox instead.
- **Flax (NNX) / Haiku**: Flax works, but Equinox keeps the physics, the solvers (Optimistix, Lineax, Diffrax) and the ML in one PyTree model. Haiku is inactive.
- **jax-finufft**: its error is relative to Σ|c_j| = N, so ε ≈ 1e-14 leaves |ρ(k)| errors of ~1e-10. That is too coarse for a Φ < 1e-22 stopping criterion. At most it is useful as a coarse first stage if N grows beyond 10⁴.
- **cola**: matrix-free linear algebra, but last pushed in 2025-08. matfree covers the same need and is actively maintained.
- **kfac-jax, laplax, traceax**: curvature tools aimed at neural-network training; matfree is the general-purpose option. traceax is small and Lineax-based; keep it in mind only for stochastic traces.
- **mpi4jax / JAXMg**: not needed at N ≤ 10⁴. `vmap` and single-GPU batching cover the throughput.
- **TDA (ripser.py, gudhi, giotto-ph)**: not selected. ripser and gudhi are active CPU libraries if persistent homology is wanted later; giotto-ph has not been updated since 2024.

## Design notes that affect library choice

1. **The stealthy energy is a sum of squares.** Φ = ‖r‖² with residuals r = √(v(k)/V)·[Re ρ(k), Im ρ(k)]. The soft-core term is also a sum of squares: √ε(1 − r_ij/σ) for overlapping pairs. So Optimistix Gauss–Newton/LM apply directly, and should converge quadratically on this zero-residual problem. Caveat: different minimizers land on different points of the degenerate manifold. Trajectory-dependent sampling studies must fix the minimizer, as Dawley & Torquato did.
2. **Hessian at a ground state.** H = 2JᵀJ, with J the 2M × dN Jacobian of r. The tangent dimension of the ground-state manifold is dN − rank J (generically d(N−1) − 2M + d, counting the d translations). Compute the rank with an SVD of J (dense) or with matfree on JᵀJ products, rather than forming H.
3. **Batch first.** The main workload is 10⁴ configurations per (χ, φ) cell at N = 100. Write every kernel for a single configuration and get batches from `vmap`. Use `jit` with static k-vector sets. Different χ values need different M, so pad and mask by M or recompile per χ.
4. **float64 throughout.** Set `jax.config.update("jax_enable_x64", True)` at import. Add a test that asserts every array is float64; one silent float32 cast caps Φ at ~1e-12.

## Gaps (to build in-house)

- Saddle search (dimer or eigenvector following), basin assignment, and landscape connectivity graphs: no JAX library covers these. Build them from Optimistix, matfree and Diffrax.
- k-vector enumeration for general (non-cubic) cells and χ ↔ K conversion.
- ConfigPack I/O compatible with the existing C++ ensembles.
- Initial-condition generators: RSA, URL, HIP, equiluminous and Poisson patterns.
