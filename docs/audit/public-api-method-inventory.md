# Numra public API and numerical method inventory

**Source of truth:** crate `lib.rs` `pub use` / `pub mod` and the facade `[numra/src/lib.rs](../../numra/src/lib.rs)`.  
**Workspace members:** root `[Cargo.toml](../../Cargo.toml)`.  
**Version:** workspace `0.1.0`.

This document lists **public numerical methods and primary APIs** intended for users. Internal helpers and private modules are out of scope.

Legend: **Import** = typical path via facade `numra::…` unless only subcrate path exists.

---

## Facade `numra`


| Module alias         | Crate             | Notes                                                        |
| -------------------- | ----------------- | ------------------------------------------------------------ |
| `numra::core` traits | `numra-core`      | Re-exports `Scalar`, `Vector`, `Signal`, errors, uncertainty |
| `numra::linalg`      | `numra-linalg`    |                                                              |
| `numra::nonlinear`   | `numra-nonlinear` |                                                              |
| `numra::ode`         | `numra-ode`       |                                                              |
| `numra::sde`         | `numra-sde`       |                                                              |
| `numra::dde`         | `numra-dde`       |                                                              |
| `numra::fde`         | `numra-fde`       |                                                              |
| `numra::ide`         | `numra-ide`       |                                                              |
| `numra::pde`         | `numra-pde`       |                                                              |
| `numra::spde`        | `numra-spde`      |                                                              |
| `numra::autodiff`    | `numra-autodiff`  |                                                              |
| `numra::optim`       | `numra-optim`     |                                                              |
| `numra::ocp`         | `numra-ocp`       |                                                              |
| `numra::integrate`   | `numra-integrate` |                                                              |
| `numra::interp`      | `numra-interp`    |                                                              |
| `numra::special`     | `numra-special`   |                                                              |
| `numra::fft`         | `numra-fft`       |                                                              |
| `numra::stats`       | `numra-stats`     |                                                              |
| `numra::fit`         | `numra-fit`       |                                                              |
| `numra::dsp`         | `numra-signal`    | **Not** `numra::signal`                                      |


---

## `numra-core`


| Category    | Public API                                                                           | Role              |
| ----------- | ------------------------------------------------------------------------------------ | ----------------- |
| Scalar      | `Scalar`, `to_f64_vec`, `from_f64_vec`                                               | Float abstraction |
| Vector      | `Vector`                                                                             | Vector ops trait  |
| Signal      | `Signal`, `signal::*` (`Harmonic`, `Step`, …)                                        | Forcing / signals |
| Errors      | `NumraError`, `NumraResult`, `LinalgError`                                           |                   |
| Uncertainty | `Uncertain`, `Interval`, `compute_sensitivities`, `SensitivityResult`, `Sensitivity` |                   |


---

## `numra-linalg`


| Method / API                                | Import                                                                            |
| ------------------------------------------- | --------------------------------------------------------------------------------- |
| Dense matrix, trait `Matrix`, `DenseMatrix` | `numra::linalg::…`                                                                |
| LU                                          | `LUFactorization`, `LUSolver`                                                     |
| QR                                          | `QRFactorization`                                                                 |
| Cholesky                                    | `CholeskyFactorization`                                                           |
| Sparse                                      | `SparseMatrix`, `SparseLU`, `SparseCholesky`                                      |
| Eigen                                       | `SymEigenDecomposition`, `EigenDecomposition`                                     |
| SVD                                         | `SvdDecomposition`, `ThinSvdDecomposition`                                        |
| Iterative                                   | `cg`, `pcg`, `gmres`, `bicgstab`, `minres`, `IterativeOptions`, `IterativeResult` |
| Preconditioners                             | `Preconditioner`, `Jacobi`, `Ilu0`, `Ssor`, `IdentityPreconditioner`              |


---

## `numra-nonlinear`


| Method                       | Import                                                                       |
| ---------------------------- | ---------------------------------------------------------------------------- |
| Newton-Raphson + line search | `Newton`, `NewtonOptions`, `NewtonResult`, `NonlinearSystem`, `newton_solve` |
| Wolfe line search            | `wolfe_line_search`, `WolfeOptions`, `LineSearchResult`, `LineSearchError`   |


---

## `numra-ode`


| Method                  | Type                                                                                                                                                | Import                   |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| Dormand–Prince 5(4)     | `DoPri5`                                                                                                                                            | `numra::ode::DoPri5`     |
| Tsitouras 5(4)          | `Tsit5`                                                                                                                                             | `numra::ode::Tsit5`      |
| Verner 6(5), 7(6), 8(7) | `Vern6`, `Vern7`, `Vern8`                                                                                                                           | `numra::ode::Vern6` …    |
| Radau IIA (stiff)       | `Radau5`                                                                                                                                            | `numra::ode::Radau5`     |
| ESDIRK                  | `Esdirk32`, `Esdirk43`, `Esdirk54`                                                                                                                  | `numra::ode::Esdirk32` … |
| BDF variable order 1–5  | `Bdf`                                                                                                                                               | `numra::ode::Bdf`        |
| Auto selection          | `Auto`, `auto_solve`, `auto_solve_with_hints`, `SolverHints`, `Stiffness`, `Accuracy`                                                               |                          |
| Problem / solver API    | `OdeProblem`, `OdeSystem`, `DaeProblem`, `Solver`, `SolverOptions`, `SolverResult`, `SolverStats`, `SolverError`                                    |                          |
| Step / dense / events   | `StepController`, `PIController`, `DenseOutput`, `events` module                                                                                    |                          |
| Sensitivity             | `SensitivityEquations`, `SensitivityState`, `SensitivityResult`, `AugmentedSystem`                                                                  |                          |
| DAE init                | `compute_consistent_initial`, `compute_consistent_initial_tol`                                                                                      |                          |
| Index reduction         | `analyze_dae_index`, `analyze_system`, `reduce_index`, `reduce_dae_problem`, `detect_structure`, `DaeIndexInfo`, `DaeStructure`, `ReducedDaeSystem` |                          |
| Uncertainty wrappers    | `solve_with_uncertainty`, `solve_trajectory`, `solve_monte_carlo`, `UncertaintyMode`, `UncertainParam`, `UncertainSolverResult`                     |                          |


---

## `numra-sde`


| Method                      | Import                                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Euler–Maruyama              | `EulerMaruyama`                                                                                                                       |
| Milstein                    | `Milstein`                                                                                                                            |
| Strong 1.5 / adaptive (SRA) | `Sra1`                                                                                                                                |
| Weak order 2 / adaptive     | `Sra2`                                                                                                                                |
| System / options            | `SdeSystem`, `SdeOptions`, `SdeResult`, `SdeSolver`, `NoiseType`                                                                      |
| Wiener                      | `WienerProcess`, `WienerIncrement`, `create_wiener`                                                                                   |
| Ensemble / stats            | `EnsembleRunner`, `EnsembleResult`, `EnsembleStats`, `RunningStats`, `Percentiles`, `mean`, `std`, `variance`, `percentile`, `median` |


**Note:** Literature may cite “SRI/SRA” families; Rust public types are `**Sra1`**, `**Sra2`** (not `SriW1`).

---

## `numra-dde`


| Method                                        | Import                                                                                                               |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Method of steps (embedded RK 5(4) internally) | `MethodOfSteps`                                                                                                      |
| System / history                              | `DdeSystem`, `DdeOptions`, `DdeResult`, `DdeSolver`, `DdeStats`, `History`, `HistoryFunction`, `HermiteInterpolator` |


**Note:** There is **no** public type `DdeDoPri5`; the solver type is `**MethodOfSteps`**.

---

## `numra-fde`


| Method             | Import                                                          |
| ------------------ | --------------------------------------------------------------- |
| L1 scheme (Caputo) | `L1Solver`                                                      |
| Helpers            | `caputo_weights`, `mittag_leffler`, `mittag_leffler_1`, `gamma` |
| System             | `FdeSystem`, `FdeOptions`, `FdeResult`, `FdeSolver`             |


---

## `numra-ide`


| Method                    | Import                                                                                  |
| ------------------------- | --------------------------------------------------------------------------------------- |
| Volterra quadrature-style | `VolterraSolver`                                                                        |
| Volterra + RK4 coupling   | `VolterraRK4Solver`                                                                     |
| Prony / exponentials      | `PronySolver`, `PronySystem`                                                            |
| Kernels                   | `Kernel`, `ExponentialKernel`, `PowerLawKernel`, `PronyKernel` in `numra::ide::kernels` |
| System                    | `IdeSystem`, `IdeOptions`, `IdeResult`, `IdeSolver`, `IdeStats`                         |


---

## `numra-pde`


| Area            | Public API (representative)                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------ |
| Grids           | `Grid1D`, `Grid2D`, `Grid3D`                                                                                       |
| FDM / MOL       | `FDM`, `DifferenceScheme`, `Stencil`, `MOLSystem`, `MOLSystem2D`, `PdeSystem`                                      |
| BCs             | `DirichletBC`, `NeumannBC`, `RobinBC`, `PeriodicBC`, `BoundaryConditions2D`, `BoundaryConditions3D`                |
| Preset PDEs     | `HeatEquation1D`, `DiffusionReaction1D`, `HeatEquation2D`, `AdvectionDiffusion2D`, `ReactionDiffusion2D`           |
| Sparse helpers  | `assemble_laplacian_2d`, `assemble_operator_2d`, `assemble_laplacian_3d`, `Operator2DCoefficients`, `SparseScalar` |
| Moving boundary | `Domain1D`, `Bound`, `MovingBound`, `StefanCondition`, `CoordinateTransform`                                       |


---

## `numra-spde`


| Method          | Import                                                                                                    |
| --------------- | --------------------------------------------------------------------------------------------------------- |
| MOL + SDE       | `SpdeSolver`, `MolSdeSolver`, `SpdeEnsemble`                                                              |
| Options / stats | `SpdeOptions`, `SpdeResult`, `SpdeStats`, `SpdeMethod`                                                    |
| System / noise  | `SpdeSystem`, `NoiseCorrelation`, `SpaceTimeNoise`, `WhiteNoise`, `ColoredNoise`, `ColoredNoiseGenerator` |
| Re-export       | `Grid1D` (from `numra-pde`)                                                                               |


---

## `numra-autodiff`


| API                | Import                                       |
| ------------------ | -------------------------------------------- |
| Forward dual       | `Dual`                                       |
| Forward grad / Jac | `gradient`, `jacobian`                       |
| Reverse            | `numra::autodiff::reverse` (`grad`, …)       |
| Bridge             | `gradient_closure`, `model_jacobian_closure` |


---

## `numra-optim`


| Method               | Import                                                               |
| -------------------- | -------------------------------------------------------------------- |
| BFGS                 | `bfgs_minimize`, `Bfgs`                                              |
| L-BFGS               | `lbfgs_minimize`, `Lbfgs`, `LbfgsOptions`                            |
| LM least squares     | `lm_minimize`, `LmOptions`                                           |
| L-BFGS-B             | `lbfgsb_minimize`, `LbfgsBOptions`                                   |
| Augmented Lagrangian | `augmented_lagrangian_minimize`, `AugLagOptions`                     |
| Problem builder      | `OptimProblem`, `ObjectiveKind`, `Constraint`, …                     |
| LP / MILP / QP       | `simplex_solve`, `milp_solve`, `active_set_qp_solve` + options types |
| Global               | `de_minimize`, `DEOptions`                                           |
| Multi-objective      | `nsga2_optimize`, `NsgaIIOptions`                                    |
| Derivative-free      | `nelder_mead`, `powell` + options                                    |
| CMA-ES               | `cmaes_minimize`, `CmaEsOptions`                                     |
| SQP                  | `sqp_minimize`, `SqpOptions`                                         |
| Robust / stochastic  | `RobustProblem`, `StochasticProblem`, …                              |
| Param sensitivity    | `compute_param_sensitivity`, `ParamSensitivity`                      |
| Solver hint enum     | `SolverChoice`                                                       |


---

## `numra-ocp`


| Method               | Import                                                         |
| -------------------- | -------------------------------------------------------------- |
| Adjoint              | `adjoint_gradient`, `AdjointResult`                            |
| Collocation          | `CollocationProblem`, `CollocationResult`, `CollocationScheme` |
| Shooting             | `ShootingProblem`, `ShootingResult`                            |
| Multiple shooting    | `MultipleShootingProblem`, `MultipleShootingResult`            |
| Parameter estimation | `ParamEstProblem`, `ParamEstResult`, `OdeSolverChoice`         |
| Forward sensitivity  | `forward_sensitivity`, `SensitivityResult`                     |
| Errors               | `OcpError`                                                     |


---

## `numra-integrate`


| Method          | Import                                                                            |
| --------------- | --------------------------------------------------------------------------------- |
| Adaptive `quad` | `quad`, `QuadOptions`, `QuadResult`                                               |
| Gauss rules     | `gauss_legendre`, `gauss_laguerre`, `gauss_hermite`                               |
| Composite       | `trapezoid`, `simpson`, `romberg`, `cumulative_trapezoid`, `trapezoid_nonuniform` |
| 2D              | `dblquad`                                                                         |
| Errors          | `IntegrationError`                                                                |


---

## `numra-interp`


| Method               | Import                                      |
| -------------------- | ------------------------------------------- |
| Piecewise linear     | `Linear`                                    |
| Cubic spline         | `CubicSpline`                               |
| PCHIP / Akima        | `Pchip`, `Akima`                            |
| Barycentric Lagrange | `BarycentricLagrange`                       |
| Trait / selector     | `Interpolant`, `Interp1dMethod`, `interp1d` |
| Errors               | `InterpError`                               |


---

## `numra-special`

Large surface of **free functions** (see `[numra-special/src/lib.rs](../../numra-special/src/lib.rs)`): gamma family, erf, Bessel, elliptic, Airy, hypergeometric, orthogonal polynomials, zeta, Dawson, Fresnel, **Mittag–Leffler** (`mittag_leffler`, `mittag_leffler_1`).  
Errors: `SpecialError`.

---

## `numra-fft`


| API          | Import                                                      |
| ------------ | ----------------------------------------------------------- |
| Complex, FFT | `Complex`, `ComplexF64`, `fft`, `ifft`, `fft2`              |
| Real FFT     | `rfft`, `irfft`                                             |
| Spectral     | `psd`, `welch`, `stft`, `StftResult`                        |
| Convolution  | `fftconvolve`, `fftcorrelate`                               |
| Utils        | `fftfreq`, `fftshift`, `ifftshift`, `window_func`, `Window` |


---

## `numra-stats`


| Area          | Public types / functions                                                                                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Distributions | `Normal`, `Uniform`, `Exponential`, `GammaDist`, `BetaDist`, `ChiSquared`, `StudentT`, `FDist`, `Poisson`, `Binomial`, `LogNormal`, traits `ContinuousDistribution`, `DiscreteDistribution` |
| Descriptive   | `mean`, `variance`, `std_dev`, `median`, `percentile`, `skewness`, `kurtosis`, `covariance`, `covariance_matrix`                                                                            |
| Tests         | `ttest_1samp`, `ttest_ind`, `ttest_rel`, `chi2_test`, `ks_test`, `anova_oneway`, `TestResult`                                                                                               |
| Regression    | `linregress`, `multiple_linregress`, `polyfit`, `RegressionResult`                                                                                                                          |
| Correlation   | `pearson_r`, `spearman_r`                                                                                                                                                                   |
| Errors        | `StatsError`                                                                                                                                                                                |


---

## `numra-fit`


| API           | Import                                                       |
| ------------- | ------------------------------------------------------------ |
| Nonlinear fit | `curve_fit`, `curve_fit_weighted`, `curve_fit_with_jacobian` |
| Polynomial    | `polyfit`, `polyval`                                         |
| Types         | `FitResult`, `FitOptions`, `FitError`                        |


---

## `numra-signal` (facade: `numra::dsp`)


| API        | Import                                           |
| ---------- | ------------------------------------------------ |
| IIR design | `butter`, `cheby1`                               |
| Filtering  | `SosFilter`, `sosfilt`, `filtfilt`               |
| FIR        | `firwin`, `fir_filter`                           |
| Resample   | `resample`                                       |
| Hilbert    | `hilbert`, `envelope`, `instantaneous_frequency` |
| Peaks      | `find_peaks`, `PeakOptions`                      |
| Errors     | `SignalError`                                    |


---

## `numra-bench`

Criterion benchmarks for ODE performance (not a user library crate).

---

## Maintenance

When adding a public solver or numerical entry point:

1. Export it from the crate `lib.rs`.
2. Re-export from `numra` if user-facing.
3. Add a row here and in `[book-coverage-matrix.md](book-coverage-matrix.md)`.
4. Extend `[../scripts/check_book_inventory.py](../scripts/check_book_inventory.py)` denylist / checks if needed.

