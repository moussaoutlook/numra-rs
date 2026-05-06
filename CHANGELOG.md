# Changelog

All notable public changes to Numra are recorded here. The project follows semantic-versioning intent, with extra care around solver behavior, public re-exports, and documented book examples while the `0.1.x` API is still settling.

## 0.1.0 - Unreleased

### Added

- Workspace facade crate covering ODE, SDE, DDE, FDE, IDE, PDE, SPDE, optimization, optimal control, linear algebra, quadrature, interpolation, special functions, FFT, statistics, fitting, signal processing, and autodiff.
- Public release audit artifacts under `docs/audit/`, including API inventory, book coverage matrix, correctness test map, and release checklist.
- CI and local audit gates for formatting, clippy, tests, docs, MSRV, and supply-chain checks.
- `SolverResult.dense_output: Option<DenseOutput<S>>` field so callers can actually use the dense interpolant they requested via `SolverOptions::dense()`.

### Changed

- `numra-core`: renamed `Sensitivity` → `ParameterSensitivity` and `SensitivityResult` → `ParameterSensitivityResult` (and the corresponding `numra` facade re-exports) to disambiguate from the new ODE forward-sensitivity types. The `compute_sensitivities` function name is preserved; only the type names changed.
- `numra-ode`: renamed the `SensitivityEquations` trait to `ParametricOdeSystem` and reshaped it to take `params()` and `rhs_with_params(t, y, p, dydt)` (which lets the trait drive its own FD on `J_p`); added FD-default `jacobian_y` and `jacobian_p` overrides so a minimal implementation supplies only `n_states`, `n_params`, `params`, and `rhs_with_params`. Removed the unused `SensitivityState` helper. `AugmentedSystem` now implements `OdeSystem` directly and can be solved by any `Solver`. Sensitivity flattening is column-major (parameter-major), matching CVODES indexing.
- `numra-ode`: `SensitivityResult` reshaped to flat row-major over time and column-major over parameters within each time block, with `y_at`, `sensitivity_at`, `sensitivity_for_param`, `dyi_dpj`, `final_state`, `final_sensitivity`, and `normalized_sensitivity_at` accessors.

> **Forward-sensitivity v1 scope.** This release lays the trait + augmented-system foundation. The user-facing solve entry points (`solve_forward_sensitivity` and `solve_forward_sensitivity_with`), regression suite, Robertson example, and book chapter land in subsequent commits on this branch. The augmented Jacobian's block-diagonal structure (which would let implicit solvers reuse a single LU of `M = (1/γh)·I_N - J_y` across all `N_s + 1` sub-systems) is not yet exploited — Radau5 and BDF perform dense LU on the full `N(N_s+1) × N(N_s+1)` augmented matrix in v1. Block-aware factorisation is tracked as a named follow-up.

### Fixed

- `numra-ode`: DoPri5 was building a `DenseOutput` when `SolverOptions::dense()` was set but never returning it, so the interpolant was silently dropped at the end of integration. The new `SolverResult.dense_output` field is now populated on both the normal exit and the early-termination event path.
- `numra-ode`: Radau5 step controller rewritten against Hairer–Wanner ODE II §IV.8 and the SciPy `Radau` reference. Eight bugs fixed (most importantly: the `error_estimate` forcing term used `y` instead of `f(t,y)`; Newton's initial guess now extrapolates the previous step's collocation polynomial; LU is reused unless the step ratio leaves [1.0, 1.2]; Gustafsson predictive controller). Step counts on the standard reference suite are now within ~1.5–2× of SciPy's, and Van der Pol μ=10 at rtol=1e-4 runs in ~0.66 ms (vs ~200 ms before, and ~12 ms for SciPy).

### Policy

- Root `Cargo.lock` is committed for reproducible public CI.
- The user-facing book is built from `website/book/` (Astro + Starlight) and deployed to `book.numra-rs.org` by the `Website` workflow.
