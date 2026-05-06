# Changelog

All notable public changes to Numra are recorded here. The project follows semantic-versioning intent, with extra care around solver behavior, public re-exports, and documented book examples while the `0.1.x` API is still settling.

## 0.1.0 - Unreleased

### Added

- Workspace facade crate covering ODE, SDE, DDE, FDE, IDE, PDE, SPDE, optimization, optimal control, linear algebra, quadrature, interpolation, special functions, FFT, statistics, fitting, signal processing, and autodiff.
- Public release audit artifacts under `docs/audit/`, including API inventory, book coverage matrix, correctness test map, and release checklist.
- CI and local audit gates for formatting, clippy, tests, docs, MSRV, and supply-chain checks.
- `SolverResult.dense_output: Option<DenseOutput<S>>` field so callers can actually use the dense interpolant they requested via `SolverOptions::dense()`.

### Fixed

- `numra-ode`: DoPri5 was building a `DenseOutput` when `SolverOptions::dense()` was set but never returning it, so the interpolant was silently dropped at the end of integration. The new `SolverResult.dense_output` field is now populated on both the normal exit and the early-termination event path.
- `numra-ode`: Radau5 step controller rewritten against Hairer–Wanner ODE II §IV.8 and the SciPy `Radau` reference. Eight bugs fixed (most importantly: the `error_estimate` forcing term used `y` instead of `f(t,y)`; Newton's initial guess now extrapolates the previous step's collocation polynomial; LU is reused unless the step ratio leaves [1.0, 1.2]; Gustafsson predictive controller). Step counts on the standard reference suite are now within ~1.5–2× of SciPy's, and Van der Pol μ=10 at rtol=1e-4 runs in ~0.66 ms (vs ~200 ms before, and ~12 ms for SciPy).

### Policy

- Root `Cargo.lock` is committed for reproducible public CI.
- The user-facing book is built from `website/book/` (Astro + Starlight) and deployed to `book.numra-rs.org` by the `Website` workflow.
