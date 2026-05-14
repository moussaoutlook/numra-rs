---
title: "Announcing Numra 0.1.0"
description: "Numra 0.1.0 is on crates.io: 21 native-Rust crates of numerical methods that compose under one API — ODE, SDE, DDE, FDE, IDE, PDE, SPDE, optimization, autodiff, FFT, statistics, and more."
pubDate: 2026-05-14
tags:
  - release
  - numerical-methods
  - rust
---

Numra 0.1.0 is on [crates.io](https://crates.io/crates/numra). One `cargo add numra` gives you ordinary and stochastic differential equations, delay equations, fractional and integral equations, PDEs and SPDEs, optimization, automatic differentiation, FFT, statistics, signal processing, and quadrature — implemented in Rust, sharing one set of abstractions, designed to compose without glue.

```bash
cargo add numra
```

This post is the *why*. The [changelog](/changelog) is the *what*.

## The thesis: one workspace, native Rust, methods that compose

Most numerical computing in Rust today is a thin shell over decades-old C or Fortran libraries. That works, and it gets you battle-tested code, but it has costs you only notice once you try to *compose* methods:

- An ODE solver returning a `Vec<f64>` that an optimizer can't ingest without conversion.
- An autodiff library whose gradients an FFI solver can't accept because the type doesn't cross the boundary.
- A PDE discretization that gives you a sparse matrix in one crate's format and a linear solver in another's.

Numra is the bet that if you implement the numerics natively in Rust, share a small core of abstractions (`Scalar`, `Vector`, `Signal`, `Uncertain`) across every subsystem, and design every public type to be a first-class citizen everywhere else in the workspace, the resulting library is *qualitatively* different to use.

You stop writing adapters. You write the math.

## What composability looks like in practice

Here's a complete, self-contained example: solve a stiff ODE, build a cubic spline through the trajectory, then integrate the spline back to verify conservation — three subsystems, no glue.

```rust
use numra::integrate::{quad, QuadOptions};
use numra::interp::{CubicSpline, Interpolant};
use numra::ode::{DoPri5, OdeProblem, Solver, SolverOptions};

// y' = -y, y(0) = 1 on [0, 3]
let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| { dydt[0] = -y[0]; },
    0.0, 3.0, vec![1.0],
);
let opts = SolverOptions::default().rtol(1e-8).dense();
let result = DoPri5::solve(&problem, 0.0, 3.0, &[1.0], &opts).unwrap();

// ODE -> Interpolation: just hand the trajectory to a spline.
let y_series = result.component(0).unwrap();
let spline = CubicSpline::natural(&result.t, &y_series).unwrap();

// Interpolation -> Quadrature: the spline IS a callable f64 -> f64.
let q = quad(|t| spline.interpolate(t), 0.0, 3.0, &QuadOptions::default()).unwrap();

// integral_0^3 e^(-t) dt = 1 - e^(-3) ≈ 0.9502
assert!((q.value - (1.0 - (-3.0_f64).exp())).abs() < 1e-3);
```

No type conversions. No `Into` impls in your code. No "wrap this with a closure that does the dance." This is the workflow that's [enforced by an integration test](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/interop_workflows.rs) so it stays that way.

Five other workflows are checked the same way: ODE → FFT → signal processing → peak detection, parameter estimation → sensitivity → uncertainty, autodiff → optimization → curve fitting, PDE → statistics, and stats sampling → Monte Carlo ODE.

## What's in 0.1.0

| Subsystem | What it gives you |
| --- | --- |
| `numra::ode` | DoPri5, Tsit5, Verner 6/7/8/9, Radau5, ESDIRK, BDF, `Auto` (stiffness-aware), forward sensitivity, event detection, DAE |
| `numra::sde` | Euler-Maruyama, Milstein, stochastic Runge-Kutta, multiple noise correlations |
| `numra::dde` | Method-of-steps with continuous extension; handles state-dependent delays |
| `numra::fde` | Caputo and Riemann-Liouville fractional derivatives, predictor-corrector schemes |
| `numra::ide` | Volterra and Fredholm integro-differential equations |
| `numra::pde` | Method-of-lines on structured 2D/3D grids, sparse operator assembly |
| `numra::spde` | Stochastic PDEs combining `pde` discretization with `sde` time integration |
| `numra::linalg` | Dense linear algebra via [faer](https://github.com/sarah-ek/faer-rs); decompositions, eigensolvers |
| `numra::nonlinear` | Newton, Broyden, trust-region methods |
| `numra::optim` | First- and second-order unconstrained, constrained, and stochastic optimizers |
| `numra::ocp` | Parameter estimation and optimal-control problems against ODE/DAE dynamics |
| `numra::integrate` | Adaptive quadrature (Gauss-Kronrod), tanh-sinh, Romberg |
| `numra::interp` | Linear, cubic, Akima, B-spline, RBF |
| `numra::special` | Gamma, beta, error, Bessel, Mittag-Leffler families |
| `numra::fft` | Radix-2/4 and mixed-radix FFTs, real and complex |
| `numra::stats` | Distributions, descriptive statistics, hypothesis tests |
| `numra::fit` | Linear and nonlinear least-squares curve fitting with analytical or AD Jacobians |
| `numra::autodiff` | Forward-mode dual numbers, gradients and Jacobians as closures |
| `numra::dsp` | IIR/FIR filter design, peak detection, windowing |
| `numra::core` | The shared abstractions everything else builds on |

Twenty member crates plus the `numra` facade. The full inventory and per-method coverage matrix is in the [public API inventory](https://github.com/moussaoutlook/numra-rs/blob/main/docs/audit/public-api-method-inventory.md).

## The design constraints I refused to compromise

A few decisions defined what 0.1.0 had to be before I'd ship it:

**Native Rust everywhere.** No FFI to LAPACK, SUNDIALS, FFTW, or similar. Every algorithm in this release is implemented in Rust source you can read in this repo. Dense linear algebra builds on [faer](https://github.com/sarah-ek/faer-rs), itself a native-Rust project.

**One shared `Scalar` trait.** The same generic numeric trait is used by every solver, every quadrature rule, every optimizer. That's what lets you flip between `f64`, `f32`, and (eventually) extended-precision or AD types without rewriting your problem.

**Solvers as first-class values.** `DoPri5`, `Tsit5`, `Radau5`, and the rest are types you can name, store, and choose between at runtime via the `Auto` selector or your own logic. No string-keyed dispatch, no factory functions, no enum-of-everything.

**Composability tested, not promised.** Six end-to-end workflows are checked by CI as integration tests. If a future change breaks composability between subsystems, the tests fail.

## Installation and citation

```bash
cargo add numra
```

Documentation: the [user book](https://book.numra-rs.org/) for narrative, [docs.rs](https://docs.rs/numra/0.1.0) for the API reference, the [examples gallery](https://examples.numra-rs.org/) for end-to-end programs.

If Numra contributes to published work, please cite it. The repository has a [`CITATION.cff`](https://github.com/moussaoutlook/numra-rs/blob/main/CITATION.cff), and Zenodo has minted two DOIs:

- Concept DOI (always resolves to the latest version): [`10.5281/zenodo.20159709`](https://doi.org/10.5281/zenodo.20159709)
- Version DOI for 0.1.0 (use this if reproducibility against a specific implementation matters): [`10.5281/zenodo.20159710`](https://doi.org/10.5281/zenodo.20159710)

BibTeX, RIS, and a longer note on why citation matters live on the [Cite page](/cite).

## Licensing

Numra ships under the **Numra Academic & Research License (Non-Commercial)**: free for academic and research use, source-available, and a separate commercial license is required for production or for-profit use. The full text is in [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE), and the [Commercial licensing page](/commercial) explains the why.

## What's next

The 0.1.0 release establishes the surface area. The next few releases will deepen it:

- Sparse linear algebra (matrix-free and assembled), unlocking larger PDE problems
- More PDE discretizations beyond method-of-lines (finite volume, spectral)
- Adjoint sensitivity to complement the forward sensitivity already in `numra::ode`
- A constraint solver and DAE index-2/3 reduction in `numra::ode`
- GPU acceleration as an opt-in feature, once the CPU story is fully stable

The [roadmap page](/roadmap) tracks the current direction, and the [issue tracker](https://github.com/moussaoutlook/numra-rs/issues) is the right place to push back on it.

If you build something with Numra, [tell me](https://github.com/moussaoutlook/numra-rs/discussions). 0.1.0 is the *start* of the conversation about what composable numerical methods in Rust should look like.
