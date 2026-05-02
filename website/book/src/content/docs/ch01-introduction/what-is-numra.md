---
title: "What is Numra?"
---

Numra is a comprehensive numerical methods library for Rust. It provides
production-quality implementations of algorithms for differential equations,
optimization, linear algebra, signal processing, statistics, and more -- all in
pure Rust with no mandatory C or Fortran dependencies.

## Why Numra?

The scientific computing ecosystem in Rust is growing rapidly, but most libraries
focus on a single domain: ODE solvers, or linear algebra, or FFT. Numra takes
a different approach. It provides a *unified workspace* where crates are designed
to work together, sharing common traits and types across the entire stack.

This means you can:

- Solve a stiff ODE system with `Radau5`, then analyze the frequency content
  of the solution with `fft::psd`
- Fit model parameters to data with `optim::lm_minimize`, where the model
  itself is an ODE integration
- Propagate parameter uncertainty through a differential equation solve
- Design an optimal controller using `ocp::shooting` that calls the ODE
  solver internally

All of this works seamlessly because every crate in Numra speaks the same
language: the `Scalar` trait for generic `f32`/`f64` computation, the `Vector`
trait for BLAS-like operations, and consistent error types throughout.

## What Does Numra Cover?

| Domain | Crate | Highlights |
|--------|-------|------------|
| **Core traits** | `numra-core` | `Scalar`, `Vector`, `Signal`, uncertainty propagation |
| **Linear algebra** | `numra-linalg` | Dense/sparse matrices, LU/QR/Cholesky/SVD, iterative solvers |
| **Nonlinear solvers** | `numra-nonlinear` | Newton-Raphson with Wolfe line search |
| **ODE solvers** | `numra-ode` | 11 solvers: DoPri5, Tsit5, Vern6/7/8, Radau5, ESDIRK, BDF, Auto |
| **SDE solvers** | `numra-sde` | Euler-Maruyama, Milstein, SRA1/SRA2 |
| **DDE solvers** | `numra-dde` | Method of Steps with discontinuity tracking |
| **FDE solvers** | `numra-fde` | L1 scheme for Caputo fractional derivatives |
| **IDE solvers** | `numra-ide` | Volterra equations, Prony series kernels |
| **PDE solvers** | `numra-pde` | Method of Lines, moving boundary (Stefan) problems |
| **SPDE solvers** | `numra-spde` | MOL + stochastic time stepping |
| **Autodiff** | `numra-autodiff` | Forward-mode (Dual) and reverse-mode (tape-based) |
| **Optimization** | `numra-optim` | BFGS, L-BFGS, LM, SQP, CMA-ES, LP, MILP, QP, NSGA-II |
| **Optimal control** | `numra-ocp` | Shooting, collocation, adjoint methods |
| **Integration** | `numra-integrate` | Gauss-Kronrod, Gauss-Legendre/Laguerre/Hermite, Romberg |
| **Interpolation** | `numra-interp` | Linear, cubic spline, PCHIP, Akima, barycentric Lagrange |
| **Special functions** | `numra-special` | Gamma, Bessel, Erf, elliptic integrals, Airy, hypergeometric |
| **FFT** | `numra-fft` | FFT/IFFT, PSD, Welch, STFT, convolution |
| **Statistics** | `numra-stats` | 11 distributions, hypothesis tests, regression, correlation |
| **Curve fitting** | `numra-fit` | Nonlinear least squares, polynomial fitting |
| **Signal processing** | `numra-signal` | Butterworth/Chebyshev filters, FIR, Hilbert transform |

## Design Principles

**Generic over scalar type.** Every algorithm works with both `f32` and `f64`
through the `Scalar` trait. Choose precision at the call site, not at the
library level.

**no_std compatible core.** The `numra-core` crate uses `libm` for math
functions and can run on embedded systems without the standard library.

**Pure Rust.** No mandatory C, Fortran, or LAPACK dependencies. The linear
algebra backend is [faer](https://github.com/sarah-ek/faer-rs), a
high-performance pure-Rust library.

**Composable.** Crates are designed to interoperate. An ODE solver result
can be fed directly into an FFT, an optimizer can call a differential equation
solver, and uncertainty propagation works across the entire stack.

**Correct first, then fast.** Every algorithm is validated against published
references, analytical solutions, and cross-checked with established tools
like SciPy and MATLAB.
