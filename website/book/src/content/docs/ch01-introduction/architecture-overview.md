---
title: "Architecture Overview"
---

Numra is organized as a Cargo workspace with 22 library crates plus a facade
crate and a benchmark crate. The crates form a layered architecture where
dependencies flow strictly downward.

## Crate Dependency Graph

```text
                          numra (facade)
                              |
          +---+---+---+---+---+---+---+---+---+
          |   |   |   |   |   |   |   |   |   |
        ode sde dde fde ide pde spde optim ocp ...
          |       |           |   |    |    |
          +-------+-----------+   |    |    +-- numra-autodiff
          |                       |    |
     numra-nonlinear          numra-pde |
          |                       |    |
     numra-linalg  numra-fft  numra-sde
          |            |
     numra-core    numra-core
```

### Layer 0: Foundation

**numra-core** is the bedrock. It defines three foundational traits:

- **`Scalar`** -- Abstraction over `f32` and `f64`. Provides all math
  operations (`sin`, `exp`, `sqrt`, `gamma_fn`, etc.) using `libm` for
  `no_std` compatibility.

- **`Vector`** -- BLAS-like operations on `Vec<S>`: `axpy`, `dot`, `norm2`,
  `scale`, `weighted_rms_norm`. Every solver in Numra uses these operations
  instead of manual loops.

- **`Signal`** -- Time-dependent forcing functions. Includes `Harmonic`,
  `Step`, `Ramp`, `Chirp`, `Tabulated`, and composites (`Sum`, `Product`,
  `Scaled`).

`numra-core` also provides `Uncertain<S>` for uncertainty propagation,
`Interval<S>` for interval arithmetic, and the error type hierarchy.

### Layer 1: Linear Algebra

**numra-linalg** wraps [faer](https://github.com/sarah-ek/faer-rs) with
Numra's `Matrix` trait. It provides:

- `DenseMatrix<S>` with row-major and column-major constructors
- Factorizations: LU, QR, Cholesky, SVD (full and thin), eigendecomposition
- `SparseMatrix<S>` in CSC format
- Iterative solvers: CG, PCG, GMRES, BiCGSTAB, MINRES
- Preconditioners: Jacobi, ILU(0), SSOR

**numra-nonlinear** provides Newton-Raphson with Wolfe line search for solving
F(x) = 0. From the facade, use `numra::nonlinear` (for example `Newton`,
`NonlinearSystem`, and `newton_solve`). Implicit solvers in `numra-ode` use
this stack internally for nonlinear systems at each time step.

### Layer 2: Differential Equation Solvers

Seven crates handle different types of differential equations:

| Crate | Equation Type | Methods |
|-------|--------------|---------|
| `numra-ode` | y' = f(t, y) | DoPri5, Tsit5, Vern6/7/8, Radau5, ESDIRK32/43/54, BDF, Auto |
| `numra-sde` | dX = f dt + g dW | Euler-Maruyama, Milstein, SRA1, SRA2 |
| `numra-dde` | y'(t) = f(t, y(t), y(t-τ)) | Method of Steps |
| `numra-fde` | D^α y = f(t, y) | L1 scheme (Caputo) |
| `numra-ide` | y'(t) = f(t, y) + ∫K(t,s,y)ds | Volterra, Prony series |
| `numra-pde` | u_t = F(u, u_x, u_xx) | Method of Lines |
| `numra-spde` | du = F dt + G dW | MOL + SDE stepping |

### Layer 3: Analysis & Optimization

- **numra-autodiff** -- Forward-mode (`Dual<S>`) and reverse-mode (tape-based
  `Var<S>`) automatic differentiation.

- **numra-optim** -- 15+ optimization algorithms: unconstrained (BFGS, L-BFGS,
  Nelder-Mead, Powell), constrained (SQP, augmented Lagrangian, L-BFGS-B),
  global (Differential Evolution, CMA-ES), linear/quadratic programming
  (Simplex, MILP, active-set QP), and multi-objective (NSGA-II).

- **numra-ocp** -- Optimal control via shooting, collocation, and adjoint methods.

### Layer 4: Data Analysis

- **numra-integrate** -- Adaptive quadrature (Gauss-Kronrod), fixed Gaussian
  rules, composite rules (trapezoid, Simpson, Romberg), double integrals.

- **numra-interp** -- Piecewise interpolation: linear, cubic spline (natural,
  clamped, not-a-knot), PCHIP, Akima, barycentric Lagrange.

- **numra-special** -- Gamma, Beta, Bessel, Erf, elliptic integrals, Airy,
  hypergeometric, orthogonal polynomials, Riemann zeta, Mittag-Leffler.

- **numra-fft** -- FFT/IFFT, real FFT, PSD, Welch's method, STFT,
  convolution, windowing functions.

- **numra-stats** -- 11 distributions, descriptive statistics, hypothesis
  tests (t-test, chi-squared, KS, ANOVA), regression, correlation.

- **numra-fit** -- Nonlinear curve fitting (Levenberg-Marquardt), polynomial
  fitting (SVD-based).

- **numra-signal** -- IIR filter design (Butterworth, Chebyshev), FIR
  (windowed sinc), filtering (`sosfilt`, `filtfilt`), Hilbert transform,
  resampling, peak detection.

## The Facade Crate

The top-level `numra` crate re-exports everything under clean namespaces:

For a practical method-selection view across these crates, see the
[appendix method cards](../appendix/method-cards).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode;          // ODE solvers
use numra::sde;          // SDE solvers
use numra::dde;          // DDE solvers
use numra::optim;        // Optimization
use numra::linalg;       // Linear algebra
use numra::nonlinear;    // Nonlinear systems F(x)=0 (Newton + line search)
use numra::autodiff;     // Automatic differentiation
use numra::integrate;    // Numerical integration
use numra::interp;       // Interpolation
use numra::special;      // Special functions
use numra::fft;          // FFT
use numra::stats;        // Statistics
use numra::fit;          // Curve fitting
use numra::dsp;          // Signal processing (digital signal processing)
use numra::ocp;          // Optimal control
```

Core traits are re-exported at the top level:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::{Scalar, Vector, Signal};
use numra::{Uncertain, Interval, compute_sensitivities};
```

## Key Design Patterns

### Builder Pattern for Options

All solver options use the builder pattern with sensible defaults:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let options = SolverOptions::default()
    .rtol(1e-8)       // relative tolerance
    .atol(1e-10)      // absolute tolerance
    .h0(Some(1e-3))   // initial step size (None = auto)
    .max_steps(50000)  // safety limit
    .dense();          // enable dense output
```

### Row-Major Solution Storage

ODE solution data is stored in row-major format for cache-friendly
sequential access:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let result = DoPri5::solve(&problem, t0, tf, &y0, &options)?;

// Access time point i, component j:
let y_ij = result.y[i * result.dim + j];

// Or use the helper:
let y_at_i = result.y_at(i);  // returns &[S] slice

// Extract a single component across all times:
let x_component = result.component(0);  // Vec<S>
```

### Generic Over Scalar

All algorithms are generic over `S: Scalar`, meaning the same code
works for both `f32` and `f64`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
fn solve_exponential<S: Scalar>(rtol: S) -> S {
    let problem = OdeProblem::new(
        |_t, y: &[S], dydt: &mut [S]| { dydt[0] = -y[0]; },
        S::ZERO, S::ONE, &[S::ONE],
    );
    let options = SolverOptions::default().rtol(rtol);
    let result = DoPri5::solve(&problem, S::ZERO, S::ONE, &[S::ONE], &options).unwrap();
    result.y_final()[0]
}

let f64_result: f64 = solve_exponential(1e-10);
let f32_result: f32 = solve_exponential(1e-5);
```
