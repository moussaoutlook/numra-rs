# Installation

## Adding Numra to Your Project

Add the facade crate to your `Cargo.toml`:

```toml
[dependencies]
numra = "0.1"
```

This gives you access to all Numra components through a single dependency.
The facade crate re-exports everything under organized namespaces:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};
use numra::optim::{bfgs_minimize, OptimOptions};
use numra::linalg::{DenseMatrix, Matrix};
use numra::dsp::butter;   // signal processing
use numra::fft;
use numra::stats;
// ... and more
```

## Using Individual Crates

If you only need specific functionality, depend on individual crates to
minimize compile times:

```toml
[dependencies]
# Just ODE solvers
numra-ode = "0.1"

# Just linear algebra
numra-linalg = "0.1"

# Just FFT
numra-fft = "0.1"
```

The crate hierarchy ensures you only pull in what you need:

```text
numra-core          (always required -- Scalar, Vector traits)
numra-linalg        (adds matrices, factorizations)
numra-nonlinear     (adds Newton solver)
numra-ode           (adds ODE solvers)
numra-sde           (adds SDE solvers, independent of numra-ode)
```

## Minimum Supported Rust Version

Numra requires **Rust 2021 edition** (1.56+). We recommend using the latest
stable release for best performance.

## Feature Flags

The `numra-core` crate supports `no_std`:

```toml
[dependencies]
numra-core = { version = "0.1", default-features = false }
```

With `default-features = false`, the `std` feature is disabled and `numra-core`
uses `libm` for all math functions. This is suitable for embedded targets.

## Verifying Your Installation

Create a simple test to verify everything works:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

fn main() {
    // Solve dy/dt = -y, y(0) = 1 (exact solution: y = e^(-t))
    let problem = OdeProblem::new(
        |_t, y: &[f64], dydt: &mut [f64]| { dydt[0] = -y[0]; },
        0.0, 1.0, vec![1.0],
    );
    let options = SolverOptions::default();
    let result = DoPri5::solve(&problem, 0.0, 1.0, &[1.0], &options).unwrap();

    let y_final = result.y_final().unwrap()[0];
    let exact = (-1.0_f64).exp();
    println!("Computed: {y_final:.10}");
    println!("Exact:    {exact:.10}");
    println!("Error:    {:.2e}", (y_final - exact).abs());
}
```

You should see an error on the order of 1e-7 or smaller.

## Dependencies

Numra's key external dependencies:

| Dependency | Version | Purpose |
|------------|---------|---------|
| [faer](https://crates.io/crates/faer) | 0.20 | Linear algebra backend |
| [rustfft](https://crates.io/crates/rustfft) | 6.2 | FFT backend |
| [libm](https://crates.io/crates/libm) | 0.2 | no_std math functions |
| [rand](https://crates.io/crates/rand) | 0.8 | Random number generation |
| [thiserror](https://crates.io/crates/thiserror) | 1.0 | Error handling |
| [rayon](https://crates.io/crates/rayon) | 1.10 | Parallel Monte Carlo |

All dependencies are pure Rust. No system libraries, C compilers, or LAPACK
installations are needed.
