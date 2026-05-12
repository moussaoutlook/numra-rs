# numra-ide

**Volterra integro-differential equation solvers for the [Numra](https://numra-rs.org/) workspace — general quadrature solver plus a Prony-series fast path for exponential kernels.**

[![Crates.io](https://img.shields.io/crates/v/numra-ide.svg)](https://crates.io/crates/numra-ide)
[![docs.rs](https://docs.rs/numra-ide/badge.svg)](https://docs.rs/numra-ide)

Solves Volterra IDEs `y'(t) = f(t, y) + ∫₀ᵗ K(t, s, y(s)) ds` for history-dependent dynamics. The general `VolterraSolver` handles arbitrary memory kernels via quadrature; the specialized `PronySolver` exploits sums of exponentials for an `O(N)` rather than `O(N²)` step cost.

## Example

```rust
use numra_ide::kernels::ExponentialKernel;
use numra_ide::{IdeOptions, IdeSolver, IdeSystem, Kernel, VolterraSolver};

// Viscoelastic: y' = -k·y + ∫ K(t-s)·y(s) ds with exponential kernel
struct Viscoelastic {
    k: f64,
    kernel: ExponentialKernel<f64>,
}

impl IdeSystem<f64> for Viscoelastic {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -self.k * y[0];
    }
    fn kernel(&self, t: f64, s: f64, y: &[f64], k: &mut [f64]) {
        k[0] = self.kernel.evaluate(t - s) * y[0];
    }
}
```

## What's in this crate

**Solvers:**

- **`VolterraSolver`** — general Volterra IDE solver using quadrature
- **`VolterraRK4Solver`** — RK4-based variant
- **`PronySolver`** — `O(N)` fast path for sum-of-exponentials kernels via Prony reduction

**Kernels:**

- **`ExponentialKernel`** — `K(t, s) = a · exp(-b(t-s))`
- **`PowerLawKernel`** — `K(t, s) = (t-s)^{-α}`, fractional memory
- **`PronyKernel`** — sum-of-exponentials representation
- **`Kernel` trait** — user-defined kernels

## Composes with

- [`numra-fde`](https://docs.rs/numra-fde) — power-law kernels share the long-memory machinery of fractional derivatives
- [`numra-interp`](https://docs.rs/numra-interp) for post-hoc resampling of viscoelastic relaxation curves
- [`numra-fit`](https://docs.rs/numra-fit) for fitting Prony coefficients to measured kernel responses

See [cross-formalism tests](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/composition_tests.rs) for IDE-reduces-to-ODE consistency and memory-kernel composition.

## Install

```toml
[dependencies]
numra-ide = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-ide>
- **Book**: [Integro-DEs](https://book.numra-rs.org/ch04-beyond-odes/integro-des/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-ide>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
