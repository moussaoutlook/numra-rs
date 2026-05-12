# numra-core

**Foundational traits and types for the [Numra](https://numra-rs.org/) workspace — `Scalar`, `Vector`, `Signal`, `Uncertain<S>`, and the workspace error model.**

[![Crates.io](https://img.shields.io/crates/v/numra-core.svg)](https://crates.io/crates/numra-core)
[![docs.rs](https://docs.rs/numra-core/badge.svg)](https://docs.rs/numra-core)

`numra-core` defines the numeric vocabulary every other Numra crate shares: a `Scalar` trait that abstracts over `f32` / `f64`, a `Vector` trait for state representations, a `Signal` trait for time-dependent forcings, and an `Uncertain<S>` primitive that propagates first-order mean / variance through arbitrary scalar pipelines. Optional `no-std` support for embedded targets.

## Example

```rust
use numra_core::Uncertain;

// 1.0 ± 0.1 combined with 2.0 ± 0.2 — means add, variances add
let x = Uncertain::from_std(1.0_f64, 0.1);
let y = Uncertain::from_std(2.0_f64, 0.2);
let sum = x.add(&y);

assert!((sum.mean - 3.0).abs() < 1e-12);
assert!((sum.std() - (0.01_f64 + 0.04).sqrt()).abs() < 1e-12);
```

## What's in this crate

- **`Scalar` trait** — `f32` / `f64` abstraction with `from_f64`, `to_f64`, common constants, trig / exp / log methods
- **`Vector` trait** — state-vector backbone for solvers and discretizations
- **`Signal` trait** — `Harmonic`, `Chirp`, `Pulse`, `Ramp`, `Step`, `Tabulated` time-dependent forcings (plus `FromFile` with `std`)
- **`Uncertain<S>`** — mean ± std primitive with `add` / `sub` / `mul` / `div` / `scale` / `add_const` methods
- **`Interval<S>`** — interval-arithmetic primitive for worst-case error bounds
- **`compute_sensitivities` / `ParameterSensitivity` / `ParameterSensitivityResult`** — finite-difference parameter-sensitivity helper and result type
- **`NumraError` / `NumraResult` / `LinalgError`** — workspace-wide error model
- **`std` feature** (default-on) — disable for `no-std` targets

## Composes with

Every crate in the Numra workspace builds on `numra-core`. The `Scalar` and `Vector` traits define the shared numeric vocabulary that lets ODE / SDE / PDE solvers, optimizers, autodiff, and statistics exchange data without conversion or `f64` bottlenecks. `Signal` lets a time-dependent forcing drop into any solver. `Uncertain<S>` carries first-order uncertainty through arbitrary scalar pipelines.

## Install

```toml
[dependencies]
numra-core = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-core>
- **Book**: [Architecture overview](https://book.numra-rs.org/ch01-introduction/architecture-overview/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-core>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
