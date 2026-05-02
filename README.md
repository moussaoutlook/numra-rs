# Numra

**Numerical methods for Rust — differential equations, optimization, linear algebra, and more in one coherent workspace.**

[![CI](https://github.com/moussaoutlook/numra-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/moussaoutlook/numra-rs/actions/workflows/ci.yml)
[![Website](https://img.shields.io/badge/website-numra--rs.org-blue)](https://numra-rs.org/)
[![Book](https://img.shields.io/badge/book-book.numra--rs.org-blue)](https://book.numra-rs.org/)
[![MSRV](https://img.shields.io/badge/MSRV-1.83-blue)](Cargo.toml)
[![License](https://img.shields.io/badge/license-Academic%20%26%20Research%20NC-orange)](LICENSE)

Numra is a **Cargo workspace** with a unified `[numra](numra)` facade: one dependency surface, shared `Scalar` / `Vector` abstractions, and **native Rust** implementations of the solvers and methods in-tree (contrasted with stacks that lean on FFI to classic C/Fortran libraries for the same breadth). Dense linear algebra builds on **[faer](https://github.com/sarah-ek/faer-rs)**.

> *“Numra is the only comprehensive native-Rust source-available numerical stack for differential equations and related methods we are aware of”* — to the best of our knowledge after surveying Rust projects; *“comprehensive”* here means the breadth documented in this repository’s [public API inventory](docs/audit/public-api-method-inventory.md) and the [mdBook](numra-book).

---

## Composability

Numra is built so **domains compose under one API**: you can chain solvers, uncertainty, sensitivity, interpolation, quadrature, optimization, autodiff, DSP, FFT, statistics, and PDE discretizations without glue layers or ad hoc type bridges.

End-to-end workflows are **enforced by integration tests**, including:


| Workflow                                                | Verified in                                                                      |
| ------------------------------------------------------- | -------------------------------------------------------------------------------- |
| ODE → cubic spline interpolation → numerical quadrature | `[workflow_ode_interp_integrate](numra/tests/interop_workflows.rs)`              |
| ODE → FFT → signal processing → peak detection          | `[workflow_ode_fft_signal_peaks](numra/tests/interop_workflows.rs)`              |
| Parameter estimation → sensitivity → uncertainty        | `[workflow_param_est_sensitivity_uncertainty](numra/tests/interop_workflows.rs)` |
| Autodiff → optimization → curve fitting                 | `[workflow_autodiff_optim_fit](numra/tests/interop_workflows.rs)`                |
| PDE (method of lines) → statistics                      | `[workflow_pde_statistics](numra/tests/interop_workflows.rs)`                    |
| Sampling → Monte Carlo ODE                              | `[workflow_stats_monte_carlo_ode](numra/tests/interop_workflows.rs)`             |


Cross-formalism checks (ODE / FDE / IDE / DDE / PDE / SPDE building blocks) live in `[composition_tests.rs](numra/tests/composition_tests.rs)`. Uncertainty, sensitivity, and ODE solves (including automatic stiff/nonstiff selection) are composed in `[integration_tests.rs](numra/tests/integration_tests.rs)` (`test_full_composition`, `test_week4_full_composition`).

For narrative patterns (ODE + optimization, AD, uncertainty, …), see the book chapter **[Composing solvers](numra-book/src/ch12-advanced-topics/composing-solvers.md)**.

```bash
cargo test -p numra --test interop_workflows
cargo test -p numra --test composition_tests
```

---

## Scope at a glance


| Area                                                                                   | Crate / module (facade)      |
| -------------------------------------------------------------------------------------- | ---------------------------- |
| Core (`Scalar`, `Vector`, `Signal`, errors, uncertainty)                               | `numra::core` → `numra-core` |
| ODE / DAE (DoPri5, Tsit5, Verner, Radau5, ESDIRK, BDF, `Auto`, events, sensitivity, …) | `numra::ode`                 |
| SDE                                                                                    | `numra::sde`                 |
| DDE                                                                                    | `numra::dde`                 |
| FDE                                                                                    | `numra::fde`                 |
| IDE                                                                                    | `numra::ide`                 |
| PDE                                                                                    | `numra::pde`                 |
| SPDE                                                                                   | `numra::spde`                |
| Linear algebra                                                                         | `numra::linalg`              |
| Nonlinear solvers                                                                      | `numra::nonlinear`           |
| Optimization                                                                           | `numra::optim`               |
| Optimal control / parameter estimation                                                 | `numra::ocp`                 |
| Quadrature                                                                             | `numra::integrate`           |
| Interpolation                                                                          | `numra::interp`              |
| Special functions                                                                      | `numra::special`             |
| FFT                                                                                    | `numra::fft`                 |
| Statistics                                                                             | `numra::stats`               |
| Curve fitting                                                                          | `numra::fit`                 |
| Autodiff                                                                               | `numra::autodiff`            |
| Signal processing                                                                      | `numra::dsp`                 |


Details: [public API inventory](docs/audit/public-api-method-inventory.md).

---

## Quick start

```rust
use numra::ode::{DoPri5, OdeProblem, Solver, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -y[0];
    },
    0.0,
    2.0,
    vec![1.0],
);
let opts = SolverOptions::default().rtol(1e-8);
let result = DoPri5::solve(&problem, 0.0, 2.0, &[1.0], &opts).expect("solve");

assert!(result.success);
let y_end = result.y_final().expect("trajectory")[0];
assert!((y_end - (-2.0_f64).exp()).abs() < 1e-5);
```

```bash
cargo build
cargo test --workspace
cargo run -p numra --example lorenz
```

More examples: `cargo run -p numra --example van_der_pol`, `solver_zoo`, `heat_equation`, `gbm_monte_carlo`, … (see `[numra/Cargo.toml](numra/Cargo.toml)` `[[example]]` list).

---

## Documentation

Primary website: `https://numra-rs.org/`  
User docs (book): `https://book.numra-rs.org/`


| Resource            | How                                                                                                                     |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **mdBook**          | Online: `https://book.numra-rs.org/` · Local: `(cd numra-book && mdbook build)` · Sources: `[numra-book/](numra-book/)` |
| **Rust API**        | `cargo doc --workspace --no-deps --open`                                                                                |
| **Changelog**       | `[CHANGELOG.md](CHANGELOG.md)`                                                                                          |
| **Release / audit** | `[docs/audit/](docs/audit/)` — inventory, coverage matrices, checklists                                                 |


---

## Contributing and release quality

CI (GitHub Actions) runs `rustfmt`, `clippy` with `-D warnings`, full workspace tests and doctests, example builds, `rustdoc` with `-D warnings`, book content checks (drift, snippet harness, math lint), and `cargo deny`. The mdBook itself (`numra-book/`) is built locally with `mdbook build`; the online book at `book.numra-rs.org` is the Astro/Starlight version under `website/book/`.

Local parity with the release bar:

```bash
bash scripts/audit_release.sh
```

**Third-party licenses:** the workspace allow-list lives in `[deny.toml](deny.toml)` (enforced in CI). To audit dependency SPDX expressions against that list:

```bash
cargo deny check licenses
# full advisories + bans + licenses + sources:
cargo deny check
```

---

## License

Numra is licensed under the **Numra Academic & Research License (Non-Commercial)**.

- **Academic & research use**: permitted at no cost
- **Commercial / for-profit use**: requires a separate commercial license — contact `contact@spectralautomata.com`

See `[LICENSE](LICENSE)`.

## CLA

All contributions require agreeing to the Contributor License Agreement: `[CLA.md](CLA.md)`.