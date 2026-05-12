# numra-special

**Special mathematical functions for the [Numra](https://numra-rs.org/) workspace — gamma, error functions, Bessel, elliptic integrals, Airy, hypergeometric, orthogonal polynomials, zeta, and more.**

[![Crates.io](https://img.shields.io/crates/v/numra-special.svg)](https://crates.io/crates/numra-special)
[![docs.rs](https://docs.rs/numra-special/badge.svg)](https://docs.rs/numra-special)

A compact, native-Rust library of special functions sufficient to back the distribution PDFs in `numra-stats`, the fractional-derivative analytics in `numra-fde`, and the analytical reference solutions used throughout the workspace's tests.

## Example

```rust
use numra_special::{erf, gamma};

// Γ(1/2) = √π
assert!((gamma(0.5_f64) - std::f64::consts::PI.sqrt()).abs() < 1e-10);

// erf(1) ≈ 0.8427
assert!((erf(1.0_f64) - 0.8427007929_f64).abs() < 1e-8);
```

## What's in this crate

- **Gamma family**: `gamma`, `lgamma`, `digamma`, `beta`, `gammainc`, `gammaincc`, `betainc`
- **Error functions**: `erf`, `erfc`, `erfinv`, `erfcinv`
- **Bessel**: `besselj`, `bessely`, `besseli`, `besselk`
- **Elliptic integrals**: `ellipk` (complete K), `ellipe` (complete E), `ellipf` (incomplete F)
- **Airy**: `airy_ai`, `airy_bi`
- **Hypergeometric**: `hyp1f1`, `hyp2f1`
- **Orthogonal polynomials**: `legendre_p`, `legendre_plm`, `hermite_h`, `laguerre_l`, `chebyshev_t`
- **Zeta**: `zeta`, `hurwitz_zeta`
- **Miscellaneous**: `dawson`, `fresnel_c`, `fresnel_s`, `mittag_leffler`, `mittag_leffler_1`

## Composes with

- [`numra-stats`](https://docs.rs/numra-stats) — `gamma`, `beta`, `erf` inside continuous-distribution PDFs and CDFs (direct dependency)
- [`numra-fde`](https://docs.rs/numra-fde) — shares the Mittag-Leffler function family; the two crates ship independent implementations with different signatures, useful for cross-checking
- [`numra-integrate`](https://docs.rs/numra-integrate) — special-function integrands appearing in quadrature applications

## Install

```toml
[dependencies]
numra-special = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-special>
- **Book**: [Special functions](https://book.numra-rs.org/ch07-calculus-and-analysis/special-functions/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-special>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
