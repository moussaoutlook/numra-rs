# no_std Usage

Numra's core crate supports `no_std` environments, making it usable on
embedded systems, WASM targets, and other constrained platforms.

## What is `no_std`?

Rust's standard library (`std`) depends on an operating system for features
like heap allocation, file I/O, and threading. The `no_std` attribute removes
this dependency, enabling bare-metal usage.

Numra uses a layered approach:
- **`numra-core`**: Fully `no_std` compatible (with `alloc`)
- **Higher crates** (`numra-ode`, `numra-linalg`, etc.): Require `std`

## The `std` Feature

`numra-core` uses a Cargo feature flag:

```toml
[dependencies]
numra-core = { version = "0.1", default-features = false }
```

With `default-features = false`, the crate compiles without `std`. It still
requires the `alloc` crate for `Vec` and `String`.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
#![no_std]
extern crate alloc;

use numra_core::{Scalar, Uncertain, Interval};
use alloc::vec;
use alloc::vec::Vec;
```

## The Scalar Trait in `no_std`

The `Scalar` trait uses `libm` for math functions when `std` is unavailable.
All transcendental functions (sin, cos, exp, ln, sqrt, etc.) work identically
in both modes:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_core::Scalar;

fn compute_decay<S: Scalar>(k: S, t: S) -> S {
    (-k * t).exp()  // Uses libm::exp in no_std, std::f64::exp in std
}
```

### Available math operations

All `Scalar` methods work in `no_std`:

| Category | Functions |
|----------|-----------|
| **Basic** | `abs`, `sqrt`, `cbrt`, `powi`, `powf` |
| **Trig** | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` |
| **Hyperbolic** | `sinh`, `cosh`, `tanh` |
| **Exponential** | `exp`, `ln`, `log2`, `log10` |
| **Rounding** | `floor`, `ceil`, `round` |
| **Comparison** | `min_val`, `max_val`, `clamp` |
| **Classification** | `is_finite`, `is_nan`, `is_infinite` |

## f32 Support

All Numra algorithms are generic over `S: Scalar`, meaning they work with both
`f32` and `f64`. Use `f32` to halve memory usage on memory-constrained targets:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_core::Uncertain;

// f32 uncertainty propagation
let x = Uncertain::<f32>::from_std(10.0, 0.5);
let y = Uncertain::<f32>::from_std(5.0, 0.3);
let result = x.mul(&y);
println!("f32: {} +/- {}", result.mean, result.std());

// f64 for full precision
let x = Uncertain::<f64>::from_std(10.0, 0.5);
let y = Uncertain::<f64>::from_std(5.0, 0.3);
let result = x.mul(&y);
println!("f64: {} +/- {}", result.mean, result.std());
```

### f32 vs f64 precision guide

| Aspect | f32 | f64 |
|--------|-----|-----|
| **Digits** | ~7 decimal | ~15 decimal |
| **Epsilon** | 1.19e-7 | 2.22e-16 |
| **Range** | 1.18e-38 to 3.4e+38 | 2.23e-308 to 1.8e+308 |
| **Memory** | 4 bytes | 8 bytes |
| **Best for** | Embedded, GPU, bulk data | Scientific computing |

**Caution**: For ODE solving, `f32` limits achievable accuracy. Tolerances
below `rtol ~ 1e-5` may not be meaningful with `f32`.

## Crate Compatibility

| Crate | `no_std` | Notes |
|-------|----------|-------|
| `numra-core` | Yes (with alloc) | Scalar, Vector, Signal, Uncertain, Interval |
| `numra-ode` | No | Requires std for complex solvers |
| `numra-linalg` | No | Uses faer (requires std) |
| `numra-nonlinear` | No | Depends on numra-linalg |
| `numra-integrate` | No | Requires std |
| `numra-interp` | No | Requires std |
| `numra-autodiff` | No | Requires std |
| `numra-special` | No | Requires std |
| `numra-fft` | No | Uses rustfft (requires std) |
| `numra-signal` | No | Depends on numra-fft |
| `numra-stats` | No | Requires std |
| `numra-fit` | No | Depends on numra-linalg |
| `numra-optimize` | No | Depends on numra-linalg |
| `numra-ocp` | No | Depends on numra-ode |

## Signal Types in `no_std`

The `Signal` trait and most signal types work in `no_std`. The exception is
`FromFile`, which requires filesystem access:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_core::signal::{Harmonic, Step, Ramp, Pulse};

// These all work in no_std
let sine = Harmonic::new(1.0, 10.0, 0.0);
let step = Step::new(1.0, 0.5);   // amplitude=1, at t=0.5
let ramp = Ramp::new(0.0, 1.0);   // from t=0 to t=1
```

## WASM Targets

`numra-core` compiles to WebAssembly without modification:

```toml
# .cargo/config.toml
[build]
target = "wasm32-unknown-unknown"
```

For WASM with a JavaScript host, you can use `wasm-bindgen` to expose Numra
functions:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use wasm_bindgen::prelude::*;
use numra_core::{Uncertain, Scalar};

#[wasm_bindgen]
pub fn propagate_uncertainty(mean: f64, std: f64, scale: f64) -> Vec<f64> {
    let x = Uncertain::<f64>::from_std(mean, std);
    let result = x.scale(scale);
    vec![result.mean, result.std()]
}
```

## Best Practices

1. **Use `numra-core` directly** for embedded projects; don't pull in the
   full `numra` facade crate
2. **Prefer `f64`** unless memory is truly constrained -- the precision
   difference matters for numerical methods
3. **Test with both features** -- run `cargo test` with and without the `std`
   feature to catch accidental `std` dependencies
4. **Pre-allocate** vectors where possible to reduce allocator pressure on
   constrained targets
