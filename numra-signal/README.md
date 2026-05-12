# numra-signal

**Digital signal processing for the [Numra](https://numra-rs.org/) workspace — IIR (Butterworth, Chebyshev I) and FIR filter design, zero-phase filtering, resampling, Hilbert transform, and peak detection.**

[![Crates.io](https://img.shields.io/crates/v/numra-signal.svg)](https://crates.io/crates/numra-signal)
[![docs.rs](https://docs.rs/numra-signal/badge.svg)](https://docs.rs/numra-signal)

Re-exposed at the umbrella level as `numra::dsp`. SOS filter representation, FFT-based resampling and Hilbert transform (via `numra-fft`), and peak detection with height / distance / prominence constraints.

## Example

```rust
use numra_signal::{butter, filtfilt, find_peaks, PeakOptions};

// Design a 4th-order Butterworth lowpass at 10 Hz, sampled at 100 Hz
let sos = butter(4, 10.0, 100.0).unwrap();

// Generate a noisy 3-Hz signal contaminated with 40-Hz noise
let pi2 = 2.0 * std::f64::consts::PI;
let x: Vec<f64> = (0..200).map(|i| {
    let t = i as f64 / 100.0;
    (pi2 * 3.0 * t).sin() + 0.3 * (pi2 * 40.0 * t).sin()
}).collect();

// Zero-phase filter and detect peaks
let y = filtfilt(&sos, &x);
let peaks = find_peaks(&y, &PeakOptions::default().height(0.5));

assert!(!peaks.is_empty());
```

## What's in this crate

- **Filter design**: `butter` (Butterworth), `cheby1` (Chebyshev Type I) — both return second-order-section coefficients
- **FIR design**: `fir_filter`, `firwin` (windowed-sinc)
- **Filter application**: `sosfilt` (forward), `filtfilt` (zero-phase forward-backward), `SosFilter`
- **Resampling**: `resample` (FFT-based, non-integer ratios)
- **Hilbert transform**: `hilbert` (analytic signal), `envelope`, `instantaneous_frequency`
- **Peak detection**: `find_peaks` with `PeakOptions` (height, distance, prominence)
- **`SignalError`** — DSP-specific error reporting

## Composes with

- [`numra-fft`](https://docs.rs/numra-fft) — FFT backbone for `resample`, `hilbert`, and convolution-based filtering
- [`numra-ode`](https://docs.rs/numra-ode) — peak detection on time-domain ODE solutions
- [`numra-sde`](https://docs.rs/numra-sde) — filtering noisy SDE realizations before peak detection
- [`numra-stats`](https://docs.rs/numra-stats) — descriptive statistics on detected peak heights and inter-peak spacings

See [interop workflows](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/interop_workflows.rs) for the verified ODE → FFT → signal-processing → peak-detection workflow.

## Install

```toml
[dependencies]
numra-signal = "0.1"
```

Or via the umbrella crate (re-exported as `numra::dsp`):

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-signal>
- **Book**: [Filtering signals](https://book.numra-rs.org/ch08-signal-processing/filtering-signals/) · [Filter design](https://book.numra-rs.org/ch08-signal-processing/filter-design/) · [Hilbert transform](https://book.numra-rs.org/ch08-signal-processing/hilbert-transform/) · [Resampling and peaks](https://book.numra-rs.org/ch08-signal-processing/resampling-peaks/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-signal>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
