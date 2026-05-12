# numra-fft

**FFT and spectral analysis for the [Numra](https://numra-rs.org/) workspace — complex / real FFT, IFFT, 2D FFT, convolution, PSD, Welch, STFT, and windowing — built on [rustfft](https://github.com/ejmahler/RustFFT).**

[![Crates.io](https://img.shields.io/crates/v/numra-fft.svg)](https://crates.io/crates/numra-fft)
[![docs.rs](https://docs.rs/numra-fft/badge.svg)](https://docs.rs/numra-fft)

Generic over `Scalar` (`f32` / `f64`), with FFT computation internally in `f64` for numerical stability. Covers the spectrum from raw FFT/IFFT through windowed power-spectral-density and short-time Fourier transforms.

## Example

```rust
use numra_fft::{fft, ifft, Complex};

let signal: Vec<Complex<f64>> = vec![
    Complex::new(1.0_f64, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
];
let spectrum = fft(&signal);
let recovered = ifft(&spectrum);

for (a, b) in signal.iter().zip(recovered.iter()) {
    assert!((a.re - b.re).abs() < 1e-12);
}
```

## What's in this crate

- **Core FFT**: `fft`, `ifft`, `fft2` (complex 1D / 2D)
- **Real FFT**: `rfft`, `irfft` (efficient real-to-complex)
- **Complex type**: `Complex`, `ComplexF64`
- **Spectral analysis**: `psd` (power spectral density), `welch` (Welch's method), `stft` / `StftResult` (short-time Fourier transform)
- **Convolution**: `fftconvolve`, `fftcorrelate`
- **Utilities**: `fftfreq`, `fftshift`, `ifftshift`, `window_func`, `Window` (Hann, Hamming, Blackman, Bartlett, …)

## Composes with

- [`numra-ode`](https://docs.rs/numra-ode) — spectral analysis of time-series ODE solutions
- [`numra-signal`](https://docs.rs/numra-signal) — FFT backbone for `resample`, `hilbert`, and FIR windowed-sinc design (`filtfilt` itself is SOS-based IIR and stays in the time domain)
- [`numra-stats`](https://docs.rs/numra-stats) — PSD treated as a frequency-domain statistic
- [`numra-interp`](https://docs.rs/numra-interp) — resampling onto uniform grids before FFT

See [interop workflows](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/interop_workflows.rs) for the verified ODE → FFT → signal-processing → peak-detection workflow.

## Install

```toml
[dependencies]
numra-fft = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-fft>
- **Book**: [FFT and spectral](https://book.numra-rs.org/ch08-signal-processing/fft-and-spectral/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-fft>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
