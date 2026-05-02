---
title: "FFT & Spectral Analysis"
---

The Fast Fourier Transform (FFT) is the backbone of digital signal processing.
Numra's `numra-fft` crate provides a full FFT toolkit -- forward and inverse
transforms, real-valued optimized variants, power spectral density, Welch's
method, the short-time Fourier transform, windowing functions, and
frequency-domain convolution. Internally it delegates to `rustfft` for the
heavy lifting, while presenting a clean generic API over `Scalar` types.

## Core FFT Functions

### Complex FFT and IFFT

The `fft` function computes the discrete Fourier transform of a complex
sequence. The `ifft` function computes the inverse, normalized by $1/N$ so
that `ifft(fft(x)) == x`.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::{fft, ifft, Complex};

// Build a 4-sample complex signal
let signal = vec![
    Complex::new(1.0_f64, 0.0),
    Complex::new(2.0, 0.0),
    Complex::new(3.0, 0.0),
    Complex::new(4.0, 0.0),
];

// Forward transform
let spectrum = fft(&signal);
assert_eq!(spectrum.len(), 4);

// DC component = sum of all samples
assert!((spectrum[0].re - 10.0).abs() < 1e-12);

// Inverse transform recovers the original
let recovered = ifft(&spectrum);
for (a, b) in signal.iter().zip(recovered.iter()) {
    assert!((a.re - b.re).abs() < 1e-12);
}
```

The DFT is defined as:

$$
X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j 2\pi k n / N}
$$

and the inverse:

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \, e^{j 2\pi k n / N}
$$

### Real-Valued FFT

When the input is a real signal (no imaginary part), the DFT has conjugate
symmetry: $X[N-k] = X[k]^*$. The `rfft` function exploits this and returns
only the $N/2+1$ non-redundant complex coefficients. The `irfft` function
reconstructs the original real signal.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::{rfft, irfft};

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let spectrum = rfft(&x);
assert_eq!(spectrum.len(), 5); // N/2 + 1 = 5

// Round-trip
let recovered = irfft(&spectrum, x.len());
for (a, b) in x.iter().zip(recovered.iter()) {
    assert!((a - b).abs() < 1e-12);
}
```

### 2-D FFT

For image processing or 2-D data, `fft2` computes a row-major 2-D DFT by
performing FFTs along rows, then along columns.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::{fft2, Complex};

// 2x2 constant matrix -- all energy in DC corner
let data = vec![
    Complex::new(1.0, 0.0), Complex::new(1.0, 0.0),
    Complex::new(1.0, 0.0), Complex::new(1.0, 0.0),
];
let result = fft2(&data, 2, 2);
assert!((result[0].re - 4.0).abs() < 1e-12); // DC bin
assert!(result[1].abs() < 1e-12);             // all other bins ~0
```

## Frequency Utilities

### fftfreq

`fftfreq` returns the DFT sample frequencies in cycles per unit of sample
spacing. The layout matches the standard FFT output ordering:
$[0, 1, \ldots, N/2-1, -N/2, \ldots, -1] / (dt \cdot N)$.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::fftfreq;

let freqs: Vec<f64> = fftfreq(8, 1.0);
// freqs = [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
assert!((freqs[1] - 0.125).abs() < 1e-14);
assert!((freqs[4] - (-0.5)).abs() < 1e-14);
```

### fftshift and ifftshift

`fftshift` rearranges the FFT output so that the zero-frequency component is
in the center -- useful for plotting. `ifftshift` reverses the operation.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::{fftshift, ifftshift};

let mut x = vec![0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0];
fftshift(&mut x);
// x is now [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

ifftshift(&mut x);
// x is back to [0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0]
```

## Window Functions

Applying a window before the FFT reduces spectral leakage. Numra provides six
window types via the `Window` enum.

| Window | Formula | Sidelobe (dB) | Use Case |
|---|---|---|---|
| `Rectangular` | $w[n] = 1$ | -13 | Transient analysis, exact-bin signals |
| `Hann` | $0.5(1 - \cos\frac{2\pi n}{N-1})$ | -31 | General-purpose spectral analysis |
| `Hamming` | $0.54 - 0.46\cos\frac{2\pi n}{N-1}$ | -43 | Speech processing, filter design |
| `Blackman` | $0.42 - 0.5\cos\frac{2\pi n}{N-1} + 0.08\cos\frac{4\pi n}{N-1}$ | -58 | High dynamic range spectra |
| `Kaiser(beta)` | Bessel-based parametric | variable | Adjustable leakage/resolution trade-off |
| `FlatTop` | Five-term cosine sum | -93 | Accurate amplitude measurement |

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::{window_func, Window};

let hann: Vec<f64> = window_func(&Window::Hann, 64);
assert!(hann[0].abs() < 1e-14);   // tapers to zero at edges
assert!(hann[63].abs() < 1e-14);

let kaiser: Vec<f64> = window_func(&Window::Kaiser(5.0), 64);
// Kaiser is symmetric
assert!((kaiser[0] - kaiser[63]).abs() < 1e-12);
```

## Power Spectral Density

### Periodogram

The `psd` function computes a one-sided power spectral density using a single
windowed periodogram. It returns `(frequencies, psd_values)` in units of Hz
and power/Hz.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::{psd, Window};

// Generate a 10 Hz sine at 256 Hz sample rate
let n = 256;
let fs = 256.0;
let pi2 = 2.0 * std::f64::consts::PI;
let x: Vec<f64> = (0..n)
    .map(|k| (pi2 * 10.0 * k as f64 / fs).sin())
    .collect();

let (freqs, power) = psd(&x, fs, &Window::Hann);

// Find the peak frequency
let peak_idx = power.iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .unwrap().0;
println!("Peak at {} Hz", freqs[peak_idx]); // ~10 Hz
```

### Welch's Method

Welch's method averages periodograms of overlapping segments for reduced
variance. This is the recommended approach for noisy data.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::{welch, Window};

let n = 1024;
let fs = 100.0;
let pi2 = 2.0 * std::f64::consts::PI;
let x: Vec<f64> = (0..n)
    .map(|k| {
        let t = k as f64 / fs;
        (pi2 * 10.0 * t).sin() + 0.5 * (pi2 * 25.0 * t).sin()
    })
    .collect();

let (freqs, power) = welch(&x, fs, 256, 128, &Window::Hann);
assert!(!freqs.is_empty());
// All PSD values are non-negative
assert!(power.iter().all(|&v| v >= 0.0));
```

**Parameters:**

| Parameter | Description |
|---|---|
| `fs` | Sampling frequency in Hz |
| `nperseg` | Segment length (FFT size) |
| `noverlap` | Number of overlapping samples between segments |
| `window` | Window function to apply to each segment |

**Rule of thumb:** Use 50% overlap (`noverlap = nperseg / 2`) with a Hann
window for a good bias-variance trade-off.

## Short-Time Fourier Transform

The STFT reveals how the frequency content of a signal changes over time. It
slides a window across the signal and computes a local FFT for each position.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::{stft, Window};

let n = 512;
let fs = 100.0;
let pi2 = 2.0 * std::f64::consts::PI;

// Chirp signal: frequency sweeps from 5 to 40 Hz
let x: Vec<f64> = (0..n)
    .map(|k| {
        let t = k as f64 / fs;
        let freq = 5.0 + 35.0 * t / (n as f64 / fs);
        (pi2 * freq * t).sin()
    })
    .collect();

let result = stft(&x, fs, 64, 48, &Window::Hann);

println!("Time bins:      {}", result.times.len());
println!("Frequency bins: {}", result.frequencies.len());
println!("Spectrogram:    {} x {}", result.magnitude.len(),
         result.magnitude[0].len());
```

The `StftResult` struct contains:

| Field | Type | Description |
|---|---|---|
| `times` | `Vec<S>` | Time values for each window center |
| `frequencies` | `Vec<S>` | Frequency bin centers in Hz |
| `magnitude` | `Vec<Vec<S>>` | Magnitude matrix `[time_idx][freq_idx]` |

## Convolution and Correlation

FFT-based convolution is asymptotically faster than direct convolution for
long signals: $O(N \log N)$ vs $O(N^2)$.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::{fftconvolve, fftcorrelate};

let a = vec![1.0, 1.0];
let b = vec![1.0, 1.0, 1.0];

// Convolution: [1,1] * [1,1,1] = [1, 2, 2, 1]
let conv = fftconvolve(&a, &b);
assert_eq!(conv.len(), 4); // len(a) + len(b) - 1
assert!((conv[1] - 2.0).abs() < 1e-12);

// Cross-correlation (equivalent to convolve(a, reverse(b)))
let corr = fftcorrelate(&a, &b);
assert_eq!(corr.len(), 4);
```

## Complete Example: Multi-Tone Frequency Analysis

Putting it all together -- analyze a signal composed of three sine waves
embedded in noise.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::{rfft, fftfreq, window_func, welch, Window};

fn main() {
    let fs = 1000.0;          // 1 kHz sampling
    let n = 4096;
    let pi2 = 2.0 * std::f64::consts::PI;

    // Three tones at 50, 120, and 300 Hz, plus noise
    let x: Vec<f64> = (0..n)
        .map(|k| {
            let t = k as f64 / fs;
            1.0 * (pi2 * 50.0 * t).sin()
          + 0.5 * (pi2 * 120.0 * t).sin()
          + 0.3 * (pi2 * 300.0 * t).sin()
          + 0.2 * ((k as f64 * 1.23456).sin()) // pseudo-noise
        })
        .collect();

    // --- Method 1: Windowed FFT ---
    let win = window_func(&Window::Hann, n);
    let windowed: Vec<f64> = x.iter().zip(win.iter())
        .map(|(&xi, &wi)| xi * wi)
        .collect();
    let spectrum = rfft(&windowed);
    let magnitudes: Vec<f64> = spectrum.iter()
        .map(|c| 2.0 * c.abs() / n as f64)
        .collect();
    println!("FFT peak magnitude at 50 Hz bin: {:.4}", magnitudes[50 * n / 1000]);

    // --- Method 2: Welch PSD ---
    let (freqs, power) = welch(&x, fs, 512, 256, &Window::Hann);
    let peak_idx = power.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap().0;
    println!("Welch peak frequency: {:.1} Hz", freqs[peak_idx]);
}
```

## Function Reference

| Function | Signature | Description |
|---|---|---|
| `fft` | `fn fft<S>(x: &[Complex<S>]) -> Vec<Complex<S>>` | Forward DFT |
| `ifft` | `fn ifft<S>(x: &[Complex<S>]) -> Vec<Complex<S>>` | Inverse DFT (normalized) |
| `fft2` | `fn fft2<S>(x: &[Complex<S>], rows, cols) -> Vec<Complex<S>>` | 2-D DFT |
| `rfft` | `fn rfft<S>(x: &[S]) -> Vec<Complex<S>>` | Real-input FFT (N/2+1 bins) |
| `irfft` | `fn irfft<S>(x: &[Complex<S>], n) -> Vec<S>` | Inverse real FFT |
| `psd` | `fn psd<S>(x: &[S], fs, window) -> (Vec<S>, Vec<S>)` | Periodogram PSD |
| `welch` | `fn welch<S>(x, fs, nperseg, noverlap, window) -> (Vec<S>, Vec<S>)` | Welch PSD |
| `stft` | `fn stft<S>(x, fs, nperseg, noverlap, window) -> StftResult<S>` | Short-time Fourier transform |
| `fftfreq` | `fn fftfreq<S>(n, dt) -> Vec<S>` | DFT sample frequencies |
| `fftshift` | `fn fftshift<S>(x: &mut [S])` | Shift zero-freq to center |
| `ifftshift` | `fn ifftshift<S>(x: &mut [S])` | Undo fftshift |
| `fftconvolve` | `fn fftconvolve<S>(a, b) -> Vec<S>` | FFT-based convolution |
| `fftcorrelate` | `fn fftcorrelate<S>(a, b) -> Vec<S>` | FFT-based cross-correlation |
| `window_func` | `fn window_func<S>(window, n) -> Vec<S>` | Generate window coefficients |
