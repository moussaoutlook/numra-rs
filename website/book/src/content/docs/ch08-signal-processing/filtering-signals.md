---
title: "Filtering Signals"
---

Once you have designed a filter (see [Filter Design](./filter-design)),
you need to apply it to data. Numra provides three application methods:
forward-only IIR filtering (`sosfilt`), zero-phase forward-backward filtering
(`filtfilt`), and direct-convolution FIR filtering (`fir_filter`). There is
also FFT-based convolution for long sequences (`fftconvolve`).

## Forward IIR Filtering: sosfilt

The `sosfilt` function applies an `SosFilter` to a signal in a single forward
pass. It uses the Direct Form II Transposed structure for each second-order
section, which is efficient and numerically stable.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::{butter, sosfilt};

let fs = 100.0;
let sos = butter(4, 10.0, fs).unwrap();

// Generate a test signal: 5 Hz + 40 Hz
let pi2 = 2.0 * std::f64::consts::PI;
let x: Vec<f64> = (0..500).map(|i| {
    let t = i as f64 / fs;
    (pi2 * 5.0 * t).sin() + (pi2 * 40.0 * t).sin()
}).collect();

let y = sosfilt(&sos, &x);
assert_eq!(y.len(), x.len());

// After transients, the 40 Hz component is heavily attenuated
let max_amp: f64 = y[200..].iter().map(|v| v.abs()).fold(0.0, f64::max);
assert!(max_amp < 1.2); // mostly just the 5 Hz component remains
```

### How Direct Form II Transposed Works

For each second-order section with coefficients $[b_0, b_1, b_2, 1, a_1, a_2]$,
the algorithm maintains two delay elements $d_1, d_2$ and processes each
sample as:

$$
\begin{aligned}
y[n] &= b_0 \, x[n] + d_1 \\\\
d_1 &= b_1 \, x[n] - a_1 \, y[n] + d_2 \\\\
d_2 &= b_2 \, x[n] - a_2 \, y[n]
\end{aligned}
$$

This requires only 2 multiply-adds per sample per section -- highly efficient
for real-time processing.

### Cascading Sections

When the filter has multiple SOS sections, `sosfilt` processes the signal
through each section sequentially. The output of section $k$ becomes the
input of section $k+1$:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::{SosFilter, sosfilt};

// Two identity sections cascaded = identity
let sos = SosFilter::new(vec![
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
]);
let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = sosfilt(&sos, &x);
for (a, b) in x.iter().zip(y.iter()) {
    assert!((a - b).abs() < 1e-12);
}
```

## Zero-Phase Filtering: filtfilt

Forward-only filtering introduces phase distortion -- different frequency
components are delayed by different amounts. The `filtfilt` function eliminates
this by filtering the signal forward, then backward, resulting in zero phase
distortion and squared magnitude response (doubled effective order).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::{butter, filtfilt};

let fs = 100.0;
let n = 200;
let pi2 = 2.0 * std::f64::consts::PI;

// 5 Hz signal + 40 Hz noise
let x: Vec<f64> = (0..n).map(|i| {
    let t = i as f64 / fs;
    (pi2 * 5.0 * t).sin() + 0.5 * (pi2 * 40.0 * t).sin()
}).collect();

let sos = butter(4, 10.0, fs).unwrap();
let y = filtfilt(&sos, &x);
assert_eq!(y.len(), x.len());
```

### sosfilt vs filtfilt

| Property | `sosfilt` | `filtfilt` |
|---|---|---|
| Phase | Nonlinear distortion | Zero phase |
| Causality | Causal (real-time capable) | Non-causal (needs full signal) |
| Effective order | $N$ | $2N$ (doubled) |
| Transient handling | Startup transient | Reduced via signal extension |
| Use case | Real-time streaming | Offline analysis |

### How filtfilt Reduces Edge Effects

The `filtfilt` implementation pads the signal using reflection at both ends
before filtering. For a signal $x[0..N]$:

1. **Left pad:** Mirror values $2 x[0] - x[\text{padlen}], \ldots, 2 x[0] - x[1]$
2. **Signal:** Original $x[0..N]$
3. **Right pad:** Mirror values $2 x[N{-}1] - x[N{-}2], \ldots$

The forward pass runs, then the result is reversed and passed through the
filter again. Finally, the padding is stripped to return a signal of the
original length.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::{butter, filtfilt};

// DC signal should pass through unchanged
let sos = butter(4, 10.0, 100.0).unwrap();
let x = vec![3.0_f64; 100];
let y = filtfilt(&sos, &x);

// Interior samples should be very close to 3.0
for &yi in &y[20..80] {
    assert!((yi - 3.0).abs() < 0.01);
}
```

## FIR Filtering: fir_filter

The `fir_filter` function applies FIR filter coefficients to a signal via
direct convolution. This is a causal operation -- the output at sample $n$
depends only on current and past inputs:

$$
y[n] = \sum_{k=0}^{M-1} h[k] \, x[n-k]
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::{firwin, fir_filter};
use numra::fft::Window;

// Design a 63-tap FIR lowpass at 10 Hz, fs=100 Hz
let taps: Vec<f64> = firwin(63, 10.0, 100.0, &Window::Hamming).unwrap();

// Apply to a 40 Hz sine -- should be heavily attenuated
let fs = 100.0;
let pi2 = 2.0 * std::f64::consts::PI;
let x: Vec<f64> = (0..500).map(|i| {
    (pi2 * 40.0 * i as f64 / fs).sin()
}).collect();

let y = fir_filter(&taps, &x);
assert_eq!(y.len(), x.len());

// After transient (taps.len() samples), output should be small
let max_amp: f64 = y[200..].iter().map(|v| v.abs()).fold(0.0, f64::max);
assert!(max_amp < 0.1);
```

### Impulse Response

Feeding a unit impulse through `fir_filter` recovers the filter taps:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::fir_filter;

let taps = vec![0.25, 0.5, 0.25];
let impulse = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
let y = fir_filter(&taps, &impulse);

// Impulse response is the taps shifted by the impulse position
assert!((y[2] - 0.25).abs() < 1e-12);
assert!((y[3] - 0.5).abs() < 1e-12);
assert!((y[4] - 0.25).abs() < 1e-12);
```

## FFT-Based Convolution

For long signals or long filter kernels, FFT-based convolution is much faster
than direct convolution: $O(N \log N)$ vs $O(NM)$.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::fft::fftconvolve;

// Convolve a long signal with a long kernel
let signal = vec![1.0_f64; 10000];
let kernel = vec![0.01_f64; 100]; // 100-point moving average

let y = fftconvolve(&signal, &kernel);
assert_eq!(y.len(), signal.len() + kernel.len() - 1);
```

### When to Use Each Method

| Method | Time Complexity | Best For |
|---|---|---|
| `fir_filter` | $O(NM)$ | Short kernels (M < 100), real-time |
| `fftconvolve` | $O(N \log N)$ | Long kernels, offline analysis |
| `sosfilt` | $O(NK)$ per section | IIR filters, real-time streaming |
| `filtfilt` | $O(NK)$ doubled | IIR filters, zero-phase offline |

## Complete Example: Cleaning a Noisy Signal

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::{butter, cheby1, filtfilt, firwin, fir_filter};
use numra::fft::Window;

fn main() {
    let fs = 1000.0;  // 1 kHz sample rate
    let n = 2000;
    let pi2 = 2.0 * std::f64::consts::PI;

    // Clean signal: sum of 10 Hz and 50 Hz
    // Noise: 200 Hz and 400 Hz
    let x: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / fs;
        0.8 * (pi2 * 10.0 * t).sin()
      + 0.5 * (pi2 * 50.0 * t).sin()
      + 0.3 * (pi2 * 200.0 * t).sin()
      + 0.2 * (pi2 * 400.0 * t).sin()
    }).collect();

    // --- Method 1: Butterworth + filtfilt ---
    let sos_butter = butter(4, 80.0, fs).unwrap();
    let y1 = filtfilt(&sos_butter, &x);

    // --- Method 2: Chebyshev + filtfilt ---
    let sos_cheby = cheby1(4, 0.5, 80.0, fs).unwrap();
    let y2 = filtfilt(&sos_cheby, &x);

    // --- Method 3: FIR ---
    let taps: Vec<f64> = firwin(101, 80.0, fs, &Window::Hamming).unwrap();
    let y3 = fir_filter(&taps, &x);

    // All methods should preserve the 10 Hz and 50 Hz components
    // while attenuating the 200 Hz and 400 Hz noise
    println!("Butterworth max output: {:.3}",
        y1[500..].iter().map(|v| v.abs()).fold(0.0, f64::max));
    println!("Chebyshev max output:   {:.3}",
        y2[500..].iter().map(|v| v.abs()).fold(0.0, f64::max));
    println!("FIR max output:         {:.3}",
        y3[500..].iter().map(|v| v.abs()).fold(0.0, f64::max));
}
```

## Function Reference

| Function | Signature | Description |
|---|---|---|
| `sosfilt` | `fn sosfilt<S>(sos: &SosFilter<S>, x: &[S]) -> Vec<S>` | Forward IIR filtering |
| `filtfilt` | `fn filtfilt<S>(sos: &SosFilter<S>, x: &[S]) -> Vec<S>` | Zero-phase forward-backward filtering |
| `fir_filter` | `fn fir_filter<S>(taps: &[S], x: &[S]) -> Vec<S>` | FIR filtering via direct convolution |
| `fftconvolve` | `fn fftconvolve<S>(a: &[S], b: &[S]) -> Vec<S>` | FFT-based convolution |
