---
title: "Hilbert Transform & Envelope"
---

The Hilbert transform converts a real signal into its analytic signal -- a
complex-valued signal whose magnitude is the instantaneous amplitude (envelope)
and whose phase derivative gives the instantaneous frequency. Numra provides
the Hilbert transform, envelope extraction, and instantaneous frequency
estimation in `numra-signal`.

## The Analytic Signal

Given a real signal $x(t)$, its analytic signal is:

$$
z(t) = x(t) + j\,\hat{x}(t)
$$

where $\hat{x}(t)$ is the Hilbert transform of $x(t)$. In the frequency
domain, the analytic signal is obtained by:

1. Compute the FFT of $x$
2. Zero out all negative frequency components
3. Double the positive frequency components (leave DC and Nyquist unchanged)
4. Compute the IFFT

This is exactly what Numra's `hilbert` function does.

## Computing the Hilbert Transform

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::hilbert;

// Pure 5 Hz sine wave, 64 samples
let n = 64;
let pi2 = 2.0 * std::f64::consts::PI;
let x: Vec<f64> = (0..n)
    .map(|i| (pi2 * 5.0 * i as f64 / n as f64).sin())
    .collect();

let z = hilbert(&x);
assert_eq!(z.len(), n);

// The real part equals the original signal
for (i, (&xi, zi)) in x.iter().zip(z.iter()).enumerate() {
    assert!((xi - zi.re).abs() < 1e-10, "sample {}: mismatch", i);
}

// The envelope of a pure sine is approximately 1.0
for zi in &z[5..59] {
    assert!((zi.abs() - 1.0).abs() < 0.15);
}
```

### Why the Envelope of a Pure Sine is 1.0

For $x(t) = \sin(2\pi f t)$, the analytic signal is:

$$
z(t) = \sin(2\pi f t) + j(-\cos(2\pi f t)) = -j\,e^{j 2\pi f t}
$$

so $|z(t)| = 1$ everywhere. The Hilbert transform shifts the phase by
$-90^{\circ}$ for positive frequencies.

## Envelope Detection

The `envelope` function extracts the instantaneous amplitude -- the magnitude
of the analytic signal:

$$
A(t) = |z(t)| = \sqrt{x(t)^2 + \hat{x}(t)^2}
$$

This is the primary tool for demodulating amplitude-modulated (AM) signals.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::envelope;

let n = 256;
let pi2 = 2.0 * std::f64::consts::PI;

// AM signal: carrier at 30 Hz, modulated by a 2 Hz envelope
// Envelope = 1 + 0.5 * cos(2*pi*2*t)
let x: Vec<f64> = (0..n).map(|i| {
    let t = i as f64 / 256.0;
    (1.0 + 0.5 * (pi2 * 2.0 * t).cos()) * (pi2 * 30.0 * t).sin()
}).collect();

let env = envelope(&x);
assert_eq!(env.len(), n);

// Envelope should be positive everywhere
assert!(env.iter().all(|&e| e >= 0.0));

// Peak envelope should be near 1.5, minimum near 0.5
let max_env: f64 = env[20..n - 20].iter().copied().fold(0.0, f64::max);
let min_env: f64 = env[20..n - 20].iter().copied().fold(f64::MAX, f64::min);
assert!(max_env > 1.2);
assert!(min_env < 0.8);
```

### Applications of Envelope Detection

| Application | Description |
|---|---|
| AM demodulation | Recover the message signal from an AM carrier |
| Vibration analysis | Track amplitude variations in mechanical systems |
| Speech processing | Extract the speech envelope for recognition |
| Ultrasound imaging | Demodulate RF signals for B-mode images |
| Seismology | Compute signal envelopes for event detection |

## Instantaneous Frequency

The instantaneous frequency is the time derivative of the analytic signal's
phase, divided by $2\pi$:

$$
f_{\text{inst}}(t) = \frac{1}{2\pi} \frac{d\phi(t)}{dt}
$$

where $\phi(t) = \arg(z(t))$ is the unwrapped instantaneous phase.

Numra's `instantaneous_frequency` function:

1. Computes the analytic signal via the Hilbert transform
2. Extracts the phase angle at each sample
3. Unwraps the phase to remove $2\pi$ jumps
4. Differentiates using central differences (boundary: forward/backward)

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::instantaneous_frequency;

let n = 128;
let fs = 128.0;
let freq = 10.0;
let pi2 = 2.0 * std::f64::consts::PI;

// Pure 10 Hz sine wave
let x: Vec<f64> = (0..n)
    .map(|i| (pi2 * freq * i as f64 / fs).sin())
    .collect();

let inst_freq = instantaneous_frequency(&x, fs);

// Away from edges, instantaneous frequency should be ~10 Hz
for &f in &inst_freq[10..n - 10] {
    assert!((f - 10.0).abs() < 1.0, "expected ~10 Hz, got {} Hz", f);
}
```

### Chirp Signal Analysis

Instantaneous frequency is particularly useful for analyzing chirp signals
where the frequency changes over time:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::{envelope, instantaneous_frequency};

fn main() {
    let n = 1024;
    let fs = 1000.0;
    let pi2 = 2.0 * std::f64::consts::PI;

    // Linear chirp: 50 Hz to 200 Hz over 1 second
    let x: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / fs;
        let f = 50.0 + 150.0 * t; // instantaneous frequency
        (pi2 * (50.0 * t + 75.0 * t * t)).sin()
    }).collect();

    let env = envelope(&x);
    let inst_freq = instantaneous_frequency(&x, fs);

    // Check a few points in the middle
    for &i in &[250, 500, 750] {
        let t = i as f64 / fs;
        let expected_freq = 50.0 + 150.0 * t;
        println!("t={:.3}s: expected={:.1} Hz, measured={:.1} Hz, envelope={:.3}",
            t, expected_freq, inst_freq[i], env[i]);
    }
}
```

## Phase Unwrapping

The `instantaneous_frequency` function includes automatic phase unwrapping.
Without unwrapping, the phase angle $\arg(z)$ wraps around at $\pm\pi$,
causing discontinuities in the derivative. The unwrapping algorithm detects
jumps larger than $\pi$ and corrects them:

$$
\phi_{\text{unwrapped}}[n] = \phi_{\text{unwrapped}}[n-1] + \text{wrap}(\phi[n] - \phi[n-1])
$$

where $\text{wrap}(d)$ adjusts $d$ into $[-\pi, \pi)$.

## Practical Considerations

### Signal Length

The Hilbert transform works best with signals that are long relative to the
lowest frequency of interest. Very short signals or signals with strong
discontinuities at the boundaries will produce edge effects.

### Windowing

For non-periodic signals, consider applying a window (e.g., Hann) before the
Hilbert transform to reduce Gibbs-like ringing at the edges.

### Narrowband Assumption

The instantaneous frequency concept is most meaningful for narrowband signals
-- signals whose frequency content varies slowly compared to the carrier
frequency. For wideband signals, the STFT (see [FFT & Spectral Analysis](./fft-and-spectral))
may be more appropriate.

## Complete Example: AM Signal Demodulation

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::{hilbert, envelope, instantaneous_frequency};

fn main() {
    let n = 512;
    let fs = 512.0;
    let pi2 = 2.0 * std::f64::consts::PI;

    // AM signal: message = 3 Hz sine, carrier = 40 Hz
    // x(t) = [1 + m*cos(2*pi*f_m*t)] * cos(2*pi*f_c*t)
    let f_carrier = 40.0;
    let f_message = 3.0;
    let modulation_depth = 0.7;

    let x: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / fs;
        let msg = 1.0 + modulation_depth * (pi2 * f_message * t).cos();
        msg * (pi2 * f_carrier * t).cos()
    }).collect();

    // Extract the analytic signal
    let z = hilbert(&x);

    // Envelope recovers the message signal (plus DC offset)
    let env = envelope(&x);

    // Carrier frequency should be ~40 Hz
    let inst_freq = instantaneous_frequency(&x, fs);

    println!("Carrier frequency (mid-signal): {:.1} Hz", inst_freq[n / 2]);
    println!("Envelope range: {:.2} to {:.2}",
        env[20..n - 20].iter().copied().fold(f64::MAX, f64::min),
        env[20..n - 20].iter().copied().fold(0.0, f64::max));
    // Expected: envelope ranges from (1-0.7)=0.3 to (1+0.7)=1.7
}
```

## Function Reference

| Function | Signature | Description |
|---|---|---|
| `hilbert` | `fn hilbert<S>(x: &[S]) -> Vec<Complex<S>>` | Analytic signal via Hilbert transform |
| `envelope` | `fn envelope<S>(x: &[S]) -> Vec<S>` | Instantaneous amplitude (magnitude of analytic signal) |
| `instantaneous_frequency` | `fn instantaneous_frequency<S>(x: &[S], fs: f64) -> Vec<S>` | Instantaneous frequency in Hz |
