---
title: "Filter Design"
---

Numra's `numra-signal` crate provides IIR and FIR digital filter design.
IIR filters (Butterworth, Chebyshev Type I) are designed via analog prototyping
and bilinear transformation. FIR filters use the windowed sinc method. All IIR
filters are output in numerically stable second-order sections (SOS) format.

## IIR Filter Design Pipeline

The classical IIR design approach implemented in Numra:

1. **Specify** digital filter requirements (order, cutoff, sample rate)
2. **Pre-warp** the cutoff frequency for the bilinear transform:
   $\omega_c = 2 f_s \tan\!\bigl(\pi f_c / f_s\bigr)$
3. **Compute** analog prototype poles (Butterworth or Chebyshev)
4. **Bilinear transform** each analog pole to the z-plane:
   $z_k = \frac{1 + p_k / (2 f_s)}{1 - p_k / (2 f_s)}$
5. **Pair** conjugate poles into second-order sections for numerical stability

## Butterworth Filters

The Butterworth filter has a maximally flat magnitude response in the passband.
Its squared magnitude response is:

$$
|H(j\omega)|^2 = \frac{1}{1 + (\omega / \omega_c)^{2N}}
$$

where $N$ is the filter order.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::butter;

// 4th-order Butterworth lowpass at 10 Hz, sampled at 100 Hz
let sos = butter(4, 10.0, 100.0).unwrap();

// Returns 2 second-order sections (order 4 / 2 = 2)
assert_eq!(sos.n_sections(), 2);

// DC gain is unity: the filter passes DC without attenuation
let dc_gain: f64 = sos.sections.iter().map(|s| {
    (s[0] + s[1] + s[2]) / (s[3] + s[4] + s[5])
}).product();
assert!((dc_gain - 1.0).abs() < 1e-6);
```

### Butterworth Characteristics

| Property | Description |
|---|---|
| Passband | Maximally flat (no ripple) |
| Stopband | Monotonically decreasing |
| Roll-off | $-20N$ dB/decade, where $N$ = order |
| Phase | Nonlinear (use `filtfilt` for zero phase) |
| Stability | Always stable for valid parameters |

### Choosing the Order

Higher order means sharper transition but more phase distortion and greater
computational cost per sample.

| Order | Roll-off | Sections | Typical Use |
|---|---|---|---|
| 1 | -20 dB/dec | 1 | Gentle smoothing |
| 2 | -40 dB/dec | 1 | General filtering |
| 4 | -80 dB/dec | 2 | Standard DSP tasks |
| 6 | -120 dB/dec | 3 | Sharp cutoff required |
| 8 | -160 dB/dec | 4 | Very demanding applications |

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::butter;

// Odd orders produce one first-order section plus conjugate pairs
let sos_5 = butter(5, 10.0, 100.0).unwrap();
assert_eq!(sos_5.n_sections(), 3); // 2 conjugate pairs + 1 real pole

let sos_2 = butter(2, 10.0, 100.0).unwrap();
assert_eq!(sos_2.n_sections(), 1); // 1 conjugate pair
```

## Chebyshev Type I Filters

Chebyshev Type I filters trade passband flatness for a steeper roll-off. They
allow equiripple in the passband up to a specified maximum deviation in dB.

$$
|H(j\omega)|^2 = \frac{1}{1 + \varepsilon^2 \, T_N^2(\omega / \omega_c)}
$$

where $T_N$ is the Chebyshev polynomial of order $N$ and
$\varepsilon = \sqrt{10^{R_p/10} - 1}$ for ripple $R_p$ dB.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::cheby1;

// 4th-order Chebyshev I, 1 dB passband ripple, cutoff 10 Hz, fs=100 Hz
let sos = cheby1(4, 1.0, 10.0, 100.0).unwrap();
assert_eq!(sos.n_sections(), 2);
```

### Butterworth vs Chebyshev Comparison

| Property | Butterworth | Chebyshev I |
|---|---|---|
| Passband | Maximally flat | Equiripple (user-specified) |
| Transition band | Wider | Narrower (steeper roll-off) |
| Stopband | Monotonic | Monotonic |
| Group delay | Moderate variation | Larger variation |
| Best for | Smooth passband needed | Sharp cutoff needed |

## FIR Filter Design

FIR filters have finite impulse responses and inherently linear phase (when
symmetric). Numra uses the windowed sinc method: truncate the ideal sinc
lowpass impulse response and apply a window to control sidelobes.

The ideal lowpass impulse response is:

$$
h_{\text{ideal}}[n] = \frac{\sin(\pi f_c (n - M/2))}{\pi (n - M/2)}
$$

where $f_c$ is the normalized cutoff and $M$ is the number of taps minus one.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::firwin;
use numra::fft::Window;

// 31-tap FIR lowpass at 10 Hz, fs=100 Hz, Hamming window
let taps: Vec<f64> = firwin(31, 10.0, 100.0, &Window::Hamming).unwrap();
assert_eq!(taps.len(), 31);

// DC gain is normalized to 1.0
let dc_gain: f64 = taps.iter().sum();
assert!((dc_gain - 1.0).abs() < 0.01);

// Linear-phase FIR is symmetric
for i in 0..15 {
    assert!((taps[i] - taps[30 - i]).abs() < 1e-12);
}
```

### FIR vs IIR Comparison

| Property | FIR (`firwin`) | IIR (`butter`/`cheby1`) |
|---|---|---|
| Phase | Exactly linear | Nonlinear (use `filtfilt` for zero-phase) |
| Stability | Always stable | Can be unstable if poorly designed |
| Order for sharp cutoff | High (many taps) | Low (few sections) |
| Memory | Stores all taps | Stores 2 delay elements per section |
| Latency | $M/2$ samples | Very low |

### Choosing the Window for FIR Design

| Window | Transition Width | Sidelobe Attn | Taps Needed |
|---|---|---|---|
| `Rectangular` | Narrowest | -13 dB | Fewest, but poor sidelobes |
| `Hamming` | Moderate | -43 dB | Good default choice |
| `Blackman` | Wider | -58 dB | More taps for same cutoff |
| `Kaiser(beta)` | Adjustable | Adjustable | Tunable via beta parameter |

## Second-Order Sections (SOS) Format

Both `butter` and `cheby1` return an `SosFilter` -- a cascade of biquad
sections. Each section stores six coefficients `[b0, b1, b2, a0, a1, a2]`
representing the transfer function:

$$
H_k(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}
$$

The overall filter is the product:

$$
H(z) = \prod_{k=1}^{K} H_k(z)
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::{butter, SosFilter};

let sos = butter(4, 10.0, 100.0).unwrap();

// Inspect sections
for (i, section) in sos.sections.iter().enumerate() {
    println!("Section {}: b=[{:.4}, {:.4}, {:.4}]  a=[{:.4}, {:.4}, {:.4}]",
        i, section[0], section[1], section[2],
        section[3], section[4], section[5]);
}

// The overall filter order
println!("Filter order: {}", sos.order()); // 4
```

**Why SOS?** Direct-form transfer function representations `(b, a)` suffer
from coefficient sensitivity for high-order filters. A tiny rounding error in
the denominator polynomial can shift poles dramatically. SOS factors the
problem into small, well-conditioned biquads.

## Error Handling

Both design functions validate their inputs and return `SignalError`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::{butter, cheby1};

// Order must be >= 1
assert!(butter(0, 10.0, 100.0).is_err());

// Cutoff must be in (0, fs/2) -- the Nyquist interval
assert!(butter(4, 0.0, 100.0).is_err());
assert!(butter(4, 50.0, 100.0).is_err());   // 50 Hz = Nyquist
assert!(butter(4, -1.0, 100.0).is_err());

// Chebyshev ripple must be positive
assert!(cheby1(4, 0.0, 10.0, 100.0).is_err());
assert!(cheby1(4, -1.0, 10.0, 100.0).is_err());

// FIR: numtaps must be > 0, cutoff in valid range
use numra::signal::firwin;
use numra::fft::Window;
assert!(firwin::<f64>(0, 10.0, 100.0, &Window::Hamming).is_err());
assert!(firwin::<f64>(31, 50.0, 100.0, &Window::Hamming).is_err());
```

## Complete Example: Comparing Butterworth Orders

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::signal::{butter, sosfilt};

fn main() {
    let fs = 100.0;
    let n = 500;
    let pi2 = 2.0 * std::f64::consts::PI;

    // Test signal: 5 Hz (passband) + 40 Hz (stopband)
    let x: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / fs;
        (pi2 * 5.0 * t).sin() + (pi2 * 40.0 * t).sin()
    }).collect();

    for order in [2, 4, 6, 8] {
        let sos = butter(order, 10.0, fs).unwrap();
        let y = sosfilt(&sos, &x);

        // Measure residual 40 Hz amplitude after transient
        let max_40hz: f64 = y[200..].iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        println!("Order {}: sections={}, 40 Hz residual = {:.4}",
            order, sos.n_sections(), max_40hz);
    }
    // Higher order => more attenuation of 40 Hz
}
```

## Function Reference

| Function | Signature | Description |
|---|---|---|
| `butter` | `fn butter(order, cutoff, fs) -> Result<SosFilter<f64>, SignalError>` | Butterworth lowpass design |
| `cheby1` | `fn cheby1(order, ripple_db, cutoff, fs) -> Result<SosFilter<f64>, SignalError>` | Chebyshev Type I lowpass design |
| `firwin` | `fn firwin<S>(numtaps, cutoff, fs, window) -> Result<Vec<S>, SignalError>` | FIR lowpass via windowed sinc |
