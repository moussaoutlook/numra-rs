# Resampling & Peak Detection

Numra provides FFT-based signal resampling for changing sample rates, and a
flexible peak detection algorithm with height, distance, and prominence
constraints.

## FFT-Based Resampling

The `resample` function changes the number of samples in a signal using the
Fourier method. It works by transforming to the frequency domain, zero-padding
(upsampling) or truncating (downsampling) the spectrum, and transforming back.

### How It Works

For upsampling from \\(N\\) to \\(M > N\\) samples:

1. Compute the FFT of the input (\\(N\\) bins)
2. Create a new spectrum of length \\(M\\), initialized to zero
3. Copy positive frequencies to the beginning, negative frequencies to the end
4. Split the Nyquist bin if \\(N\\) is even
5. IFFT and scale by \\(M/N\\)

For downsampling (\\(M < N\\)), the spectrum is truncated instead of zero-padded.

### Upsampling

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::resample;

// Upsample a 4-sample signal to 8 samples
let x = vec![1.0, 2.0, 3.0, 4.0];
let y = resample(&x, 8);
assert_eq!(y.len(), 8);
```

### Downsampling

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::resample;

// Downsample a 16-sample DC signal to 8 samples
let x = vec![3.0_f64; 16];
let y = resample(&x, 8);
assert_eq!(y.len(), 8);

// DC value should be preserved
for &yi in &y {
    assert!((yi - 3.0).abs() < 1e-10);
}
```

### Sine Wave Resampling

The Fourier method perfectly preserves bandlimited content:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::resample;

let n = 32;
let n_out = 64;
let pi2 = 2.0 * std::f64::consts::PI;
let freq = 3.0; // 3 cycles in the signal

let x: Vec<f64> = (0..n)
    .map(|i| (pi2 * freq * i as f64 / n as f64).sin())
    .collect();

let y = resample(&x, n_out);

// Compare with the analytic sine at the new sample points
let expected: Vec<f64> = (0..n_out)
    .map(|i| (pi2 * freq * i as f64 / n_out as f64).sin())
    .collect();

for (i, (&yi, &ei)) in y.iter().zip(expected.iter()).enumerate() {
    assert!((yi - ei).abs() < 0.05, "sample {}: {} vs {}", i, yi, ei);
}
```

### Practical Considerations

| Scenario | Recommendation |
|---|---|
| Upsampling | FFT resample preserves all frequency content |
| Downsampling | Apply a lowpass filter first to avoid aliasing |
| Non-periodic signals | Apply a window before resampling, or use zero-padding |
| Identity (same length) | `resample(&x, x.len())` returns a copy of the input |

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::resample;

// Identity: same length in = same signal out
let x = vec![1.0, 2.0, 3.0, 4.0];
let y = resample(&x, 4);
for (a, b) in x.iter().zip(y.iter()) {
    assert!((a - b).abs() < 1e-10);
}
```

### Anti-Aliased Downsampling

When downsampling, frequencies above the new Nyquist rate will alias. Apply a
lowpass filter before resampling to avoid this:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::{butter, filtfilt, resample};

let fs_original = 1000.0;
let fs_target = 250.0;
let n = 2000;

// Signal with content at 10 Hz (keep) and 200 Hz (would alias)
let pi2 = 2.0 * std::f64::consts::PI;
let x: Vec<f64> = (0..n).map(|i| {
    let t = i as f64 / fs_original;
    (pi2 * 10.0 * t).sin() + (pi2 * 200.0 * t).sin()
}).collect();

// Anti-alias filter: lowpass below new Nyquist (125 Hz)
let sos = butter(6, 100.0, fs_original).unwrap();
let x_filtered = filtfilt(&sos, &x);

// Now safe to downsample
let n_out = n * fs_target as usize / fs_original as usize;
let y = resample(&x_filtered, n_out);
assert_eq!(y.len(), n_out);
```

## Peak Detection

The `find_peaks` function locates local maxima in a signal. A peak is defined
as a sample that is strictly greater than both its immediate neighbors. Peaks
can be filtered by minimum height, minimum distance, and minimum prominence.

### Basic Usage

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::{find_peaks, PeakOptions};

let x = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];

// Find all peaks
let peaks = find_peaks(&x, &PeakOptions::default());
assert_eq!(peaks, vec![1, 3, 5]);

// Values at peak positions
for &p in &peaks {
    println!("Peak at index {}: value = {}", p, x[p]);
}
```

### Height Constraint

Filter peaks by a minimum absolute height:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::{find_peaks, PeakOptions};

let x = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];

// Only peaks with value >= 1.8
let peaks = find_peaks(&x, &PeakOptions::default().height(1.8));
assert_eq!(peaks, vec![3]); // only the peak at index 3 (value 2.0)
```

### Distance Constraint

Enforce a minimum separation between peaks. When two peaks are too close, the
taller one is kept (greedy algorithm sorted by height):

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::{find_peaks, PeakOptions};

let x = vec![0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0];

// All peaks
let all = find_peaks(&x, &PeakOptions::default());
assert_eq!(all, vec![1, 3, 5]);

// Minimum distance of 3 samples between peaks
let peaks = find_peaks(&x, &PeakOptions::default().distance(3));
assert_eq!(peaks, vec![1, 5]); // 3.0 at idx=1, then 1.0 at idx=5 (distance 4)
```

### Prominence Constraint

Prominence measures how much a peak stands out relative to nearby valleys.
It is computed as the peak height minus the highest valley on either side
before reaching a taller peak:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::{find_peaks, PeakOptions};

let x = vec![0.0, 5.0, 4.5, 4.8, 0.0];
// Peak at index 1 (5.0): prominent
// Peak at index 3 (4.8): low prominence (4.8 - 4.5 = 0.3)

let peaks = find_peaks(&x, &PeakOptions::default().prominence(1.0));
assert_eq!(peaks, vec![1]); // only the prominent peak
```

### Combining Constraints

All three constraints can be combined using the builder pattern:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::{find_peaks, PeakOptions};

let x: Vec<f64> = (0..200).map(|i| {
    let t = i as f64 / 200.0;
    (2.0 * std::f64::consts::PI * 5.0 * t).sin()
        + 0.3 * (2.0 * std::f64::consts::PI * 20.0 * t).sin()
}).collect();

let peaks = find_peaks(&x, &PeakOptions::default()
    .height(0.5)
    .distance(10)
    .prominence(0.3));

println!("Found {} peaks", peaks.len());
```

### PeakOptions API

| Method | Type | Description |
|---|---|---|
| `height(h)` | `S` | Minimum peak value (absolute) |
| `distance(d)` | `usize` | Minimum separation in samples |
| `prominence(p)` | `S` | Minimum prominence |

### Edge Cases

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::{find_peaks, PeakOptions};

// Empty or short signals: no peaks possible
assert!(find_peaks::<f64>(&[], &PeakOptions::default()).is_empty());
assert!(find_peaks(&[1.0], &PeakOptions::default()).is_empty());
assert!(find_peaks(&[1.0, 2.0], &PeakOptions::default()).is_empty());

// Monotonic signal: no local maxima
let monotone: Vec<f64> = (0..10).map(|i| i as f64).collect();
assert!(find_peaks(&monotone, &PeakOptions::default()).is_empty());

// Plateaus are NOT peaks (not strictly greater than neighbors)
let plateau = vec![0.0, 1.0, 1.0, 0.0];
assert!(find_peaks(&plateau, &PeakOptions::default()).is_empty());
```

## Complete Example: Finding Peaks in a Filtered Signal

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::signal::{butter, filtfilt, find_peaks, PeakOptions};

fn main() {
    let fs = 100.0;
    let pi2 = 2.0 * std::f64::consts::PI;

    // Generate a noisy multi-frequency signal
    let x: Vec<f64> = (0..500).map(|i| {
        let t = i as f64 / fs;
        (pi2 * 3.0 * t).sin()
            + 0.3 * (pi2 * 40.0 * t).sin()  // high-freq noise
    }).collect();

    // Step 1: Lowpass filter to remove the 40 Hz noise
    let sos = butter(4, 10.0, fs).unwrap();
    let y = filtfilt(&sos, &x);

    // Step 2: Find peaks in the cleaned signal
    let peaks = find_peaks(&y, &PeakOptions::default()
        .height(0.5)
        .distance(20));

    println!("Detected {} peaks in the 3 Hz signal over 5 seconds", peaks.len());
    // 3 Hz * 5 seconds = ~15 peaks expected

    for &p in &peaks {
        let t = p as f64 / fs;
        println!("  Peak at t = {:.2}s, amplitude = {:.3}", t, y[p]);
    }
}
```

## Function Reference

| Function | Signature | Description |
|---|---|---|
| `resample` | `fn resample<S>(x: &[S], num_out: usize) -> Vec<S>` | FFT-based resampling |
| `find_peaks` | `fn find_peaks<S>(x: &[S], opts: &PeakOptions<S>) -> Vec<usize>` | Peak detection with constraints |
