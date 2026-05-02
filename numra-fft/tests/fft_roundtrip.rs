//! FFT / IFFT round-trip on a short signal.
//!
//! Author: Moussa Leblouba
//! Date: 26 April 2026
//! Modified: 2 May 2026

use numra_fft::{fft, ifft, Complex};

#[test]
fn ifft_recovers_impulse() {
    let n = 16usize;
    let signal: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            if i == 3 {
                Complex::new(1.0, -0.5)
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect();

    let spec = fft(&signal);
    let recovered = ifft(&spec);
    for i in 0..n {
        assert!(
            (signal[i].re - recovered[i].re).abs() < 1e-11
                && (signal[i].im - recovered[i].im).abs() < 1e-11,
            "mismatch at {}: {:?} vs {:?}",
            i,
            signal[i],
            recovered[i]
        );
    }
}
