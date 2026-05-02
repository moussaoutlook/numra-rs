//! Core FFT and IFFT using rustfft backend.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::complex::Complex;
use numra_core::Scalar;
use rustfft::num_complex::Complex as RComplex;
use rustfft::FftPlanner;

/// Compute the discrete Fourier transform of a complex sequence.
///
/// Returns a vector of the same length as the input.
pub fn fft<S: Scalar>(x: &[Complex<S>]) -> Vec<Complex<S>> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // Convert to f64 for rustfft
    let mut buffer: Vec<RComplex<f64>> = x
        .iter()
        .map(|c| RComplex::new(c.re.to_f64(), c.im.to_f64()))
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft_plan = planner.plan_fft_forward(n);
    fft_plan.process(&mut buffer);

    // Convert back to Complex<S>
    buffer
        .into_iter()
        .map(|c| Complex::new(S::from_f64(c.re), S::from_f64(c.im)))
        .collect()
}

/// Compute the inverse discrete Fourier transform of a complex sequence.
///
/// Normalizes by 1/N so that `ifft(fft(x)) == x`.
pub fn ifft<S: Scalar>(x: &[Complex<S>]) -> Vec<Complex<S>> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // Convert to f64 for rustfft
    let mut buffer: Vec<RComplex<f64>> = x
        .iter()
        .map(|c| RComplex::new(c.re.to_f64(), c.im.to_f64()))
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let ifft_plan = planner.plan_fft_inverse(n);
    ifft_plan.process(&mut buffer);

    let norm = S::ONE / S::from_usize(n);
    buffer
        .into_iter()
        .map(|c| Complex::new(S::from_f64(c.re), S::from_f64(c.im)) * norm)
        .collect()
}

/// 2D FFT of a row-major complex array.
///
/// Performs FFT along each row, then along each column.
pub fn fft2<S: Scalar>(x: &[Complex<S>], rows: usize, cols: usize) -> Vec<Complex<S>> {
    assert_eq!(
        x.len(),
        rows * cols,
        "fft2: input length must equal rows * cols"
    );

    // Convert to f64 for rustfft
    let mut data: Vec<RComplex<f64>> = x
        .iter()
        .map(|c| RComplex::new(c.re.to_f64(), c.im.to_f64()))
        .collect();

    let mut planner = FftPlanner::<f64>::new();

    // FFT along rows
    let row_plan = planner.plan_fft_forward(cols);
    for r in 0..rows {
        let start = r * cols;
        row_plan.process(&mut data[start..start + cols]);
    }

    // FFT along columns (need to gather/scatter)
    let col_plan = planner.plan_fft_forward(rows);
    let mut col_buf = vec![RComplex::new(0.0, 0.0); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = data[r * cols + c];
        }
        col_plan.process(&mut col_buf);
        for r in 0..rows {
            data[r * cols + c] = col_buf[r];
        }
    }

    // Convert back to Complex<S>
    data.into_iter()
        .map(|c| Complex::new(S::from_f64(c.re), S::from_f64(c.im)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_ifft_roundtrip() {
        let signal = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let spectrum = fft(&signal);
        let recovered = ifft(&spectrum);

        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a.re - b.re).abs() < 1e-12);
            assert!((a.im - b.im).abs() < 1e-12);
        }
    }

    #[test]
    fn test_fft_dc() {
        // Constant signal: FFT should have all energy in DC bin
        let n = 8;
        let signal: Vec<Complex<f64>> = (0..n).map(|_| Complex::new(1.0, 0.0)).collect();
        let spectrum = fft(&signal);
        assert!((spectrum[0].re - n as f64).abs() < 1e-12);
        for k in 1..n {
            assert!(spectrum[k].abs() < 1e-12);
        }
    }

    #[test]
    fn test_fft_single_frequency() {
        let n = 16;
        let freq = 3; // 3 cycles in N samples
        let pi2 = 2.0 * core::f64::consts::PI;
        let signal: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::new((pi2 * freq as f64 * k as f64 / n as f64).cos(), 0.0))
            .collect();
        let spectrum = fft(&signal);
        // Energy should be at bins freq and N-freq
        let amp = n as f64 / 2.0;
        assert!((spectrum[freq].abs() - amp).abs() < 1e-10);
        assert!((spectrum[n - freq].abs() - amp).abs() < 1e-10);
        // Other bins should be approximately zero
        for k in 0..n {
            if k != freq && k != n - freq {
                assert!(
                    spectrum[k].abs() < 1e-10,
                    "bin {} = {}",
                    k,
                    spectrum[k].abs()
                );
            }
        }
    }

    #[test]
    fn test_fft_empty() {
        assert!(fft::<f64>(&[]).is_empty());
        assert!(ifft::<f64>(&[]).is_empty());
    }

    #[test]
    fn test_fft2_basic() {
        // 2x2 constant matrix: FFT2 should put all energy in (0,0)
        let data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        let result = fft2(&data, 2, 2);
        assert!((result[0].re - 4.0).abs() < 1e-12);
        assert!(result[1].abs() < 1e-12);
        assert!(result[2].abs() < 1e-12);
        assert!(result[3].abs() < 1e-12);
    }

    #[test]
    fn test_fft_parseval() {
        // Parseval's theorem: sum|x|^2 = (1/N)*sum|X|^2
        let signal = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 1.0),
            Complex::new(-1.0, 0.5),
            Complex::new(0.0, -1.0),
        ];
        let n = signal.len() as f64;
        let time_energy: f64 = signal.iter().map(|c| c.norm_sqr()).sum();
        let spectrum = fft(&signal);
        let freq_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();
        assert!((time_energy - freq_energy / n).abs() < 1e-10);
    }
}
