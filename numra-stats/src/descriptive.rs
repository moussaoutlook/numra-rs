//! Descriptive statistics: mean, variance, standard deviation, percentiles, etc.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::error::StatsError;

/// Arithmetic mean.
pub fn mean<S: Scalar>(data: &[S]) -> Result<S, StatsError> {
    if data.is_empty() {
        return Err(StatsError::EmptyData);
    }
    let n = S::from_usize(data.len());
    let sum: S = data.iter().copied().fold(S::ZERO, |a, b| a + b);
    Ok(sum / n)
}

/// Sample variance (with Bessel's correction, divides by N-1).
pub fn variance<S: Scalar>(data: &[S]) -> Result<S, StatsError> {
    if data.len() < 2 {
        return Err(StatsError::EmptyData);
    }
    let m = mean(data)?;
    let n = S::from_usize(data.len());
    let sum_sq: S = data
        .iter()
        .copied()
        .fold(S::ZERO, |a, x| a + (x - m) * (x - m));
    Ok(sum_sq / (n - S::ONE))
}

/// Sample standard deviation.
pub fn std_dev<S: Scalar>(data: &[S]) -> Result<S, StatsError> {
    Ok(variance(data)?.sqrt())
}

/// Median (middle value of sorted data).
pub fn median<S: Scalar>(data: &[S]) -> Result<S, StatsError> {
    if data.is_empty() {
        return Err(StatsError::EmptyData);
    }
    let mut sorted: Vec<S> = data.to_vec();
    sorted.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    let n = sorted.len();
    if n % 2 == 1 {
        Ok(sorted[n / 2])
    } else {
        let half = S::HALF;
        Ok((sorted[n / 2 - 1] + sorted[n / 2]) * half)
    }
}

/// Percentile (p in [0, 100]) using linear interpolation.
pub fn percentile<S: Scalar>(data: &[S], p: S) -> Result<S, StatsError> {
    if data.is_empty() {
        return Err(StatsError::EmptyData);
    }
    let p_f64 = p.to_f64();
    if !(0.0..=100.0).contains(&p_f64) {
        return Err(StatsError::InvalidParameter(
            "percentile must be in [0, 100]".into(),
        ));
    }
    let mut sorted: Vec<S> = data.to_vec();
    sorted.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    let n = sorted.len();
    if n == 1 {
        return Ok(sorted[0]);
    }
    // Linear interpolation
    let rank = p_f64 / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        Ok(sorted[lo])
    } else {
        let frac = S::from_f64(rank - lo as f64);
        Ok(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
    }
}

/// Skewness (Fisher's definition, adjusted for sample bias).
pub fn skewness<S: Scalar>(data: &[S]) -> Result<S, StatsError> {
    if data.len() < 3 {
        return Err(StatsError::EmptyData);
    }
    let m = mean(data)?;
    let n = data.len() as f64;
    let s = std_dev(data)?;
    let m3: S = data
        .iter()
        .copied()
        .fold(S::ZERO, |a, x| a + (x - m).powi(3));
    let m3_avg = m3 / S::from_f64(n);
    let s3 = s * s * s;
    // Adjusted skewness: n / ((n-1)(n-2)) * sum((x-mean)^3) / s^3
    let adjust = n * n / ((n - 1.0) * (n - 2.0));
    Ok(m3_avg / s3 * S::from_f64(adjust))
}

/// Excess kurtosis (Fisher's definition).
pub fn kurtosis<S: Scalar>(data: &[S]) -> Result<S, StatsError> {
    if data.len() < 4 {
        return Err(StatsError::EmptyData);
    }
    let m = mean(data)?;
    let n = data.len() as f64;
    let v = variance(data)?;
    let m4: S = data
        .iter()
        .copied()
        .fold(S::ZERO, |a, x| a + (x - m).powi(4));
    let m4_avg = m4 / S::from_f64(n);
    let v2 = v * v;
    // Adjusted excess kurtosis
    let term1 = n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
    let term2 = 3.0 * (n - 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0));
    // Using sample moments: n * sum((x-mean)^4) / (sum((x-mean)^2)^2) - 3 (plus bias correction)
    Ok(S::from_f64(term1) * m4_avg * S::from_f64(n) / v2 - S::from_f64(term2))
}

/// Sample covariance of two data sets.
pub fn covariance<S: Scalar>(x: &[S], y: &[S]) -> Result<S, StatsError> {
    if x.is_empty() {
        return Err(StatsError::EmptyData);
    }
    if x.len() != y.len() {
        return Err(StatsError::LengthMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    if x.len() < 2 {
        return Err(StatsError::EmptyData);
    }
    let mx = mean(x)?;
    let my = mean(y)?;
    let n = S::from_usize(x.len());
    let sum: S = x
        .iter()
        .zip(y.iter())
        .fold(S::ZERO, |a, (&xi, &yi)| a + (xi - mx) * (yi - my));
    Ok(sum / (n - S::ONE))
}

/// Covariance matrix for p variables, each with n observations.
///
/// `data` is a slice of p vectors, each of length n.
/// Returns a row-major p x p matrix.
pub fn covariance_matrix<S: Scalar>(data: &[Vec<S>]) -> Result<Vec<S>, StatsError> {
    if data.is_empty() {
        return Err(StatsError::EmptyData);
    }
    let p = data.len();
    let n = data[0].len();
    for v in data {
        if v.len() != n {
            return Err(StatsError::LengthMismatch {
                expected: n,
                got: v.len(),
            });
        }
    }
    if n < 2 {
        return Err(StatsError::EmptyData);
    }

    let mut cov = vec![S::ZERO; p * p];
    for i in 0..p {
        for j in i..p {
            let c = covariance(&data[i], &data[j])?;
            cov[i * p + j] = c;
            cov[j * p + i] = c;
        }
    }
    Ok(cov)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data).unwrap() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_variance() {
        let data = vec![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((variance(&data).unwrap() - 4.571428571428571).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = std_dev(&data).unwrap();
        assert!((s * s - variance(&data).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_median_odd() {
        let data = vec![3.0_f64, 1.0, 2.0];
        assert!((median(&data).unwrap() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_median_even() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        assert!((median(&data).unwrap() - 2.5).abs() < 1e-14);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 0.0).unwrap() - 1.0).abs() < 1e-14);
        assert!((percentile(&data, 50.0).unwrap() - 3.0).abs() < 1e-14);
        assert!((percentile(&data, 100.0).unwrap() - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric distribution should have ~0 skewness
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert!(skewness(&data).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_kurtosis_normal_like() {
        // For a uniform-ish distribution, kurtosis should be negative
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let k = kurtosis(&data).unwrap();
        assert!(k < 0.0, "kurtosis = {}", k); // Uniform has excess kurtosis = -1.2
    }

    #[test]
    fn test_covariance_self() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let cov = covariance(&data, &data).unwrap();
        let var = variance(&data).unwrap();
        assert!((cov - var).abs() < 1e-12);
    }

    #[test]
    fn test_covariance_matrix_diagonal() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0_f64, 4.0, 3.0, 2.0, 1.0];
        let cov = covariance_matrix(&[x.clone(), y.clone()]).unwrap();
        assert_eq!(cov.len(), 4);
        assert!((cov[0] - variance(&x).unwrap()).abs() < 1e-12);
        assert!((cov[3] - variance(&y).unwrap()).abs() < 1e-12);
        // Off-diagonal should be negative (inversely correlated)
        assert!(cov[1] < 0.0);
        assert!((cov[1] - cov[2]).abs() < 1e-12); // symmetric
    }

    #[test]
    fn test_empty_data() {
        assert!(mean::<f64>(&[]).is_err());
        assert!(variance::<f64>(&[]).is_err());
        assert!(median::<f64>(&[]).is_err());
    }
}
