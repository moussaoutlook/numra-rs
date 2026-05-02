//! Statistics utilities for ensemble SDE results.
//!
//! Provides tools for computing statistics from Monte Carlo simulations.
//!
//! Author: Moussa Leblouba
//! Date: 3 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Statistics computed from an ensemble of trajectories.
#[derive(Clone, Debug)]
pub struct EnsembleStats<S: Scalar> {
    /// Number of samples
    pub n_samples: usize,
    /// Sample mean
    pub mean: S,
    /// Sample standard deviation
    pub std: S,
    /// Sample variance
    pub variance: S,
    /// Minimum value
    pub min: S,
    /// Maximum value
    pub max: S,
    /// Percentiles (5th, 25th, 50th, 75th, 95th)
    pub percentiles: Percentiles<S>,
}

/// Common percentiles.
#[derive(Clone, Debug)]
pub struct Percentiles<S: Scalar> {
    pub p5: S,
    pub p25: S,
    pub p50: S, // Median
    pub p75: S,
    pub p95: S,
}

impl<S: Scalar> EnsembleStats<S> {
    /// Compute statistics from a vector of samples.
    pub fn from_samples(samples: &[S]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let n = samples.len();
        let n_f = S::from_usize(n);

        // Mean
        let sum: S = samples.iter().fold(S::ZERO, |acc, &x| acc + x);
        let mean = sum / n_f;

        // Variance and std
        let var_sum: S = samples.iter().fold(S::ZERO, |acc, &x| {
            let diff = x - mean;
            acc + diff * diff
        });
        let variance = if n > 1 {
            var_sum / S::from_usize(n - 1) // Bessel correction
        } else {
            S::ZERO
        };
        let std = variance.sqrt();

        // Min and max
        let mut min = samples[0];
        let mut max = samples[0];
        for &x in samples.iter().skip(1) {
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        }

        // Percentiles (need sorted data)
        let mut sorted = samples.to_vec();
        // Sort by converting to f64 for comparison
        sorted.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());

        let percentiles = Percentiles {
            p5: percentile_sorted(&sorted, 5.0),
            p25: percentile_sorted(&sorted, 25.0),
            p50: percentile_sorted(&sorted, 50.0),
            p75: percentile_sorted(&sorted, 75.0),
            p95: percentile_sorted(&sorted, 95.0),
        };

        Some(Self {
            n_samples: n,
            mean,
            std,
            variance,
            min,
            max,
            percentiles,
        })
    }

    /// Compute standard error of the mean.
    pub fn standard_error(&self) -> S {
        self.std / S::from_usize(self.n_samples).sqrt()
    }

    /// Compute confidence interval for the mean at given level (e.g., 0.95 for 95%).
    ///
    /// Uses normal approximation (valid for large samples).
    pub fn confidence_interval(&self, level: S) -> (S, S) {
        // z-score for two-tailed test
        // For 95%: z ≈ 1.96, for 99%: z ≈ 2.576
        let alpha = (S::ONE - level) / S::from_f64(2.0);
        let z = normal_quantile(S::ONE - alpha);
        let margin = z * self.standard_error();
        (self.mean - margin, self.mean + margin)
    }

    /// Interquartile range (IQR = Q3 - Q1).
    pub fn iqr(&self) -> S {
        self.percentiles.p75 - self.percentiles.p25
    }

    /// Median (50th percentile).
    pub fn median(&self) -> S {
        self.percentiles.p50
    }
}

/// Compute percentile from sorted data.
fn percentile_sorted<S: Scalar>(sorted: &[S], p: f64) -> S {
    let n = sorted.len();
    if n == 0 {
        return S::ZERO;
    }
    if n == 1 {
        return sorted[0];
    }

    // Linear interpolation method
    let rank = (p / 100.0) * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;

    if lower == upper {
        sorted[lower]
    } else {
        let frac = S::from_f64(rank - lower as f64);
        sorted[lower] + frac * (sorted[upper] - sorted[lower])
    }
}

/// Approximate inverse normal CDF (quantile function).
///
/// Uses Abramowitz and Stegun approximation.
fn normal_quantile<S: Scalar>(p: S) -> S {
    // Rational approximation for 0 < p < 1
    let p_f = p.to_f64();
    if p_f <= 0.0 || p_f >= 1.0 {
        return S::ZERO;
    }

    #[allow(clippy::excessive_precision)]
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let q = if p_f < p_low {
        let q = (-2.0 * p_f.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p_f <= p_high {
        let q = p_f - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p_f).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    };

    S::from_f64(q)
}

/// Running statistics using Welford's online algorithm.
///
/// Memory-efficient for streaming data or very large ensembles.
#[derive(Clone, Debug)]
pub struct RunningStats<S: Scalar> {
    n: usize,
    mean: S,
    m2: S, // Sum of squared deviations
    min: S,
    max: S,
}

impl<S: Scalar> RunningStats<S> {
    /// Create a new running statistics accumulator.
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: S::ZERO,
            m2: S::ZERO,
            min: S::INFINITY,
            max: S::NEG_INFINITY,
        }
    }

    /// Update statistics with a new value (Welford's algorithm).
    pub fn update(&mut self, value: S) {
        self.n += 1;
        let n_f = S::from_usize(self.n);

        let delta = value - self.mean;
        self.mean += delta / n_f;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Number of samples seen.
    pub fn count(&self) -> usize {
        self.n
    }

    /// Current mean estimate.
    pub fn mean(&self) -> S {
        self.mean
    }

    /// Current variance estimate (sample variance with Bessel correction).
    pub fn variance(&self) -> S {
        if self.n < 2 {
            S::ZERO
        } else {
            self.m2 / S::from_usize(self.n - 1)
        }
    }

    /// Current standard deviation estimate.
    pub fn std(&self) -> S {
        self.variance().sqrt()
    }

    /// Standard error of the mean.
    pub fn standard_error(&self) -> S {
        self.std() / S::from_usize(self.n).sqrt()
    }

    /// Minimum value seen.
    pub fn min(&self) -> S {
        self.min
    }

    /// Maximum value seen.
    pub fn max(&self) -> S {
        self.max
    }

    /// Merge another RunningStats into this one (parallel reduction).
    pub fn merge(&mut self, other: &RunningStats<S>) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
            return;
        }

        let n_a = S::from_usize(self.n);
        let n_b = S::from_usize(other.n);
        let n_total = n_a + n_b;

        let delta = other.mean - self.mean;
        let new_mean = (n_a * self.mean + n_b * other.mean) / n_total;

        // Chan's parallel algorithm for M2
        let new_m2 = self.m2 + other.m2 + delta * delta * n_a * n_b / n_total;

        self.n += other.n;
        self.mean = new_mean;
        self.m2 = new_m2;

        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
    }
}

impl<S: Scalar> Default for RunningStats<S> {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions exported for convenience

/// Compute mean of a slice.
#[inline]
pub fn mean<S: Scalar>(data: &[S]) -> S {
    if data.is_empty() {
        return S::ZERO;
    }
    data.iter().fold(S::ZERO, |acc, &x| acc + x) / S::from_usize(data.len())
}

/// Compute sample standard deviation.
pub fn std<S: Scalar>(data: &[S]) -> S {
    variance(data).sqrt()
}

/// Compute sample variance.
pub fn variance<S: Scalar>(data: &[S]) -> S {
    if data.len() < 2 {
        return S::ZERO;
    }
    let m = mean(data);
    let sum_sq: S = data.iter().fold(S::ZERO, |acc, &x| {
        let diff = x - m;
        acc + diff * diff
    });
    sum_sq / S::from_usize(data.len() - 1)
}

/// Compute percentile (0-100 scale).
pub fn percentile<S: Scalar>(data: &[S], p: f64) -> S {
    if data.is_empty() {
        return S::ZERO;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    percentile_sorted(&sorted, p)
}

/// Compute median.
pub fn median<S: Scalar>(data: &[S]) -> S {
    percentile(data, 50.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((mean(&data) - 3.0).abs() < 1e-10);
        assert!((variance(&data) - 2.5).abs() < 1e-10); // Sample variance
        assert!((std(&data) - 2.5_f64.sqrt()).abs() < 1e-10);
        assert!((median(&data) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentiles() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();

        assert!((percentile(&data, 50.0) - 50.5).abs() < 0.5);
        assert!((percentile(&data, 25.0) - 25.0).abs() < 1.0);
        assert!((percentile(&data, 75.0) - 75.0).abs() < 1.0);
    }

    #[test]
    fn test_ensemble_stats() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let stats = EnsembleStats::from_samples(&data).unwrap();

        assert_eq!(stats.n_samples, 100);
        assert!((stats.mean - 50.5).abs() < 0.01);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_running_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut rs = RunningStats::<f64>::new();

        for &x in &data {
            rs.update(x);
        }

        assert_eq!(rs.count(), 5);
        assert!((rs.mean() - 3.0).abs() < 1e-10);
        assert!((rs.variance() - 2.5).abs() < 1e-10);
        assert!((rs.min() - 1.0).abs() < 1e-10);
        assert!((rs.max() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_running_stats_merge() {
        let data1 = vec![1.0, 2.0, 3.0];
        let data2 = vec![4.0, 5.0, 6.0];

        let mut rs1 = RunningStats::<f64>::new();
        let mut rs2 = RunningStats::<f64>::new();

        for &x in &data1 {
            rs1.update(x);
        }
        for &x in &data2 {
            rs2.update(x);
        }

        rs1.merge(&rs2);

        // Should equal stats of combined data
        let combined = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(rs1.count(), 6);
        assert!((rs1.mean() - mean(&combined)).abs() < 1e-10);
        assert!((rs1.variance() - variance(&combined)).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_interval() {
        // Large normal sample should have mean close to 0
        let data: Vec<f64> = (0..1000)
            .map(|i| {
                // Pseudo-normal using Box-Muller would be better, but just use uniform here
                (i as f64 / 1000.0 - 0.5) * 2.0
            })
            .collect();

        let stats = EnsembleStats::from_samples(&data).unwrap();
        let (lo, hi) = stats.confidence_interval(0.95);

        // Interval should contain mean
        assert!(lo < stats.mean);
        assert!(hi > stats.mean);
        // Interval should be narrower than total range
        assert!(hi - lo < stats.max - stats.min);
    }
}
