//! Correlation analysis: Pearson, Spearman.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::descriptive;
use crate::distributions::{student_t::StudentT, ContinuousDistribution};
use crate::error::StatsError;

/// Pearson correlation coefficient with p-value.
///
/// Returns `(r, p_value)` where r is in [-1, 1].
pub fn pearson_r<S: Scalar>(x: &[S], y: &[S]) -> Result<(S, S), StatsError> {
    if x.len() != y.len() {
        return Err(StatsError::LengthMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    if x.len() < 3 {
        return Err(StatsError::EmptyData);
    }
    let n = x.len();
    let cov = descriptive::covariance(x, y)?;
    let sx = descriptive::std_dev(x)?;
    let sy = descriptive::std_dev(y)?;
    let r = cov / (sx * sy);

    // p-value via t-distribution: t = r * sqrt(n-2) / sqrt(1-r^2)
    let ns = S::from_usize(n);
    let two = S::TWO;
    let r2 = r * r;
    let t_stat = r * (ns - two).sqrt() / (S::ONE - r2).sqrt();
    let df = S::from_usize(n - 2);
    let t_dist = StudentT::new(df);
    let p_value = two * (S::ONE - t_dist.cdf(t_stat.abs()));

    Ok((r, p_value))
}

/// Spearman rank correlation coefficient with p-value.
///
/// Returns `(rho, p_value)`.
pub fn spearman_r<S: Scalar>(x: &[S], y: &[S]) -> Result<(S, S), StatsError> {
    if x.len() != y.len() {
        return Err(StatsError::LengthMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    if x.len() < 3 {
        return Err(StatsError::EmptyData);
    }

    // Compute ranks
    let rx = rank(x);
    let ry = rank(y);

    // Pearson correlation of ranks
    pearson_r(&rx, &ry)
}

/// Compute ranks (1-based, average rank for ties).
fn rank<S: Scalar>(data: &[S]) -> Vec<S> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.to_f64()))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n - 1 && (indexed[j + 1].1 - indexed[j].1).abs() < 1e-14 {
            j += 1;
        }
        // Average rank for tied values
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for idx in &indexed[i..=j] {
            ranks[idx.0] = avg_rank;
        }
        i = j + 1;
    }

    ranks.into_iter().map(|r| S::from_f64(r)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_positive() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (r, p) = pearson_r(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-12);
        assert!(p < 0.01);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let (r, _) = pearson_r(&x, &y).unwrap();
        assert!((r + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_spearman_monotone() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x^2, monotone
        let (rho, _) = spearman_r(&x, &y).unwrap();
        assert!((rho - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_spearman_with_ties() {
        let x = vec![1.0_f64, 2.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        let (rho, _) = spearman_r(&x, &y).unwrap();
        assert!((rho - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_rank_basic() {
        let data = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0];
        let r = rank::<f64>(&data);
        // 1.0 appears twice -> rank 1.5
        // 3.0 -> rank 3
        // 4.0 -> rank 4
        // 5.0 -> rank 5
        assert!((r[0] - 3.0).abs() < 1e-14); // 3.0
        assert!((r[1] - 1.5).abs() < 1e-14); // 1.0 (tied)
        assert!((r[2] - 4.0).abs() < 1e-14); // 4.0
        assert!((r[3] - 1.5).abs() < 1e-14); // 1.0 (tied)
        assert!((r[4] - 5.0).abs() < 1e-14); // 5.0
    }
}
