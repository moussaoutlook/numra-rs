//! Hypothesis testing: t-tests, chi-squared, Kolmogorov-Smirnov, ANOVA.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::descriptive;
use crate::distributions::{
    chi_squared::ChiSquared, f_dist::FDist, student_t::StudentT, ContinuousDistribution,
};
use crate::error::StatsError;

/// Result of a hypothesis test.
#[derive(Clone, Debug)]
pub struct TestResult<S: Scalar> {
    /// Test statistic.
    pub statistic: S,
    /// p-value.
    pub p_value: S,
    /// Whether to reject H0 at the given significance level.
    pub reject: bool,
}

/// One-sample t-test: H0: mean = mu0.
pub fn ttest_1samp<S: Scalar>(data: &[S], mu0: S, alpha: S) -> Result<TestResult<S>, StatsError> {
    if data.len() < 2 {
        return Err(StatsError::EmptyData);
    }
    let n = data.len();
    let m = descriptive::mean(data)?;
    let s = descriptive::std_dev(data)?;
    let sqrt_n = S::from_usize(n).sqrt();
    let t_stat = (m - mu0) / (s / sqrt_n);
    let df = S::from_usize(n - 1);
    let t_dist = StudentT::new(df);
    // Two-tailed p-value
    let p_value = S::TWO * (S::ONE - t_dist.cdf(t_stat.abs()));
    Ok(TestResult {
        statistic: t_stat,
        p_value,
        reject: p_value < alpha,
    })
}

/// Independent two-sample t-test (Welch's t-test): H0: mean1 = mean2.
pub fn ttest_ind<S: Scalar>(
    data1: &[S],
    data2: &[S],
    alpha: S,
) -> Result<TestResult<S>, StatsError> {
    if data1.len() < 2 || data2.len() < 2 {
        return Err(StatsError::EmptyData);
    }
    let n1 = data1.len();
    let n2 = data2.len();
    let m1 = descriptive::mean(data1)?;
    let m2 = descriptive::mean(data2)?;
    let v1 = descriptive::variance(data1)?;
    let v2 = descriptive::variance(data2)?;
    let n1s = S::from_usize(n1);
    let n2s = S::from_usize(n2);

    let se = (v1 / n1s + v2 / n2s).sqrt();
    let t_stat = (m1 - m2) / se;

    // Welch-Satterthwaite degrees of freedom
    let vn1 = v1 / n1s;
    let vn2 = v2 / n2s;
    let num = (vn1 + vn2) * (vn1 + vn2);
    let denom = vn1 * vn1 / (n1s - S::ONE) + vn2 * vn2 / (n2s - S::ONE);
    let df = num / denom;

    let t_dist = StudentT::new(df);
    let p_value = S::TWO * (S::ONE - t_dist.cdf(t_stat.abs()));
    Ok(TestResult {
        statistic: t_stat,
        p_value,
        reject: p_value < alpha,
    })
}

/// Paired t-test: H0: mean difference = 0.
pub fn ttest_rel<S: Scalar>(
    data1: &[S],
    data2: &[S],
    alpha: S,
) -> Result<TestResult<S>, StatsError> {
    if data1.len() != data2.len() {
        return Err(StatsError::LengthMismatch {
            expected: data1.len(),
            got: data2.len(),
        });
    }
    let diffs: Vec<S> = data1
        .iter()
        .zip(data2.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    ttest_1samp(&diffs, S::ZERO, alpha)
}

/// Chi-squared goodness-of-fit test.
pub fn chi2_test<S: Scalar>(
    observed: &[S],
    expected: &[S],
    alpha: S,
) -> Result<TestResult<S>, StatsError> {
    if observed.len() != expected.len() {
        return Err(StatsError::LengthMismatch {
            expected: observed.len(),
            got: expected.len(),
        });
    }
    if observed.len() < 2 {
        return Err(StatsError::EmptyData);
    }
    let chi2_stat: S = observed
        .iter()
        .zip(expected.iter())
        .fold(S::ZERO, |a, (&o, &e)| {
            let d = o - e;
            a + d * d / e
        });
    let df = S::from_usize(observed.len() - 1);
    let chi2_dist = ChiSquared::new(df);
    let p_value = S::ONE - chi2_dist.cdf(chi2_stat);
    Ok(TestResult {
        statistic: chi2_stat,
        p_value,
        reject: p_value < alpha,
    })
}

/// Kolmogorov-Smirnov test: compare sample to a continuous distribution.
pub fn ks_test<S: Scalar>(
    data: &[S],
    dist: &dyn ContinuousDistribution<S>,
    alpha: S,
) -> Result<TestResult<S>, StatsError> {
    if data.is_empty() {
        return Err(StatsError::EmptyData);
    }
    let n = data.len();
    let mut sorted: Vec<S> = data.to_vec();
    sorted.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());

    let mut d_max = S::ZERO;
    let ns = S::from_usize(n);
    for (i, &x) in sorted.iter().enumerate() {
        let f_x = dist.cdf(x);
        let ecdf_above = S::from_usize(i + 1) / ns;
        let ecdf_below = S::from_usize(i) / ns;
        let d1 = (ecdf_above - f_x).abs();
        let d2 = (ecdf_below - f_x).abs();
        let d = if d1 > d2 { d1 } else { d2 };
        if d > d_max {
            d_max = d;
        }
    }

    // Approximate p-value using asymptotic distribution
    let sqrt_n = ns.sqrt();
    let z = (sqrt_n + S::from_f64(0.12) + S::from_f64(0.11) / sqrt_n) * d_max;
    let p_value = ks_pvalue(z);

    Ok(TestResult {
        statistic: d_max,
        p_value,
        reject: p_value < alpha,
    })
}

/// Asymptotic p-value for KS statistic (Kolmogorov distribution).
fn ks_pvalue<S: Scalar>(z: S) -> S {
    let z_f64 = z.to_f64();
    if z_f64 < 0.27 {
        return S::ONE;
    }
    if z_f64 > 3.1 {
        return S::ZERO;
    }
    // Compute using the series: P(D > z) = 2 * sum_{k=1}^inf (-1)^(k+1) * exp(-2*k^2*z^2)
    let mut sum = 0.0;
    for k in 1..=100 {
        let kf = k as f64;
        let term = (-2.0 * kf * kf * z_f64 * z_f64).exp();
        if k % 2 == 1 {
            sum += term;
        } else {
            sum -= term;
        }
        if term < 1e-15 {
            break;
        }
    }
    S::from_f64((2.0 * sum).clamp(0.0, 1.0))
}

/// One-way ANOVA: H0: all group means are equal.
pub fn anova_oneway<S: Scalar>(groups: &[&[S]], alpha: S) -> Result<TestResult<S>, StatsError> {
    if groups.len() < 2 {
        return Err(StatsError::InvalidParameter(
            "ANOVA requires at least 2 groups".into(),
        ));
    }
    let k = groups.len();
    let mut total_n = 0;
    let mut grand_sum = S::ZERO;
    let mut group_means = Vec::with_capacity(k);
    let mut group_sizes = Vec::with_capacity(k);

    for g in groups {
        if g.is_empty() {
            return Err(StatsError::EmptyData);
        }
        let m = descriptive::mean(g)?;
        group_means.push(m);
        group_sizes.push(g.len());
        total_n += g.len();
        grand_sum += g.iter().copied().fold(S::ZERO, |a, b| a + b);
    }
    let grand_mean = grand_sum / S::from_usize(total_n);

    // Between-group sum of squares
    let ss_between: S = group_means
        .iter()
        .zip(group_sizes.iter())
        .fold(S::ZERO, |a, (&m, &n)| {
            a + S::from_usize(n) * (m - grand_mean) * (m - grand_mean)
        });

    // Within-group sum of squares
    let ss_within: S = groups
        .iter()
        .zip(group_means.iter())
        .fold(S::ZERO, |a, (g, &m)| {
            a + g
                .iter()
                .copied()
                .fold(S::ZERO, |b, x| b + (x - m) * (x - m))
        });

    let df_between = S::from_usize(k - 1);
    let df_within = S::from_usize(total_n - k);

    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;

    let f_stat = ms_between / ms_within;
    let f_dist = FDist::new(df_between, df_within);
    let p_value = S::ONE - f_dist.cdf(f_stat);

    Ok(TestResult {
        statistic: f_stat,
        p_value,
        reject: p_value < alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttest_1samp_no_effect() {
        // Data centered at 0 — should not reject H0: mean=0
        let data = vec![-1.0_f64, -0.5, 0.0, 0.5, 1.0];
        let result = ttest_1samp(&data, 0.0, 0.05).unwrap();
        assert!(!result.reject);
        assert!(result.statistic.abs() < 1e-12);
    }

    #[test]
    fn test_ttest_1samp_reject() {
        let data = vec![10.0_f64, 11.0, 12.0, 10.5, 11.5];
        let result = ttest_1samp(&data, 0.0, 0.05).unwrap();
        assert!(result.reject);
    }

    #[test]
    fn test_ttest_ind_same_distribution() {
        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.5_f64, 2.5, 3.5, 4.5, 5.5];
        let result = ttest_ind(&a, &b, 0.05).unwrap();
        // Means differ by 0.5 but with these small samples, should not reject
        assert!(!result.reject);
    }

    #[test]
    fn test_ttest_rel() {
        let before = vec![200.0_f64, 220.0, 190.0, 210.0, 230.0];
        let after = vec![195.0, 215.0, 185.0, 205.0, 225.0];
        let result = ttest_rel(&before, &after, 0.05).unwrap();
        // Consistent 5-unit decrease
        assert!(result.reject);
    }

    #[test]
    fn test_chi2_test_uniform() {
        // Observed counts close to expected (uniform die)
        let obs = vec![18.0_f64, 16.0, 17.0, 15.0, 17.0, 17.0];
        let exp = vec![16.67_f64; 6];
        let result = chi2_test(&obs, &exp, 0.05).unwrap();
        assert!(!result.reject); // Good fit
    }

    #[test]
    fn test_ks_test_normal() {
        use crate::distributions::normal::Normal;
        use rand::SeedableRng;
        // Generate data from N(0,1) with a fixed seed for reproducibility
        let n = Normal::<f64>::standard();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data = n.sample_n(&mut rng, 200);
        let result = ks_test(&data, &n, 0.05).unwrap();
        // Should not reject since data came from the test distribution
        assert!(
            !result.reject,
            "KS test unexpectedly rejected: stat={}, p={}",
            result.statistic, result.p_value
        );
    }

    #[test]
    fn test_anova_equal_groups() {
        let g1 = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![1.5_f64, 2.5, 3.5, 4.5, 5.5];
        let g3 = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let result = anova_oneway(&[&g1, &g2, &g3], 0.05).unwrap();
        assert!(!result.reject);
    }

    #[test]
    fn test_anova_different_groups() {
        let g1 = vec![1.0_f64, 2.0, 3.0];
        let g2 = vec![10.0, 11.0, 12.0];
        let g3 = vec![20.0, 21.0, 22.0];
        let result = anova_oneway(&[&g1, &g2, &g3], 0.05).unwrap();
        assert!(result.reject);
    }
}
