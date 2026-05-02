//! Statistics and probability distributions for Numra.
//!
//! This crate provides:
//! - **Distributions**: Normal, Uniform, Exponential, Gamma, Beta, Chi-squared,
//!   Student's t, F, Poisson, Binomial, Log-normal
//! - **Descriptive statistics**: mean, variance, std_dev, median, percentile,
//!   skewness, kurtosis, covariance, covariance_matrix
//! - **Hypothesis testing**: t-tests (one-sample, two-sample, paired),
//!   chi-squared, Kolmogorov-Smirnov, one-way ANOVA
//! - **Regression**: linear, multiple linear, polynomial
//! - **Correlation**: Pearson, Spearman
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

pub mod correlation;
pub mod descriptive;
pub mod distributions;
pub mod error;
pub mod hypothesis;
pub mod regression;

pub use error::StatsError;

// Distribution traits
pub use distributions::{ContinuousDistribution, DiscreteDistribution};

// Distribution types
pub use distributions::{
    BetaDist, Binomial, ChiSquared, Exponential, FDist, GammaDist, LogNormal, Normal, Poisson,
    StudentT, Uniform,
};

// Descriptive statistics
pub use descriptive::{
    covariance, covariance_matrix, kurtosis, mean, median, percentile, skewness, std_dev, variance,
};

// Hypothesis testing
pub use hypothesis::{
    anova_oneway, chi2_test, ks_test, ttest_1samp, ttest_ind, ttest_rel, TestResult,
};

// Regression
pub use regression::{linregress, multiple_linregress, polyfit, RegressionResult};

// Correlation
pub use correlation::{pearson_r, spearman_r};
