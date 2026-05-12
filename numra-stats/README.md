# numra-stats

**Statistics for the [Numra](https://numra-rs.org/) workspace — probability distributions, descriptive statistics, hypothesis tests, regression, and correlation.**

[![Crates.io](https://img.shields.io/crates/v/numra-stats.svg)](https://crates.io/crates/numra-stats)
[![docs.rs](https://docs.rs/numra-stats/badge.svg)](https://docs.rs/numra-stats)

Eleven distributions (continuous and discrete) with PDF / CDF / quantile / sample, a full descriptive suite, the standard tests (t, chi², KS, ANOVA), and three regression flavors.

## Example

```rust
use numra_stats::{mean, std_dev, ttest_1samp};

let samples = vec![5.2, 4.8, 5.1, 4.9, 5.0, 5.3, 4.7];

let m = mean(&samples).unwrap();
let s = std_dev(&samples).unwrap();

// One-sample t-test against H0: μ = 5.0 at α = 0.05
let t = ttest_1samp(&samples, 5.0, 0.05).unwrap();
assert!(t.p_value > 0.05);  // Cannot reject H0
```

## What's in this crate

**Distributions** (each with `pdf`, `cdf`, `quantile`, `sample`, `mean`, `variance`):

- Continuous: `Normal`, `Uniform`, `Exponential`, `GammaDist`, `BetaDist`, `ChiSquared`, `StudentT`, `FDist`, `LogNormal`
- Discrete: `Poisson`, `Binomial`

**Descriptive:** `mean`, `median`, `percentile`, `variance`, `std_dev`, `skewness`, `kurtosis`, `covariance`, `covariance_matrix`

**Hypothesis testing:** `ttest_1samp`, `ttest_ind`, `ttest_rel`, `chi2_test`, `ks_test`, `anova_oneway`, `TestResult`

**Regression:** `linregress`, `multiple_linregress`, `polyfit`, `RegressionResult`

**Correlation:** `pearson_r`, `spearman_r`

## Composes with

- [`numra-ode`](https://docs.rs/numra-ode) — trajectory statistics and Monte Carlo summaries
- [`numra-sde`](https://docs.rs/numra-sde) — SDE ensemble mean / variance / percentile
- [`numra-pde`](https://docs.rs/numra-pde) — spatial-distribution statistics at the final time
- [`numra-fit`](https://docs.rs/numra-fit) — regression as a statistical model
- [`numra-special`](https://docs.rs/numra-special) — distribution PDFs back onto `gamma`, `beta`, `erf`

See [interop workflows](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/interop_workflows.rs) for verified PDE → statistics and stats-sampling → Monte Carlo ODE workflows.

## Install

```toml
[dependencies]
numra-stats = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-stats>
- **Book**: [Descriptive stats](https://book.numra-rs.org/ch09-statistics/descriptive-stats/) · [Distributions](https://book.numra-rs.org/ch09-statistics/distributions/) · [Hypothesis testing](https://book.numra-rs.org/ch09-statistics/hypothesis-testing/) · [Regression](https://book.numra-rs.org/ch09-statistics/regression/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-stats>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
