# Hypothesis Testing

Numra provides a suite of hypothesis tests for common statistical inference
tasks: t-tests for means, the chi-squared test for categorical data, the
Kolmogorov-Smirnov test for distribution fitting, and one-way ANOVA for
comparing group means.

## TestResult

All hypothesis tests return a `TestResult<S>` struct:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub struct TestResult<S: Scalar> {
    pub statistic: S,  // test statistic (t, chi2, D, F, ...)
    pub p_value: S,    // p-value
    pub reject: bool,  // whether to reject H0 at the given alpha
}
```

The decision rule: reject \\(H_0\\) when `p_value < alpha`.

## One-Sample t-Test

Tests whether the population mean equals a hypothesized value \\(\mu_0\\).

**Hypotheses:**
- \\(H_0: \mu = \mu_0\\)
- \\(H_1: \mu \neq \mu_0\\) (two-tailed)

**Test statistic:**

\\[
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
\\]

with \\(n - 1\\) degrees of freedom.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::ttest_1samp;

// Data centered around 0 -- should not reject H0: mean = 0
let data = vec![-1.0_f64, -0.5, 0.0, 0.5, 1.0];
let result = ttest_1samp(&data, 0.0, 0.05).unwrap();
assert!(!result.reject);
assert!(result.statistic.abs() < 1e-12); // t = 0 exactly

// Data clearly not centered at 0
let data = vec![10.0_f64, 11.0, 12.0, 10.5, 11.5];
let result = ttest_1samp(&data, 0.0, 0.05).unwrap();
assert!(result.reject);
println!("t = {:.4}, p = {:.6}", result.statistic, result.p_value);
```

## Independent Two-Sample t-Test (Welch's)

Compares the means of two independent groups without assuming equal variances.

**Hypotheses:**
- \\(H_0: \mu_1 = \mu_2\\)
- \\(H_1: \mu_1 \neq \mu_2\\)

**Test statistic:**

\\[
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}
\\]

Degrees of freedom are estimated using the Welch-Satterthwaite approximation:

\\[
\nu = \frac{(s_1^2/n_1 + s_2^2/n_2)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::ttest_ind;

let group_a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let group_b = vec![1.5_f64, 2.5, 3.5, 4.5, 5.5];
let result = ttest_ind(&group_a, &group_b, 0.05).unwrap();

// Small difference, small sample: should not reject
assert!(!result.reject);
println!("t = {:.4}, p = {:.4}", result.statistic, result.p_value);
```

### When Groups Are Clearly Different

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::ttest_ind;

let control = vec![5.0_f64, 5.5, 4.8, 5.2, 5.1, 4.9, 5.3];
let treatment = vec![8.0, 8.5, 7.8, 8.2, 8.1, 7.9, 8.3];
let result = ttest_ind(&control, &treatment, 0.05).unwrap();

assert!(result.reject); // significant difference
println!("t = {:.4}, p = {:.2e}", result.statistic, result.p_value);
```

## Paired t-Test

Compares two related measurements on the same subjects (before/after,
left/right, etc.).

**Hypotheses:**
- \\(H_0: \mu_d = 0\\) (mean difference is zero)
- \\(H_1: \mu_d \neq 0\\)

Internally, this computes the differences and applies a one-sample t-test
with \\(\mu_0 = 0\\).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::ttest_rel;

let before = vec![200.0_f64, 220.0, 190.0, 210.0, 230.0];
let after  = vec![195.0,     215.0, 185.0, 205.0, 225.0];

let result = ttest_rel(&before, &after, 0.05).unwrap();
// Consistent 5-unit decrease: should reject
assert!(result.reject);
println!("Paired t = {:.4}, p = {:.4}", result.statistic, result.p_value);
```

## t-Test Comparison

| Test | Function | Inputs | Pairing | Use Case |
|---|---|---|---|---|
| One-sample | `ttest_1samp` | One sample + \\(\mu_0\\) | N/A | Is the mean a specific value? |
| Independent | `ttest_ind` | Two samples | No | Compare two group means |
| Paired | `ttest_rel` | Two samples | Yes | Before/after on same subjects |

## Chi-Squared Goodness-of-Fit Test

Tests whether observed categorical frequencies match expected frequencies.

**Hypotheses:**
- \\(H_0\\): Observed data follows the expected distribution
- \\(H_1\\): Observed data does not follow the expected distribution

**Test statistic:**

\\[
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
\\]

with \\(k - 1\\) degrees of freedom.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::chi2_test;

// Fair die: observed vs expected (100 total rolls)
let observed = vec![18.0_f64, 16.0, 17.0, 15.0, 17.0, 17.0];
let expected = vec![16.67; 6]; // 100/6 each

let result = chi2_test(&observed, &expected, 0.05).unwrap();
assert!(!result.reject); // good fit -- die appears fair
println!("chi2 = {:.4}, p = {:.4}", result.statistic, result.p_value);
```

### Biased Die Example

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::chi2_test;

// Heavily biased die: face 1 appears too often
let observed = vec![35.0_f64, 10.0, 12.0, 15.0, 14.0, 14.0];
let expected = vec![16.67; 6];

let result = chi2_test(&observed, &expected, 0.05).unwrap();
assert!(result.reject); // significant deviation from uniform
```

## Kolmogorov-Smirnov Test

Tests whether a sample comes from a specified continuous distribution by
measuring the maximum distance between the empirical CDF and the theoretical
CDF.

**Hypotheses:**
- \\(H_0\\): Sample is drawn from the specified distribution
- \\(H_1\\): Sample is not drawn from the specified distribution

**Test statistic:**

\\[
D = \sup_x |F_n(x) - F(x)|
\\]

where \\(F_n\\) is the empirical CDF and \\(F\\) is the theoretical CDF.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{ks_test, Normal, ContinuousDistribution};
use rand::SeedableRng;

// Generate data from N(0, 1)
let dist = Normal::<f64>::standard();
let mut rng = rand::rngs::StdRng::seed_from_u64(42);
let data = dist.sample_n(&mut rng, 200);

// Test against N(0, 1) -- should not reject
let result = ks_test(&data, &dist, 0.05).unwrap();
assert!(!result.reject);
println!("D = {:.4}, p = {:.4}", result.statistic, result.p_value);
```

### Testing Against the Wrong Distribution

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{ks_test, Normal, Uniform, ContinuousDistribution};
use rand::SeedableRng;

// Generate uniform data but test against normal
let uniform = Uniform::new(0.0_f64, 1.0);
let mut rng = rand::rngs::StdRng::seed_from_u64(42);
let data = uniform.sample_n(&mut rng, 200);

let normal = Normal::new(0.5, 0.3);
let result = ks_test(&data, &normal, 0.05).unwrap();
// Should reject: data is uniform, not normal
println!("D = {:.4}, p = {:.4}, reject = {}", result.statistic, result.p_value, result.reject);
```

## One-Way ANOVA

Tests whether the means of three or more groups are all equal.

**Hypotheses:**
- \\(H_0: \mu_1 = \mu_2 = \ldots = \mu_k\\)
- \\(H_1\\): At least one mean differs

**Test statistic:**

\\[
F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}
= \frac{\text{SS}_{\text{between}} / (k-1)}{\text{SS}_{\text{within}} / (N-k)}
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::anova_oneway;

// Three groups with similar means -- should not reject
let g1 = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let g2 = vec![1.5, 2.5, 3.5, 4.5, 5.5];
let g3 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let result = anova_oneway(&[&g1, &g2, &g3], 0.05).unwrap();
assert!(!result.reject);
```

### Groups With Different Means

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::anova_oneway;

let g1 = vec![1.0_f64, 2.0, 3.0];
let g2 = vec![10.0, 11.0, 12.0];
let g3 = vec![20.0, 21.0, 22.0];
let result = anova_oneway(&[&g1, &g2, &g3], 0.05).unwrap();
assert!(result.reject);
println!("F = {:.4}, p = {:.2e}", result.statistic, result.p_value);
```

### ANOVA Decomposition

| Source | Sum of Squares | df | Mean Square |
|---|---|---|---|
| Between groups | \\(\sum n_i (\bar{x}_i - \bar{x})^2\\) | \\(k - 1\\) | \\(\text{SS}_B / (k-1)\\) |
| Within groups | \\(\sum\sum (x_{ij} - \bar{x}_i)^2\\) | \\(N - k\\) | \\(\text{SS}_W / (N-k)\\) |
| Total | \\(\sum\sum (x_{ij} - \bar{x})^2\\) | \\(N - 1\\) | |

## Choosing the Right Test

| Question | Test | Function |
|---|---|---|
| Is the mean a specific value? | One-sample t-test | `ttest_1samp` |
| Do two independent groups differ? | Welch's t-test | `ttest_ind` |
| Do paired measurements differ? | Paired t-test | `ttest_rel` |
| Do categorical frequencies match? | Chi-squared | `chi2_test` |
| Does data follow a distribution? | Kolmogorov-Smirnov | `ks_test` |
| Do 3+ group means differ? | One-way ANOVA | `anova_oneway` |

## Understanding p-Values

The p-value is the probability of observing a test statistic as extreme as
(or more extreme than) the one computed, assuming \\(H_0\\) is true.

| p-value | Interpretation |
|---|---|
| \\(> 0.10\\) | No evidence against \\(H_0\\) |
| \\(0.05 \text{--} 0.10\\) | Weak evidence against \\(H_0\\) |
| \\(0.01 \text{--} 0.05\\) | Moderate evidence against \\(H_0\\) |
| \\(< 0.01\\) | Strong evidence against \\(H_0\\) |
| \\(< 0.001\\) | Very strong evidence against \\(H_0\\) |

## Complete Example: A/B Test Analysis

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{ttest_ind, mean, std_dev};

fn main() {
    // Conversion rates (as proportions) from an A/B test
    let control = vec![
        0.12, 0.15, 0.11, 0.13, 0.14, 0.12, 0.13, 0.11,
        0.14, 0.12, 0.13, 0.15, 0.12, 0.14, 0.11, 0.13,
    ];
    let treatment = vec![
        0.15, 0.18, 0.14, 0.16, 0.17, 0.15, 0.16, 0.14,
        0.17, 0.15, 0.16, 0.18, 0.15, 0.17, 0.14, 0.16,
    ];

    let m_c = mean(&control).unwrap();
    let m_t = mean(&treatment).unwrap();
    let s_c = std_dev(&control).unwrap();
    let s_t = std_dev(&treatment).unwrap();

    println!("Control:   mean = {:.4}, std = {:.4}", m_c, s_c);
    println!("Treatment: mean = {:.4}, std = {:.4}", m_t, s_t);
    println!("Lift:      {:.1}%", (m_t - m_c) / m_c * 100.0);

    let result = ttest_ind(&control, &treatment, 0.05).unwrap();
    println!("\nWelch's t-test:");
    println!("  t-statistic = {:.4}", result.statistic);
    println!("  p-value     = {:.6}", result.p_value);
    println!("  Reject H0?  {}", if result.reject { "Yes" } else { "No" });
}
```

## Function Reference

| Function | Signature | Description |
|---|---|---|
| `ttest_1samp` | `fn ttest_1samp<S>(data, mu0, alpha) -> Result<TestResult<S>>` | One-sample t-test |
| `ttest_ind` | `fn ttest_ind<S>(data1, data2, alpha) -> Result<TestResult<S>>` | Independent two-sample t-test |
| `ttest_rel` | `fn ttest_rel<S>(data1, data2, alpha) -> Result<TestResult<S>>` | Paired t-test |
| `chi2_test` | `fn chi2_test<S>(observed, expected, alpha) -> Result<TestResult<S>>` | Chi-squared goodness-of-fit |
| `ks_test` | `fn ks_test<S>(data, dist, alpha) -> Result<TestResult<S>>` | Kolmogorov-Smirnov test |
| `anova_oneway` | `fn anova_oneway<S>(groups, alpha) -> Result<TestResult<S>>` | One-way ANOVA |
