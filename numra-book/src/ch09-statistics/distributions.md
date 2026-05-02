# Probability Distributions

Numra's `numra-stats` crate provides a rich collection of probability
distributions, both continuous and discrete. Each distribution implements a
common trait with PDF/PMF, CDF, quantile, mean, variance, and random sampling.

## Distribution Traits

### ContinuousDistribution

All continuous distributions implement `ContinuousDistribution<S>`:

| Method | Signature | Description |
|---|---|---|
| `pdf(x)` | `fn pdf(&self, x: S) -> S` | Probability density function |
| `cdf(x)` | `fn cdf(&self, x: S) -> S` | Cumulative distribution function \\(P(X \le x)\\) |
| `quantile(p)` | `fn quantile(&self, p: S) -> S` | Inverse CDF (\\(p \in [0, 1]\\)) |
| `mean()` | `fn mean(&self) -> S` | Distribution mean |
| `variance()` | `fn variance(&self) -> S` | Distribution variance |
| `std_dev()` | `fn std_dev(&self) -> S` | Standard deviation (\\(\sqrt{\text{variance}}\\)) |
| `sample(rng)` | `fn sample(&self, rng) -> S` | Generate a single random sample |
| `sample_n(rng, n)` | `fn sample_n(&self, rng, n) -> Vec<S>` | Generate \\(n\\) random samples |

### DiscreteDistribution

Discrete distributions implement `DiscreteDistribution<S>`:

| Method | Signature | Description |
|---|---|---|
| `pmf(k)` | `fn pmf(&self, k: usize) -> S` | Probability mass function |
| `cdf(k)` | `fn cdf(&self, k: usize) -> S` | Cumulative \\(P(X \le k)\\) |
| `mean()` | `fn mean(&self) -> S` | Distribution mean |
| `variance()` | `fn variance(&self) -> S` | Distribution variance |
| `sample(rng)` | `fn sample(&self, rng) -> usize` | Generate a single random sample |
| `sample_n(rng, n)` | `fn sample_n(&self, rng, n) -> Vec<usize>` | Generate \\(n\\) random samples |

## Continuous Distributions

### Normal (Gaussian)

The most fundamental continuous distribution. Parameterized by mean \\(\mu\\)
and standard deviation \\(\sigma\\):

\\[
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{Normal, ContinuousDistribution};

let n = Normal::new(0.0_f64, 1.0);  // standard normal N(0, 1)
// or equivalently:
let n = Normal::<f64>::standard();

// PDF at the mean is the peak value
let peak = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
assert!((n.pdf(0.0) - peak).abs() < 1e-12);

// CDF: P(X <= 0) = 0.5 for standard normal
assert!((n.cdf(0.0) - 0.5).abs() < 1e-12);

// Quantile: inverse CDF
let x = n.quantile(0.975);
assert!((x - 1.96).abs() < 0.01); // 97.5th percentile ~ 1.96

// Mean and variance
assert!((n.mean()).abs() < 1e-14);
assert!((n.variance() - 1.0).abs() < 1e-14);
```

### Student's t

Used for hypothesis testing and confidence intervals with small samples.
Parameterized by degrees of freedom \\(\nu\\):

\\[
f(x) = \frac{\Gamma\!\bigl(\frac{\nu+1}{2}\bigr)}{\sqrt{\nu\pi}\;\Gamma\!\bigl(\frac{\nu}{2}\bigr)}
\left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{StudentT, ContinuousDistribution};

let t = StudentT::new(5.0_f64); // 5 degrees of freedom

// Symmetric about zero
assert!((t.pdf(1.0) - t.pdf(-1.0)).abs() < 1e-12);
assert!((t.cdf(0.0) - 0.5).abs() < 1e-10);

// Variance = nu / (nu - 2) for nu > 2
assert!((t.variance() - 5.0 / 3.0).abs() < 1e-14);

// Quantile round-trip
for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
    let x = t.quantile(p);
    assert!((t.cdf(x) - p).abs() < 1e-6);
}
```

### Chi-Squared

Special case of the Gamma distribution. Used in goodness-of-fit tests and
confidence intervals for variance. Parameterized by degrees of freedom \\(k\\):

\\[
f(x) = \frac{x^{k/2-1} e^{-x/2}}{2^{k/2} \Gamma(k/2)}, \quad x > 0
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{ChiSquared, ContinuousDistribution};

let chi2 = ChiSquared::new(5.0_f64);

assert!((chi2.mean() - 5.0).abs() < 1e-14);       // mean = k
assert!((chi2.variance() - 10.0).abs() < 1e-14);   // variance = 2k
assert!(chi2.cdf(0.0).abs() < 1e-10);               // CDF(0) = 0
```

### F-Distribution

The ratio of two chi-squared distributions. Used in ANOVA and regression
testing. Parameterized by \\(d_1, d_2\\) degrees of freedom:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{FDist, ContinuousDistribution};

let f = FDist::new(5.0_f64, 10.0);

// Mean = d2 / (d2 - 2) for d2 > 2
assert!((f.mean() - 10.0 / 8.0).abs() < 1e-12);

assert!(f.cdf(0.0).abs() < 1e-10);
```

### Uniform

Constant density on the interval \\([a, b]\\):

\\[
f(x) = \frac{1}{b - a}, \quad a \le x \le b
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{Uniform, ContinuousDistribution};

let u = Uniform::new(0.0_f64, 10.0);

assert!((u.pdf(5.0) - 0.1).abs() < 1e-14);
assert!((u.cdf(5.0) - 0.5).abs() < 1e-14);
assert!((u.mean() - 5.0).abs() < 1e-14);

// Variance = (b-a)^2 / 12
assert!((u.variance() - 100.0 / 12.0).abs() < 1e-12);
```

### Exponential

Models the time between events in a Poisson process. Parameterized by rate
\\(\lambda\\):

\\[
f(x) = \lambda e^{-\lambda x}, \quad x \ge 0
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{Exponential, ContinuousDistribution};

let e = Exponential::new(2.0_f64); // rate = 2

assert!((e.pdf(0.0) - 2.0).abs() < 1e-12);    // PDF at 0 = lambda
assert!((e.mean() - 0.5).abs() < 1e-14);       // mean = 1/lambda
assert!((e.variance() - 0.25).abs() < 1e-14);  // variance = 1/lambda^2

// CDF: P(X <= 1) = 1 - exp(-2)
let expected = 1.0 - (-2.0_f64).exp();
assert!((e.cdf(1.0) - expected).abs() < 1e-12);
```

### Other Continuous Distributions

| Distribution | Constructor | Parameters | Mean | Variance |
|---|---|---|---|---|
| `GammaDist` | `GammaDist::new(alpha, beta)` | Shape \\(\alpha\\), rate \\(\beta\\) | \\(\alpha/\beta\\) | \\(\alpha/\beta^2\\) |
| `BetaDist` | `BetaDist::new(alpha, beta)` | \\(\alpha, \beta > 0\\) | \\(\frac{\alpha}{\alpha+\beta}\\) | \\(\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}\\) |
| `LogNormal` | `LogNormal::new(mu, sigma)` | Log-mean, log-std | \\(e^{\mu+\sigma^2/2}\\) | \\((e^{\sigma^2}-1)e^{2\mu+\sigma^2}\\) |

## Discrete Distributions

### Poisson

Models the number of events in a fixed interval. Parameterized by rate
\\(\lambda\\):

\\[
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{Poisson, DiscreteDistribution};

let p = Poisson::new(5.0_f64);

// Mean = variance = lambda
assert!((p.mean() - 5.0).abs() < 1e-14);
assert!((p.variance() - 5.0).abs() < 1e-14);

// PMF sums to 1
let sum: f64 = (0..30).map(|k| p.pmf(k)).sum();
assert!((sum - 1.0).abs() < 1e-8);

// CDF is monotone
let mut prev = 0.0;
for k in 0..20 {
    let c = p.cdf(k);
    assert!(c >= prev);
    prev = c;
}
```

### Binomial

Models the number of successes in \\(n\\) independent Bernoulli trials:

\\[
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{Binomial, DiscreteDistribution};

let b = Binomial::new(10, 0.3_f64); // n=10 trials, p=0.3

// Mean = np, Variance = np(1-p)
assert!((b.mean() - 3.0).abs() < 1e-12);
assert!((b.variance() - 2.1).abs() < 1e-12);
```

## Random Sampling

All distributions support random sampling through the `rand` crate's `RngCore`
trait. Use a seeded RNG for reproducible results.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{Normal, ContinuousDistribution};
use rand::SeedableRng;

let dist = Normal::new(0.0_f64, 1.0);
let mut rng = rand::rngs::StdRng::seed_from_u64(42);

// Single sample
let x = dist.sample(&mut rng);

// Multiple samples
let samples = dist.sample_n(&mut rng, 10000);

// Sample mean should be close to 0
let sample_mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
assert!(sample_mean.abs() < 0.1);
```

### Sampling from Different Distributions

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{
    Normal, Uniform, Exponential, Poisson,
    ContinuousDistribution, DiscreteDistribution,
};

fn main() {
    let mut rng = rand::thread_rng();

    // Continuous samples
    let normal_samples = Normal::new(100.0, 15.0).sample_n(&mut rng, 1000);
    let uniform_samples = Uniform::new(0.0, 1.0).sample_n(&mut rng, 1000);
    let exp_samples = Exponential::new(0.5).sample_n(&mut rng, 1000);

    // Discrete samples
    let poisson_samples = Poisson::new(7.0).sample_n(&mut rng, 1000);

    println!("Normal:      mean = {:.2}",
        normal_samples.iter().sum::<f64>() / 1000.0);
    println!("Uniform:     mean = {:.2}",
        uniform_samples.iter().sum::<f64>() / 1000.0);
    println!("Exponential: mean = {:.2}",
        exp_samples.iter().sum::<f64>() / 1000.0);
    println!("Poisson:     mean = {:.2}",
        poisson_samples.iter().sum::<usize>() as f64 / 1000.0);
}
```

## Distribution Summary

### Continuous Distributions

| Distribution | Constructor | Key Use |
|---|---|---|
| `Normal` | `Normal::new(mu, sigma)` | General modeling, central limit theorem |
| `Normal::standard()` | N(0, 1) | Z-scores, standard reference |
| `StudentT` | `StudentT::new(df)` | Small-sample inference |
| `ChiSquared` | `ChiSquared::new(df)` | Goodness-of-fit, variance testing |
| `FDist` | `FDist::new(df1, df2)` | ANOVA, regression F-tests |
| `Uniform` | `Uniform::new(a, b)` | Simulation, random number generation |
| `Exponential` | `Exponential::new(lambda)` | Waiting times, reliability |
| `GammaDist` | `GammaDist::new(alpha, beta)` | Bayesian priors, queuing theory |
| `BetaDist` | `BetaDist::new(alpha, beta)` | Bayesian priors for proportions |
| `LogNormal` | `LogNormal::new(mu, sigma)` | Financial modeling, multiplicative processes |

### Discrete Distributions

| Distribution | Constructor | Key Use |
|---|---|---|
| `Poisson` | `Poisson::new(lambda)` | Count data, rare events |
| `Binomial` | `Binomial::new(n, p)` | Success/failure experiments |

## Complete Example: Distribution Comparison

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{
    Normal, StudentT, ContinuousDistribution,
};

fn main() {
    let normal = Normal::<f64>::standard();

    // As degrees of freedom increase, Student's t approaches the normal
    for df in [1, 5, 10, 30, 100] {
        let t = StudentT::new(df as f64);
        let x = 1.96;
        let p_normal = normal.cdf(x);
        let p_t = t.cdf(x);
        println!("df={:3}: P(T <= 1.96) = {:.6}   P(Z <= 1.96) = {:.6}   diff = {:.6}",
            df, p_t, p_normal, (p_t - p_normal).abs());
    }
    // At df=100, the difference should be very small
}
```
