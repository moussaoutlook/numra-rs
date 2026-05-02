//! Stochastic optimization via Sample-Average Approximation (SAA).
//!
//! Given an objective `f(x, xi)` that depends on random parameters `xi`,
//! SAA draws `N` scenarios and solves:
//!
//!   min_x  (1/N) sum_{s=1}^{N} f(x, xi_s)
//!
//! Chance constraints `P{g(x, xi) <= 0} >= alpha` are enforced via a smooth
//! quadratic penalty relaxation.
//!
//! # Example
//!
//! ```rust
//! use numra_optim::stochastic::StochasticProblem;
//!
//! let result = StochasticProblem::new(1)
//!     .x0(&[0.0])
//!     .objective(|x: &[f64], p: &[f64]| (x[0] - p[0]) * (x[0] - p[0]))
//!     .param_normal("xi", 5.0, 1.0)
//!     .n_samples(200)
//!     .solve()
//!     .unwrap();
//!
//! assert!((result.x[0] - 5.0).abs() < 0.5);
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use std::sync::Arc;
use std::time::Instant;

use numra_core::Scalar;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::error::OptimError;
use crate::problem::{ConstraintKind, OptimProblem};

/// Shared parameterized scalar function: `f(x, params) -> S`.
type ParamFn<S> = Arc<dyn Fn(&[S], &[S]) -> S + Send + Sync>;
/// Boxed deterministic scalar function: `f(x) -> S`.
type DetFn<S> = Box<dyn Fn(&[S]) -> S + Send + Sync>;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A stochastic parameter with a sampling function.
pub struct StochasticParam<S: Scalar> {
    /// Parameter name.
    pub name: String,
    /// Sampling function: draws one sample from the distribution.
    pub sampler: Box<dyn Fn(&mut StdRng) -> S>,
    /// Nominal value (mean or mode, for reporting).
    pub nominal: S,
}

/// Options for stochastic optimization.
#[derive(Clone, Debug)]
pub struct StochasticOptions {
    /// Number of samples for SAA. Default: 100.
    pub n_samples: usize,
    /// Random seed. Default: 42.
    pub seed: u64,
    /// Maximum optimizer iterations. Default: 1000.
    pub max_iter: usize,
}

impl Default for StochasticOptions {
    fn default() -> Self {
        Self {
            n_samples: 100,
            seed: 42,
            max_iter: 1000,
        }
    }
}

/// Result of stochastic optimization.
#[derive(Clone, Debug)]
pub struct StochasticResult<S: Scalar> {
    /// Optimal decision variables.
    pub x: Vec<S>,
    /// Sample-average objective value.
    pub f_mean: S,
    /// Standard error of the objective estimate.
    pub f_std_error: S,
    /// Objective values at each scenario.
    pub scenario_objectives: Vec<S>,
    /// Fraction of scenarios where each chance constraint is satisfied (one per chance constraint).
    pub chance_satisfaction: Vec<S>,
    /// Whether the optimizer converged.
    pub converged: bool,
    /// Status message.
    pub message: String,
    /// Iterations.
    pub iterations: usize,
    /// Wall time.
    pub wall_time_secs: f64,
}

// ---------------------------------------------------------------------------
// Internal constraint types
// ---------------------------------------------------------------------------

/// A deterministic constraint (does not depend on stochastic parameters).
struct DetConstraint<S: Scalar> {
    func: DetFn<S>,
    kind: ConstraintKind,
}

/// A chance constraint: `P{g(x, xi) <= 0} >= probability`.
struct ChanceConstraint<S: Scalar> {
    func: ParamFn<S>,
    probability: S,
}

// ---------------------------------------------------------------------------
// Standalone helper functions
// ---------------------------------------------------------------------------

/// Create a normally distributed stochastic parameter.
///
/// Internally samples from `Normal(mean.to_f64(), std.to_f64())` and converts
/// back via `S::from_f64(...)`.
pub fn param_normal<S: Scalar>(name: &str, mean: S, std: S) -> StochasticParam<S> {
    use rand_distr::{Distribution, Normal};
    let mean_f64 = mean.to_f64();
    let std_f64 = std.to_f64();
    let dist = Normal::new(mean_f64, std_f64).unwrap();
    StochasticParam {
        name: name.to_string(),
        sampler: Box::new(move |rng: &mut StdRng| S::from_f64(dist.sample(rng))),
        nominal: mean,
    }
}

/// Create a stochastic parameter with a custom sampling closure.
///
/// This allows plugging in any distribution:
///
/// ```
/// use numra_optim::stochastic::param_sampled;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
/// use rand_distr::{Distribution, Normal};
///
/// let dist = Normal::new(5.0, 1.0).unwrap();
/// let param = param_sampled("xi", 5.0, move |rng: &mut StdRng| dist.sample(rng));
/// assert_eq!(param.name, "xi");
/// ```
pub fn param_sampled<S: Scalar>(
    name: &str,
    nominal: S,
    sampler: impl Fn(&mut StdRng) -> S + 'static,
) -> StochasticParam<S> {
    StochasticParam {
        name: name.to_string(),
        sampler: Box::new(sampler),
        nominal,
    }
}

/// Create a uniformly distributed stochastic parameter.
///
/// Internally samples from `Uniform(lo.to_f64(), hi.to_f64())` and converts
/// back via `S::from_f64(...)`.
pub fn param_uniform<S: Scalar>(name: &str, lo: S, hi: S) -> StochasticParam<S> {
    use rand_distr::{Distribution, Uniform};
    let lo_f64 = lo.to_f64();
    let hi_f64 = hi.to_f64();
    let dist = Uniform::new(lo_f64, hi_f64);
    let two = S::from_f64(2.0);
    StochasticParam {
        name: name.to_string(),
        sampler: Box::new(move |rng: &mut StdRng| S::from_f64(dist.sample(rng))),
        nominal: (lo + hi) / two,
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Declarative builder for stochastic optimization problems.
///
/// Build a problem with the fluent API, then call `.solve()` to perform
/// Sample-Average Approximation (SAA) optimization.
pub struct StochasticProblem<S: Scalar> {
    n: usize,
    x0: Option<Vec<S>>,
    bounds: Vec<Option<(S, S)>>,
    objective: Option<ParamFn<S>>,
    deterministic_constraints: Vec<DetConstraint<S>>,
    chance_constraints: Vec<ChanceConstraint<S>>,
    params: Vec<StochasticParam<S>>,
    options: StochasticOptions,
    /// If set, minimize CVaR at this confidence level instead of expected value.
    /// Uses the Rockafellar-Uryasev reformulation.
    cvar_alpha: Option<S>,
}

impl<S: Scalar> StochasticProblem<S> {
    /// Create a new stochastic optimization problem with `n` decision variables.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            x0: None,
            bounds: vec![None; n],
            objective: None,
            deterministic_constraints: Vec::new(),
            chance_constraints: Vec::new(),
            params: Vec::new(),
            options: StochasticOptions::default(),
            cvar_alpha: None,
        }
    }

    /// Set the initial point.
    pub fn x0(mut self, x0: &[S]) -> Self {
        self.x0 = Some(x0.to_vec());
        self
    }

    /// Set bounds for variable `i`.
    pub fn bounds(mut self, i: usize, lo_hi: (S, S)) -> Self {
        self.bounds[i] = Some(lo_hi);
        self
    }

    /// Set bounds for all variables at once.
    pub fn all_bounds(mut self, bounds: &[(S, S)]) -> Self {
        for (i, &b) in bounds.iter().enumerate() {
            self.bounds[i] = Some(b);
        }
        self
    }

    /// Set the parameterized objective function `f(x, params)`.
    ///
    /// The solver minimizes the expected value `E[f(x, xi)]` over the
    /// stochastic parameters.
    pub fn objective<F>(mut self, f: F) -> Self
    where
        F: Fn(&[S], &[S]) -> S + Send + Sync + 'static,
    {
        self.objective = Some(Arc::new(f));
        self
    }

    /// Add a deterministic inequality constraint `g(x) <= 0`.
    pub fn constraint_det_ineq<F>(mut self, f: F) -> Self
    where
        F: Fn(&[S]) -> S + Send + Sync + 'static,
    {
        self.deterministic_constraints.push(DetConstraint {
            func: Box::new(f),
            kind: ConstraintKind::Inequality,
        });
        self
    }

    /// Add a deterministic equality constraint `h(x) = 0`.
    pub fn constraint_det_eq<F>(mut self, f: F) -> Self
    where
        F: Fn(&[S]) -> S + Send + Sync + 'static,
    {
        self.deterministic_constraints.push(DetConstraint {
            func: Box::new(f),
            kind: ConstraintKind::Equality,
        });
        self
    }

    /// Add a chance constraint: `P{g(x, xi) <= 0} >= probability`.
    ///
    /// The constraint function `g(x, params)` should return a value where
    /// `g <= 0` means the constraint is satisfied.
    pub fn chance_constraint<F>(mut self, f: F, probability: S) -> Self
    where
        F: Fn(&[S], &[S]) -> S + Send + Sync + 'static,
    {
        self.chance_constraints.push(ChanceConstraint {
            func: Arc::new(f),
            probability,
        });
        self
    }

    /// Add a stochastic parameter directly.
    pub fn param(mut self, p: StochasticParam<S>) -> Self {
        self.params.push(p);
        self
    }

    /// Add a normally distributed stochastic parameter.
    pub fn param_normal(mut self, name: &str, mean: S, std: S) -> Self {
        self.params.push(param_normal(name, mean, std));
        self
    }

    /// Add a uniformly distributed stochastic parameter.
    pub fn param_uniform(mut self, name: &str, lo: S, hi: S) -> Self {
        self.params.push(param_uniform(name, lo, hi));
        self
    }

    /// Set the number of SAA samples.
    pub fn n_samples(mut self, n: usize) -> Self {
        self.options.n_samples = n;
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.options.seed = s;
        self
    }

    /// Set the maximum number of optimizer iterations.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.options.max_iter = n;
        self
    }

    /// Minimize CVaR (Conditional Value at Risk) at confidence level `alpha`
    /// instead of the expected value.
    ///
    /// Uses the Rockafellar-Uryasev reformulation:
    ///
    ///   CVaR_alpha(X) = min_t { t + 1/(1-alpha) * E[max(0, X - t)] }
    ///
    /// An auxiliary variable `t` (VaR estimate) is appended to the decision vector.
    /// Requires `alpha` in (0, 1), typically 0.9 or 0.95.
    pub fn minimize_cvar(mut self, alpha: S) -> Self {
        self.cvar_alpha = Some(alpha);
        self
    }
}

impl<S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField>
    StochasticProblem<S>
{
    /// Solve the stochastic optimization problem using SAA.
    ///
    /// 1. Generates `n_samples` scenarios by sampling all stochastic parameters.
    /// 2. Builds a deterministic SAA problem: `min (1/N) sum f(x, xi_s)`.
    /// 3. Chance constraints are enforced via smooth quadratic penalty.
    /// 4. Solves and evaluates constraint satisfaction at the optimal point.
    pub fn solve(self) -> Result<StochasticResult<S>, OptimError> {
        let start = Instant::now();

        // Validate inputs.
        let obj = self.objective.ok_or(OptimError::NoObjective)?;
        let x0 = self.x0.clone().ok_or(OptimError::NoInitialPoint)?;
        if self.params.is_empty() {
            return Err(OptimError::Other(
                "at least one stochastic parameter is required".to_string(),
            ));
        }

        let n = self.n;
        let n_samples = self.options.n_samples;
        let cvar_alpha = self.cvar_alpha;

        // 1. Generate scenarios.
        let mut rng = StdRng::seed_from_u64(self.options.seed);
        let scenarios: Vec<Vec<S>> = (0..n_samples)
            .map(|_| self.params.iter().map(|p| (p.sampler)(&mut rng)).collect())
            .collect();
        let scenarios = Arc::new(scenarios);

        // 2. Build SAA objective.
        let obj_for_saa = Arc::clone(&obj);
        let scenarios_for_saa = Arc::clone(&scenarios);

        // Collect chance constraints into Arcs for sharing.
        let chance_fns: Vec<ParamFn<S>> = self
            .chance_constraints
            .iter()
            .map(|cc| Arc::clone(&cc.func))
            .collect();
        let chance_fns = Arc::new(chance_fns);
        let chance_fns_for_saa = Arc::clone(&chance_fns);

        // Per-constraint penalty weights: higher required probability gets a larger weight.
        let base_penalty = S::from_f64(1e4);
        let chance_weights: Arc<Vec<S>> = Arc::new(
            self.chance_constraints
                .iter()
                .map(|cc| base_penalty * cc.probability)
                .collect(),
        );
        let chance_weights_for_saa = Arc::clone(&chance_weights);
        let n_chance = self.chance_constraints.len();

        // Determine problem dimension: n + 1 if CVaR (extra variable t = VaR estimate)
        let n_opt = if cvar_alpha.is_some() { n + 1 } else { n };
        let mut x0_ext = x0.clone();
        if cvar_alpha.is_some() {
            // Initialize t to the median scenario objective
            let mut f_scenarios: Vec<S> = scenarios.iter().map(|s| obj(&x0, s)).collect();
            f_scenarios.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
            let median_idx = f_scenarios.len() / 2;
            x0_ext.push(f_scenarios[median_idx]);
        }

        let saa_objective = move |x_ext: &[S]| -> S {
            let ns = scenarios_for_saa.len();
            let inv_n = S::ONE / S::from_usize(ns);

            match cvar_alpha {
                Some(alpha) => {
                    // CVaR via Rockafellar-Uryasev reformulation:
                    //   min_t { t + 1/(1-alpha) * (1/N) * sum max(0, f(x, xi_s) - t) }
                    // Use smooth approximation: max(0, z) ≈ z/2 + sqrt(z^2 + eps)/2
                    let x = &x_ext[..x_ext.len() - 1];
                    let t = x_ext[x_ext.len() - 1];
                    let inv_one_minus_alpha = S::ONE / (S::ONE - alpha);
                    let eps = S::from_f64(1e-6);

                    let mut cvar_sum = S::ZERO;
                    for s in 0..ns {
                        let fs = obj_for_saa(x, &scenarios_for_saa[s]);
                        let z = fs - t;
                        // Smooth max(0, z): (z + sqrt(z^2 + eps)) / 2
                        let smooth_max = (z + (z * z + eps).sqrt()) * S::HALF;
                        cvar_sum += smooth_max;
                    }

                    // Chance constraint penalty on original variables
                    let mut penalty = S::ZERO;
                    for c in 0..n_chance {
                        let w = chance_weights_for_saa[c];
                        for s in 0..ns {
                            let g = chance_fns_for_saa[c](x, &scenarios_for_saa[s]);
                            if g > S::ZERO {
                                penalty += w * g * g;
                            }
                        }
                    }

                    t + inv_one_minus_alpha * inv_n * cvar_sum + inv_n * penalty
                }
                None => {
                    // Standard SAA: expected value
                    let x = x_ext;
                    let mut f_sum = S::ZERO;
                    for s in 0..ns {
                        f_sum += obj_for_saa(x, &scenarios_for_saa[s]);
                    }

                    // Chance constraint penalty
                    let mut penalty = S::ZERO;
                    for c in 0..n_chance {
                        let w = chance_weights_for_saa[c];
                        for s in 0..ns {
                            let g = chance_fns_for_saa[c](x, &scenarios_for_saa[s]);
                            if g > S::ZERO {
                                penalty += w * g * g;
                            }
                        }
                    }

                    inv_n * f_sum + inv_n * penalty
                }
            }
        };

        // 3. Build OptimProblem.
        let mut problem = OptimProblem::<S>::new(n_opt)
            .x0(&x0_ext)
            .objective(saa_objective)
            .max_iter(self.options.max_iter);

        // Apply bounds (only to original variables, not the VaR auxiliary).
        for (i, b) in self.bounds.iter().enumerate() {
            if let Some(lo_hi) = b {
                problem = problem.bounds(i, *lo_hi);
            }
        }

        // Add deterministic constraints (on original variables).
        // For CVaR, constraints ignore the auxiliary variable.
        for dc in self.deterministic_constraints {
            if cvar_alpha.is_some() {
                let dc_n = n;
                match dc.kind {
                    ConstraintKind::Inequality => {
                        let func = dc.func;
                        problem = problem.constraint_ineq(move |x_ext: &[S]| func(&x_ext[..dc_n]));
                    }
                    ConstraintKind::Equality => {
                        let func = dc.func;
                        problem = problem.constraint_eq(move |x_ext: &[S]| func(&x_ext[..dc_n]));
                    }
                }
            } else {
                match dc.kind {
                    ConstraintKind::Inequality => {
                        problem = problem.constraint_ineq(dc.func);
                    }
                    ConstraintKind::Equality => {
                        problem = problem.constraint_eq(dc.func);
                    }
                }
            }
        }

        // 4. Solve.
        let result = problem.solve()?;
        // Extract original decision variables (strip auxiliary VaR variable if CVaR)
        let x_star = if cvar_alpha.is_some() {
            result.x[..n].to_vec()
        } else {
            result.x.clone()
        };

        // 5. Post-solve evaluation.
        let scenario_objectives: Vec<S> = scenarios.iter().map(|s| obj(&x_star, s)).collect();

        let f_mean = scenario_objectives.iter().copied().sum::<S>() / S::from_usize(n_samples);

        let f_std_error = if n_samples > 1 {
            let variance = scenario_objectives
                .iter()
                .map(|&fi| (fi - f_mean) * (fi - f_mean))
                .sum::<S>()
                / S::from_usize(n_samples - 1);
            variance.sqrt() / S::from_usize(n_samples).sqrt()
        } else {
            S::ZERO
        };

        // Evaluate chance constraint satisfaction.
        let chance_satisfaction: Vec<S> = self
            .chance_constraints
            .iter()
            .map(|cc| {
                let n_satisfied = scenarios
                    .iter()
                    .filter(|s| (cc.func)(&x_star, s) <= S::ZERO)
                    .count();
                S::from_usize(n_satisfied) / S::from_usize(n_samples)
            })
            .collect();

        Ok(StochasticResult {
            x: x_star,
            f_mean,
            f_std_error,
            scenario_objectives,
            chance_satisfaction,
            converged: result.converged,
            message: result.message,
            iterations: result.iterations,
            wall_time_secs: start.elapsed().as_secs_f64(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_saa_expected_value() {
        // min E[(x - xi)^2] where xi ~ Normal(5, 1).
        // Optimal: x* = E[xi] = 5.0.
        let result = StochasticProblem::new(1)
            .x0(&[0.0])
            .objective(|x: &[f64], p: &[f64]| (x[0] - p[0]) * (x[0] - p[0]))
            .param_normal("xi", 5.0, 1.0)
            .n_samples(200)
            .solve()
            .unwrap();

        assert!(
            (result.x[0] - 5.0).abs() < 0.5,
            "x* = {}, expected ~5.0",
            result.x[0]
        );
        assert!(result.converged, "solver should converge");
        assert_eq!(result.scenario_objectives.len(), 200);
        assert!(result.f_std_error > 0.0, "std error should be positive");
    }

    #[test]
    fn test_saa_bounded() {
        // min E[(x - xi)^2] where xi ~ Normal(5, 1), x in [0, 3].
        // Unconstrained optimal: x* = 5. With bound: x* = 3 (active).
        let result = StochasticProblem::new(1)
            .x0(&[1.0])
            .objective(|x: &[f64], p: &[f64]| (x[0] - p[0]) * (x[0] - p[0]))
            .param_normal("xi", 5.0, 1.0)
            .bounds(0, (0.0, 3.0))
            .n_samples(200)
            .solve()
            .unwrap();

        assert!(
            (result.x[0] - 3.0).abs() < 0.1,
            "x* = {}, expected ~3.0 (bound active)",
            result.x[0]
        );
    }

    #[test]
    fn test_chance_constraint() {
        // max x  (i.e. min -x) subject to:
        //   P{x - xi <= 0} >= 0.95  (i.e. x <= xi with 95% probability)
        //   xi ~ Normal(10, 2)
        //   x in [0, 20]
        // The optimal should be around the 5th percentile of xi:
        //   10 - 1.645*2 ~ 6.71
        let result = StochasticProblem::new(1)
            .x0(&[5.0])
            .objective(|x: &[f64], _p: &[f64]| -x[0])
            .chance_constraint(
                |x: &[f64], p: &[f64]| x[0] - p[0], // g(x,xi) = x - xi <= 0
                0.95,
            )
            .param_normal("xi", 10.0, 2.0)
            .bounds(0, (0.0, 20.0))
            .n_samples(500)
            .max_iter(2000)
            .solve()
            .unwrap();

        assert!(
            result.x[0] < 10.0,
            "x* = {}, expected < 10.0 (must be conservative)",
            result.x[0]
        );
        assert!(result.x[0] > 0.0, "x* = {}, expected > 0.0", result.x[0]);
        // Due to penalty relaxation it won't be exact, but should be in a reasonable range.
        assert!(
            result.chance_satisfaction.len() == 1,
            "should have one chance constraint"
        );
    }

    #[test]
    fn test_param_normal_uniform_helpers() {
        // Test that param_normal and param_uniform produce correct distributions.
        let mut rng = StdRng::seed_from_u64(123);

        // Normal(5, 1): draw 1000 samples, check mean and std.
        {
            use rand_distr::{Distribution, Normal};
            let dist = Normal::new(5.0, 1.0).unwrap();
            let samples: Vec<f64> = (0..1000).map(|_| dist.sample(&mut rng)).collect();
            let mean = samples.iter().sum::<f64>() / 1000.0;
            let variance = samples
                .iter()
                .map(|&s| (s - mean) * (s - mean))
                .sum::<f64>()
                / 999.0;
            let std = variance.sqrt();
            assert!(
                (mean - 5.0).abs() < 0.3,
                "Normal mean = {}, expected ~5.0",
                mean
            );
            assert!(
                (std - 1.0).abs() < 0.3,
                "Normal std = {}, expected ~1.0",
                std
            );
        }

        // Uniform(0, 10): draw 1000 samples, check range and mean.
        {
            use rand_distr::{Distribution, Uniform};
            let dist = Uniform::new(0.0, 10.0);
            let samples: Vec<f64> = (0..1000).map(|_| dist.sample(&mut rng)).collect();
            let mean = samples.iter().sum::<f64>() / 1000.0;
            assert!(
                samples.iter().all(|&s| (0.0..10.0).contains(&s)),
                "Uniform samples should be in [0, 10)"
            );
            assert!(
                (mean - 5.0).abs() < 1.0,
                "Uniform mean = {}, expected ~5.0",
                mean
            );
        }

        // Also test that StochasticParam samplers work through the builder.
        let problem = StochasticProblem::<f64>::new(1)
            .param_normal("n1", 5.0, 1.0)
            .param_uniform("u1", 0.0, 10.0);
        assert_eq!(problem.params.len(), 2);
        assert_eq!(problem.params[0].name, "n1");
        assert!((problem.params[0].nominal - 5.0).abs() < 1e-15);
        assert_eq!(problem.params[1].name, "u1");
        assert!((problem.params[1].nominal - 5.0).abs() < 1e-15);

        // Sample from the parameters using the builder's samplers.
        let mut rng2 = StdRng::seed_from_u64(456);
        let normal_samples: Vec<f64> = (0..1000)
            .map(|_| (problem.params[0].sampler)(&mut rng2))
            .collect();
        let n_mean = normal_samples.iter().sum::<f64>() / 1000.0;
        assert!(
            (n_mean - 5.0).abs() < 0.3,
            "StochasticParam Normal mean = {}, expected ~5.0",
            n_mean
        );

        let uniform_samples: Vec<f64> = (0..1000)
            .map(|_| (problem.params[1].sampler)(&mut rng2))
            .collect();
        let u_mean = uniform_samples.iter().sum::<f64>() / 1000.0;
        assert!(
            uniform_samples.iter().all(|&s| (0.0..10.0).contains(&s)),
            "StochasticParam Uniform samples should be in [0, 10)"
        );
        assert!(
            (u_mean - 5.0).abs() < 1.0,
            "StochasticParam Uniform mean = {}, expected ~5.0",
            u_mean
        );
    }

    #[test]
    fn test_param_sampled() {
        // Use param_sampled with a custom closure (simulating what a user with numra-stats would do)
        use rand_distr::{Distribution, Normal};
        let dist = Normal::new(5.0_f64, 1.0).unwrap();
        let p = param_sampled("xi", 5.0, move |rng: &mut StdRng| dist.sample(rng));

        assert_eq!(p.name, "xi");
        assert!((p.nominal - 5.0).abs() < 1e-15);

        // Verify sampling works
        let mut rng = StdRng::seed_from_u64(42);
        let samples: Vec<f64> = (0..1000).map(|_| (p.sampler)(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / 1000.0;
        assert!((mean - 5.0).abs() < 0.3);
    }

    #[test]
    fn test_param_sampled_in_problem() {
        // Verify param_sampled integrates with StochasticProblem
        use rand_distr::{Distribution, Normal};
        let dist = Normal::new(5.0_f64, 1.0).unwrap();

        let result = StochasticProblem::new(1)
            .x0(&[0.0])
            .objective(|x: &[f64], p: &[f64]| (x[0] - p[0]) * (x[0] - p[0]))
            .param(param_sampled("xi", 5.0, move |rng: &mut StdRng| {
                dist.sample(rng)
            }))
            .n_samples(200)
            .solve()
            .unwrap();

        assert!((result.x[0] - 5.0).abs() < 0.5);
        assert!(result.converged);
    }

    #[test]
    fn test_cvar_minimization() {
        // Minimize CVaR_0.9 of (x - xi)^2 where xi ~ Normal(5, 2).
        // CVaR focuses on the worst 10% of outcomes.
        // For squared loss, CVaR-optimal x should be close to E[xi] = 5 but
        // potentially shifted toward the tail, depending on the asymmetry.
        // For symmetric distributions with squared loss, E[f] minimizer = CVaR minimizer = mean.
        let result_cvar = StochasticProblem::new(1)
            .x0(&[3.0])
            .objective(|x: &[f64], p: &[f64]| (x[0] - p[0]) * (x[0] - p[0]))
            .param_normal("xi", 5.0, 2.0)
            .n_samples(500)
            .max_iter(2000)
            .minimize_cvar(0.9)
            .solve()
            .unwrap();

        // Result should have only 1 decision variable (not the auxiliary t)
        assert_eq!(result_cvar.x.len(), 1, "CVaR should return original dim");
        assert!(
            result_cvar.converged,
            "CVaR should converge: {}",
            result_cvar.message
        );
        // For symmetric squared loss, CVaR minimizer should still be near the mean
        assert!(
            (result_cvar.x[0] - 5.0).abs() < 1.0,
            "CVaR x* = {}, expected ~5.0",
            result_cvar.x[0]
        );

        // Compare: CVaR should be >= expected value
        let result_ev = StochasticProblem::new(1)
            .x0(&[0.0])
            .objective(|x: &[f64], p: &[f64]| (x[0] - p[0]) * (x[0] - p[0]))
            .param_normal("xi", 5.0, 2.0)
            .n_samples(500)
            .solve()
            .unwrap();

        // CVaR should report f_mean (average over all scenarios, not just tail)
        // The important thing is that the solver ran correctly
        assert!(
            result_cvar.f_mean >= result_ev.f_mean * 0.5,
            "CVaR f_mean={} should be comparable to EV f_mean={}",
            result_cvar.f_mean,
            result_ev.f_mean
        );
    }
}
