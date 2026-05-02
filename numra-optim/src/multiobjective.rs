//! NSGA-II multi-objective optimizer.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use crate::error::OptimError;
use crate::types::{OptimResult, OptimStatus, ParetoPoint, ParetoResult};
use numra_core::Scalar;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Type alias for objective function references used in NSGA-II.
type ObjFnSlice<'a, S> = &'a [&'a dyn Fn(&[S]) -> S];

/// Options for NSGA-II multi-objective optimization.
#[derive(Clone, Debug)]
pub struct NsgaIIOptions<S: Scalar> {
    /// Population size (must be even). Default: 100.
    pub pop_size: usize,
    /// Maximum number of generations. Default: 200.
    pub max_generations: usize,
    /// SBX crossover distribution index. Default: 20.0.
    pub crossover_eta: S,
    /// Polynomial mutation distribution index. Default: 20.0.
    pub mutation_eta: S,
    /// Crossover probability. Default: 0.9.
    pub crossover_prob: S,
    /// Mutation probability per variable. Default: None (= 1/n).
    pub mutation_prob: Option<S>,
    /// Random seed for reproducibility. Default: 42.
    pub seed: u64,
    /// Print progress information. Default: false.
    pub verbose: bool,
}

impl<S: Scalar> Default for NsgaIIOptions<S> {
    fn default() -> Self {
        Self {
            pop_size: 100,
            max_generations: 200,
            crossover_eta: S::from_f64(20.0),
            mutation_eta: S::from_f64(20.0),
            crossover_prob: S::from_f64(0.9),
            mutation_prob: None,
            seed: 42,
            verbose: false,
        }
    }
}

/// Perform non-dominated sorting on a population given their objective values.
///
/// Returns a vector of fronts, where each front is a vector of individual indices.
/// Front 0 contains the non-dominated set, front 1 the next layer, etc.
fn non_dominated_sort<S: Scalar>(objectives: &[Vec<S>]) -> Vec<Vec<usize>> {
    let n = objectives.len();
    if n == 0 {
        return vec![];
    }

    // For each individual p:
    //   s_p: set of individuals that p dominates
    //   n_p: number of individuals that dominate p
    let mut s_p: Vec<Vec<usize>> = vec![vec![]; n];
    let mut n_p: Vec<usize> = vec![0; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dom_ij = dominates(&objectives[i], &objectives[j]);
            let dom_ji = dominates(&objectives[j], &objectives[i]);
            if dom_ij {
                s_p[i].push(j);
                n_p[j] += 1;
            } else if dom_ji {
                s_p[j].push(i);
                n_p[i] += 1;
            }
        }
    }

    let mut fronts: Vec<Vec<usize>> = vec![];

    // First front: all individuals with n_p == 0
    let mut current_front: Vec<usize> = (0..n).filter(|&i| n_p[i] == 0).collect();

    while !current_front.is_empty() {
        let mut next_front = vec![];
        for &p in &current_front {
            for &q in &s_p[p] {
                n_p[q] -= 1;
                if n_p[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

/// Returns true if `a` dominates `b` (all objectives of a <= b, at least one strictly <).
fn dominates<S: Scalar>(a: &[S], b: &[S]) -> bool {
    let mut any_strictly_less = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if *ai > *bi {
            return false;
        }
        if *ai < *bi {
            any_strictly_less = true;
        }
    }
    any_strictly_less
}

/// Compute crowding distances for individuals in a single front.
///
/// `front` contains indices into the `objectives` array.
/// Returns a vector of crowding distances, one per element in `front`.
fn crowding_distance<S: Scalar>(front: &[usize], objectives: &[Vec<S>]) -> Vec<S> {
    let n = front.len();
    if n <= 2 {
        return vec![S::INFINITY; n];
    }

    let n_obj = objectives[front[0]].len();
    let mut distances = vec![S::ZERO; n];

    #[allow(clippy::needless_range_loop)]
    for m in 0..n_obj {
        // Sort front indices by objective m
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            let fa = objectives[front[a]][m];
            let fb = objectives[front[b]][m];
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let obj_min = objectives[front[sorted_indices[0]]][m];
        let obj_max = objectives[front[sorted_indices[n - 1]]][m];
        let range = obj_max - obj_min;

        // Endpoints get infinity
        distances[sorted_indices[0]] = S::INFINITY;
        distances[sorted_indices[n - 1]] = S::INFINITY;

        if range < S::from_f64(1e-30) {
            // All values essentially the same for this objective; skip contribution
            continue;
        }

        for i in 1..(n - 1) {
            let prev_obj = objectives[front[sorted_indices[i - 1]]][m];
            let next_obj = objectives[front[sorted_indices[i + 1]]][m];
            distances[sorted_indices[i]] += (next_obj - prev_obj) / range;
        }
    }

    distances
}

/// Binary tournament selection.
/// Pick 2 random individuals. Lower front rank wins; if tied, higher crowding distance wins.
fn tournament_select<S: Scalar>(
    rng: &mut SmallRng,
    ranks: &[usize],
    crowding: &[S],
    pop_size: usize,
) -> usize {
    let a = rng.gen_range(0..pop_size);
    let b = rng.gen_range(0..pop_size);
    if ranks[a] < ranks[b] {
        a
    } else if ranks[b] < ranks[a] {
        b
    } else if crowding[a] > crowding[b] {
        a
    } else {
        b
    }
}

/// Simulated Binary Crossover (SBX) for two parents.
/// Produces two children, clipped to bounds.
fn sbx_crossover<S: Scalar>(
    p1: &[S],
    p2: &[S],
    bounds: &[(S, S)],
    eta: S,
    prob: S,
    rng: &mut SmallRng,
) -> (Vec<S>, Vec<S>) {
    let n = p1.len();
    let mut c1 = p1.to_vec();
    let mut c2 = p2.to_vec();

    if S::from_f64(rng.gen::<f64>()) > prob {
        return (c1, c2);
    }

    let half = S::HALF;
    let one = S::ONE;

    for j in 0..n {
        if (p1[j] - p2[j]).abs() < S::from_f64(1e-14) {
            continue;
        }

        let u = S::from_f64(rng.gen::<f64>());
        let exp = one / (eta + one);
        let beta = if u <= half {
            (S::TWO * u).powf(exp)
        } else {
            (one / (S::TWO * (one - u))).powf(exp)
        };

        c1[j] = half * ((one + beta) * p1[j] + (one - beta) * p2[j]);
        c2[j] = half * ((one - beta) * p1[j] + (one + beta) * p2[j]);

        // Clip to bounds
        c1[j] = c1[j].clamp(bounds[j].0, bounds[j].1);
        c2[j] = c2[j].clamp(bounds[j].0, bounds[j].1);
    }

    (c1, c2)
}

/// Polynomial mutation.
/// Mutates each variable with given probability, clipped to bounds.
fn polynomial_mutation<S: Scalar>(
    x: &mut [S],
    bounds: &[(S, S)],
    eta: S,
    prob: S,
    rng: &mut SmallRng,
) {
    let half = S::HALF;
    let one = S::ONE;
    let two = S::TWO;

    for j in 0..x.len() {
        if S::from_f64(rng.gen::<f64>()) >= prob {
            continue;
        }

        let u = S::from_f64(rng.gen::<f64>());
        let exp = one / (eta + one);
        let delta = if u < half {
            (two * u).powf(exp) - one
        } else {
            one - (two * (one - u)).powf(exp)
        };

        let range = bounds[j].1 - bounds[j].0;
        x[j] += delta * range;
        x[j] = x[j].clamp(bounds[j].0, bounds[j].1);
    }
}

/// Evaluate all objectives for a single individual.
fn evaluate<S: Scalar>(objectives: ObjFnSlice<'_, S>, x: &[S]) -> Vec<S> {
    objectives.iter().map(|f| f(x)).collect()
}

/// NSGA-II multi-objective optimization.
///
/// Minimizes multiple objectives simultaneously, returning a set of Pareto-optimal solutions.
///
/// # Arguments
///
/// * `objectives` - Slice of objective functions, each taking `&[S]` and returning `S`.
/// * `bounds` - Variable bounds as `(lower, upper)` pairs, one per decision variable.
/// * `opts` - Algorithm options.
///
/// # Returns
///
/// An `OptimResult` with:
/// - `x`: the Pareto point with the best (minimum) first objective value.
/// - `f`: that point's first objective value.
/// - `pareto`: the full Pareto front as `Some(ParetoResult)`.
pub fn nsga2_optimize<S: Scalar>(
    objectives: ObjFnSlice<'_, S>,
    bounds: &[(S, S)],
    opts: &NsgaIIOptions<S>,
) -> Result<OptimResult<S>, OptimError> {
    let start = std::time::Instant::now();
    let n = bounds.len();
    let pop_size = opts.pop_size;
    let mutation_prob = opts
        .mutation_prob
        .unwrap_or_else(|| S::ONE / S::from_usize(n));
    let mut rng = SmallRng::seed_from_u64(opts.seed);

    // 1. Initialize population uniformly in bounds
    let mut population: Vec<Vec<S>> = (0..pop_size)
        .map(|_| {
            (0..n)
                .map(|j| {
                    let (lo, hi) = bounds[j];
                    lo + S::from_f64(rng.gen::<f64>()) * (hi - lo)
                })
                .collect()
        })
        .collect();

    // 2. Evaluate all objectives
    let mut obj_values: Vec<Vec<S>> = population.iter().map(|x| evaluate(objectives, x)).collect();

    // 3. Main generational loop
    for gen in 0..opts.max_generations {
        // Compute ranks and crowding distances for current population
        let fronts = non_dominated_sort(&obj_values);
        let mut ranks = vec![0_usize; pop_size];
        let mut crowding = vec![S::ZERO; pop_size];
        for (rank, front) in fronts.iter().enumerate() {
            let cd = crowding_distance(front, &obj_values);
            for (i, &idx) in front.iter().enumerate() {
                ranks[idx] = rank;
                crowding[idx] = cd[i];
            }
        }

        if opts.verbose && gen % 10 == 0 {
            let n_front0 = fronts.first().map_or(0, |f| f.len());
            eprintln!(
                "NSGA-II gen {}: front 0 size = {}, pop = {}",
                gen, n_front0, pop_size
            );
        }

        // 3a. Create offspring population via tournament + SBX + mutation
        let mut offspring: Vec<Vec<S>> = Vec::with_capacity(pop_size);
        while offspring.len() < pop_size {
            let p1_idx = tournament_select(&mut rng, &ranks, &crowding, pop_size);
            let p2_idx = tournament_select(&mut rng, &ranks, &crowding, pop_size);

            let (mut c1, mut c2) = sbx_crossover(
                &population[p1_idx],
                &population[p2_idx],
                bounds,
                opts.crossover_eta,
                opts.crossover_prob,
                &mut rng,
            );

            polynomial_mutation(&mut c1, bounds, opts.mutation_eta, mutation_prob, &mut rng);
            polynomial_mutation(&mut c2, bounds, opts.mutation_eta, mutation_prob, &mut rng);

            offspring.push(c1);
            if offspring.len() < pop_size {
                offspring.push(c2);
            }
        }

        // 3b. Evaluate offspring objectives
        let offspring_obj: Vec<Vec<S>> =
            offspring.iter().map(|x| evaluate(objectives, x)).collect();

        // 3c. Combine parent + offspring (2N individuals)
        let mut combined_pop: Vec<Vec<S>> = population;
        combined_pop.extend(offspring);
        let mut combined_obj: Vec<Vec<S>> = obj_values;
        combined_obj.extend(offspring_obj);

        // 3d. Non-dominated sort the combined population
        let combined_fronts = non_dominated_sort(&combined_obj);

        // 3e-f. Fill next generation from fronts
        let mut new_population: Vec<Vec<S>> = Vec::with_capacity(pop_size);
        let mut new_obj: Vec<Vec<S>> = Vec::with_capacity(pop_size);

        for front in &combined_fronts {
            if new_population.len() + front.len() <= pop_size {
                // Entire front fits
                for &idx in front {
                    new_population.push(combined_pop[idx].clone());
                    new_obj.push(combined_obj[idx].clone());
                }
            } else {
                // Partial front: sort by crowding distance, take the best
                let remaining = pop_size - new_population.len();
                let cd = crowding_distance(front, &combined_obj);
                let mut sorted: Vec<usize> = (0..front.len()).collect();
                sorted.sort_by(|&a, &b| {
                    cd[b]
                        .partial_cmp(&cd[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for &i in sorted.iter().take(remaining) {
                    let idx = front[i];
                    new_population.push(combined_pop[idx].clone());
                    new_obj.push(combined_obj[idx].clone());
                }
                break;
            }
        }

        population = new_population;
        obj_values = new_obj;
    }

    // 4. Extract front 0 from final population
    let final_fronts = non_dominated_sort(&obj_values);
    let front0 = &final_fronts[0];

    let pareto_points: Vec<ParetoPoint<S>> = front0
        .iter()
        .map(|&idx| ParetoPoint {
            x: population[idx].clone(),
            objectives: obj_values[idx].clone(),
        })
        .collect();

    // Find the point with best (minimum) first objective
    let best_idx = front0
        .iter()
        .min_by(|&&a, &&b| {
            obj_values[a][0]
                .partial_cmp(&obj_values[b][0])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
        .unwrap_or(0);

    let result = OptimResult {
        x: population[best_idx].clone(),
        f: obj_values[best_idx][0],
        grad: vec![],
        iterations: opts.max_generations,
        n_feval: 0,
        n_geval: 0,
        converged: true,
        message: format!(
            "NSGA-II completed {} generations, Pareto front size = {}",
            opts.max_generations,
            pareto_points.len()
        ),
        status: OptimStatus::MaxIterations,
        history: vec![],
        lambda_eq: vec![],
        lambda_ineq: vec![],
        active_bounds: vec![],
        constraint_violation: S::ZERO,
        wall_time_secs: 0.0,
        pareto: Some(ParetoResult {
            points: pareto_points,
        }),
        sensitivity: None,
    }
    .with_wall_time(start);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    type ObjRef<'a> = Vec<&'a dyn Fn(&[f64]) -> f64>;

    #[test]
    fn test_nsga2_zdt1() {
        // ZDT1 in 3D (reduced for speed)
        let n = 3;
        let bounds = vec![(0.0, 1.0); n];
        let f1 = |x: &[f64]| x[0];
        let f2 = |x: &[f64]| {
            let g = 1.0 + 9.0 * x[1..].iter().copied().sum::<f64>() / (x.len() - 1) as f64;
            g * (1.0 - (x[0] / g).sqrt())
        };
        let objectives: ObjRef<'_> = vec![&f1, &f2];
        let opts = NsgaIIOptions {
            pop_size: 50,
            max_generations: 100,
            ..Default::default()
        };
        let result = nsga2_optimize(&objectives, &bounds, &opts).unwrap();
        let pareto = result.pareto.as_ref().unwrap();
        assert!(
            pareto.points.len() >= 5,
            "too few points: {}",
            pareto.points.len()
        );
        for p in &pareto.points {
            assert_eq!(p.objectives.len(), 2);
        }
    }

    #[test]
    fn test_nsga2_simple_biobj() {
        // f1 = x^2, f2 = (x-2)^2, Pareto front: x in [0, 2]
        let bounds = vec![(0.0, 4.0)];
        let f1 = |x: &[f64]| x[0] * x[0];
        let f2 = |x: &[f64]| (x[0] - 2.0).powi(2);
        let objectives: ObjRef<'_> = vec![&f1, &f2];
        let opts = NsgaIIOptions {
            pop_size: 50,
            max_generations: 100,
            ..Default::default()
        };
        let result = nsga2_optimize(&objectives, &bounds, &opts).unwrap();
        let pareto = result.pareto.as_ref().unwrap();
        assert!(pareto.points.len() >= 3);
        for p in &pareto.points {
            assert!(p.x[0] >= -0.5 && p.x[0] <= 2.5, "x={}", p.x[0]);
        }
    }

    #[test]
    fn test_nsga2_three_objectives() {
        let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
        let f1 = |x: &[f64]| x[0] * x[0];
        let f2 = |x: &[f64]| x[1] * x[1];
        let f3 = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2);
        let objectives: ObjRef<'_> = vec![&f1, &f2, &f3];
        let opts = NsgaIIOptions {
            pop_size: 50,
            max_generations: 100,
            ..Default::default()
        };
        let result = nsga2_optimize(&objectives, &bounds, &opts).unwrap();
        let pareto = result.pareto.as_ref().unwrap();
        assert!(!pareto.points.is_empty());
        for p in &pareto.points {
            assert_eq!(p.objectives.len(), 3);
        }
    }

    #[test]
    fn test_nsga2_deterministic() {
        let bounds = vec![(0.0, 1.0); 2];
        let f1 = |x: &[f64]| x[0];
        let f2 = |x: &[f64]| 1.0 - x[0] + x[1];
        let objectives: ObjRef<'_> = vec![&f1, &f2];
        let opts = NsgaIIOptions {
            pop_size: 20,
            max_generations: 50,
            ..Default::default()
        };
        let r1 = nsga2_optimize(&objectives, &bounds, &opts).unwrap();
        let r2 = nsga2_optimize(&objectives, &bounds, &opts).unwrap();
        assert_eq!(
            r1.pareto.unwrap().points.len(),
            r2.pareto.unwrap().points.len()
        );
    }

    #[test]
    fn test_nsga2_returns_nondominated() {
        let bounds = vec![(0.0, 5.0); 2];
        let f1 = |x: &[f64]| x[0] * x[0] + x[1];
        let f2 = |x: &[f64]| x[1] * x[1] + x[0];
        let objectives: ObjRef<'_> = vec![&f1, &f2];
        let opts = NsgaIIOptions {
            pop_size: 30,
            max_generations: 80,
            ..Default::default()
        };
        let result = nsga2_optimize(&objectives, &bounds, &opts).unwrap();
        let pareto = result.pareto.as_ref().unwrap();
        for (i, pi) in pareto.points.iter().enumerate() {
            for (j, pj) in pareto.points.iter().enumerate() {
                if i == j {
                    continue;
                }
                let all_leq = pi
                    .objectives
                    .iter()
                    .zip(&pj.objectives)
                    .all(|(a, b)| a <= b);
                let any_lt = pi.objectives.iter().zip(&pj.objectives).any(|(a, b)| a < b);
                assert!(
                    !(all_leq && any_lt),
                    "point {} dominates point {}: {:?} vs {:?}",
                    i,
                    j,
                    pi.objectives,
                    pj.objectives
                );
            }
        }
    }
}
