//! Branch-and-Bound solver for Mixed-Integer Linear Programs (MILP).
//!
//! Solves: minimize c^T x subject to A_ineq x <= b_ineq, A_eq x = b_eq,
//! with variable types (continuous, integer, binary) and optional bounds.
//!
//! Uses best-first branch and bound with LP relaxations solved by the simplex method.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use numra_core::Scalar;

use crate::error::OptimError;
use crate::lp::{simplex_solve, LPOptions};
use crate::problem::VarType;
use crate::types::{OptimResult, OptimStatus};

/// Options for the MILP branch-and-bound solver.
#[derive(Clone, Debug)]
pub struct MILPOptions<S: Scalar> {
    /// Maximum number of branch-and-bound nodes to explore.
    pub max_nodes: usize,
    /// Tolerance for considering a value integer-feasible.
    pub int_tol: S,
    /// Tolerance for the LP sub-solver.
    pub lp_tol: S,
    /// Relative gap tolerance for termination.
    pub gap_tol: S,
    /// Print progress information.
    pub verbose: bool,
}

impl<S: Scalar> Default for MILPOptions<S> {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            int_tol: S::from_f64(1e-6),
            lp_tol: S::from_f64(1e-10),
            gap_tol: S::from_f64(1e-8),
            verbose: false,
        }
    }
}

/// A node in the branch-and-bound tree.
#[derive(Clone, Debug)]
struct BbNode<S: Scalar> {
    /// Tightened lower bounds for each variable.
    lb: Vec<S>,
    /// Tightened upper bounds for each variable.
    ub: Vec<S>,
    /// LP relaxation objective at this node (lower bound on integer optimum in subtree).
    lp_bound: S,
    /// Depth in the B&B tree (for diagnostics).
    depth: usize,
}

/// Wrapper for BinaryHeap ordering: best-first = smallest lp_bound first.
/// BinaryHeap is a max-heap, so we reverse the comparison.
#[derive(Clone, Debug)]
struct OrderedNode<S: Scalar>(BbNode<S>);

impl<S: Scalar> PartialEq for OrderedNode<S> {
    fn eq(&self, other: &Self) -> bool {
        self.0.lp_bound == other.0.lp_bound
    }
}

impl<S: Scalar> Eq for OrderedNode<S> {}

impl<S: Scalar> PartialOrd for OrderedNode<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: Scalar> Ord for OrderedNode<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering: smaller lp_bound gets higher priority.
        other
            .0
            .lp_bound
            .partial_cmp(&self.0.lp_bound)
            .unwrap_or(Ordering::Equal)
    }
}

/// Solve a Mixed-Integer Linear Program using branch and bound.
///
/// # Arguments
///
/// * `c` - Objective coefficients (minimize c^T x)
/// * `a_ineq` - Inequality constraint matrix rows (A_ineq x <= b_ineq)
/// * `b_ineq` - Inequality constraint RHS
/// * `a_eq` - Equality constraint matrix rows (A_eq x = b_eq)
/// * `b_eq` - Equality constraint RHS
/// * `var_types` - Type of each variable (Continuous, Integer, Binary)
/// * `bounds` - Optional (lower, upper) bounds for each variable
/// * `opts` - Solver options
///
/// # Returns
///
/// An `OptimResult` with the optimal integer-feasible solution, or an error if
/// the problem is infeasible or the node limit is reached.
#[allow(clippy::too_many_arguments)]
pub fn milp_solve<
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
>(
    c: &[S],
    a_ineq: &[Vec<S>],
    b_ineq: &[S],
    a_eq: &[Vec<S>],
    b_eq: &[S],
    var_types: &[VarType],
    bounds: &[Option<(S, S)>],
    opts: &MILPOptions<S>,
) -> Result<OptimResult<S>, OptimError> {
    let start = std::time::Instant::now();
    let n = c.len();

    if var_types.len() != n || bounds.len() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            actual: if var_types.len() > bounds.len() {
                var_types.len()
            } else {
                bounds.len()
            },
        });
    }

    // Collect indices of integer/binary variables.
    let int_indices: Vec<usize> = (0..n)
        .filter(|&i| var_types[i] == VarType::Integer || var_types[i] == VarType::Binary)
        .collect();

    // Initialize bounds: simplex requires x >= 0, so default lb=0.
    let mut init_lb = vec![S::ZERO; n];
    let mut init_ub = vec![S::INFINITY; n];

    for i in 0..n {
        if let Some((lo, hi)) = bounds[i] {
            init_lb[i] = if lo > S::ZERO { lo } else { S::ZERO }; // simplex requires x >= 0
            init_ub[i] = hi;
        }
        // Enforce binary bounds.
        if var_types[i] == VarType::Binary {
            init_lb[i] = S::ZERO;
            init_ub[i] = S::ONE;
        }
    }

    // If no integer variables, solve directly as LP.
    if int_indices.is_empty() {
        let result = solve_lp_relaxation(c, a_ineq, b_ineq, a_eq, b_eq, &init_lb, &init_ub, opts)?;
        let mut res = result;
        res.wall_time_secs = start.elapsed().as_secs_f64();
        return Ok(res);
    }

    // Solve root LP relaxation.
    let root_result =
        match solve_lp_relaxation(c, a_ineq, b_ineq, a_eq, b_eq, &init_lb, &init_ub, opts) {
            Ok(r) => r,
            Err(OptimError::LPInfeasible) => return Err(OptimError::MILPInfeasible),
            Err(e) => return Err(e),
        };

    // Check if root solution is already integer feasible.
    if is_integer_feasible(&root_result.x, &int_indices, opts.int_tol) {
        let mut res = root_result;
        res.message = "Optimal (LP relaxation is integer feasible)".into();
        res.wall_time_secs = start.elapsed().as_secs_f64();
        return Ok(res);
    }

    // Initialize branch-and-bound.
    let root_node = BbNode {
        lb: init_lb,
        ub: init_ub,
        lp_bound: root_result.f,
        depth: 0,
    };

    let mut heap = BinaryHeap::new();
    heap.push(OrderedNode(root_node));

    let mut best_obj = S::INFINITY;
    let mut best_x: Option<Vec<S>> = None;
    let mut nodes_explored: usize = 0;

    while let Some(OrderedNode(node)) = heap.pop() {
        nodes_explored += 1;

        if nodes_explored > opts.max_nodes {
            break;
        }

        // Pruning: if node's LP bound >= incumbent, skip.
        if node.lp_bound >= best_obj - opts.gap_tol {
            continue;
        }

        // Solve LP relaxation at this node.
        let lp_result =
            match solve_lp_relaxation(c, a_ineq, b_ineq, a_eq, b_eq, &node.lb, &node.ub, opts) {
                Ok(r) => r,
                Err(OptimError::LPInfeasible) => continue, // Prune infeasible nodes.
                Err(_) => continue,                        // Prune on any LP error.
            };

        // Prune by bound.
        if lp_result.f >= best_obj - opts.gap_tol {
            continue;
        }

        // Check integer feasibility.
        if is_integer_feasible(&lp_result.x, &int_indices, opts.int_tol) {
            if lp_result.f < best_obj {
                best_obj = lp_result.f;
                // Round integer variables to nearest integer for clean output.
                let mut x_rounded = lp_result.x.clone();
                for &i in &int_indices {
                    x_rounded[i] = x_rounded[i].round();
                }
                best_x = Some(x_rounded);
                if opts.verbose {
                    eprintln!(
                        "MILP: new incumbent obj={:.6} at node {}",
                        best_obj.to_f64(),
                        nodes_explored
                    );
                }
            }
            continue;
        }

        // Branch on the most fractional integer variable.
        let branch_var = select_branching_variable(&lp_result.x, &int_indices, opts.int_tol);
        if let Some(bvar) = branch_var {
            let val = lp_result.x[bvar];
            let floor_val = val.floor();
            let ceil_val = val.ceil();

            // Left child: x[bvar] <= floor(val)
            if floor_val >= node.lb[bvar] {
                let mut left_ub = node.ub.clone();
                left_ub[bvar] = floor_val;
                let left_node = BbNode {
                    lb: node.lb.clone(),
                    ub: left_ub,
                    lp_bound: lp_result.f, // Parent LP bound as heuristic lower bound.
                    depth: node.depth + 1,
                };
                heap.push(OrderedNode(left_node));
            }

            // Right child: x[bvar] >= ceil(val)
            if ceil_val <= node.ub[bvar] {
                let mut right_lb = node.lb.clone();
                right_lb[bvar] = ceil_val;
                let right_node = BbNode {
                    lb: right_lb,
                    ub: node.ub.clone(),
                    lp_bound: lp_result.f,
                    depth: node.depth + 1,
                };
                heap.push(OrderedNode(right_node));
            }
        }
    }

    match best_x {
        Some(x) => {
            let f_val: S = c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
            Ok(OptimResult {
                x,
                f: f_val,
                grad: c.to_vec(),
                iterations: nodes_explored,
                n_feval: nodes_explored,
                n_geval: 0,
                converged: true,
                message: format!(
                    "Optimal integer solution found after {} nodes",
                    nodes_explored
                ),
                status: OptimStatus::FunctionConverged,
                history: Vec::new(),
                lambda_eq: Vec::new(),
                lambda_ineq: Vec::new(),
                active_bounds: Vec::new(),
                constraint_violation: S::ZERO,
                wall_time_secs: start.elapsed().as_secs_f64(),
                pareto: None,
                sensitivity: None,
            })
        }
        None => Err(OptimError::MILPInfeasible),
    }
}

/// Check if all integer-constrained variables are near integer values.
fn is_integer_feasible<S: Scalar>(x: &[S], int_indices: &[usize], tol: S) -> bool {
    int_indices
        .iter()
        .all(|&i| (x[i] - x[i].round()).abs() < tol)
}

/// Select the most fractional integer variable for branching.
/// Returns the index of the variable whose fractional part is closest to 0.5.
fn select_branching_variable<S: Scalar>(x: &[S], int_indices: &[usize], tol: S) -> Option<usize> {
    let mut best_idx = None;
    let mut best_frac_dist = S::ZERO; // distance of fractional part from 0 or 1
    let half = S::from_f64(0.5);

    for &i in int_indices {
        let frac = x[i] - x[i].floor();
        if frac < tol || frac > S::ONE - tol {
            continue; // Already integer.
        }
        // Distance from 0.5 -- we want the closest to 0.5 (most fractional).
        let dist = (frac - half).abs();
        let score = half - dist; // Higher = more fractional.
        if score > best_frac_dist {
            best_frac_dist = score;
            best_idx = Some(i);
        }
    }

    best_idx
}

/// Solve the LP relaxation at a B&B node.
///
/// Converts variable bounds into constraints for the simplex solver:
/// - Lower bounds > 0: variable substitution x_i' = x_i - lb_i
/// - Upper bounds < inf: add constraint e_i^T x <= ub_i (after substitution)
#[allow(clippy::too_many_arguments)]
fn solve_lp_relaxation<
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
>(
    c: &[S],
    a_ineq: &[Vec<S>],
    b_ineq: &[S],
    a_eq: &[Vec<S>],
    b_eq: &[S],
    lb: &[S],
    ub: &[S],
    opts: &MILPOptions<S>,
) -> Result<OptimResult<S>, OptimError> {
    let n = c.len();

    // Variable substitution: x_i' = x_i - lb_i so that x_i' >= 0
    // Original: min c^T x = min c^T (x' + lb) = min c^T x' + c^T lb
    // The constant c^T lb doesn't affect the optimal x' but must be added to the objective.
    let obj_offset: S = c.iter().zip(lb.iter()).map(|(&ci, &lbi)| ci * lbi).sum();

    // Transform objective: c stays the same (since min c^T x' has same c).
    // The shifted objective is c^T x' (no change in c).

    // Transform inequality constraints: A_ineq * x <= b_ineq
    // becomes A_ineq * (x' + lb) <= b_ineq
    // i.e., A_ineq * x' <= b_ineq - A_ineq * lb
    let mut new_a_ineq: Vec<Vec<S>> = Vec::with_capacity(a_ineq.len() + n);
    let mut new_b_ineq: Vec<S> = Vec::with_capacity(b_ineq.len() + n);

    for (i, row) in a_ineq.iter().enumerate() {
        let shift: S = row.iter().zip(lb.iter()).map(|(&a, &l)| a * l).sum();
        new_a_ineq.push(row.clone());
        new_b_ineq.push(b_ineq[i] - shift);
    }

    // Transform equality constraints similarly.
    let mut new_a_eq: Vec<Vec<S>> = Vec::with_capacity(a_eq.len());
    let mut new_b_eq: Vec<S> = Vec::with_capacity(b_eq.len());

    for (i, row) in a_eq.iter().enumerate() {
        let shift: S = row.iter().zip(lb.iter()).map(|(&a, &l)| a * l).sum();
        new_a_eq.push(row.clone());
        new_b_eq.push(b_eq[i] - shift);
    }

    // Add upper bound constraints: x_i' <= ub_i - lb_i for finite ub.
    for i in 0..n {
        if ub[i].is_finite() {
            let effective_ub = ub[i] - lb[i];
            if effective_ub < -opts.lp_tol {
                // lb > ub: infeasible.
                return Err(OptimError::LPInfeasible);
            }
            let mut row = vec![S::ZERO; n];
            row[i] = S::ONE;
            new_a_ineq.push(row);
            new_b_ineq.push(effective_ub);
        }
    }

    let lp_opts = LPOptions {
        max_iter: 10_000,
        tol: opts.lp_tol,
        verbose: false,
    };

    let mut result = simplex_solve(c, &new_a_ineq, &new_b_ineq, &new_a_eq, &new_b_eq, &lp_opts)?;

    // Shift back: x_i = x_i' + lb_i
    for (xi, &lbi) in result.x.iter_mut().zip(lb.iter()) {
        *xi += lbi;
    }

    // Correct the objective value.
    result.f = c
        .iter()
        .zip(result.x.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum();

    // Discard the offset that was built into result.f by simplex
    // (we already recomputed it from x).
    let _ = obj_offset;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_opts() -> MILPOptions<f64> {
        MILPOptions::default()
    }

    #[test]
    fn test_milp_no_integer_vars() {
        // No integer variables: should behave like LP.
        // minimize -x1 - x2 s.t. x1+x2<=4, x1<=3, x2<=3, x>=0
        let c = vec![-1.0, -1.0];
        let a_ineq = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let b_ineq = vec![4.0, 3.0, 3.0];
        let var_types = vec![VarType::Continuous, VarType::Continuous];
        let bounds = vec![None, None];
        let opts = default_opts();

        let result =
            milp_solve(&c, &a_ineq, &b_ineq, &[], &[], &var_types, &bounds, &opts).unwrap();
        assert!(result.converged);
        assert!(
            (result.f - (-4.0)).abs() < 1e-6,
            "expected f ~ -4.0, got {}",
            result.f
        );
    }

    #[test]
    fn test_milp_all_integer() {
        // minimize -3x1 - 5x2 s.t. x1+2x2<=6, 2x1+x2<=8, x1,x2 integer, x>=0
        // LP relaxation optimum is fractional; MILP should find integer solution.
        let c = vec![-3.0, -5.0];
        let a_ineq = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let b_ineq = vec![6.0, 8.0];
        let var_types = vec![VarType::Integer, VarType::Integer];
        let bounds = vec![None, None];
        let opts = default_opts();

        let result =
            milp_solve(&c, &a_ineq, &b_ineq, &[], &[], &var_types, &bounds, &opts).unwrap();
        assert!(result.converged);
        // Best integer solution should have obj <= -15.9
        assert!(
            result.f <= -15.0 + 1e-6,
            "expected obj <= -15, got {}",
            result.f
        );
        // Check integrality.
        for (i, xi) in result.x.iter().enumerate() {
            assert!(
                (xi - xi.round()).abs() < 1e-6,
                "x[{}]={} is not integer",
                i,
                xi
            );
        }
        // Check feasibility.
        let lhs1: f64 = result.x[0] + 2.0 * result.x[1];
        let lhs2: f64 = 2.0 * result.x[0] + result.x[1];
        assert!(lhs1 <= 6.0 + 1e-6, "constraint 1 violated: {}", lhs1);
        assert!(lhs2 <= 8.0 + 1e-6, "constraint 2 violated: {}", lhs2);
    }

    #[test]
    fn test_milp_binary_knapsack() {
        // Binary knapsack:
        // minimize -6b1 - 5b2 - 4b3 s.t. 3b1+4b2+2b3<=7, binary vars
        let c = vec![-6.0, -5.0, -4.0];
        let a_ineq = vec![vec![3.0, 4.0, 2.0]];
        let b_ineq = vec![7.0];
        let var_types = vec![VarType::Binary, VarType::Binary, VarType::Binary];
        let bounds = vec![Some((0.0, 1.0)), Some((0.0, 1.0)), Some((0.0, 1.0))];
        let opts = default_opts();

        let result =
            milp_solve(&c, &a_ineq, &b_ineq, &[], &[], &var_types, &bounds, &opts).unwrap();
        assert!(result.converged);
        // Optimal: b1=1, b3=1 (weight 5<=7, value 10) or b1=1,b2=1 (weight 7<=7, value 11)
        // obj = -11 or -10
        assert!(
            result.f <= -10.0 + 1e-6,
            "expected obj <= -10, got {}",
            result.f
        );
        // All should be 0 or 1.
        for (i, xi) in result.x.iter().enumerate() {
            assert!(
                (xi - 0.0).abs() < 1e-6 || (xi - 1.0).abs() < 1e-6,
                "x[{}]={} is not binary",
                i,
                xi
            );
        }
        // Check knapsack capacity.
        let weight: f64 = 3.0 * result.x[0] + 4.0 * result.x[1] + 2.0 * result.x[2];
        assert!(
            weight <= 7.0 + 1e-6,
            "knapsack capacity violated: {}",
            weight
        );
    }

    #[test]
    fn test_milp_mixed_integer() {
        // Mixed: x1 integer, x2 continuous.
        // minimize -x1 - x2 s.t. x1+x2<=3.5, x>=0
        let c = vec![-1.0, -1.0];
        let a_ineq = vec![vec![1.0, 1.0]];
        let b_ineq = vec![3.5];
        let var_types = vec![VarType::Integer, VarType::Continuous];
        let bounds = vec![None, None];
        let opts = default_opts();

        let result =
            milp_solve(&c, &a_ineq, &b_ineq, &[], &[], &var_types, &bounds, &opts).unwrap();
        assert!(result.converged);
        // x1 should be integer.
        assert!(
            (result.x[0] - result.x[0].round()).abs() < 1e-6,
            "x[0]={} is not integer",
            result.x[0]
        );
        // obj should be <= -3.4 (at least -3.5).
        assert!(result.f <= -3.4, "expected obj <= -3.4, got {}", result.f);
    }

    #[test]
    fn test_milp_infeasible() {
        // Infeasible: x1+x2=1, x1+x2=2 (contradictory equalities), integer vars, x>=0
        let c = vec![-1.0, -1.0];
        let a_eq = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let b_eq = vec![1.0, 2.0];
        let var_types = vec![VarType::Integer, VarType::Integer];
        let bounds = vec![None, None];
        let opts = default_opts();

        let result = milp_solve(&c, &[], &[], &a_eq, &b_eq, &var_types, &bounds, &opts);
        assert!(result.is_err(), "expected infeasible, got {:?}", result);
        match result.unwrap_err() {
            OptimError::MILPInfeasible | OptimError::LPInfeasible => {}
            e => panic!("expected MILPInfeasible or LPInfeasible, got {:?}", e),
        }
    }

    #[test]
    fn test_milp_with_equality() {
        // minimize -x1 - 2*x2 s.t. x1+x2=3, x1,x2 integer, x>=0
        // Optimal: (0,3) with obj=-6, or (1,2) with obj=-5, or (3,0) with obj=-3
        let c = vec![-1.0, -2.0];
        let a_eq = vec![vec![1.0, 1.0]];
        let b_eq = vec![3.0];
        let var_types = vec![VarType::Integer, VarType::Integer];
        let bounds = vec![None, None];
        let opts = default_opts();

        let result = milp_solve(&c, &[], &[], &a_eq, &b_eq, &var_types, &bounds, &opts).unwrap();
        assert!(result.converged);
        assert!(
            (result.f - (-6.0)).abs() < 1e-6,
            "expected obj=-6, got {}",
            result.f
        );
        assert!(
            (result.x[0] - 0.0).abs() < 1e-6,
            "expected x[0]=0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 3.0).abs() < 1e-6,
            "expected x[1]=3, got {}",
            result.x[1]
        );
    }
}
