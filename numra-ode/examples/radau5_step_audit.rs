//! Step-count audit for the corrected Radau5 implementation.
//!
//! Prints `n_accept / n_reject / n_eval / n_jac / n_lu` for the five reference
//! problems used to validate Radau5 against SciPy's port of Hairer's radau5.f.

use numra_ode::problem::OdeProblem;
use numra_ode::radau5::Radau5;
use numra_ode::solver::{Solver, SolverOptions};

fn run<F>(name: &str, rhs: F, t0: f64, tf: f64, y0: Vec<f64>, rtol: f64, atol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]) + Copy,
{
    let problem = OdeProblem::new(rhs, t0, tf, y0.clone());
    let opts = SolverOptions::default().rtol(rtol).atol(atol);
    let res = Radau5::solve(&problem, t0, tf, &y0, &opts).expect("solve failed");
    println!(
        "{:<32} acc={:>4}  rej={:>3}  rhs={:>5}  jac={:>3}  lu={:>4}",
        name,
        res.stats.n_accept,
        res.stats.n_reject,
        res.stats.n_eval,
        res.stats.n_jac,
        res.stats.n_lu,
    );
}

fn main() {
    println!("Radau5 step-count audit (after corrections)");
    println!("--------------------------------------------");

    run(
        "stiff decay (rtol=1e-2)",
        |_, y, dy| dy[0] = -100.0 * y[0],
        0.0,
        0.1,
        vec![1.0],
        1e-2,
        1e-4,
    );

    run(
        "exponential (rtol=1e-6)",
        |_, y, dy| dy[0] = y[0],
        0.0,
        1.0,
        vec![1.0],
        1e-6,
        1e-8,
    );

    run(
        "linear 2D (rtol=1e-4)",
        |_, y, dy| {
            dy[0] = -y[0] + y[1];
            dy[1] = -y[0] - y[1];
        },
        0.0,
        1.0,
        vec![1.0, 0.0],
        1e-4,
        1e-6,
    );

    let mu_mild = 10.0;
    run(
        "van der Pol mu=10 (rtol=1e-4)",
        move |_, y, dy| {
            dy[0] = y[1];
            dy[1] = mu_mild * (1.0 - y[0] * y[0]) * y[1] - y[0];
        },
        0.0,
        2.0,
        vec![2.0, 0.0],
        1e-4,
        1e-6,
    );

    let mu_stiff = 100.0;
    run(
        "van der Pol mu=100 (rtol=1e-3)",
        move |_, y, dy| {
            dy[0] = y[1];
            dy[1] = mu_stiff * (1.0 - y[0] * y[0]) * y[1] - y[0];
        },
        0.0,
        20.0,
        vec![2.0, 0.0],
        1e-3,
        1e-5,
    );
}
