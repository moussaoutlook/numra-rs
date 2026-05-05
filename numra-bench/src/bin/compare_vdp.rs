//! Single-shot Numra implicit-solver timer on the Van der Pol oscillator.
//!
//! Invoked by `website/figures/comparisons/vdp_stiff.py` once per
//! (solver, rtol, mu) tuple. Emits a single JSON line on stdout — no
//! pretty-printing, no log noise — so the Python harness can `json.loads`
//! it directly.
//!
//! Usage:
//!   cargo run --release --bin compare_vdp -- --solver esdirk54 --mu 10 \
//!     --rtol 1e-6 --reps 5
//!
//! Supported solvers: radau5, bdf, esdirk54. The first run is treated as
//! a warm-up and discarded.

use numra_ode::{Bdf, Esdirk54, OdeProblem, Radau5, Solver, SolverOptions};
use std::env;
use std::time::Instant;

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == flag {
            if let Some(v) = iter.next() {
                if let Ok(parsed) = v.parse() {
                    return parsed;
                }
            }
        }
    }
    default
}

fn parse_str(args: &[String], flag: &str, default: &str) -> String {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == flag {
            if let Some(v) = iter.next() {
                return v.clone();
            }
        }
    }
    default.to_string()
}

macro_rules! run_solver {
    ($solver:ty, $label:expr, $problem:expr, $tf:expr, $y0:expr, $options:expr,
     $rtol:expr, $atol:expr, $mu:expr, $reps:expr) => {{
        let _ = <$solver>::solve(&$problem, 0.0, $tf, $y0, $options);

        let mut samples_ns: Vec<u128> = Vec::with_capacity($reps);
        let mut last_state = (0.0_f64, 0.0_f64);
        let mut last_stats = (0_usize, 0_usize, 0_usize, 0_usize, 0_usize);

        for _ in 0..$reps {
            let start = Instant::now();
            let result = <$solver>::solve(&$problem, 0.0, $tf, $y0, $options);
            let elapsed = start.elapsed().as_nanos();
            match result {
                Ok(r) => {
                    let yf = r.y_final().expect("final state");
                    last_state = (yf[0], yf[1]);
                    last_stats = (
                        r.stats.n_accept,
                        r.stats.n_reject,
                        r.stats.n_eval,
                        r.stats.n_jac,
                        r.stats.n_lu,
                    );
                    samples_ns.push(elapsed);
                }
                Err(e) => {
                    eprintln!(
                        "compare_vdp: {} failed rtol={} mu={}: {e}",
                        $label, $rtol, $mu
                    );
                    std::process::exit(2);
                }
            }
        }

        samples_ns.sort_unstable();
        let median_ns = samples_ns[samples_ns.len() / 2];
        let mean_ns: f64 =
            samples_ns.iter().map(|&n| n as f64).sum::<f64>() / samples_ns.len() as f64;
        let min_ns = samples_ns[0];
        let max_ns = *samples_ns.last().unwrap();

        println!(
            "{{\"library\":\"numra-{lab}\",\"problem\":\"vdp\",\"mu\":{mu},\
             \"rtol\":{rtol},\"atol\":{atol},\"tf\":{tf},\"reps\":{reps},\
             \"mean_ns\":{mean_ns},\"median_ns\":{median_ns},\
             \"min_ns\":{min_ns},\"max_ns\":{max_ns},\
             \"final_x\":{x},\"final_xprime\":{xp},\
             \"n_accept\":{na},\"n_reject\":{nr},\"n_eval\":{ne},\
             \"n_jac\":{nj},\"n_lu\":{nl}}}",
            lab = $label,
            mu = $mu,
            rtol = $rtol,
            atol = $atol,
            tf = $tf,
            reps = $reps,
            mean_ns = mean_ns,
            median_ns = median_ns,
            min_ns = min_ns,
            max_ns = max_ns,
            x = last_state.0,
            xp = last_state.1,
            na = last_stats.0,
            nr = last_stats.1,
            ne = last_stats.2,
            nj = last_stats.3,
            nl = last_stats.4,
        );
    }};
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let solver = parse_str(&args, "--solver", "radau5");
    let mu: f64 = parse_arg(&args, "--mu", 10.0_f64);
    let rtol: f64 = parse_arg(&args, "--rtol", 1e-6_f64);
    let atol: f64 = parse_arg(&args, "--atol", rtol * 1e-3);
    let reps: usize = parse_arg(&args, "--reps", 5_usize);

    let tf = 2.0 * mu;
    let y0 = vec![2.0_f64, 0.0_f64];

    let rhs = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    };

    let problem = OdeProblem::new(rhs, 0.0, tf, y0.clone());
    let options = SolverOptions::default()
        .rtol(rtol)
        .atol(atol)
        .max_steps(5_000_000);

    match solver.as_str() {
        "radau5" => run_solver!(Radau5, "radau5", problem, tf, &y0, &options, rtol, atol, mu, reps),
        "bdf" => run_solver!(Bdf, "bdf", problem, tf, &y0, &options, rtol, atol, mu, reps),
        "esdirk54" => {
            run_solver!(Esdirk54, "esdirk54", problem, tf, &y0, &options, rtol, atol, mu, reps)
        }
        other => {
            eprintln!("compare_vdp: unknown solver '{other}' (radau5|bdf|esdirk54)");
            std::process::exit(64);
        }
    }
}
