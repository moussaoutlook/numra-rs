//! Benchmarks
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use numra_ode::{Bdf, DoPri5, Esdirk54, OdeProblem, Radau5, Solver, SolverOptions, Tsit5, Vern6};
use std::time::Duration;

/// Publication-grade measurement window. Per SPEC §5.2, every shipped
/// chart's underlying bench needs a long enough sampling window that
/// the reported confidence intervals are not warm-up noise. 20s gives
/// Criterion roughly 100+ inner iterations on the slowest groups.
const MEASUREMENT_TIME: Duration = Duration::from_secs(20);
const WARM_UP_TIME: Duration = Duration::from_secs(3);

// ---------------------------------------------------------------------------
// Right-hand side functions
// ---------------------------------------------------------------------------

/// Lorenz system (sigma=10, rho=28, beta=8/3) -- classic non-stiff chaotic ODE.
fn lorenz_rhs(_t: f64, y: &[f64], dydt: &mut [f64]) {
    let sigma = 10.0;
    let rho = 28.0;
    let beta = 8.0 / 3.0;
    dydt[0] = sigma * (y[1] - y[0]);
    dydt[1] = y[0] * (rho - y[2]) - y[1];
    dydt[2] = y[0] * y[1] - beta * y[2];
}

/// Simple exponential decay -- minimal overhead benchmark.
fn exponential_decay(_t: f64, y: &[f64], dydt: &mut [f64]) {
    dydt[0] = -y[0];
}

/// N-body linear coupling -- scalable dimension benchmark.
/// dy_i/dt = -alpha * y_i + beta * (y_{i-1} + y_{i+1})
fn coupled_linear(n: usize, alpha: f64, beta: f64) -> impl Fn(f64, &[f64], &mut [f64]) {
    move |_t: f64, y: &[f64], dydt: &mut [f64]| {
        for i in 0..n {
            let left = if i > 0 { y[i - 1] } else { 0.0 };
            let right = if i < n - 1 { y[i + 1] } else { 0.0 };
            dydt[i] = -alpha * y[i] + beta * (left + right);
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// Benchmark DoPri5 on the Lorenz system integrated to t=100.
fn bench_dopri5_lorenz(c: &mut Criterion) {
    let y0 = vec![1.0, 1.0, 1.0];
    let problem = OdeProblem::new(lorenz_rhs, 0.0, 100.0, y0.clone());
    let options = SolverOptions::default().rtol(1e-6).atol(1e-9);

    let mut group = c.benchmark_group("dopri5_lorenz_t100");
    group
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARM_UP_TIME);
    group.bench_function("dopri5_lorenz_t100", |b| {
        b.iter(|| DoPri5::solve(black_box(&problem), 0.0, 100.0, black_box(&y0), &options))
    });
    group.finish();
}

/// Compare DoPri5 vs Tsit5 vs Vern6 on non-stiff exponential decay.
fn bench_explicit_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("explicit_nonstiff");
    group
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARM_UP_TIME);

    let y0 = vec![1.0];
    let problem = OdeProblem::new(exponential_decay, 0.0, 10.0, y0.clone());
    let options = SolverOptions::default().rtol(1e-6).atol(1e-9);

    group.bench_function("dopri5", |b| {
        b.iter(|| DoPri5::solve(black_box(&problem), 0.0, 10.0, &y0, &options))
    });

    group.bench_function("tsit5", |b| {
        b.iter(|| Tsit5::solve(black_box(&problem), 0.0, 10.0, &y0, &options))
    });

    group.bench_function("vern6", |b| {
        b.iter(|| Vern6::solve(black_box(&problem), 0.0, 10.0, &y0, &options))
    });

    group.finish();
}

/// Benchmark stiff solvers (Radau5, BDF, ESDIRK54) on Van der Pol with
/// increasing stiffness parameter mu.
fn bench_stiff_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("van_der_pol_stiff");
    group
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARM_UP_TIME);

    for mu in [10.0, 100.0, 1000.0] {
        let mu_val = mu;
        let y0 = vec![2.0, 0.0];
        let tf = 2.0 * mu;
        let problem = OdeProblem::new(
            move |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = mu_val * (1.0 - y[0] * y[0]) * y[1] - y[0];
            },
            0.0,
            tf,
            y0.clone(),
        );
        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);

        group.bench_with_input(BenchmarkId::new("radau5", mu), &mu, |b, _| {
            b.iter(|| Radau5::solve(black_box(&problem), 0.0, black_box(tf), &y0, &options))
        });

        group.bench_with_input(BenchmarkId::new("bdf", mu), &mu, |b, _| {
            b.iter(|| Bdf::solve(black_box(&problem), 0.0, black_box(tf), &y0, &options))
        });

        group.bench_with_input(BenchmarkId::new("esdirk54", mu), &mu, |b, _| {
            b.iter(|| Esdirk54::solve(black_box(&problem), 0.0, black_box(tf), &y0, &options))
        });
    }

    group.finish();
}

/// Benchmark dimension scaling on the coupled linear ODE.
///
/// DoPri5 (explicit) is run across n ∈ {2, 10, 50, 200} — its per-step
/// cost is O(n) so the n=200 case still completes inside Criterion's
/// budget. Radau5 (implicit) factors a dense Jacobian at every Newton
/// iterate (O(n³)); n=200 is impractical at publication-grade sampling
/// and is omitted, keeping the implicit-vs-explicit slope comparison
/// honest within the n ∈ {2, 10, 50} range.
fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension_scaling");
    group
        .sample_size(20) // Fewer samples for large systems
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARM_UP_TIME);

    for n in [2_usize, 10, 50, 200] {
        let rhs = coupled_linear(n, 2.0, 0.5);
        let y0: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let problem = OdeProblem::new(rhs, 0.0, 5.0, y0.clone());
        let options = SolverOptions::default().rtol(1e-6).atol(1e-9);

        group.bench_with_input(BenchmarkId::new("dopri5", n), &n, |b, _| {
            b.iter(|| DoPri5::solve(black_box(&problem), 0.0, 5.0, &y0, &options))
        });

        // Radau5 is omitted at n=200: with a dense Jacobian factorisation
        // the cost-per-step blows up, and a single 20s Criterion sample
        // turns into many minutes of wall-clock for a single point.
        if n < 200 {
            group.bench_with_input(BenchmarkId::new("radau5", n), &n, |b, _| {
                b.iter(|| Radau5::solve(black_box(&problem), 0.0, 5.0, &y0, &options))
            });
        }
    }

    group.finish();
}

/// Benchmark dense output overhead.
fn bench_dense_output(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_output");
    group
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARM_UP_TIME);

    let y0 = vec![1.0, 1.0, 1.0];
    let problem = OdeProblem::new(lorenz_rhs, 0.0, 10.0, y0.clone());

    let opts_no_dense = SolverOptions::default().rtol(1e-6).atol(1e-9);
    let opts_dense = SolverOptions::default().rtol(1e-6).atol(1e-9).dense();

    group.bench_function("dopri5_no_dense", |b| {
        b.iter(|| DoPri5::solve(black_box(&problem), 0.0, 10.0, &y0, &opts_no_dense))
    });

    group.bench_function("dopri5_dense", |b| {
        b.iter(|| DoPri5::solve(black_box(&problem), 0.0, 10.0, &y0, &opts_dense))
    });

    group.finish();
}

/// Benchmark tolerance scaling: same problem at different accuracies.
fn bench_tolerance_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tolerance_scaling");
    group
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARM_UP_TIME);

    let y0 = vec![1.0, 1.0, 1.0];
    let problem = OdeProblem::new(lorenz_rhs, 0.0, 20.0, y0.clone());

    for rtol_exp in [3, 4, 5, 6, 7, 8, 9] {
        let rtol = 10.0_f64.powi(-rtol_exp);
        let atol = rtol * 1e-3;
        let options = SolverOptions::default().rtol(rtol).atol(atol);

        group.bench_with_input(
            BenchmarkId::new("dopri5", format!("1e-{}", rtol_exp)),
            &rtol_exp,
            |b, _| b.iter(|| DoPri5::solve(black_box(&problem), 0.0, 20.0, &y0, &options)),
        );

        group.bench_with_input(
            BenchmarkId::new("tsit5", format!("1e-{}", rtol_exp)),
            &rtol_exp,
            |b, _| b.iter(|| Tsit5::solve(black_box(&problem), 0.0, 20.0, &y0, &options)),
        );

        group.bench_with_input(
            BenchmarkId::new("vern6", format!("1e-{}", rtol_exp)),
            &rtol_exp,
            |b, _| b.iter(|| Vern6::solve(black_box(&problem), 0.0, 20.0, &y0, &options)),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dopri5_lorenz,
    bench_explicit_comparison,
    bench_stiff_solvers,
    bench_dimension_scaling,
    bench_dense_output,
    bench_tolerance_scaling
);
criterion_main!(benches);
