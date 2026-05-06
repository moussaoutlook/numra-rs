//! Forward-sensitivity benchmarks.
//!
//! Three measurement axes, each producing one figure for the
//! performance chapter:
//!
//!   1. `sensitivity_param_scaling` — wall-clock vs `n_params` on a fixed
//!      2-state Lotka–Volterra-flavoured system. Quantifies the marginal
//!      cost of adding parameters (one extra state-block per parameter).
//!
//!   2. `sensitivity_jacobian_mode` — wall-clock as a function of how
//!      Jacobians are supplied: FD-default, analytical-`J_y`-only, or
//!      analytical-`J_y` + analytical-`J_p`. Same problem (4-param LV)
//!      driven by `Radau5` so Jacobian quality matters.
//!
//!   3. `sensitivity_solver_choice` — wall-clock for the Robertson stiff
//!      problem (3 states × 3 params, 8 orders of magnitude in rate
//!      constants) across explicit and implicit solvers.
//!
//! All three groups use the publication-grade 20s/3s windows.
//!
//! Author: Moussa Leblouba
//! Date: 6 May 2026

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use numra_ode::sensitivity::{solve_forward_sensitivity, ParametricOdeSystem};
use numra_ode::{Bdf, DoPri5, Radau5, SolverOptions, Tsit5};
use std::time::Duration;

const MEASUREMENT_TIME: Duration = Duration::from_secs(20);
const WARM_UP_TIME: Duration = Duration::from_secs(3);

// ---------------------------------------------------------------------------
// Parameter-scaling system: 2-state oscillator with N adjustable rate constants.
// ---------------------------------------------------------------------------

/// Parameterised damped harmonic oscillator with `n_p` rate constants
/// multiplying additive perturbations. State dimension is fixed at 2 so
/// only the parameter count varies across the sweep.
struct OscillatorN {
    p: Vec<f64>,
}

impl ParametricOdeSystem<f64> for OscillatorN {
    fn n_states(&self) -> usize {
        2
    }
    fn n_params(&self) -> usize {
        self.p.len()
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
        // dy0/dt = y1
        // dy1/dt = -y0 - 0.05*y1 + sum_k p_k * sin(k * y0)
        dy[0] = y[1];
        let mut driven = -y[0] - 0.05 * y[1];
        for (k, &pk) in p.iter().enumerate() {
            driven += pk * ((k + 1) as f64 * y[0]).sin();
        }
        dy[1] = driven;
    }
    // Analytical state Jacobian: row-major.
    fn jacobian_y(&self, _t: f64, y: &[f64], jy: &mut [f64]) {
        let mut d11 = -1.0;
        for (k, &pk) in self.p.iter().enumerate() {
            let kk = (k + 1) as f64;
            d11 += pk * kk * (kk * y[0]).cos();
        }
        jy[0] = 0.0;
        jy[1] = 1.0;
        jy[2] = d11;
        jy[3] = -0.05;
    }
    // Analytical parameter Jacobian: column-major (k*N + i).
    fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
        let n = 2;
        for (k, _pk) in self.p.iter().enumerate() {
            jp[k * n] = 0.0; // ∂(dy0)/∂p_k = 0
            jp[k * n + 1] = ((k + 1) as f64 * y[0]).sin();
        }
    }
    fn has_analytical_jacobian_y(&self) -> bool {
        true
    }
    fn has_analytical_jacobian_p(&self) -> bool {
        true
    }
}

fn bench_sensitivity_param_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sensitivity_param_scaling");
    group
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARM_UP_TIME);

    let opts = SolverOptions::default().rtol(1e-8).atol(1e-10);
    let y0 = [1.0_f64, 0.0];

    for &n_p in &[1usize, 2, 4, 8] {
        let p: Vec<f64> = (0..n_p).map(|k| 0.1 / (k + 1) as f64).collect();
        let sys = OscillatorN { p };
        group.bench_with_input(BenchmarkId::new("dopri5", n_p), &n_p, |b, &_n_p_inner| {
            b.iter(|| {
                solve_forward_sensitivity::<DoPri5, f64, _>(
                    black_box(&sys),
                    0.0,
                    10.0,
                    black_box(&y0),
                    &opts,
                )
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Jacobian-mode: same LV problem, three different supply modes.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum JacMode {
    Fd,
    AnalyticalY,
    AnalyticalYp,
}

struct LvJac {
    p: [f64; 4],
    mode: JacMode,
}

impl ParametricOdeSystem<f64> for LvJac {
    fn n_states(&self) -> usize {
        2
    }
    fn n_params(&self) -> usize {
        4
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
        let (x, yy) = (y[0], y[1]);
        dy[0] = p[0] * x - p[1] * x * yy;
        dy[1] = p[2] * x * yy - p[3] * yy;
    }
    fn jacobian_y(&self, _t: f64, y: &[f64], jy: &mut [f64]) {
        let (x, yy) = (y[0], y[1]);
        let (alpha, beta, delta, gamma) = (self.p[0], self.p[1], self.p[2], self.p[3]);
        jy[0] = alpha - beta * yy;
        jy[1] = -beta * x;
        jy[2] = delta * yy;
        jy[3] = delta * x - gamma;
    }
    fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
        let (x, yy) = (y[0], y[1]);
        // Column-major (N=2, N_s=4): jp[k*2 + i] = ∂f_i/∂p_k.
        jp[0] = x;
        jp[1] = 0.0; // ∂/∂α
        jp[2] = -x * yy;
        jp[3] = 0.0; // ∂/∂β
        jp[4] = 0.0;
        jp[5] = x * yy; // ∂/∂δ
        jp[6] = 0.0;
        jp[7] = -yy; // ∂/∂γ
    }
    fn has_analytical_jacobian_y(&self) -> bool {
        matches!(self.mode, JacMode::AnalyticalY | JacMode::AnalyticalYp)
    }
    fn has_analytical_jacobian_p(&self) -> bool {
        matches!(self.mode, JacMode::AnalyticalYp)
    }
}

fn bench_sensitivity_jacobian_mode(c: &mut Criterion) {
    let mut group = c.benchmark_group("sensitivity_jacobian_mode");
    group
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARM_UP_TIME);

    let opts = SolverOptions::default().rtol(1e-8).atol(1e-10);
    let y0 = [10.0_f64, 5.0];

    for (mode, label) in [
        (JacMode::Fd, "fd"),
        (JacMode::AnalyticalY, "analytical_y"),
        (JacMode::AnalyticalYp, "analytical_yp"),
    ] {
        let sys = LvJac {
            p: [1.0, 0.1, 0.075, 1.5],
            mode,
        };
        group.bench_with_input(
            BenchmarkId::new("radau5", label),
            label,
            |b, _label_inner| {
                b.iter(|| {
                    solve_forward_sensitivity::<Radau5, f64, _>(
                        black_box(&sys),
                        0.0,
                        5.0,
                        black_box(&y0),
                        &opts,
                    )
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Solver-choice: Robertson stiff (3 states × 3 params), four solvers.
// ---------------------------------------------------------------------------

struct Robertson {
    k: [f64; 3],
}

impl ParametricOdeSystem<f64> for Robertson {
    fn n_states(&self) -> usize {
        3
    }
    fn n_params(&self) -> usize {
        3
    }
    fn params(&self) -> &[f64] {
        &self.k
    }
    fn rhs_with_params(&self, _t: f64, y: &[f64], k: &[f64], dy: &mut [f64]) {
        let (y0, y1, y2) = (y[0], y[1], y[2]);
        dy[0] = -k[0] * y0 + k[2] * y1 * y2;
        dy[1] = k[0] * y0 - k[1] * y1 * y1 - k[2] * y1 * y2;
        dy[2] = k[1] * y1 * y1;
    }
    fn jacobian_y(&self, _t: f64, y: &[f64], jy: &mut [f64]) {
        let (y1, y2) = (y[1], y[2]);
        let (k1, k2, k3) = (self.k[0], self.k[1], self.k[2]);
        jy[0] = -k1;
        jy[1] = k3 * y2;
        jy[2] = k3 * y1;
        jy[3] = k1;
        jy[4] = -2.0 * k2 * y1 - k3 * y2;
        jy[5] = -k3 * y1;
        jy[6] = 0.0;
        jy[7] = 2.0 * k2 * y1;
        jy[8] = 0.0;
    }
    fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
        let (y0, y1, y2) = (y[0], y[1], y[2]);
        // Column-major (N=3, N_s=3).
        jp[0] = -y0;
        jp[1] = y0;
        jp[2] = 0.0;
        jp[3] = 0.0;
        jp[4] = -y1 * y1;
        jp[5] = y1 * y1;
        jp[6] = y1 * y2;
        jp[7] = -y1 * y2;
        jp[8] = 0.0;
    }
    fn has_analytical_jacobian_y(&self) -> bool {
        true
    }
    fn has_analytical_jacobian_p(&self) -> bool {
        true
    }
}

fn bench_sensitivity_solver_choice(c: &mut Criterion) {
    let mut group = c.benchmark_group("sensitivity_solver_choice");
    group
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARM_UP_TIME);

    let opts = SolverOptions::default().rtol(1e-6).atol(1e-9);
    let y0 = [1.0_f64, 0.0, 0.0];
    let t_final = 40.0;

    let sys = Robertson {
        k: [0.04, 3.0e7, 1.0e4],
    };

    // Note: DoPri5 / Tsit5 are explicit and will struggle on Robertson at
    // wider tolerances. We keep them in the chart precisely so the gap is
    // visible — the chart's narrative is "use Radau5/BDF for stiff sens".
    group.bench_with_input(BenchmarkId::new("radau5", "robertson"), &(), |b, _| {
        b.iter(|| {
            solve_forward_sensitivity::<Radau5, f64, _>(
                black_box(&sys),
                0.0,
                t_final,
                black_box(&y0),
                &opts,
            )
        });
    });
    group.bench_with_input(BenchmarkId::new("bdf", "robertson"), &(), |b, _| {
        b.iter(|| {
            solve_forward_sensitivity::<Bdf, f64, _>(
                black_box(&sys),
                0.0,
                t_final,
                black_box(&y0),
                &opts,
            )
        });
    });
    // Explicit solvers on Robertson at default step caps will burn many
    // steps; we run them at a shorter horizon (t=1) with the same problem
    // to keep wall-clock bounded while still showing the cost ratio.
    let t_short = 1.0_f64;
    group.bench_with_input(BenchmarkId::new("dopri5", "robertson_t1"), &(), |b, _| {
        b.iter(|| {
            solve_forward_sensitivity::<DoPri5, f64, _>(
                black_box(&sys),
                0.0,
                t_short,
                black_box(&y0),
                &opts,
            )
        });
    });
    group.bench_with_input(BenchmarkId::new("tsit5", "robertson_t1"), &(), |b, _| {
        b.iter(|| {
            solve_forward_sensitivity::<Tsit5, f64, _>(
                black_box(&sys),
                0.0,
                t_short,
                black_box(&y0),
                &opts,
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sensitivity_param_scaling,
    bench_sensitivity_jacobian_mode,
    bench_sensitivity_solver_choice,
);
criterion_main!(benches);
