//! PDE method-of-lines scaling benchmarks.
//!
//! Discretises the 1D heat equation on a uniform grid of N interior
//! points and integrates with DoPri5. The bench scans N to measure
//! how wall-clock scales with spatial resolution — which doubles as a
//! stiffness study, because finer grids make the semi-discretised ODE
//! progressively stiffer.
//!
//! Author: Moussa Leblouba

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use numra_ode::{DoPri5, Solver, SolverOptions};
use numra_pde::{boundary::DirichletBC, Grid1D, HeatEquation1D, MOLSystem};
use std::time::Duration;

const MEASUREMENT_TIME: Duration = Duration::from_secs(20);
const WARM_UP_TIME: Duration = Duration::from_secs(3);

/// MOL spatial-resolution scan: N interior points ∈ {21, 51, 101, 201}.
fn bench_pde_mol_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("pde_mol_scaling");
    group
        .sample_size(20)
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARM_UP_TIME);

    let alpha = 0.01_f64;
    let t_final = 0.5_f64;
    let options = SolverOptions::default().rtol(1e-6).atol(1e-9);

    for &n_total in &[21_usize, 51, 101, 201] {
        let n_interior = n_total - 2;
        let grid = Grid1D::uniform(0.0, 1.0, n_total);
        let pde = HeatEquation1D::new(alpha);
        let bc_left = DirichletBC::new(1.0);
        let bc_right = DirichletBC::new(0.0);
        let mol = MOLSystem::new(pde, grid.clone(), bc_left, bc_right);

        // Linear initial condition on interior points (matches the
        // example in numra/examples/heat_equation.rs).
        let u0: Vec<f64> = grid.interior_points().iter().map(|&x| 1.0 - x).collect();

        group.bench_with_input(
            BenchmarkId::new("dopri5_heat_1d", n_interior),
            &n_interior,
            |b, _| {
                b.iter(|| {
                    DoPri5::solve(black_box(&mol), 0.0, t_final, black_box(&u0), &options)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_pde_mol_scaling);
criterion_main!(benches);
