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
use numra_ode::{DoPri5, OdeSystem, Radau5, Solver, SolverOptions};
use numra_pde::{
    boundary::DirichletBC, BoundaryConditions2D, Grid1D, Grid2D, HeatEquation1D, MOLSystem,
    MOLSystem2D,
};
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
                b.iter(|| DoPri5::solve(black_box(&mol), 0.0, t_final, black_box(&u0), &options))
            },
        );
    }

    group.finish();
}

/// Wrapper that delegates rhs to MOLSystem2D but deliberately does NOT
/// override `jacobian`, so the trait-default FD path is exercised. Used
/// only by the benchmark to measure the analytical-Jacobian win against
/// the path that would have shipped without the override.
struct FdJacobianMol2D(MOLSystem2D<f64>);

impl OdeSystem<f64> for FdJacobianMol2D {
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn rhs(&self, t: f64, y: &[f64], dydt: &mut [f64]) {
        self.0.rhs(t, y, dydt)
    }
    // No jacobian override here — Radau5 will fall through to the
    // trait-default forward-FD path.
}

/// Quantifies the win from the MOLSystem2D analytical-Jacobian override
/// on a stiff 2D heat-with-reaction workload. The reference workload
/// matches the user-cited candidate: stiff diffusion + nonlinear
/// reaction on a moderately-resolved grid, integrated with Radau5.
fn bench_mol2d_radau5_jacobian_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("mol2d_radau5_jacobian");
    group
        .sample_size(20)
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARM_UP_TIME);

    // 2D heat with cubic reaction (Allen-Cahn-like). α is large enough
    // to make explicit solvers prohibitive at this grid; Radau5 is the
    // intended path. The grid is sized so FD-Jacobian's N+1 rhs sweeps
    // per rebuild are visible against the LU baseline.
    let n = 21_usize; // 19² = 361 interior points
    let alpha = 0.5_f64;
    let grid = Grid2D::uniform(0.0, 1.0, n, 0.0, 1.0, n);
    let bc = BoundaryConditions2D::all_zero_dirichlet();
    let make_mol = || {
        MOLSystem2D::heat(grid.clone(), alpha, &bc)
            .with_reaction(|_t, _x, _y, u: f64| u - u * u * u)
    };
    let mol_analytical = make_mol();
    let mol_fd = FdJacobianMol2D(make_mol());

    let nx_int = n - 2;
    let n_int = nx_int * nx_int;
    let pi = std::f64::consts::PI;
    let u0: Vec<f64> = (0..n_int)
        .map(|idx| {
            let ii = idx % nx_int;
            let jj = idx / nx_int;
            let x = grid.x_grid.points()[ii + 1];
            let y = grid.y_grid.points()[jj + 1];
            (pi * x).sin() * (pi * y).sin()
        })
        .collect();

    let t_final = 0.05_f64;
    let opts = SolverOptions::default().rtol(1e-6).atol(1e-9);

    group.bench_function("analytical_jacobian", |b| {
        b.iter(|| {
            Radau5::solve(black_box(&mol_analytical), 0.0, t_final, &u0, &opts)
        })
    });

    group.bench_function("fd_jacobian", |b| {
        b.iter(|| Radau5::solve(black_box(&mol_fd), 0.0, t_final, &u0, &opts))
    });

    group.finish();
}

criterion_group!(benches, bench_pde_mol_scaling, bench_mol2d_radau5_jacobian_path);
criterion_main!(benches);
