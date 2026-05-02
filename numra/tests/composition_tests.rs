//! Cross-crate composition tests for Numra.
//!
//! These tests verify that all the different equation types
//! (ODE, SDE, DDE, FDE, IDE, PDE) compose and work seamlessly together.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra::dde::{DdeOptions, DdeSolver, DdeSystem, MethodOfSteps};
use numra::fde::{FdeOptions, FdeSolver, FdeSystem, L1Solver};
use numra::ide::{IdeOptions, IdeSolver, IdeSystem, VolterraSolver};
use numra::ode::{DoPri5, OdeSystem, Solver, SolverOptions};
use numra::pde::boundary::DirichletBC;
use numra::pde::{Grid1D, HeatEquation1D, MOLSystem};
use numra::sde::{EulerMaruyama, NoiseType, SdeOptions, SdeSolver, SdeSystem};
use numra::spde::{MolSdeSolver, SpdeOptions, SpdeSolver, SpdeSystem};

// ============================================================================
// Test 1: ODE + Signal composition
// ============================================================================

/// ODE with time-varying forcing (Signal composition)
struct ForcedOscillator {
    omega: f64,
    gamma: f64,
}

impl OdeSystem<f64> for ForcedOscillator {
    fn dim(&self) -> usize {
        2
    }

    fn rhs(&self, t: f64, y: &[f64], f: &mut [f64]) {
        // x'' + gamma*x' + omega²*x = sin(t)
        f[0] = y[1];
        f[1] = t.sin() - self.gamma * y[1] - self.omega * self.omega * y[0];
    }
}

#[test]
fn test_ode_with_forcing() {
    let system = ForcedOscillator {
        omega: 2.0,
        gamma: 0.1,
    };
    let options = SolverOptions::default().rtol(1e-8);
    let result = DoPri5::solve(&system, 0.0, 10.0, &[0.0, 0.0], &options).unwrap();

    assert!(result.success);
    assert!(!result.t.is_empty());

    // Should reach periodic oscillation
    let y_final = result.y_final().unwrap();
    assert!(y_final[0].abs() < 2.0, "Bounded oscillation expected");
}

// ============================================================================
// Test 2: ODE solutions used to verify other methods
// ============================================================================

/// Simple exponential decay: y' = -λy
struct ExponentialDecay {
    lambda: f64,
}

impl OdeSystem<f64> for ExponentialDecay {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -self.lambda * y[0];
    }
}

impl FdeSystem<f64> for ExponentialDecay {
    fn dim(&self) -> usize {
        1
    }
    fn alpha(&self) -> f64 {
        1.0 // α = 1 reduces FDE to ODE
    }
    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -self.lambda * y[0];
    }
}

#[test]
fn test_ode_fde_consistency() {
    // When α = 1, FDE should give same result as ODE
    let lambda = 1.0;
    let system = ExponentialDecay { lambda };
    let y0 = [1.0];

    // Solve as ODE
    let ode_opts = SolverOptions::default().rtol(1e-6);
    let ode_result = DoPri5::solve(&system, 0.0, 2.0, &y0, &ode_opts).unwrap();

    // Solve as FDE with α = 1
    let fde_opts = FdeOptions::default().dt(0.01);
    let fde_result = L1Solver::solve(&system, 0.0, 2.0, &y0, &fde_opts).unwrap();

    // Compare final values
    let y_ode = ode_result.y_final().unwrap()[0];
    let y_fde = fde_result.y_final().unwrap()[0];
    let exact = (-lambda * 2.0).exp();

    assert!(
        (y_ode - exact).abs() < 0.001,
        "ODE: y(2) = {}, exact = {}",
        y_ode,
        exact
    );
    assert!(
        (y_fde - exact).abs() < 0.01,
        "FDE: y(2) = {}, exact = {}",
        y_fde,
        exact
    );
}

// ============================================================================
// Test 3: DDE with small delay approximates ODE
// ============================================================================

/// DDE that approximates ODE when delay is small
struct SmallDelayDecay {
    lambda: f64,
    tau: f64,
}

impl DdeSystem<f64> for SmallDelayDecay {
    fn dim(&self) -> usize {
        1
    }

    fn delays(&self) -> Vec<f64> {
        vec![self.tau]
    }

    fn rhs(&self, _t: f64, _y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
        // y'(t) = -λ * y(t-τ)
        dydt[0] = -self.lambda * y_delayed[0][0];
    }
}

#[test]
fn test_dde_basic() {
    let system = SmallDelayDecay {
        lambda: 1.0,
        tau: 0.1,
    };
    let options = DdeOptions::default();

    // History function: y(t) = 1 for t <= 0
    let history = |_t: f64| vec![1.0];

    let result = MethodOfSteps::solve(&system, 0.0, 1.0, &history, &options).unwrap();
    let y_final = result.y_final().unwrap()[0];

    // With small delay, solution should still decay
    assert!(y_final < 1.0, "Should decay: y(1) = {}", y_final);
    assert!(y_final > 0.0, "Should remain positive: y(1) = {}", y_final);
}

// ============================================================================
// Test 4: IDE with zero kernel reduces to ODE
// ============================================================================

/// IDE that reduces to ODE when kernel contribution is zero
struct IdeAsOde;

impl IdeSystem<f64> for IdeAsOde {
    fn dim(&self) -> usize {
        1
    }

    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -y[0]; // Exponential decay
    }

    fn kernel(&self, _t: f64, _s: f64, _y_s: &[f64], k: &mut [f64]) {
        k[0] = 0.0; // Zero kernel
    }
}

#[test]
fn test_ide_zero_kernel() {
    let options = IdeOptions::default().dt(0.01);
    let result = VolterraSolver::solve(&IdeAsOde, 0.0, 1.0, &[1.0], &options).unwrap();

    let y_final = result.y_final().unwrap()[0];
    let y_exact = (-1.0_f64).exp();

    assert!(
        (y_final - y_exact).abs() < 0.05,
        "IDE with zero kernel: y(1) = {}, expected ≈ {}",
        y_final,
        y_exact
    );
}

// ============================================================================
// Test 5: PDE MOL produces ODE system
// ============================================================================

#[test]
fn test_pde_mol_integration() {
    // Heat equation on [0, 1] with Dirichlet BCs
    let grid = Grid1D::uniform(0.0, 1.0, 21);
    let pde = HeatEquation1D::new(0.1);
    let bc_left = DirichletBC::new(1.0);
    let bc_right = DirichletBC::new(0.0);

    let mol = MOLSystem::new(pde, grid.clone(), bc_left, bc_right);

    // MOL system should have correct dimension
    assert_eq!(mol.dim(), 19); // 21 - 2 boundary points

    // Initial condition: linear
    let u0: Vec<f64> = grid.interior_points().iter().map(|&x| 1.0 - x).collect();

    let options = SolverOptions::default().rtol(1e-6);
    let result = DoPri5::solve(&mol, 0.0, 0.5, &u0, &options).unwrap();

    assert!(result.success);

    // Solution should be bounded between boundary values
    let y_final = result.y_final().unwrap();
    for &y in &y_final {
        assert!(
            (-0.1..=1.1).contains(&y),
            "PDE solution out of range: {}",
            y
        );
    }
}

// ============================================================================
// Test 6: SDE with fixed seed is reproducible
// ============================================================================

struct SimpleGBM {
    mu: f64,
    sigma: f64,
}

impl SdeSystem<f64> for SimpleGBM {
    fn dim(&self) -> usize {
        1
    }
    fn noise_type(&self) -> NoiseType {
        NoiseType::Diagonal
    }
    fn drift(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = self.mu * y[0];
    }
    fn diffusion(&self, _t: f64, y: &[f64], g: &mut [f64]) {
        g[0] = self.sigma * y[0];
    }
}

#[test]
fn test_sde_reproducibility() {
    let system = SimpleGBM {
        mu: 0.05,
        sigma: 0.2,
    };
    let options = SdeOptions::default().dt(0.01);
    let seed = Some(12345_u64);

    // Run twice with same seed
    let result1 = EulerMaruyama::solve(&system, 0.0, 1.0, &[1.0], &options, seed).unwrap();
    let result2 = EulerMaruyama::solve(&system, 0.0, 1.0, &[1.0], &options, seed).unwrap();

    // Results should be identical
    let y1 = result1.y_final().unwrap()[0];
    let y2 = result2.y_final().unwrap()[0];

    assert!(
        (y1 - y2).abs() < 1e-10,
        "SDE not reproducible: y1 = {}, y2 = {}",
        y1,
        y2
    );
}

// ============================================================================
// Test 7: Mixed system - ODE and PDE building blocks work independently
// ============================================================================

#[test]
fn test_ode_pde_building_blocks() {
    // ODE part: simple decay representing a lumped system
    let ode_system = ExponentialDecay { lambda: 0.5 };
    let ode_opts = SolverOptions::default().rtol(1e-6);
    let ode_result = DoPri5::solve(&ode_system, 0.0, 1.0, &[1.0], &ode_opts).unwrap();

    // PDE part: heat equation
    let grid = Grid1D::uniform(0.0, 1.0, 11);
    let pde = HeatEquation1D::new(0.1);
    let bc_left = DirichletBC::new(1.0);
    let bc_right = DirichletBC::new(0.0);
    let mol = MOLSystem::new(pde, grid.clone(), bc_left, bc_right);

    let u0: Vec<f64> = grid.interior_points().iter().map(|&x| 1.0 - x).collect();
    let pde_result = DoPri5::solve(&mol, 0.0, 1.0, &u0, &ode_opts).unwrap();

    // Both should succeed
    assert!(ode_result.success, "ODE solve failed");
    assert!(pde_result.success, "PDE solve failed");

    // Both should give finite results
    assert!(ode_result.y_final().unwrap()[0].is_finite());
    for &y in pde_result.y_final().unwrap().iter() {
        assert!(y.is_finite());
    }
}

// ============================================================================
// Test 8: Verify all solver types handle edge cases consistently
// ============================================================================

#[test]
fn test_zero_time_span() {
    // ODE
    let ode_system = ExponentialDecay { lambda: 1.0 };
    let ode_opts = SolverOptions::default();
    let ode_result = DoPri5::solve(&ode_system, 0.0, 0.0, &[1.0], &ode_opts).unwrap();
    assert!((ode_result.y_final().unwrap()[0] - 1.0).abs() < 1e-10);

    // FDE
    let fde_system = ExponentialDecay { lambda: 1.0 };
    let fde_opts = FdeOptions::default().dt(0.01);
    let fde_result = L1Solver::solve(&fde_system, 0.0, 0.0, &[1.0], &fde_opts).unwrap();
    assert!((fde_result.y_final().unwrap()[0] - 1.0).abs() < 1e-10);

    // IDE
    let ide_opts = IdeOptions::default().dt(0.01);
    let ide_result = VolterraSolver::solve(&IdeAsOde, 0.0, 0.0, &[1.0], &ide_opts).unwrap();
    assert!((ide_result.y_final().unwrap()[0] - 1.0).abs() < 1e-10);
}

// ============================================================================
// Test 9: Verify scalar type genericity (f64 used throughout)
// ============================================================================

#[test]
fn test_scalar_consistency() {
    // All systems use f64 consistently
    let lambda: f64 = 1.0;
    let y0: [f64; 1] = [1.0];
    let t0: f64 = 0.0;
    let tf: f64 = 1.0;

    let ode = ExponentialDecay { lambda };
    let opts = SolverOptions::default().rtol(1e-6_f64);
    let result = DoPri5::solve(&ode, t0, tf, &y0, &opts).unwrap();

    assert!(result.success);
    let y_final: f64 = result.y_final().unwrap()[0];
    let exact: f64 = (-lambda * tf).exp();
    assert!((y_final - exact).abs() < 0.001);
}

// ============================================================================
// Test 10: Large system stress test
// ============================================================================

/// Large ODE system to verify scaling
struct LargeLinearSystem {
    n: usize,
}

impl OdeSystem<f64> for LargeLinearSystem {
    fn dim(&self) -> usize {
        self.n
    }

    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        // Simple decay for each component
        for i in 0..self.n {
            f[i] = -0.1 * (i as f64 + 1.0) * y[i];
        }
    }
}

#[test]
fn test_large_system() {
    let n = 100;
    let system = LargeLinearSystem { n };
    let y0: Vec<f64> = vec![1.0; n];
    let options = SolverOptions::default().rtol(1e-6);

    let result = DoPri5::solve(&system, 0.0, 1.0, &y0, &options).unwrap();
    assert!(result.success);

    let y_final = result.y_final().unwrap();
    assert_eq!(y_final.len(), n);

    // All components should decay
    for &y in &y_final {
        assert!(y > 0.0 && y <= 1.0, "Component out of range: {}", y);
    }
}

// ============================================================================
// Test 11: FDE with different fractional orders
// ============================================================================

struct FractionalDecay {
    lambda: f64,
    alpha: f64,
}

impl FdeSystem<f64> for FractionalDecay {
    fn dim(&self) -> usize {
        1
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -self.lambda * y[0];
    }
}

#[test]
fn test_fde_different_orders() {
    let lambda = 1.0;
    let y0 = [1.0];
    let fde_opts = FdeOptions::default().dt(0.01);

    // Test different fractional orders
    for &alpha in &[0.5, 0.7, 0.9, 1.0] {
        let system = FractionalDecay { lambda, alpha };
        let result = L1Solver::solve(&system, 0.0, 1.0, &y0, &fde_opts).unwrap();

        let y_final = result.y_final().unwrap()[0];

        // All solutions should decay
        assert!(
            y_final < 1.0 && y_final > 0.0,
            "α = {}: y(1) = {} not in (0, 1)",
            alpha,
            y_final
        );

        // Smaller α means slower decay (more memory)
        // We just verify the solution is reasonable
    }
}

// ============================================================================
// Test 12: IDE with memory kernel
// ============================================================================

struct IdeWithMemory {
    decay: f64,
    memory_strength: f64,
}

impl IdeSystem<f64> for IdeWithMemory {
    fn dim(&self) -> usize {
        1
    }

    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -self.decay * y[0];
    }

    fn kernel(&self, t: f64, s: f64, y_s: &[f64], k: &mut [f64]) {
        // Exponential memory kernel
        k[0] = self.memory_strength * (-(t - s)).exp() * y_s[0];
    }

    fn is_convolution_kernel(&self) -> bool {
        true
    }
}

#[test]
fn test_ide_with_memory() {
    let system = IdeWithMemory {
        decay: 1.0,
        memory_strength: 0.5,
    };
    let options = IdeOptions::default().dt(0.01);

    let result = VolterraSolver::solve(&system, 0.0, 2.0, &[1.0], &options).unwrap();
    let y_final = result.y_final().unwrap()[0];

    // With positive memory, decay should be slowed
    let y_pure_decay = (-2.0_f64).exp();
    assert!(
        y_final > y_pure_decay,
        "Memory should slow decay: y(2) = {}, pure decay = {}",
        y_final,
        y_pure_decay
    );
}

// ============================================================================
// Test 13: SPDE composition with PDE + SDE
// ============================================================================

struct StochasticHeatEq {
    alpha: f64,
    sigma: f64,
}

impl SpdeSystem<f64> for StochasticHeatEq {
    fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
        let dx = grid.dx_uniform();
        let dx2 = dx * dx;
        let n = u.len();

        for i in 0..n {
            let u_left = if i == 0 { 0.0 } else { u[i - 1] };
            let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };
            du[i] = self.alpha * (u_left - 2.0 * u[i] + u_right) / dx2;
        }
    }

    fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
        for s in sigma.iter_mut() {
            *s = self.sigma;
        }
    }
}

#[test]
fn test_spde_composition() {
    let system = StochasticHeatEq {
        alpha: 0.1,
        sigma: 0.05,
    };

    let grid = Grid1D::uniform(0.0, 1.0, 11);
    let u0: Vec<f64> = grid
        .interior_points()
        .iter()
        .map(|&x| (std::f64::consts::PI * x).sin())
        .collect();

    let options = SpdeOptions::default().dt(0.0001).n_output(10).seed(42);

    let result = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &options).unwrap();

    assert!(result.success);
    assert!(!result.t.is_empty());

    // Solution should be finite and bounded
    let y_final = result.y_final().unwrap();
    for &val in y_final {
        assert!(val.is_finite());
        assert!(val.abs() < 2.0, "Solution should remain bounded");
    }
}
