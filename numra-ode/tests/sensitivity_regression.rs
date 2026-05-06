//! Forward-sensitivity regression suite for `numra-ode`.
//!
//! The eight tests below exercise:
//!
//!  1. Closed-form sensitivity on `dy/dt = -k y` (1 param, 1 state).
//!  2. Closed-form sensitivity on `dy/dt = -a y + b` (2 params, 1 state).
//!  3. Lotka–Volterra (2 states, 4 params) cross-checked against central FD.
//!  4. Stiff Robertson kinetics via `Radau5` cross-checked against central FD.
//!  5. Solver-coverage matrix on the closed-form decay across DoPri5,
//!     Tsit5, Vern6, Vern7, Radau5, BDF, and Esdirk54.
//!  6. Analytical-Jacobian path vs FD-default-Jacobian path on LV.
//!  7. Trait form vs closure form on the same problem.
//!  8. Non-trivial initial sensitivity `y_0(p) = p` recovers `e^{-kt}`
//!     instead of the zero-IC `-t e^{-kt}`.
//!
//! The validation finite-difference scheme inside this file is **central FD**
//! (h² truncation), distinct from the **forward FD** that lives inside the
//! `ParametricOdeSystem` trait's default `jacobian_y` / `jacobian_p`. Forward
//! FD is what implicit solvers actually do at runtime; central FD is what we
//! compare *against* in tests because it has lower truncation error and
//! gives realistic agreement bounds.

use numra_ode::sensitivity::{
    solve_forward_sensitivity, solve_forward_sensitivity_with, ParametricOdeSystem,
    SensitivityResult,
};
use numra_ode::{
    Bdf, DoPri5, Esdirk54, OdeProblem, Radau5, Solver, SolverOptions, Tsit5, Vern6, Vern7,
};

// ---------------------------------------------------------------------------
// Test 1 — exp decay, closed-form sensitivity
// ---------------------------------------------------------------------------

#[test]
fn exp_decay_analytical() {
    // dy/dt = -k y, y(0) = 1, k = 0.5.
    // Analytical: y(t)  = exp(-k t),
    //             dy/dk = -t exp(-k t).
    struct Sys {
        k: f64,
    }
    impl ParametricOdeSystem<f64> for Sys {
        fn n_states(&self) -> usize {
            1
        }
        fn n_params(&self) -> usize {
            1
        }
        fn params(&self) -> &[f64] {
            std::slice::from_ref(&self.k)
        }
        fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
            dy[0] = -p[0] * y[0];
        }
        fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
            jy[0] = -self.k;
        }
        fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
            jp[0] = -y[0];
        }
    }

    let r = solve_forward_sensitivity::<DoPri5, _, _>(
        &Sys { k: 0.5 },
        0.0,
        5.0,
        &[1.0],
        &SolverOptions::default().rtol(1e-10).atol(1e-12),
    )
    .expect("solve failed");
    assert!(r.success);

    // Sample at every output time; max relative error should be well below 1e-8.
    let mut max_err: f64 = 0.0;
    for i in 1..r.len() {
        let t = r.t[i];
        let analytical = -t * (-0.5 * t).exp();
        let computed = r.dyi_dpj(i, 0, 0);
        max_err = max_err.max((computed - analytical).abs());
    }
    assert!(
        max_err < 1e-8,
        "exp_decay max sensitivity error = {:.3e}",
        max_err,
    );
}

// ---------------------------------------------------------------------------
// Test 2 — two-param linear system, both sensitivities analytical
// ---------------------------------------------------------------------------

#[test]
fn two_param_analytical() {
    // dy/dt = -a y + b,  y(0) = 1,  a = 1, b = 2.
    // Steady state: b/a = 2.
    // Analytical:
    //   y(t)    = b/a + (1 - b/a) exp(-a t) = 2 - exp(-t)
    //   ∂y/∂a   = -b/a²·(1 - exp(-a t)) - t·(1 - b/a)·exp(-a t)
    //           with a = 1, b = 2: = -2(1 - exp(-t)) + t·exp(-t)
    //   ∂y/∂b   = (1/a)·(1 - exp(-a t)) = 1 - exp(-t)
    struct Sys {
        params: [f64; 2],
    }
    impl ParametricOdeSystem<f64> for Sys {
        fn n_states(&self) -> usize {
            1
        }
        fn n_params(&self) -> usize {
            2
        }
        fn params(&self) -> &[f64] {
            &self.params
        }
        fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
            dy[0] = -p[0] * y[0] + p[1];
        }
        fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
            jy[0] = -self.params[0];
        }
        fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
            // Column-major (n=1, np=2): jp[k*1 + 0] = ∂f_0/∂p_k.
            jp[0] = -y[0]; // ∂f/∂a
            jp[1] = 1.0; // ∂f/∂b
        }
    }

    let r = solve_forward_sensitivity::<DoPri5, _, _>(
        &Sys { params: [1.0, 2.0] },
        0.0,
        4.0,
        &[1.0],
        &SolverOptions::default().rtol(1e-10).atol(1e-12),
    )
    .unwrap();

    let mut max_a: f64 = 0.0;
    let mut max_b: f64 = 0.0;
    for i in 1..r.len() {
        let t = r.t[i];
        let dyda_an = -2.0 * (1.0 - (-t).exp()) + t * (-t).exp();
        let dydb_an = 1.0 - (-t).exp();
        max_a = max_a.max((r.dyi_dpj(i, 0, 0) - dyda_an).abs());
        max_b = max_b.max((r.dyi_dpj(i, 0, 1) - dydb_an).abs());
    }
    assert!(max_a < 1e-7, "∂y/∂a max err = {:.3e}", max_a);
    assert!(max_b < 1e-7, "∂y/∂b max err = {:.3e}", max_b);
}

// ---------------------------------------------------------------------------
// Lotka–Volterra system shared by tests 3, 6, 7
// ---------------------------------------------------------------------------

struct LotkaVolterra {
    p: [f64; 4], // alpha, beta, delta, gamma
}

impl ParametricOdeSystem<f64> for LotkaVolterra {
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
        let x = y[0];
        let yy = y[1];
        dy[0] = p[0] * x - p[1] * x * yy;
        dy[1] = p[2] * x * yy - p[3] * yy;
    }
    fn jacobian_y(&self, _t: f64, y: &[f64], jy: &mut [f64]) {
        let (x, yy) = (y[0], y[1]);
        let (alpha, beta, delta, gamma) = (self.p[0], self.p[1], self.p[2], self.p[3]);
        // Row-major (N=2): jy[i*2 + j] = ∂f_i/∂y_j.
        jy[0] = alpha - beta * yy;
        jy[1] = -beta * x;
        jy[2] = delta * yy;
        jy[3] = delta * x - gamma;
    }
    fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
        let (x, yy) = (y[0], y[1]);
        // Column-major (N=2, N_s=4): jp[k*2 + i] = ∂f_i/∂p_k.
        // p_0 = alpha
        jp[0] = x; // ∂f_0/∂α
        jp[1] = 0.0; // ∂f_1/∂α
                     // p_1 = beta
        jp[2] = -x * yy; // ∂f_0/∂β
        jp[3] = 0.0; // ∂f_1/∂β
                     // p_2 = delta
        jp[4] = 0.0; // ∂f_0/∂δ
        jp[5] = x * yy; // ∂f_1/∂δ
                        // p_3 = gamma
        jp[6] = 0.0; // ∂f_0/∂γ
        jp[7] = -yy; // ∂f_1/∂γ
    }
}

/// Independent central-FD reference: re-solves the closure form at `p ± h`
/// and computes the central difference of the final state. Used as the
/// gold-standard cross-check throughout this file.
fn central_fd_final_sensitivity<Sol: Solver<f64>, F>(
    rhs_with_p: F,
    y0: &[f64],
    params: &[f64],
    t0: f64,
    tf: f64,
    h: f64,
    options: &SolverOptions<f64>,
) -> Vec<f64>
where
    F: Fn(f64, &[f64], &[f64], &mut [f64]) + Clone,
{
    let n = y0.len();
    let np = params.len();
    let mut s = vec![0.0; n * np];

    for k in 0..np {
        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        p_plus[k] += h;
        p_minus[k] -= h;

        let rhs_plus_factory = rhs_with_p.clone();
        let rhs_minus_factory = rhs_with_p.clone();

        let p_plus_for_closure = p_plus.clone();
        let problem_plus = OdeProblem::new(
            move |t: f64, y: &[f64], dy: &mut [f64]| {
                rhs_plus_factory(t, y, &p_plus_for_closure, dy)
            },
            t0,
            tf,
            y0.to_vec(),
        );
        let res_plus = Sol::solve(&problem_plus, t0, tf, y0, options).unwrap();
        let y_plus = res_plus.y_final().unwrap();

        let p_minus_for_closure = p_minus.clone();
        let problem_minus = OdeProblem::new(
            move |t: f64, y: &[f64], dy: &mut [f64]| {
                rhs_minus_factory(t, y, &p_minus_for_closure, dy)
            },
            t0,
            tf,
            y0.to_vec(),
        );
        let res_minus = Sol::solve(&problem_minus, t0, tf, y0, options).unwrap();
        let y_minus = res_minus.y_final().unwrap();

        for i in 0..n {
            s[k * n + i] = (y_plus[i] - y_minus[i]) / (2.0 * h);
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Test 3 — LV forward sensitivity matches central FD
// ---------------------------------------------------------------------------

#[test]
fn lotka_volterra_central_fd() {
    let lv = LotkaVolterra {
        p: [1.0, 0.1, 0.075, 1.5],
    };
    let opts = SolverOptions::default().rtol(1e-9).atol(1e-12);
    let r = solve_forward_sensitivity::<DoPri5, _, _>(&lv, 0.0, 5.0, &[10.0, 5.0], &opts).unwrap();
    assert!(r.success);
    let s_forward = r.final_sensitivity().to_vec();

    let s_fd = central_fd_final_sensitivity::<DoPri5, _>(
        |_t, y, p, dy| {
            let x = y[0];
            let yy = y[1];
            dy[0] = p[0] * x - p[1] * x * yy;
            dy[1] = p[2] * x * yy - p[3] * yy;
        },
        &[10.0, 5.0],
        &lv.p,
        0.0,
        5.0,
        1e-6,
        &opts,
    );

    let mut max_rel: f64 = 0.0;
    for i in 0..s_forward.len() {
        let denom = s_fd[i].abs().max(1e-3);
        let rel = (s_forward[i] - s_fd[i]).abs() / denom;
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel < 1e-4,
        "LV forward-vs-central-FD max relative error = {:.3e}",
        max_rel,
    );
}

// ---------------------------------------------------------------------------
// Robertson stiff kinetics shared by test 4
// ---------------------------------------------------------------------------

struct Robertson {
    k: [f64; 3], // k1, k2, k3
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
        // Row-major 3×3.
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
        // Column-major 3×3: jp[k*3 + i] = ∂f_i/∂k_k.
        // ∂/∂k1
        jp[0] = -y0;
        jp[1] = y0;
        jp[2] = 0.0;
        // ∂/∂k2
        jp[3] = 0.0;
        jp[4] = -y1 * y1;
        jp[5] = y1 * y1;
        // ∂/∂k3
        jp[6] = y1 * y2;
        jp[7] = -y1 * y2;
        jp[8] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Test 4 — Robertson stiff via Radau5 vs central FD reference
// ---------------------------------------------------------------------------

#[test]
fn robertson_stiff_radau5() {
    let robertson = Robertson {
        k: [0.04, 3.0e7, 1.0e4],
    };
    let y0 = [1.0, 0.0, 0.0];
    let opts = SolverOptions::default().rtol(1e-7).atol(1e-10);
    let r = solve_forward_sensitivity::<Radau5, _, _>(&robertson, 0.0, 40.0, &y0, &opts).unwrap();
    assert!(r.success, "Robertson stiff solve failed: {}", r.message);
    let s_forward = r.final_sensitivity().to_vec();

    // Central-FD reference using a relative step (parameter scales span 1e8).
    let mut s_fd = [0.0_f64; 9];
    let n = 3;
    for k in 0..3 {
        let h = 1e-5 * robertson.k[k];

        let mut k_plus = robertson.k;
        let mut k_minus = robertson.k;
        k_plus[k] += h;
        k_minus[k] -= h;

        let problem_plus = OdeProblem::new(
            move |_t: f64, y: &[f64], dy: &mut [f64]| {
                dy[0] = -k_plus[0] * y[0] + k_plus[2] * y[1] * y[2];
                dy[1] = k_plus[0] * y[0] - k_plus[1] * y[1] * y[1] - k_plus[2] * y[1] * y[2];
                dy[2] = k_plus[1] * y[1] * y[1];
            },
            0.0,
            40.0,
            y0.to_vec(),
        );
        let res_plus = Radau5::solve(&problem_plus, 0.0, 40.0, &y0, &opts).unwrap();
        let y_plus = res_plus.y_final().unwrap();

        let problem_minus = OdeProblem::new(
            move |_t: f64, y: &[f64], dy: &mut [f64]| {
                dy[0] = -k_minus[0] * y[0] + k_minus[2] * y[1] * y[2];
                dy[1] = k_minus[0] * y[0] - k_minus[1] * y[1] * y[1] - k_minus[2] * y[1] * y[2];
                dy[2] = k_minus[1] * y[1] * y[1];
            },
            0.0,
            40.0,
            y0.to_vec(),
        );
        let res_minus = Radau5::solve(&problem_minus, 0.0, 40.0, &y0, &opts).unwrap();
        let y_minus = res_minus.y_final().unwrap();

        for i in 0..n {
            s_fd[k * n + i] = (y_plus[i] - y_minus[i]) / (2.0 * h);
        }
    }

    let mut max_rel: f64 = 0.0;
    for i in 0..s_forward.len() {
        let denom = s_fd[i].abs().max(1e-3);
        let rel = (s_forward[i] - s_fd[i]).abs() / denom;
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel < 1e-2,
        "Robertson forward-vs-central-FD max relative error = {:.3e}",
        max_rel,
    );
}

// ---------------------------------------------------------------------------
// Test 5 — solver coverage matrix on the closed-form decay
// ---------------------------------------------------------------------------

struct Decay {
    k: f64,
}
impl ParametricOdeSystem<f64> for Decay {
    fn n_states(&self) -> usize {
        1
    }
    fn n_params(&self) -> usize {
        1
    }
    fn params(&self) -> &[f64] {
        std::slice::from_ref(&self.k)
    }
    fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
        dy[0] = -p[0] * y[0];
    }
    fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
        jy[0] = -self.k;
    }
    fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
        jp[0] = -y[0];
    }
}

fn solve_decay<Sol: Solver<f64>>(
    solver_name: &str,
    rtol: f64,
    atol: f64,
) -> SensitivityResult<f64> {
    solve_forward_sensitivity::<Sol, _, _>(
        &Decay { k: 0.5 },
        0.0,
        2.0,
        &[1.0],
        &SolverOptions::default().rtol(rtol).atol(atol),
    )
    .unwrap_or_else(|e| panic!("{solver_name} failed: {e:?}"))
}

#[test]
fn solver_coverage_matrix() {
    // Every solver runs at the same tight tolerance (rtol = 1e-10) and
    // compares its final-time sensitivity to the analytical answer.
    // Cross-solver agreement is implied by all solvers landing within the
    // same absolute bound of the truth.
    let analytical = -2.0_f64 * (-0.5 * 2.0_f64).exp();

    struct Case {
        name: &'static str,
        result: SensitivityResult<f64>,
    }

    let cases = vec![
        Case {
            name: "DoPri5",
            result: solve_decay::<DoPri5>("DoPri5", 1e-10, 1e-12),
        },
        Case {
            name: "Tsit5",
            result: solve_decay::<Tsit5>("Tsit5", 1e-10, 1e-12),
        },
        Case {
            name: "Vern6",
            result: solve_decay::<Vern6>("Vern6", 1e-10, 1e-12),
        },
        Case {
            name: "Vern7",
            result: solve_decay::<Vern7>("Vern7", 1e-10, 1e-12),
        },
        Case {
            name: "Radau5",
            result: solve_decay::<Radau5>("Radau5", 1e-10, 1e-12),
        },
        Case {
            name: "Esdirk54",
            result: solve_decay::<Esdirk54>("Esdirk54", 1e-10, 1e-12),
        },
        Case {
            name: "BDF",
            result: solve_decay::<Bdf>("BDF", 1e-10, 1e-12),
        },
    ];

    let bound = 1e-7;
    for c in &cases {
        assert!(c.result.success, "{} failed: {}", c.name, c.result.message);
        let s = c.result.dyi_dpj(c.result.len() - 1, 0, 0);
        let abs_err = (s - analytical).abs();
        assert!(
            abs_err < bound,
            "{}: S(t=2) = {s}, analytical = {analytical}, |err| = {abs_err:.3e} (bound {bound:.0e})",
            c.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6 — analytical-Jacobian path vs FD-default-Jacobian path on LV
// ---------------------------------------------------------------------------

#[test]
fn analytical_vs_fd_jacobian() {
    // Same dynamics as `LotkaVolterra` but without the analytical overrides,
    // so the trait's forward-FD defaults drive jacobian_y and jacobian_p.
    struct LvFd {
        p: [f64; 4],
    }
    impl ParametricOdeSystem<f64> for LvFd {
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
            let x = y[0];
            let yy = y[1];
            dy[0] = p[0] * x - p[1] * x * yy;
            dy[1] = p[2] * x * yy - p[3] * yy;
        }
        // No jacobian_y / jacobian_p override — FD defaults are exercised.
    }

    let p = [1.0, 0.1, 0.075, 1.5];
    let opts = SolverOptions::default().rtol(1e-10).atol(1e-12);

    let an = solve_forward_sensitivity::<DoPri5, _, _>(
        &LotkaVolterra { p },
        0.0,
        3.0,
        &[10.0, 5.0],
        &opts,
    )
    .unwrap();
    let fd = solve_forward_sensitivity::<DoPri5, _, _>(&LvFd { p }, 0.0, 3.0, &[10.0, 5.0], &opts)
        .unwrap();

    // Forward-FD Jacobians have ~sqrt(eps) accuracy ≈ 1.5e-8 per evaluation,
    // amplified through the integration. 1e-4 relative is comfortable.
    let an_final = an.final_sensitivity();
    let fd_final = fd.final_sensitivity();
    assert_eq!(an_final.len(), fd_final.len());

    let mut max_rel: f64 = 0.0;
    for i in 0..an_final.len() {
        let denom = an_final[i].abs().max(1e-3);
        let rel = (an_final[i] - fd_final[i]).abs() / denom;
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel < 1e-4,
        "analytical-vs-FD-Jacobian max relative error = {:.3e}",
        max_rel,
    );
}

// ---------------------------------------------------------------------------
// Test 7 — trait form vs closure form must agree to working precision
// ---------------------------------------------------------------------------

#[test]
fn closure_vs_trait_path() {
    let p = [1.0, 0.1, 0.075, 1.5];
    let opts = SolverOptions::default().rtol(1e-10).atol(1e-12);

    let trait_result = solve_forward_sensitivity::<DoPri5, _, _>(
        &LotkaVolterra { p },
        0.0,
        3.0,
        &[10.0, 5.0],
        &opts,
    )
    .unwrap();

    // Closure form does NOT supply analytical Jacobians; the trait form does.
    // For an apples-to-apples comparison we wrap LotkaVolterra so the closure
    // path inherits the same analytical Jacobians by construction — but
    // solve_forward_sensitivity_with constructs its own internal
    // ClosureSystem with FD defaults regardless. That's the path users hit;
    // confirming agreement at FD precision (1e-4 rel) is the contract.
    let closure_result = solve_forward_sensitivity_with::<DoPri5, f64, _>(
        |_t, y, p, dy| {
            let x = y[0];
            let yy = y[1];
            dy[0] = p[0] * x - p[1] * x * yy;
            dy[1] = p[2] * x * yy - p[3] * yy;
        },
        &[10.0, 5.0],
        &p,
        0.0,
        3.0,
        &opts,
    )
    .unwrap();

    let trait_final = trait_result.final_sensitivity();
    let closure_final = closure_result.final_sensitivity();
    assert_eq!(trait_final.len(), closure_final.len());

    let mut max_rel: f64 = 0.0;
    for i in 0..trait_final.len() {
        let denom = trait_final[i].abs().max(1e-3);
        let rel = (trait_final[i] - closure_final[i]).abs() / denom;
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel < 1e-4,
        "trait-vs-closure max relative error = {:.3e}",
        max_rel,
    );
}

// ---------------------------------------------------------------------------
// Test 8 — non-trivial initial sensitivity: y_0(p) = p
// ---------------------------------------------------------------------------

#[test]
fn nontrivial_initial_sensitivity() {
    // System: dy/dt = -k y, y(0) = p (so y_0 itself is the parameter and
    // ∂y_0/∂p = 1). With k fixed at 0.5, ∂y(t)/∂p = exp(-k t).
    struct Sys {
        p: [f64; 1], // p_0 = initial value
    }
    impl ParametricOdeSystem<f64> for Sys {
        fn n_states(&self) -> usize {
            1
        }
        fn n_params(&self) -> usize {
            1
        }
        fn params(&self) -> &[f64] {
            &self.p
        }
        fn rhs_with_params(&self, _t: f64, y: &[f64], _p: &[f64], dy: &mut [f64]) {
            dy[0] = -0.5 * y[0]; // k = 0.5, fixed
        }
        fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
            jy[0] = -0.5;
        }
        fn jacobian_p(&self, _t: f64, _y: &[f64], jp: &mut [f64]) {
            // f does not depend on p directly (p enters only through y_0).
            jp[0] = 0.0;
        }
        fn initial_sensitivity(&self, _y0: &[f64], s0: &mut [f64]) {
            // ∂y_0/∂p = 1 (column-major: s0[k*N + i]).
            s0[0] = 1.0;
        }
    }

    let p_nominal = 2.0_f64;
    let r = solve_forward_sensitivity::<DoPri5, _, _>(
        &Sys { p: [p_nominal] },
        0.0,
        5.0,
        &[p_nominal], // y(0) = p
        &SolverOptions::default().rtol(1e-10).atol(1e-12),
    )
    .unwrap();

    let mut max_err: f64 = 0.0;
    for i in 0..r.len() {
        let t = r.t[i];
        let analytical = (-0.5 * t).exp(); // not -t·exp(-k t)
        let computed = r.dyi_dpj(i, 0, 0);
        max_err = max_err.max((computed - analytical).abs());
    }
    assert!(
        max_err < 1e-8,
        "non-trivial IC: max sensitivity error = {:.3e}",
        max_err,
    );
}
