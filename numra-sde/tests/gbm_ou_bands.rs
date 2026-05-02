//!
//! Author: Moussa Leblouba
//! Date: 26 April 2026
//! Modified: 2 May 2026

use numra_sde::{EulerMaruyama, SdeOptions, SdeSolver, SdeSystem};

#[allow(clippy::upper_case_acronyms)]
struct GBM {
    mu: f64,
    sigma: f64,
}

impl SdeSystem<f64> for GBM {
    fn dim(&self) -> usize {
        1
    }

    fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
        f[0] = self.mu * x[0];
    }

    fn diffusion(&self, _t: f64, x: &[f64], g: &mut [f64]) {
        g[0] = self.sigma * x[0];
    }
}

struct OrnsteinUhlenbeck {
    theta: f64,
    mean: f64,
    sigma: f64,
}

impl SdeSystem<f64> for OrnsteinUhlenbeck {
    fn dim(&self) -> usize {
        1
    }

    fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
        f[0] = self.theta * (self.mean - x[0]);
    }

    fn diffusion(&self, _t: f64, _x: &[f64], g: &mut [f64]) {
        g[0] = self.sigma;
    }
}

fn sample_mean_and_variance(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, variance)
}

#[test]
fn fixed_seed_gbm_mean_and_variance_stay_in_analytic_bands() {
    let system = GBM {
        mu: 0.05,
        sigma: 0.2,
    };
    let options = SdeOptions::default().dt(0.005).save_trajectory(false);
    let paths: Vec<f64> = (0..512)
        .map(|i| {
            EulerMaruyama::solve(&system, 0.0, 1.0, &[100.0], &options, Some(10_000 + i))
                .unwrap()
                .y_final()
                .unwrap()[0]
        })
        .collect();

    let (mean, variance) = sample_mean_and_variance(&paths);
    let expected_mean = 100.0 * (system.mu).exp();
    let expected_variance = 100.0_f64.powi(2)
        * ((system.sigma * system.sigma).exp() - 1.0)
        * (2.0 * system.mu + system.sigma * system.sigma).exp();

    assert!(
        (mean - expected_mean).abs() < 4.0,
        "mean={mean}, expected={expected_mean}"
    );
    assert!(
        (variance - expected_variance).abs() < 180.0,
        "variance={variance}, expected={expected_variance}"
    );
}

#[test]
fn fixed_seed_ou_mean_and_variance_stay_in_analytic_bands() {
    let system = OrnsteinUhlenbeck {
        theta: 1.5,
        mean: 0.75,
        sigma: 0.3,
    };
    let x0 = 2.0;
    let tf = 1.0;
    let options = SdeOptions::default().dt(0.005).save_trajectory(false);
    let paths: Vec<f64> = (0..512)
        .map(|i| {
            EulerMaruyama::solve(&system, 0.0, tf, &[x0], &options, Some(20_000 + i))
                .unwrap()
                .y_final()
                .unwrap()[0]
        })
        .collect();

    let (mean, variance) = sample_mean_and_variance(&paths);
    let decay = (-system.theta * tf).exp();
    let expected_mean = system.mean + (x0 - system.mean) * decay;
    let expected_variance =
        system.sigma.powi(2) / (2.0 * system.theta) * (1.0 - (-2.0 * system.theta * tf).exp());

    assert!(
        (mean - expected_mean).abs() < 0.08,
        "mean={mean}, expected={expected_mean}"
    );
    assert!(
        (variance - expected_variance).abs() < 0.02,
        "variance={variance}, expected={expected_variance}"
    );
}
