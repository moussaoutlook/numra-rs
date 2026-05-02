//!
//! Author: Moussa Leblouba
//! Date: 26 April 2026
//! Modified: 2 May 2026

use numra_ode::{analyze_dae_index, DaeStructure};

#[test]
fn structural_index_one_matches_algebraic_variable_directly() {
    let structure = DaeStructure {
        n_diff: 1,
        n_alg: 1,
        n_diff_eqs: 1,
        n_alg_eqs: 1,
        incidence: vec![
            vec![0, 1], // y' depends on y and z
            vec![0, 1], // g(y, z) can be solved directly for z
        ],
    };

    let info = analyze_dae_index(&structure);

    assert_eq!(info.structural_index, 1);
    assert_eq!(info.n_hidden_constraints, 0);
    assert!(info.differentiation_schedule.is_empty());
}

#[test]
fn hidden_constraint_requires_differentiation_schedule() {
    let structure = DaeStructure {
        n_diff: 2,
        n_alg: 1,
        n_diff_eqs: 2,
        n_alg_eqs: 1,
        incidence: vec![
            vec![0, 2],
            vec![1, 2],
            vec![0, 1], // algebraic constraint has no direct algebraic variable
        ],
    };

    let info = analyze_dae_index(&structure);

    assert!(info.structural_index > 1);
    assert_eq!(info.n_hidden_constraints, 1);
    assert_eq!(info.differentiation_schedule, vec![(2, 1)]);
}
