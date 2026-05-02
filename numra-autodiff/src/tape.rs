//! Computation graph (Wengert list) for reverse-mode AD.
//!
//! The [`Tape`] records every operation performed on [`Var`](crate::reverse::Var) values during
//! the forward pass, then supports a backward pass to compute gradients.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use std::cell::RefCell;
use std::rc::Rc;

/// A single node in the computation graph.
///
/// Each node stores the primal value and its parents (up to two) along with
/// the partial derivatives of this node with respect to each parent.
#[derive(Clone, Debug)]
pub(crate) struct Node {
    /// Primal value of this node (used for checkpointing/introspection).
    #[allow(dead_code)]
    pub value: f64,
    /// First parent index + partial derivative (if any).
    pub parent1: Option<(usize, f64)>,
    /// Second parent index + partial derivative (if any).
    pub parent2: Option<(usize, f64)>,
}

/// Computation tape (Wengert list) that records operations for reverse-mode AD.
///
/// Create a tape, register input variables with [`Tape::var`], perform
/// computations using the returned [`Var`](crate::reverse::Var) handles, then call
/// [`Tape::gradient`] to obtain derivatives.
///
/// # Example
///
/// ```rust
/// use numra_autodiff::tape::Tape;
///
/// let tape = Tape::new();
/// let x = Tape::var(&tape, 3.0);
/// let y = Tape::var(&tape, 4.0);
/// let z = x.clone() * x + y.clone() * y;
/// let grad = Tape::gradient(&tape, &z);
/// assert!((grad[0] - 6.0).abs() < 1e-12); // dz/dx = 2*x = 6
/// assert!((grad[1] - 8.0).abs() < 1e-12); // dz/dy = 2*y = 8
/// ```
#[derive(Clone, Debug, Default)]
pub struct Tape {
    pub(crate) nodes: Vec<Node>,
    /// Number of input variables (first n_inputs nodes are inputs).
    pub(crate) n_inputs: usize,
}

/// Shared reference to a tape, used by [`Var`](crate::reverse::Var).
pub type TapeRef = Rc<RefCell<Tape>>;

impl Tape {
    /// Create a new empty tape.
    pub fn new() -> TapeRef {
        Rc::new(RefCell::new(Tape {
            nodes: Vec::new(),
            n_inputs: 0,
        }))
    }

    /// Register an input variable on this tape.
    pub fn var(tape: &TapeRef, value: f64) -> super::reverse::Var {
        let mut t = tape.borrow_mut();
        let index = t.nodes.len();
        t.nodes.push(Node {
            value,
            parent1: None,
            parent2: None,
        });
        t.n_inputs += 1;
        super::reverse::Var {
            index,
            value,
            tape: Rc::clone(tape),
        }
    }

    /// Push a new node onto the tape and return its index + value.
    pub(crate) fn push_unary(
        tape: &TapeRef,
        value: f64,
        parent: usize,
        deriv: f64,
    ) -> (usize, f64) {
        let mut t = tape.borrow_mut();
        let index = t.nodes.len();
        t.nodes.push(Node {
            value,
            parent1: Some((parent, deriv)),
            parent2: None,
        });
        (index, value)
    }

    /// Push a binary node onto the tape and return its index + value.
    pub(crate) fn push_binary(
        tape: &TapeRef,
        value: f64,
        p1: usize,
        d1: f64,
        p2: usize,
        d2: f64,
    ) -> (usize, f64) {
        let mut t = tape.borrow_mut();
        let index = t.nodes.len();
        t.nodes.push(Node {
            value,
            parent1: Some((p1, d1)),
            parent2: Some((p2, d2)),
        });
        (index, value)
    }

    /// Compute gradients of `output` with respect to all input variables.
    ///
    /// Returns a vector of length `n_inputs` where entry `i` is d(output)/d(input_i).
    pub fn gradient(tape: &TapeRef, output: &super::reverse::Var) -> Vec<f64> {
        let t = tape.borrow();
        let n = t.nodes.len();
        let mut adjoints = vec![0.0; n];
        adjoints[output.index] = 1.0;

        // Reverse pass
        for i in (0..n).rev() {
            let adj = adjoints[i];
            if adj == 0.0 {
                continue;
            }
            let node = &t.nodes[i];
            if let Some((p, d)) = node.parent1 {
                adjoints[p] += adj * d;
            }
            if let Some((p, d)) = node.parent2 {
                adjoints[p] += adj * d;
            }
        }

        adjoints[..t.n_inputs].to_vec()
    }

    /// Compute Jacobian: each row is the gradient of one output.
    ///
    /// Returns `outputs.len()` vectors, each of length `n_inputs`.
    pub fn jacobian(tape: &TapeRef, outputs: &[super::reverse::Var]) -> Vec<Vec<f64>> {
        outputs.iter().map(|o| Self::gradient(tape, o)).collect()
    }

    /// Reset the tape for reuse, clearing all nodes.
    pub fn clear(tape: &TapeRef) {
        let mut t = tape.borrow_mut();
        t.nodes.clear();
        t.n_inputs = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_basic() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 2.0);
        let y = Tape::var(&tape, 3.0);
        let z = x * y; // z = 6
        assert!((z.value - 6.0).abs() < 1e-14);
        let grad = Tape::gradient(&tape, &z);
        assert_eq!(grad.len(), 2);
        assert!((grad[0] - 3.0).abs() < 1e-14); // dz/dx = y = 3
        assert!((grad[1] - 2.0).abs() < 1e-14); // dz/dy = x = 2
    }

    #[test]
    fn test_tape_clear() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 1.0);
        let _ = x.clone() + x;
        assert!(tape.borrow().nodes.len() > 1);
        Tape::clear(&tape);
        assert_eq!(tape.borrow().nodes.len(), 0);
        assert_eq!(tape.borrow().n_inputs, 0);
    }

    #[test]
    fn test_tape_jacobian() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 1.0);
        let y = Tape::var(&tape, 2.0);
        let f1 = x.clone() + y.clone(); // f1 = x + y
        let f2 = x * y; // f2 = x * y
        let jac = Tape::jacobian(&tape, &[f1, f2]);
        assert_eq!(jac.len(), 2);
        // df1/dx=1, df1/dy=1
        assert!((jac[0][0] - 1.0).abs() < 1e-14);
        assert!((jac[0][1] - 1.0).abs() < 1e-14);
        // df2/dx=y=2, df2/dy=x=1
        assert!((jac[1][0] - 2.0).abs() < 1e-14);
        assert!((jac[1][1] - 1.0).abs() < 1e-14);
    }
}
