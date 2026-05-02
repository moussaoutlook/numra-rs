//! Peak detection in signals.
//!
//! Find local maxima with optional height and distance constraints.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Options for peak detection.
#[derive(Clone, Debug, Default)]
pub struct PeakOptions<S: Scalar> {
    /// Minimum peak height (absolute value).
    pub min_height: Option<S>,
    /// Minimum distance between peaks (in samples).
    pub min_distance: Option<usize>,
    /// Minimum prominence of peaks.
    pub min_prominence: Option<S>,
}

impl<S: Scalar> PeakOptions<S> {
    /// Set minimum height.
    pub fn height(mut self, h: S) -> Self {
        self.min_height = Some(h);
        self
    }

    /// Set minimum distance between peaks.
    pub fn distance(mut self, d: usize) -> Self {
        self.min_distance = Some(d);
        self
    }

    /// Set minimum prominence.
    pub fn prominence(mut self, p: S) -> Self {
        self.min_prominence = Some(p);
        self
    }
}

/// Find peaks (local maxima) in a signal.
///
/// A peak is a sample that is strictly greater than its immediate neighbors.
/// Optional constraints can filter peaks by minimum height and minimum
/// distance between peaks.
///
/// Returns indices of the detected peaks, sorted by position.
///
/// # Example
/// ```
/// use numra_signal::{find_peaks, PeakOptions};
///
/// let x = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
/// let peaks = find_peaks(&x, &PeakOptions::default());
/// assert_eq!(peaks, vec![1, 3, 5]);
///
/// // With minimum height
/// let peaks = find_peaks(&x, &PeakOptions::default().height(1.8));
/// assert_eq!(peaks, vec![3]);
/// ```
pub fn find_peaks<S: Scalar>(x: &[S], opts: &PeakOptions<S>) -> Vec<usize> {
    let n = x.len();
    if n < 3 {
        return vec![];
    }

    // Step 1: Find all local maxima
    let mut candidates: Vec<usize> = Vec::new();
    for i in 1..n - 1 {
        if x[i] > x[i - 1] && x[i] > x[i + 1] {
            candidates.push(i);
        }
    }

    // Step 2: Filter by minimum height
    if let Some(min_h) = opts.min_height {
        candidates.retain(|&i| x[i] >= min_h);
    }

    // Step 3: Filter by minimum prominence
    if let Some(min_prom) = opts.min_prominence {
        candidates.retain(|&i| {
            let prom = compute_prominence(x, i);
            prom >= min_prom
        });
    }

    // Step 4: Filter by minimum distance (greedy: keep tallest peaks)
    if let Some(min_dist) = opts.min_distance {
        if min_dist > 0 {
            candidates = filter_by_distance(x, &candidates, min_dist);
        }
    }

    candidates
}

/// Compute the prominence of a peak at index `idx`.
///
/// Prominence is the height of the peak relative to the highest valley
/// on either side before reaching a higher peak.
fn compute_prominence<S: Scalar>(x: &[S], idx: usize) -> S {
    let n = x.len();
    let peak_val = x[idx];

    // Search left for the highest minimum before a higher peak
    let mut left_min = peak_val;
    for i in (0..idx).rev() {
        if x[i] > peak_val {
            break;
        }
        if x[i] < left_min {
            left_min = x[i];
        }
    }

    // Search right
    let mut right_min = peak_val;
    for i in (idx + 1)..n {
        if x[i] > peak_val {
            break;
        }
        if x[i] < right_min {
            right_min = x[i];
        }
    }

    // Prominence is peak height minus the higher of the two reference levels
    let ref_level = if left_min > right_min {
        left_min
    } else {
        right_min
    };
    peak_val - ref_level
}

/// Filter peaks by minimum distance, keeping the tallest when conflicts arise.
fn filter_by_distance<S: Scalar>(x: &[S], peaks: &[usize], min_dist: usize) -> Vec<usize> {
    if peaks.is_empty() {
        return vec![];
    }

    // Sort peaks by height (descending), process tallest first
    let mut sorted: Vec<usize> = peaks.to_vec();
    sorted.sort_by(|&a, &b| {
        x[b].to_f64()
            .partial_cmp(&x[a].to_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = vec![false; x.len()];
    let mut result = Vec::new();

    for &idx in &sorted {
        // Check if any already-kept peak is within min_dist
        let lo = idx.saturating_sub(min_dist);
        let hi = (idx + min_dist).min(x.len() - 1);

        let mut conflict = false;
        for j in lo..=hi {
            if keep[j] && j != idx {
                conflict = true;
                break;
            }
        }

        if !conflict {
            keep[idx] = true;
            result.push(idx);
        }
    }

    // Sort result by position
    result.sort();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn test_find_peaks_basic() {
        let x = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
        let peaks = find_peaks(&x, &PeakOptions::default());
        assert_eq!(peaks, vec![1, 3, 5]);
    }

    #[test]
    fn test_find_peaks_with_height() {
        let x = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
        let peaks = find_peaks(&x, &PeakOptions::default().height(1.8));
        assert_eq!(peaks, vec![3]);
    }

    #[test]
    fn test_find_peaks_with_distance() {
        let x = vec![0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0];
        // All peaks without distance constraint
        let all = find_peaks(&x, &PeakOptions::default());
        assert_eq!(all, vec![1, 3, 5]);

        // With min distance 3: keep tallest (3.0 at idx=1), then 1.0 at idx=5 (distance 4)
        let peaks = find_peaks(&x, &PeakOptions::default().distance(3));
        assert_eq!(peaks, vec![1, 5]);
    }

    #[test]
    fn test_find_peaks_with_prominence() {
        let x = vec![0.0, 5.0, 4.5, 4.8, 0.0];
        // Peak at 1 (height 5.0) and peak at 3 (height 4.8)
        // Peak at 3 has low prominence (4.8 - 4.5 = 0.3)
        let peaks = find_peaks(&x, &PeakOptions::default().prominence(1.0));
        assert_eq!(peaks, vec![1]);
    }

    #[test]
    fn test_find_peaks_sine() {
        let n = 200;
        let pi2 = 2.0 * PI;
        let x: Vec<f64> = (0..n)
            .map(|i| (pi2 * 5.0 * i as f64 / n as f64).sin())
            .collect();
        let peaks = find_peaks(&x, &PeakOptions::default());
        // 5 cycles -> 5 peaks
        assert_eq!(peaks.len(), 5);
    }

    #[test]
    fn test_find_peaks_empty() {
        assert!(find_peaks::<f64>(&[], &PeakOptions::default()).is_empty());
        assert!(find_peaks(&[1.0], &PeakOptions::default()).is_empty());
        assert!(find_peaks(&[1.0, 2.0], &PeakOptions::default()).is_empty());
    }

    #[test]
    fn test_find_peaks_monotonic() {
        // No peaks in monotonic signal
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert!(find_peaks(&x, &PeakOptions::default()).is_empty());
    }

    #[test]
    fn test_find_peaks_plateau() {
        // Plateaus are not peaks (not strictly greater)
        let x = vec![0.0, 1.0, 1.0, 0.0];
        assert!(find_peaks(&x, &PeakOptions::default()).is_empty());
    }

    #[test]
    fn test_prominence_calculation() {
        // Simple case: isolated peak
        let x = vec![0.0, 0.0, 5.0, 0.0, 0.0];
        let prom = compute_prominence(&x, 2);
        assert!((prom - 5.0).abs() < 1e-10);

        // Peak on a slope
        let x = vec![0.0, 2.0, 3.0, 2.0, 4.0];
        let prom = compute_prominence(&x, 2);
        // Left min = 0.0, right min = 2.0, prominence = 3.0 - max(0.0, 2.0) = 1.0
        assert!((prom - 1.0).abs() < 1e-10);
    }
}
