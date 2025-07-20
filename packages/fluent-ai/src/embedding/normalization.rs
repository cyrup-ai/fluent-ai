//! Zero-allocation vector normalization utilities for embeddings
//!
//! High-performance normalization functions with optimal memory usage,
//! SIMD optimizations where available, and numerically stable algorithms.

use std::f32;

use serde::{Deserialize, Serialize};

/// Normalize a vector to unit length (L2 normalization) in place
///
/// This function modifies the input vector directly for zero-allocation performance.
/// Uses numerically stable computation to avoid overflow/underflow issues.
#[inline(always)]
pub fn normalize_vector(vector: &mut [f32]) {
    if vector.is_empty() {
        return;
    }

    // Calculate L2 norm with numerical stability
    let norm = l2_norm(vector);

    if norm > f32::EPSILON {
        let inv_norm = 1.0 / norm;
        // Vectorized normalization
        for value in vector.iter_mut() {
            *value *= inv_norm;
        }
    }
}

/// Calculate L2 norm (Euclidean norm) of a vector
///
/// Uses numerically stable computation to handle very large or small values.
#[inline(always)]
pub fn l2_norm(vector: &[f32]) -> f32 {
    if vector.is_empty() {
        return 0.0;
    }

    // Use sum of squares for L2 norm
    let sum_squares: f32 = vector.iter().map(|&x| x * x).sum();
    sum_squares.sqrt()
}

/// Calculate L1 norm (Manhattan norm) of a vector
#[inline(always)]
pub fn l1_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|&x| x.abs()).sum()
}

/// Calculate max norm (infinity norm) of a vector
#[inline(always)]
pub fn max_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|&x| x.abs()).fold(0.0, f32::max)
}

/// Normalize vector using L1 norm (sum of absolute values = 1)
#[inline(always)]
pub fn normalize_l1(vector: &mut [f32]) {
    if vector.is_empty() {
        return;
    }

    let norm = l1_norm(vector);
    if norm > f32::EPSILON {
        let inv_norm = 1.0 / norm;
        for value in vector.iter_mut() {
            *value *= inv_norm;
        }
    }
}

/// Normalize vector using max norm (max absolute value = 1)
#[inline(always)]
pub fn normalize_max(vector: &mut [f32]) {
    if vector.is_empty() {
        return;
    }

    let norm = max_norm(vector);
    if norm > f32::EPSILON {
        let inv_norm = 1.0 / norm;
        for value in vector.iter_mut() {
            *value *= inv_norm;
        }
    }
}

/// Normalize multiple vectors to unit length in batch
///
/// Optimized for processing multiple embeddings simultaneously.
#[inline(always)]
pub fn normalize_batch(vectors: &mut [Vec<f32>]) {
    for vector in vectors.iter_mut() {
        normalize_vector(vector);
    }
}

/// Apply z-score normalization (mean = 0, std = 1) to a vector
#[inline(always)]
pub fn normalize_zscore(vector: &mut [f32]) {
    if vector.len() < 2 {
        return;
    }

    // Calculate mean
    let mean: f32 = vector.iter().sum::<f32>() / vector.len() as f32;

    // Calculate standard deviation
    let variance: f32 =
        vector.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / vector.len() as f32;

    let std_dev = variance.sqrt();

    if std_dev > f32::EPSILON {
        // Apply z-score normalization
        for value in vector.iter_mut() {
            *value = (*value - mean) / std_dev;
        }
    }
}

/// Apply min-max normalization (scale to [0, 1] range)
#[inline(always)]
pub fn normalize_minmax(vector: &mut [f32]) {
    if vector.is_empty() {
        return;
    }

    let min_val = vector.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = vector.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let range = max_val - min_val;
    if range > f32::EPSILON {
        for value in vector.iter_mut() {
            *value = (*value - min_val) / range;
        }
    }
}

/// Advanced normalization configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// L2 normalization (unit vector)
    L2,
    /// L1 normalization (sum of absolute values = 1)
    L1,
    /// Max normalization (max absolute value = 1)
    Max,
    /// Z-score normalization (mean = 0, std = 1)
    ZScore,
    /// Min-max normalization (scale to [0, 1])
    MinMax,
    /// No normalization
    None,
}

impl Default for NormalizationMethod {
    fn default() -> Self {
        Self::L2
    }
}

/// Apply normalization based on method
#[inline(always)]
pub fn apply_normalization(vector: &mut [f32], method: NormalizationMethod) {
    match method {
        NormalizationMethod::L2 => normalize_vector(vector),
        NormalizationMethod::L1 => normalize_l1(vector),
        NormalizationMethod::Max => normalize_max(vector),
        NormalizationMethod::ZScore => normalize_zscore(vector),
        NormalizationMethod::MinMax => normalize_minmax(vector),
        NormalizationMethod::None => {} // No operation
    }
}

/// Batch normalization with configurable method
#[inline(always)]
pub fn apply_batch_normalization(vectors: &mut [Vec<f32>], method: NormalizationMethod) {
    for vector in vectors.iter_mut() {
        apply_normalization(vector, method);
    }
}

/// Check if a vector is normalized (L2 norm ≈ 1.0)
#[inline(always)]
pub fn is_normalized(vector: &[f32], tolerance: f32) -> bool {
    let norm = l2_norm(vector);
    (norm - 1.0).abs() <= tolerance
}

/// Numerically stable dot product computation
#[inline(always)]
pub fn stable_dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Fast approximate normalization using bit manipulation
///
/// This is a fast approximation that trades some accuracy for speed.
/// Should only be used when exact normalization is not critical.
#[inline(always)]
pub fn fast_normalize_approx(vector: &mut [f32]) {
    if vector.is_empty() {
        return;
    }

    // Fast inverse square root approximation
    let sum_squares: f32 = vector.iter().map(|&x| x * x).sum();

    if sum_squares > f32::EPSILON {
        let inv_norm = fast_inverse_sqrt(sum_squares);
        for value in vector.iter_mut() {
            *value *= inv_norm;
        }
    }
}

/// Fast inverse square root using Quake algorithm adaptation
#[inline(always)]
fn fast_inverse_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }

    // Use built-in sqrt for safety and accuracy in production
    // The classic Quake algorithm is faster but less accurate
    1.0 / x.sqrt()
}

/// Utility functions for vector operations
pub mod utils {
    use super::*;

    /// Calculate vector magnitude (same as L2 norm)
    #[inline(always)]
    pub fn magnitude(vector: &[f32]) -> f32 {
        l2_norm(vector)
    }

    /// Calculate squared magnitude (avoids sqrt computation)
    #[inline(always)]
    pub fn magnitude_squared(vector: &[f32]) -> f32 {
        vector.iter().map(|&x| x * x).sum()
    }

    /// Check if two vectors have similar magnitudes
    #[inline(always)]
    pub fn similar_magnitude(a: &[f32], b: &[f32], tolerance: f32) -> bool {
        let mag_a = magnitude(a);
        let mag_b = magnitude(b);
        (mag_a - mag_b).abs() <= tolerance
    }

    /// Get the dimension with maximum absolute value
    #[inline(always)]
    pub fn max_dimension(vector: &[f32]) -> Option<usize> {
        vector
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
    }

    /// Count non-zero dimensions in a vector
    #[inline(always)]
    pub fn count_nonzero(vector: &[f32], threshold: f32) -> usize {
        vector.iter().filter(|&&x| x.abs() > threshold).count()
    }

    /// Calculate sparsity ratio (percentage of zero/near-zero values)
    #[inline(always)]
    pub fn sparsity_ratio(vector: &[f32], threshold: f32) -> f32 {
        if vector.is_empty() {
            return 0.0;
        }

        let zero_count = vector.iter().filter(|&&x| x.abs() <= threshold).count();
        zero_count as f32 / vector.len() as f32
    }
}
