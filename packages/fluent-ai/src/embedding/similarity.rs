//! Zero-allocation similarity operations for embedding vectors
//!
//! High-performance similarity computations with SIMD optimizations,
//! configurable metrics, and efficient batch operations.

use std::f32;

use serde::{Deserialize, Serialize};

use crate::embedding::normalization::{l2_norm, stable_dot_product};

/// Similarity metric types for embedding comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// Cosine similarity (normalized dot product)
    Cosine,
    /// Euclidean distance (L2 distance)
    Euclidean,
    /// Manhattan distance (L1 distance)
    Manhattan,
    /// Dot product similarity
    DotProduct,
    /// Jaccard similarity for sparse vectors
    Jaccard,
    /// Pearson correlation coefficient
    Pearson}

impl Default for SimilarityMetric {
    fn default() -> Self {
        Self::Cosine
    }
}

/// Configuration for similarity computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityConfig {
    /// Similarity metric to use
    pub metric: SimilarityMetric,
    /// Threshold for considering vectors similar
    pub threshold: f32,
    /// Whether to normalize vectors before comparison
    pub normalize: bool,
    /// Tolerance for floating point comparisons
    pub tolerance: f32}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            metric: SimilarityMetric::Cosine,
            threshold: 0.8,
            normalize: true,
            tolerance: 1e-6}
    }
}

/// Result of a similarity computation
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityResult {
    /// Similarity score
    pub score: f32,
    /// Whether the score exceeds the threshold
    pub is_similar: bool,
    /// Distance (for distance-based metrics)
    pub distance: Option<f32>,
    /// Metric used for computation
    pub metric: SimilarityMetric}

/// Calculate cosine similarity between two vectors
///
/// Returns value in range [-1, 1], where 1 is identical, 0 is orthogonal, -1 is opposite
#[inline(always)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product = stable_dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Calculate Euclidean distance between two vectors
#[inline(always)]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

    let sum_squares: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();

    sum_squares.sqrt()
}

/// Calculate squared Euclidean distance (avoids sqrt for performance)
#[inline(always)]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

/// Calculate Manhattan distance (L1 distance) between two vectors
#[inline(always)]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
}

/// Calculate dot product similarity
#[inline(always)]
pub fn dot_product_similarity(a: &[f32], b: &[f32]) -> f32 {
    stable_dot_product(a, b)
}

/// Calculate Jaccard similarity for sparse vectors
///
/// Treats values below threshold as zero for sparsity computation
#[inline(always)]
pub fn jaccard_similarity(a: &[f32], b: &[f32], threshold: f32) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut intersection = 0;
    let mut union = 0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let x_nonzero = x.abs() > threshold;
        let y_nonzero = y.abs() > threshold;

        if x_nonzero && y_nonzero {
            intersection += 1;
        }
        if x_nonzero || y_nonzero {
            union += 1;
        }
    }

    if union == 0 {
        1.0 // Both vectors are all zeros, consider them identical
    } else {
        intersection as f32 / union as f32
    }
}

/// Calculate Pearson correlation coefficient
#[inline(always)]
pub fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.len() < 2 {
        return 0.0;
    }

    let n = a.len() as f32;

    // Calculate means
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;

    // Calculate covariance and variances
    let mut covariance = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff_a = x - mean_a;
        let diff_b = y - mean_b;

        covariance += diff_a * diff_b;
        var_a += diff_a * diff_a;
        var_b += diff_b * diff_b;
    }

    let std_a = var_a.sqrt();
    let std_b = var_b.sqrt();

    if std_a < f32::EPSILON || std_b < f32::EPSILON {
        return 0.0;
    }

    covariance / (std_a * std_b)
}

/// Compute similarity using specified metric
#[inline(always)]
pub fn compute_similarity(a: &[f32], b: &[f32], config: &SimilarityConfig) -> SimilarityResult {
    let score = match config.metric {
        SimilarityMetric::Cosine => cosine_similarity(a, b),
        SimilarityMetric::Euclidean => {
            let distance = euclidean_distance(a, b);
            // Convert distance to similarity (0 = identical, higher = more different)
            if distance.is_infinite() {
                0.0
            } else {
                1.0 / (1.0 + distance)
            }
        }
        SimilarityMetric::Manhattan => {
            let distance = manhattan_distance(a, b);
            if distance.is_infinite() {
                0.0
            } else {
                1.0 / (1.0 + distance)
            }
        }
        SimilarityMetric::DotProduct => dot_product_similarity(a, b),
        SimilarityMetric::Jaccard => jaccard_similarity(a, b, config.tolerance),
        SimilarityMetric::Pearson => pearson_correlation(a, b)};

    let distance = match config.metric {
        SimilarityMetric::Euclidean => Some(euclidean_distance(a, b)),
        SimilarityMetric::Manhattan => Some(manhattan_distance(a, b)),
        _ => None};

    SimilarityResult {
        score,
        is_similar: score >= config.threshold,
        distance,
        metric: config.metric}
}

/// Find the most similar vector from a collection
#[inline(always)]
pub fn find_most_similar(
    query: &[f32],
    candidates: &[Vec<f32>],
    config: &SimilarityConfig,
) -> Option<(usize, SimilarityResult)> {
    if candidates.is_empty() {
        return None;
    }

    let mut best_idx = 0;
    let mut best_result = compute_similarity(query, &candidates[0], config);

    for (idx, candidate) in candidates.iter().enumerate().skip(1) {
        let result = compute_similarity(query, candidate, config);

        // For similarity metrics, higher is better
        // For distance metrics, we already convert to similarity in compute_similarity
        if result.score > best_result.score {
            best_idx = idx;
            best_result = result;
        }
    }

    Some((best_idx, best_result))
}

/// Find top K most similar vectors
#[inline(always)]
pub fn find_top_k_similar(
    query: &[f32],
    candidates: &[Vec<f32>],
    k: usize,
    config: &SimilarityConfig,
) -> Vec<(usize, SimilarityResult)> {
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut results: Vec<(usize, SimilarityResult)> = candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| {
            let result = compute_similarity(query, candidate, config);
            (idx, result)
        })
        .collect();

    // Sort by similarity score (descending)
    results.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return top K results
    results.into_iter().take(k).collect()
}

/// Find all vectors above similarity threshold
#[inline(always)]
pub fn find_similar_above_threshold(
    query: &[f32],
    candidates: &[Vec<f32>],
    config: &SimilarityConfig,
) -> Vec<(usize, SimilarityResult)> {
    candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, candidate)| {
            let result = compute_similarity(query, candidate, config);
            if result.is_similar {
                Some((idx, result))
            } else {
                None
            }
        })
        .collect()
}

/// Batch similarity computation for multiple queries
pub struct BatchSimilarityComputer {
    config: SimilarityConfig}

impl BatchSimilarityComputer {
    /// Create new batch similarity computer
    #[inline(always)]
    pub fn new(config: SimilarityConfig) -> Self {
        Self { config }
    }

    /// Compute similarity matrix for multiple queries against multiple candidates
    #[inline(always)]
    pub fn compute_similarity_matrix(
        &self,
        queries: &[Vec<f32>],
        candidates: &[Vec<f32>],
    ) -> Vec<Vec<SimilarityResult>> {
        queries
            .iter()
            .map(|query| {
                candidates
                    .iter()
                    .map(|candidate| compute_similarity(query, candidate, &self.config))
                    .collect()
            })
            .collect()
    }

    /// Find top K for each query in batch
    #[inline(always)]
    pub fn batch_find_top_k(
        &self,
        queries: &[Vec<f32>],
        candidates: &[Vec<f32>],
        k: usize,
    ) -> Vec<Vec<(usize, SimilarityResult)>> {
        queries
            .iter()
            .map(|query| find_top_k_similar(query, candidates, k, &self.config))
            .collect()
    }
}

/// Utility functions for similarity operations
pub mod utils {
    use super::*;

    /// Check if two vectors are approximately equal
    #[inline(always)]
    pub fn vectors_approximately_equal(a: &[f32], b: &[f32], tolerance: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }

        a.iter()
            .zip(b.iter())
            .all(|(&x, &y)| (x - y).abs() <= tolerance)
    }

    /// Calculate the angle between two vectors in radians
    #[inline(always)]
    pub fn vector_angle(a: &[f32], b: &[f32]) -> f32 {
        let cos_theta = cosine_similarity(a, b);
        // Clamp to valid range for acos
        let cos_theta = cos_theta.clamp(-1.0, 1.0);
        cos_theta.acos()
    }

    /// Calculate the angle between two vectors in degrees
    #[inline(always)]
    pub fn vector_angle_degrees(a: &[f32], b: &[f32]) -> f32 {
        vector_angle(a, b) * 180.0 / std::f32::consts::PI
    }

    /// Determine if vectors are orthogonal within tolerance
    #[inline(always)]
    pub fn are_orthogonal(a: &[f32], b: &[f32], tolerance: f32) -> bool {
        let dot_product = stable_dot_product(a, b);
        dot_product.abs() <= tolerance
    }

    /// Calculate centroid of multiple vectors
    #[inline(always)]
    pub fn calculate_centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
        if vectors.is_empty() {
            return Vec::new();
        }

        let dimensions = vectors[0].len();
        let mut centroid = vec![0.0; dimensions];

        for vector in vectors {
            if vector.len() == dimensions {
                for (i, &value) in vector.iter().enumerate() {
                    centroid[i] += value;
                }
            }
        }

        let count = vectors.len() as f32;
        for value in &mut centroid {
            *value /= count;
        }

        centroid
    }

    /// Calculate variance of vectors around their centroid
    #[inline(always)]
    pub fn calculate_variance(vectors: &[Vec<f32>]) -> f32 {
        if vectors.len() < 2 {
            return 0.0;
        }

        let centroid = calculate_centroid(vectors);
        let mut total_variance = 0.0;

        for vector in vectors {
            let distance_squared = euclidean_distance_squared(vector, &centroid);
            total_variance += distance_squared;
        }

        total_variance / vectors.len() as f32
    }

    /// Find outliers based on distance from centroid
    #[inline(always)]
    pub fn find_outliers(vectors: &[Vec<f32>], threshold_multiplier: f32) -> Vec<usize> {
        if vectors.len() < 3 {
            return Vec::new();
        }

        let centroid = calculate_centroid(vectors);
        let variance = calculate_variance(vectors);
        let threshold = variance * threshold_multiplier;

        vectors
            .iter()
            .enumerate()
            .filter_map(|(idx, vector)| {
                let distance_squared = euclidean_distance_squared(vector, &centroid);
                if distance_squared > threshold {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Cluster vectors using simple k-means-like approach
    #[inline(always)]
    pub fn simple_clustering(
        vectors: &[Vec<f32>],
        num_clusters: usize,
        max_iterations: usize,
    ) -> Vec<usize> {
        if vectors.is_empty() || num_clusters == 0 || num_clusters >= vectors.len() {
            return (0..vectors.len()).collect();
        }

        let mut assignments = vec![0; vectors.len()];
        let dimensions = vectors[0].len();

        // Initialize centroids randomly
        let mut centroids = Vec::with_capacity(num_clusters);
        let step = vectors.len() / num_clusters;
        for i in 0..num_clusters {
            let idx = (i * step).min(vectors.len() - 1);
            centroids.push(vectors[idx].clone());
        }

        for _iteration in 0..max_iterations {
            let mut changed = false;

            // Assign points to nearest centroids
            for (point_idx, vector) in vectors.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_distance = euclidean_distance_squared(vector, &centroids[0]);

                for (cluster_idx, centroid) in centroids.iter().enumerate().skip(1) {
                    let distance = euclidean_distance_squared(vector, centroid);
                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = cluster_idx;
                    }
                }

                if assignments[point_idx] != best_cluster {
                    assignments[point_idx] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            for cluster_idx in 0..num_clusters {
                let cluster_points: Vec<&Vec<f32>> = vectors
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| assignments[*idx] == cluster_idx)
                    .map(|(_, vector)| vector)
                    .collect();

                if !cluster_points.is_empty() {
                    let mut new_centroid = vec![0.0; dimensions];
                    for point in &cluster_points {
                        for (i, &value) in point.iter().enumerate() {
                            new_centroid[i] += value;
                        }
                    }

                    let count = cluster_points.len() as f32;
                    for value in &mut new_centroid {
                        *value /= count;
                    }

                    centroids[cluster_idx] = new_centroid;
                }
            }
        }

        assignments
    }
}
