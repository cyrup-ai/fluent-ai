//! Similarity computation types and traits
//!
//! This module provides core types and traits for computing and working with
//! similarity metrics between vectors and other data structures.

use serde::{Deserialize, Serialize};

/// Similarity metrics that can be used for comparing vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// Cosine similarity (normalized dot product)
    Cosine,
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Dot product (unnormalized similarity)
    DotProduct,
    /// Jaccard similarity for sets
    Jaccard,
}

impl Default for SimilarityMetric {
    fn default() -> Self {
        Self::Cosine
    }
}

/// Result of a similarity computation between two vectors or embeddings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// Similarity score (higher is more similar for similarity metrics,
    /// lower is more similar for distance metrics)
    pub score: f32,
    
    /// Whether the score exceeds the configured threshold
    pub is_similar: bool,
    
    /// Distance value (for distance-based metrics)
    pub distance: Option<f32>,
    
    /// Metric used for the computation
    pub metric: SimilarityMetric,
}

impl SimilarityResult {
    /// Create a new similarity result
    #[inline]
    pub fn new(score: f32, is_similar: bool, metric: SimilarityMetric) -> Self {
        Self {
            score,
            is_similar,
            distance: None,
            metric,
        }
    }

    /// Create a new similarity result with distance
    #[inline]
    pub fn with_distance(
        score: f32,
        is_similar: bool,
        distance: f32,
        metric: SimilarityMetric,
    ) -> Self {
        Self {
            score,
            is_similar,
            distance: Some(distance),
            metric,
        }
    }

    /// Check if the similarity meets the given threshold
    #[inline]
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        match self.metric {
            // For similarity metrics, higher is better
            SimilarityMetric::Cosine | SimilarityMetric::DotProduct | SimilarityMetric::Jaccard => {
                self.score >= threshold
            }
            // For distance metrics, lower is better
            SimilarityMetric::Euclidean | SimilarityMetric::Manhattan => self.score <= threshold,
        }
    }
}

/// Trait for types that can compute similarity between themselves and other values
pub trait Similarity<T = Self> {
    /// Compute similarity between self and other
    fn similarity(&self, other: &T, metric: SimilarityMetric) -> SimilarityResult;

    /// Check if self is similar to other based on a threshold
    fn is_similar(&self, other: &T, threshold: f32, metric: SimilarityMetric) -> bool {
        self.similarity(other, metric).meets_threshold(threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_similarity_result_creation() {
        let result = SimilarityResult::new(0.8, true, SimilarityMetric::Cosine);
        assert_relative_eq!(result.score, 0.8);
        assert!(result.is_similar);
        assert_eq!(result.distance, None);
        assert_eq!(result.metric, SimilarityMetric::Cosine);

        let with_dist = SimilarityResult::with_distance(0.8, true, 0.2, SimilarityMetric::Euclidean);
        assert_relative_eq!(with_dist.score, 0.8);
        assert!(with_dist.is_similar);
        assert_relative_eq!(with_dist.distance.unwrap(), 0.2);
        assert_eq!(with_dist.metric, SimilarityMetric::Euclidean);
    }

    #[test]
    fn test_meets_threshold() {
        // Test similarity metrics (higher is better)
        let cosine_sim = SimilarityResult::new(0.7, true, SimilarityMetric::Cosine);
        assert!(cosine_sim.meets_threshold(0.5));
        assert!(!cosine_sim.meets_threshold(0.8));

        // Test distance metrics (lower is better)
        let euclidean_dist = SimilarityResult::new(0.3, true, SimilarityMetric::Euclidean);
        assert!(euclidean_dist.meets_threshold(0.5));
        assert!(!euclidean_dist.meets_threshold(0.2));
    }
}
