use fluent_ai_domain::similarity::{SimilarityResult, SimilarityMetric};
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
    assert_relative_eq!(with_dist.distance.expect("Expected distance value"), 0.2);
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