//! Analysis utilities for top-k operations
//!
//! Provides coverage analysis, entropy calculations, and advanced
//! distribution analysis for optimal k selection.

use super::core::MAX_TOP_K;
use crate::processing::traits::ProcessingResult;
use crate::processing::ProcessingError;

/// Find optimal k for target vocabulary coverage
///
/// Determines the k value that covers a specific fraction
/// of the probability mass in the logits distribution.
pub fn k_for_coverage(logits: &[f32], target_coverage: f32) -> ProcessingResult<usize> {
    if !(0.0..=1.0).contains(&target_coverage) {
        return Err(ProcessingError::configuration(
            "Target coverage must be between 0.0 and 1.0",
        ));
    }

    if logits.is_empty() {
        return Ok(0);
    }

    // Create sorted index-value pairs
    let mut indexed_logits: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &val)| (idx, val))
        .collect();

    // Sort by value descending
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Apply softmax for probability computation
    let max_logit = indexed_logits[0].1;
    let mut exp_sum = 0.0f32;
    let mut exp_logits: Vec<f32> = Vec::with_capacity(indexed_logits.len());

    for (_, logit) in &indexed_logits {
        let exp_val = (logit - max_logit).exp();
        exp_logits.push(exp_val);
        exp_sum += exp_val;
    }

    if exp_sum <= 0.0 {
        return Err(ProcessingError::numerical(
            "Invalid probability distribution",
        ));
    }

    // Find k that achieves target coverage
    let mut cumulative_prob = 0.0f32;
    for (k, &exp_logit) in exp_logits.iter().enumerate() {
        cumulative_prob += exp_logit / exp_sum;
        if cumulative_prob >= target_coverage {
            return Ok(k + 1); // +1 because k is 0-indexed
        }
    }

    // If we reach here, return full vocabulary size
    Ok(logits.len())
}

/// Estimate effective vocabulary size after top-k filtering
///
/// Provides an estimate of how many tokens will have non-negligible
/// probability after top-k filtering and softmax normalization.
pub fn estimate_effective_vocab_size(logits: &[f32], k: usize) -> ProcessingResult<usize> {
    if k == 0 || k >= logits.len() {
        return Ok(logits.len());
    }

    // For small k, the effective size is approximately k
    if k <= 10 {
        return Ok(k);
    }

    // For larger k, use heuristic based on logits distribution
    // This is an approximation - in practice, some top-k tokens
    // may have very low probability after softmax
    let effective_ratio = if k <= 50 {
        0.8 // 80% of top-k tokens are typically significant
    } else if k <= 100 {
        0.7 // 70% for medium k
    } else {
        0.6 // 60% for large k
    };

    Ok((k as f32 * effective_ratio) as usize)
}

/// Advanced coverage analysis with entropy considerations
///
/// Provides detailed analysis of probability distribution coverage
/// with entropy-based metrics for more sophisticated k selection.
pub fn entropy_based_coverage(
    logits: &[f32],
    target_entropy_retention: f32,
) -> ProcessingResult<usize> {
    if !(0.0..=1.0).contains(&target_entropy_retention) {
        return Err(ProcessingError::configuration(
            "Entropy retention must be between 0.0 and 1.0",
        ));
    }

    if logits.is_empty() {
        return Ok(0);
    }

    // Convert to probabilities
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .collect();

    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return Err(ProcessingError::numerical("Invalid probability distribution"));
    }

    for prob in &mut probs {
        *prob /= sum;
    }

    // Calculate full entropy
    let full_entropy = -probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f32>();

    if full_entropy <= 0.0 {
        return Ok(1); // Degenerate case
    }

    // Sort probabilities descending
    let mut sorted_probs = probs.clone();
    sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Find k that retains target entropy
    let target_entropy = full_entropy * target_entropy_retention;
    let mut partial_entropy = 0.0f32;

    for (k, &prob) in sorted_probs.iter().enumerate() {
        if prob > 0.0 {
            partial_entropy -= prob * prob.ln();
        }

        if partial_entropy >= target_entropy {
            return Ok((k + 1).min(MAX_TOP_K));
        }
    }

    Ok(logits.len().min(MAX_TOP_K))
}

/// Calculate distribution quality metrics
///
/// Analyzes the logits distribution to provide quality metrics
/// that can inform k selection strategies.
pub fn distribution_quality_metrics(logits: &[f32]) -> ProcessingResult<DistributionMetrics> {
    if logits.is_empty() {
        return Err(ProcessingError::configuration("Empty logits array"));
    }

    // Basic statistics
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let mean_logit: f32 = logits.iter().sum::<f32>() / logits.len() as f32;

    // Calculate standard deviation
    let variance: f32 = logits
        .iter()
        .map(|&x| (x - mean_logit).powi(2))
        .sum::<f32>()
        / logits.len() as f32;
    let std_dev = variance.sqrt();

    // Convert to probabilities for additional metrics
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .collect();

    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return Err(ProcessingError::numerical("Invalid probability distribution"));
    }

    for prob in &mut probs {
        *prob /= sum;
    }

    // Calculate entropy
    let entropy = -probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f32>();

    // Calculate concentration (max probability)
    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);

    // Calculate effective vocabulary size (number of tokens with >1% probability)
    let effective_vocab = probs.iter().filter(|&&p| p > 0.01).count();

    Ok(DistributionMetrics {
        max_logit,
        min_logit,
        mean_logit,
        std_dev,
        entropy,
        max_prob,
        effective_vocab,
        concentration_ratio: max_prob / (1.0 / logits.len() as f32)})
}

/// Distribution quality metrics for k selection
#[derive(Debug, Clone)]
pub struct DistributionMetrics {
    pub max_logit: f32,
    pub min_logit: f32,
    pub mean_logit: f32,
    pub std_dev: f32,
    pub entropy: f32,
    pub max_prob: f32,
    pub effective_vocab: usize,
    pub concentration_ratio: f32}

impl DistributionMetrics {
    /// Suggest optimal k based on distribution metrics
    pub fn suggest_optimal_k(&self, base_k: usize) -> usize {
        let mut suggested_k = base_k as f32;

        // High entropy suggests more distributed probability mass
        if self.entropy > 5.0 {
            suggested_k *= 1.2; // Allow more diversity
        } else if self.entropy < 2.0 {
            suggested_k *= 0.8; // Focus on top choices
        }

        // High concentration suggests dominant tokens
        if self.concentration_ratio > 10.0 {
            suggested_k *= 0.7; // Less diversity needed
        } else if self.concentration_ratio < 3.0 {
            suggested_k *= 1.3; // More diversity helpful
        }

        // Consider effective vocabulary size
        let eff_vocab_ratio = self.effective_vocab as f32 / base_k.max(1) as f32;
        if eff_vocab_ratio < 0.5 {
            suggested_k *= 0.6; // Effective vocab is small
        } else if eff_vocab_ratio > 2.0 {
            suggested_k *= 1.4; // Large effective vocab
        }

        (suggested_k as usize).clamp(1, MAX_TOP_K)
    }
}
