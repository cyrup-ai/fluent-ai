//! Utility functions for Top-K operations
//!
//! Advanced utility functions for adaptive k calculation, coverage analysis, and configuration validation.

use crate::processing::traits::ProcessingResult;
use crate::processing::{ProcessingContext, ProcessingError};

use super::core::MAX_TOP_K;

/// Calculate adaptive top-k based on context
///
/// Adjusts k value based on generation context to balance
/// diversity and quality. Uses context length and diversity
/// metrics to determine optimal vocabulary size.
pub fn adaptive_top_k(
    base_k: usize,
    context: &ProcessingContext,
    diversity_factor: f32,
) -> ProcessingResult<usize> {
    if base_k > MAX_TOP_K {
        return Err(ProcessingError::configuration("Base k exceeds maximum"));
    }

    if !diversity_factor.is_finite() || diversity_factor < 0.0 {
        return Err(ProcessingError::configuration("Invalid diversity factor"));
    }

    let mut adjusted_k = base_k as f32;

    // Increase k with lower diversity to encourage exploration
    let diversity_score = context.diversity_score();
    if diversity_score < 0.5 {
        adjusted_k *= 1.0 + (0.5 - diversity_score) * diversity_factor;
    }

    // Adjust based on context utilization
    let utilization = context.utilization_ratio();
    if utilization > 0.8 {
        // Near context limit, reduce k for more focused generation
        adjusted_k *= 0.8;
    } else if utilization < 0.3 {
        // Early in generation, allow more diversity
        adjusted_k *= 1.2;
    }

    // Clamp to valid range
    let final_k = (adjusted_k as usize).clamp(0, MAX_TOP_K);

    Ok(final_k)
}

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

/// Calculate perplexity-based k adjustment
///
/// Adjusts k value based on current model perplexity to balance
/// exploration and exploitation. Higher perplexity suggests more
/// uncertainty, warranting larger k for exploration.
pub fn perplexity_based_k(
    base_k: usize,
    current_perplexity: f32,
    target_perplexity: f32,
) -> ProcessingResult<usize> {
    if base_k > MAX_TOP_K {
        return Err(ProcessingError::configuration("Base k exceeds maximum"));
    }

    if !current_perplexity.is_finite() || current_perplexity <= 0.0 {
        return Err(ProcessingError::configuration("Invalid current perplexity"));
    }

    if !target_perplexity.is_finite() || target_perplexity <= 0.0 {
        return Err(ProcessingError::configuration("Invalid target perplexity"));
    }

    // Calculate perplexity ratio for adjustment
    let perplexity_ratio = current_perplexity / target_perplexity;
    
    // Adjust k based on perplexity ratio
    let adjustment_factor = if perplexity_ratio > 1.5 {
        // High perplexity - increase k for exploration
        1.0 + (perplexity_ratio - 1.0) * 0.3
    } else if perplexity_ratio < 0.7 {
        // Low perplexity - decrease k for focus
        0.7 + (perplexity_ratio - 0.7) * 0.5
    } else {
        1.0 // No adjustment needed
    };

    let adjusted_k = (base_k as f32 * adjustment_factor) as usize;
    Ok(adjusted_k.clamp(1, MAX_TOP_K))
}

/// Calculate temperature-adjusted k value
///
/// Adjusts k value based on sampling temperature to maintain
/// consistent diversity levels. Higher temperature warrants
/// smaller k to avoid over-diversification.
pub fn temperature_adjusted_k(
    base_k: usize,
    temperature: f32,
    reference_temperature: f32,
) -> ProcessingResult<usize> {
    if base_k > MAX_TOP_K {
        return Err(ProcessingError::configuration("Base k exceeds maximum"));
    }

    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(ProcessingError::configuration("Invalid temperature"));
    }

    if !reference_temperature.is_finite() || reference_temperature <= 0.0 {
        return Err(ProcessingError::configuration("Invalid reference temperature"));
    }

    // Calculate temperature ratio for adjustment
    let temp_ratio = temperature / reference_temperature;
    
    // Inverse relationship: higher temperature -> lower k
    let adjustment_factor = if temp_ratio > 1.5 {
        // High temperature - reduce k to avoid over-diversification
        1.0 / (1.0 + (temp_ratio - 1.0) * 0.4)
    } else if temp_ratio < 0.5 {
        // Low temperature - increase k for diversity
        1.0 + (1.0 - temp_ratio) * 0.6
    } else {
        1.0 // No adjustment needed
    };

    let adjusted_k = (base_k as f32 * adjustment_factor) as usize;
    Ok(adjusted_k.clamp(1, MAX_TOP_K))
}

/// Validate top-k configuration for given vocabulary size
pub fn validate_top_k_config(k: usize, vocab_size: usize) -> ProcessingResult<()> {
    if k > MAX_TOP_K {
        return Err(ProcessingError::configuration(format!(
            "Top-k {} exceeds maximum {}",
            k, MAX_TOP_K
        )));
    }

    if k > vocab_size && k != 0 {
        // This is a warning condition, not an error
        // k > vocab_size is effectively no filtering
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_top_k() {
        let context = ProcessingContext::default();
        let result = adaptive_top_k(50, &context, 0.5).unwrap();
        assert!(result <= MAX_TOP_K);
        assert!(result > 0);
    }

    #[test]
    fn test_perplexity_based_k() {
        // High perplexity should increase k
        let high_perplexity_k = perplexity_based_k(50, 10.0, 5.0).unwrap();
        assert!(high_perplexity_k >= 50);
        
        // Low perplexity should decrease k
        let low_perplexity_k = perplexity_based_k(50, 2.0, 5.0).unwrap();
        assert!(low_perplexity_k <= 50);
    }

    #[test]
    fn test_temperature_adjusted_k() {
        // High temperature should decrease k
        let high_temp_k = temperature_adjusted_k(50, 2.0, 1.0).unwrap();
        assert!(high_temp_k <= 50);
        
        // Low temperature should increase k
        let low_temp_k = temperature_adjusted_k(50, 0.3, 1.0).unwrap();
        assert!(low_temp_k >= 50);
    }

    #[test]
    fn test_k_for_coverage() {
        let logits = vec![2.0, 1.0, 0.0, -1.0, -2.0];
        let k = k_for_coverage(&logits, 0.8).unwrap();
        assert!(k <= logits.len());
        assert!(k >= 1);
    }

    #[test]
    fn test_estimate_effective_vocab_size() {
        // Small k should return approximately k
        assert_eq!(estimate_effective_vocab_size(&[1.0; 100], 5).unwrap(), 5);
        
        // Large k should apply ratio
        let result = estimate_effective_vocab_size(&[1.0; 100], 60).unwrap();
        assert!(result < 60);
        assert!(result > 30);
        
        // k=0 should return full size
        assert_eq!(estimate_effective_vocab_size(&[1.0; 100], 0).unwrap(), 100);
        
        // k >= vocab_size should return full size
        assert_eq!(estimate_effective_vocab_size(&[1.0; 50], 100).unwrap(), 50);
    }

    #[test]
    fn test_validate_top_k_config() {
        // Valid configurations
        assert!(validate_top_k_config(50, 1000).is_ok());
        assert!(validate_top_k_config(0, 1000).is_ok());
        assert!(validate_top_k_config(MAX_TOP_K, 1000).is_ok());
        
        // Invalid: k exceeds maximum
        assert!(validate_top_k_config(MAX_TOP_K + 1, 1000).is_err());
        
        // Valid but unusual: k > vocab_size (should not error)
        assert!(validate_top_k_config(100, 50).is_ok());
    }

    #[test]
    fn test_invalid_inputs() {
        let context = ProcessingContext::default();
        
        // Invalid base_k
        assert!(perplexity_based_k(MAX_TOP_K + 1, 5.0, 5.0).is_err());
        assert!(temperature_adjusted_k(MAX_TOP_K + 1, 1.0, 1.0).is_err());
        assert!(adaptive_top_k(MAX_TOP_K + 1, &context, 0.5).is_err());
        
        // Invalid perplexity
        assert!(perplexity_based_k(50, -1.0, 5.0).is_err());
        assert!(perplexity_based_k(50, f32::NAN, 5.0).is_err());
        assert!(perplexity_based_k(50, 5.0, 0.0).is_err());
        
        // Invalid temperature  
        assert!(temperature_adjusted_k(50, -1.0, 1.0).is_err());
        assert!(temperature_adjusted_k(50, f32::INFINITY, 1.0).is_err());
        assert!(temperature_adjusted_k(50, 1.0, 0.0).is_err());
        
        // Invalid diversity factor
        assert!(adaptive_top_k(50, &context, -0.1).is_err());
        assert!(adaptive_top_k(50, &context, f32::NAN).is_err());
        
        // Invalid coverage target
        assert!(k_for_coverage(&[1.0; 5], -0.1).is_err());
        assert!(k_for_coverage(&[1.0; 5], 1.1).is_err());
    }
}