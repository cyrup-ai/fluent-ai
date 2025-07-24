//! Nucleus (top-p) sampling processor with efficient probability sorting
//!
//! Implements nucleus sampling for controlling generation diversity by
//! selecting from the smallest set of tokens whose cumulative probability
//! exceeds the threshold p.

use candle_core::{D, Tensor};
use candle_nn::ops;

use super::SamplingError;
// Removed unused import: crate::processing::traits::LogitsProcessor

/// Nucleus (top-p) sampling processor
///
/// Nucleus sampling selects from the smallest set of tokens whose cumulative
/// probability mass exceeds the threshold p. This provides better quality
/// than top-k sampling by adapting to the probability distribution shape.
#[derive(Debug, Clone)]
pub struct TopPProcessor {
    top_p: f32,
    is_identity: bool, // Optimization for top_p >= 1.0
}

impl TopPProcessor {
    /// Minimum allowed top_p value (effectively deterministic)
    pub const MIN_TOP_P: f32 = 1e-6;

    /// Maximum allowed top_p value
    pub const MAX_TOP_P: f32 = 1.0;

    /// Top-p threshold above which we consider it effectively disabled
    pub const IDENTITY_THRESHOLD: f32 = 0.9999;

    /// Create a new nucleus sampling processor
    ///
    /// # Arguments
    /// * `top_p` - Cumulative probability threshold (must be in [0.0, 1.0])
    ///
    /// # Returns
    /// * `Ok(TopPProcessor)` - Successfully created processor
    /// * `Err(SamplingError::InvalidTopP)` - Invalid top_p value
    ///
    /// # Examples
    /// ```
    /// use fluent_ai_candle::sampling::TopPProcessor;
    ///
    /// let processor = TopPProcessor::new(0.9)?;
    /// ```
    #[inline(always)]
    pub fn new(top_p: f32) -> Result<Self, SamplingError> {
        // Validate top_p range
        if !top_p.is_finite() || top_p < 0.0 || top_p > 1.0 {
            return Err(SamplingError::InvalidTopP(top_p as f64));
        }

        if top_p < Self::MIN_TOP_P {
            return Err(SamplingError::InvalidTopP(top_p as f64));
        }

        let is_identity = top_p >= Self::IDENTITY_THRESHOLD;

        Ok(Self { top_p, is_identity })
    }

    /// Get the top-p value
    #[inline(always)]
    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    /// Check if this processor is effectively an identity operation
    #[inline(always)]
    pub fn is_identity(&self) -> bool {
        self.is_identity
    }

    /// Apply nucleus sampling with efficient sorting and cumulative probability calculation
    #[deprecated = "Legacy sampling module - use crate::processing::processors instead"]
    #[allow(dead_code)]
    fn apply_nucleus_sampling(&self, logits: &Tensor) -> Result<Tensor, SamplingError> {
        if self.is_identity {
            // Identity case - no filtering needed
            return Ok(logits.clone());
        }

        // Convert logits to probabilities with numerical stability
        let probs = ops::softmax(logits, D::Minus1).map_err(SamplingError::from)?;
        let prob_vec = probs.to_vec1::<f32>().map_err(SamplingError::from)?;
        let vocab_size = prob_vec.len();

        if vocab_size == 0 {
            return Err(SamplingError::EmptyLogits);
        }

        // Special case: single token
        if vocab_size == 1 {
            return Ok(logits.clone());
        }

        // Create sorted indices by probability (descending order)
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        indices.sort_by(|&a, &b| {
            prob_vec[b]
                .partial_cmp(&prob_vec[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find nucleus cutoff point using cumulative probability
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = vocab_size;

        for (i, &idx) in indices.iter().enumerate() {
            cumulative_prob += prob_vec[idx];
            if cumulative_prob >= self.top_p {
                cutoff_idx = i + 1; // Include this token
                break;
            }
        }

        // Always include at least one token to prevent empty distributions
        cutoff_idx = cutoff_idx.max(1);

        // Create mask for nucleus tokens
        let mut mask = vec![f32::NEG_INFINITY; vocab_size];
        for &idx in indices.iter().take(cutoff_idx) {
            mask[idx] = 0.0; // Keep these tokens (additive mask)
        }

        // Apply mask to logits
        let logits_vec = logits.to_vec1::<f32>().map_err(SamplingError::from)?;
        let masked_logits: Vec<f32> = logits_vec
            .iter()
            .zip(mask.iter())
            .map(|(&logit, &mask_val)| logit + mask_val)
            .collect();

        // Create masked logits tensor
        let masked_tensor = Tensor::from_vec(masked_logits, logits.shape(), logits.device())
            .map_err(SamplingError::from)?;

        // Validate output doesn't contain NaN values
        let output_vec = masked_tensor
            .to_vec1::<f32>()
            .map_err(SamplingError::from)?;
        if output_vec.iter().any(|&x| x.is_nan()) {
            return Err(SamplingError::NumericalInstability(
                "NaN values detected in nucleus sampling output".to_string(),
            ));
        }

        Ok(masked_tensor)
    }
}

// TODO: Update to new LogitsProcessor API that uses process_logits() instead of process()
// impl LogitsProcessor for TopPProcessor {
//     #[inline(always)]
//     fn process_logits(&mut self, logits: &mut [f32], context: &ProcessingContext) -> ProcessingResult<()> {
//         // Implementation needed for new API
//     }
//
//     fn validate(&self) -> ProcessingResult<()> {
//         // Implementation needed for new API
//     }
//
//     #[inline(always)]
//     fn name(&self) -> &'static str {
//         "TopPProcessor"
//     }
//
//     #[inline(always)]
//     fn is_identity(&self) -> bool {
//         self.is_identity
//     }
// }

/// Builder for nucleus sampling processor with validation and presets
#[derive(Debug, Clone)]
pub struct TopPBuilder {
    top_p: Option<f32>,
}

impl TopPBuilder {
    /// Create a new top-p builder
    #[inline(always)]
    pub fn new() -> Self {
        Self { top_p: None }
    }

    /// Set top-p value
    #[inline(always)]
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Use conservative nucleus sampling (high quality, low diversity)
    #[inline(always)]
    pub fn conservative(mut self) -> Self {
        self.top_p = Some(0.7);
        self
    }

    /// Use balanced nucleus sampling (good quality/diversity trade-off)
    #[inline(always)]
    pub fn balanced(mut self) -> Self {
        self.top_p = Some(0.9);
        self
    }

    /// Use creative nucleus sampling (high diversity)
    #[inline(always)]
    pub fn creative(mut self) -> Self {
        self.top_p = Some(0.95);
        self
    }

    /// Use exploratory nucleus sampling (maximum diversity while maintaining coherence)
    #[inline(always)]
    pub fn exploratory(mut self) -> Self {
        self.top_p = Some(0.98);
        self
    }

    /// Build the nucleus sampling processor
    pub fn build(self) -> Result<TopPProcessor, SamplingError> {
        let top_p = self.top_p.unwrap_or(0.9);
        TopPProcessor::new(top_p)
    }
}

impl Default for TopPBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced nucleus sampling utilities
pub mod utils {
    use super::*;

    /// Calculate effective nucleus size for given probability distribution
    ///
    /// Returns the number of tokens needed to reach the cumulative probability threshold.
    pub fn calculate_nucleus_size(logits: &Tensor, top_p: f32) -> Result<usize, SamplingError> {
        let probs = ops::softmax(logits, D::Minus1).map_err(SamplingError::from)?;
        let prob_vec = probs.to_vec1::<f32>().map_err(SamplingError::from)?;
        let vocab_size = prob_vec.len();

        if vocab_size == 0 {
            return Ok(0);
        }

        // Sort probabilities in descending order
        let mut probs_sorted = prob_vec.clone();
        probs_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Find nucleus size
        let mut cumulative = 0.0;
        for (i, &prob) in probs_sorted.iter().enumerate() {
            cumulative += prob;
            if cumulative >= top_p {
                return Ok(i + 1);
            }
        }

        Ok(vocab_size)
    }

    /// Adaptive top-p based on distribution entropy
    ///
    /// Adjusts top-p value based on the entropy of the probability distribution.
    /// Higher entropy (more uniform) distributions use higher top-p values.
    #[inline(always)]
    pub fn adaptive_top_p(base_top_p: f32, distribution_entropy: f32, max_entropy: f32) -> f32 {
        if max_entropy <= 0.0 {
            return base_top_p;
        }

        // Normalize entropy to [0, 1]
        let normalized_entropy = (distribution_entropy / max_entropy).clamp(0.0, 1.0);

        // Increase top-p for higher entropy distributions
        let entropy_adjustment = 0.1 * normalized_entropy;
        let adjusted = base_top_p + entropy_adjustment;

        adjusted.clamp(TopPProcessor::MIN_TOP_P, TopPProcessor::MAX_TOP_P)
    }

    /// Estimate diversity score after nucleus sampling
    ///
    /// Returns a score between 0.0 (low diversity) and 1.0 (high diversity)
    /// based on the effective vocabulary size after nucleus filtering.
    pub fn estimate_diversity_score(logits: &Tensor, top_p: f32) -> Result<f32, SamplingError> {
        let processor = TopPProcessor::new(top_p)?;
        if processor.is_identity() {
            return Ok(1.0);
        }

        let nucleus_size = calculate_nucleus_size(logits, top_p)?;
        let total_vocab = logits.shape().elem_count();

        if total_vocab == 0 {
            return Ok(0.0);
        }

        let diversity = nucleus_size as f32 / total_vocab as f32;
        Ok(diversity.clamp(0.0, 1.0))
    }

    /// Find optimal top-p for target diversity score
    ///
    /// Uses binary search to find the top-p value that achieves
    /// the desired diversity score for the given distribution.
    pub fn find_optimal_top_p(
        logits: &Tensor,
        target_diversity: f32,
        tolerance: f32,
    ) -> Result<f32, SamplingError> {
        if target_diversity < 0.0 || target_diversity > 1.0 {
            return Err(SamplingError::InvalidTopP(target_diversity as f64));
        }

        let mut low = TopPProcessor::MIN_TOP_P;
        let mut high = TopPProcessor::MAX_TOP_P;
        let max_iterations = 20;

        for _ in 0..max_iterations {
            let mid = (low + high) / 2.0;
            let diversity = estimate_diversity_score(logits, mid)?;

            if (diversity - target_diversity).abs() < tolerance {
                return Ok(mid);
            }

            if diversity < target_diversity {
                low = mid;
            } else {
                high = mid;
            }
        }

        // Return best approximation
        Ok((low + high) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;

    fn create_test_logits() -> CandleResult<Tensor> {
        let device = Device::Cpu;
        let logits = vec![1.0, 2.0, 3.0, 0.5, 1.5];
        Tensor::from_vec(logits, (1, 5), &device)
    }

    #[test]
    fn test_top_p_validation() {
        // Valid top_p values
        assert!(TopPProcessor::new(0.1).is_ok());
        assert!(TopPProcessor::new(0.9).is_ok());
        assert!(TopPProcessor::new(1.0).is_ok());

        // Invalid top_p values
        assert!(TopPProcessor::new(-0.1).is_err());
        assert!(TopPProcessor::new(1.1).is_err());
        assert!(TopPProcessor::new(f32::NAN).is_err());
        assert!(TopPProcessor::new(f32::INFINITY).is_err());
    }

    #[test]
    fn test_identity_optimization() -> Result<(), Box<dyn std::error::Error>> {
        let processor = TopPProcessor::new(1.0)?;
        assert!(processor.is_identity());

        let processor = TopPProcessor::new(0.9)?;
        assert!(!processor.is_identity());

        Ok(())
    }

    #[test]
    fn test_top_p_builder() -> Result<(), Box<dyn std::error::Error>> {
        let processor = TopPBuilder::new().balanced().build()?;
        assert_eq!(processor.top_p(), 0.9);

        let processor = TopPBuilder::new().conservative().build()?;
        assert_eq!(processor.top_p(), 0.7);

        Ok(())
    }

    #[test]
    fn test_nucleus_size_calculation() -> Result<(), Box<dyn std::error::Error>> {
        let logits = create_test_logits()?;
        let nucleus_size = utils::calculate_nucleus_size(&logits, 0.9)?;
        assert!(nucleus_size > 0);
        assert!(nucleus_size <= 5); // Should not exceed vocab size

        Ok(())
    }

    #[test]
    fn test_adaptive_top_p() {
        let base_top_p = 0.9;
        let low_entropy = 1.0;
        let high_entropy = 3.0;
        let max_entropy = 4.0;

        let adaptive_low = utils::adaptive_top_p(base_top_p, low_entropy, max_entropy);
        let adaptive_high = utils::adaptive_top_p(base_top_p, high_entropy, max_entropy);

        assert!(adaptive_high > adaptive_low); // Higher entropy should increase top_p
    }
}
