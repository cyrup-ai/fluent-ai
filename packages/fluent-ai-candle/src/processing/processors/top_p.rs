//! Top-P (Nucleus) Sampling Processor
//!
//! Implements nucleus sampling that selects from the smallest set of tokens whose cumulative
//! probability exceeds the threshold p. This provides a dynamic vocabulary size based on the
//! probability distribution shape, offering better quality than fixed top-k sampling.

use std::cmp::Ordering;

use arrayvec::ArrayVec;

use crate::processing::traits::{LogitsProcessor, ProcessingResult};
use crate::processing::{ProcessingContext, ProcessingError};

/// Nucleus sampling processor with adaptive probability-based filtering
///
/// Top-P sampling selects from the smallest set of tokens whose cumulative probability
/// exceeds the threshold p. This adapts the vocabulary size dynamically based on the
/// probability distribution, allowing for more conservative sampling when the model
/// is confident and more diverse sampling when the model is uncertain.
#[derive(Debug, Clone)]
pub struct TopPProcessor {
    /// Nucleus threshold (0.0 < p <= 1.0)
    pub p: f32,

    /// Internal buffer for probability calculations (zero allocation up to 32K tokens)
    buffer: ArrayVec<(usize, f32), 32768>,

    /// Cached identity state (optimization)
    is_identity: bool}

impl TopPProcessor {
    /// Create new Top-P processor with nucleus threshold
    ///
    /// # Arguments
    /// * `p` - Nucleus threshold (0.0 < p <= 1.0)
    ///
    /// # Returns
    /// * `Ok(TopPProcessor)` - Valid processor
    /// * `Err(ProcessingError::InvalidConfiguration)` - Invalid p value
    pub fn new(p: f32) -> ProcessingResult<Self> {
        if p <= 0.0 || p > 1.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Top-P value must be in range (0.0, 1.0], got {}",
                p
            )));
        }

        Ok(Self {
            p,
            buffer: ArrayVec::new(),
            is_identity: (p - 1.0).abs() < f32::EPSILON})
    }

    /// Apply nucleus sampling filtering with zero allocation
    #[inline]
    fn apply_nucleus_filtering(&mut self, logits: &mut [f32]) -> ProcessingResult<()> {
        if logits.is_empty() {
            return Ok(());
        }

        // Clear buffer for reuse (zero allocation)
        self.buffer.clear();

        // Convert logits to probabilities with numerical stability
        let max_logit = logits
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);

        // Collect (index, probability) pairs with overflow protection
        let mut total_prob = 0.0f64;
        for (i, &logit) in logits.iter().enumerate() {
            let stable_logit = logit - max_logit;
            let prob = stable_logit.exp();

            if prob.is_finite() && prob > 0.0 {
                total_prob += prob as f64;

                if self.buffer.is_full() {
                    return Err(ProcessingError::buffer_overflow(
                        "Vocabulary size exceeds maximum supported tokens (32768)",
                    ));
                }

                self.buffer.push((i, prob));
            }
        }

        // Handle edge cases
        if total_prob <= 0.0 || !total_prob.is_finite() {
            return Err(ProcessingError::NumericalError(
                "Invalid probability distribution - total probability is zero or infinite"
                    .to_string(),
            ));
        }

        // Normalize probabilities with numerical stability
        let total_prob_f32 = total_prob as f32;
        for (_, prob) in self.buffer.iter_mut() {
            *prob /= total_prob_f32;
        }

        // Sort by probability (descending) with proper error handling
        self.buffer
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Find nucleus cutoff with cumulative probability tracking
        let mut cumulative_prob = 0.0;
        let mut nucleus_size = 0;

        for (_, prob) in self.buffer.iter() {
            cumulative_prob += prob;
            nucleus_size += 1;

            if cumulative_prob >= self.p {
                break;
            }
        }

        // Ensure at least one token is always included (safety)
        if nucleus_size == 0 {
            nucleus_size = 1;
        }

        // Zero out logits outside nucleus with efficient batch operations
        let mut kept_indices = [false; 32768]; // Stack allocation for common case

        if logits.len() > kept_indices.len() {
            return Err(ProcessingError::buffer_overflow(
                "Vocabulary size exceeds maximum supported tokens",
            ));
        }

        // Mark nucleus tokens for keeping
        for i in 0..nucleus_size {
            if i < self.buffer.len() {
                let idx = self.buffer[i].0;
                if idx < kept_indices.len() {
                    kept_indices[idx] = true;
                }
            }
        }

        // Zero out non-nucleus tokens efficiently
        const ZERO_THRESHOLD: f32 = -1e10;
        for (i, logit) in logits.iter_mut().enumerate() {
            if i < kept_indices.len() && !kept_indices[i] {
                *logit = ZERO_THRESHOLD;
            }
        }

        Ok(())
    }

    /// Update nucleus threshold with validation
    pub fn set_p(&mut self, p: f32) -> ProcessingResult<()> {
        if p <= 0.0 || p > 1.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Top-P value must be in range (0.0, 1.0], got {}",
                p
            )));
        }

        self.p = p;
        self.is_identity = (p - 1.0).abs() < f32::EPSILON;
        Ok(())
    }

    /// Get current nucleus threshold
    #[inline(always)]
    pub fn p(&self) -> f32 {
        self.p
    }

    /// Check if this processor configuration is approximately identity
    #[inline(always)]
    pub fn is_identity_config(&self) -> bool {
        self.is_identity
    }
}

impl LogitsProcessor for TopPProcessor {
    #[inline]
    fn process_logits(
        &mut self,
        logits: &mut [f32],
        _context: &ProcessingContext,
    ) -> ProcessingResult<()> {
        if self.is_identity {
            return Ok(()); // No-op for p ≈ 1.0
        }

        self.apply_nucleus_filtering(logits)
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "TopPProcessor"
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        self.is_identity
    }

    fn validate(&self) -> ProcessingResult<()> {
        if self.p <= 0.0 || self.p > 1.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Top-P value must be in range (0.0, 1.0], got {}",
                self.p
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_top_p_creation() {
        let processor = TopPProcessor::new(0.9);
        assert!(processor.is_ok());
        assert_eq!(processor.unwrap().p(), 0.9);

        // Test invalid values
        assert!(TopPProcessor::new(0.0).is_err());
        assert!(TopPProcessor::new(1.1).is_err());
        assert!(TopPProcessor::new(-0.1).is_err());

        // Test edge case: p = 1.0 (identity)
        let identity = TopPProcessor::new(1.0).unwrap();
        assert!(identity.is_identity());
    }

    #[test]
    fn test_nucleus_filtering() {
        let mut processor = TopPProcessor::new(0.8).unwrap();
        let context = ProcessingContext::default();

        // Test basic nucleus sampling
        let mut logits = vec![1.0, 2.0, 0.5, 0.1, 0.05]; // Probabilities: high to low
        let original = logits.clone();

        processor.process_logits(&mut logits, &context).unwrap();

        // Verify that some tokens were filtered (set to very low values)
        let filtered_count = logits.iter().filter(|&&x| x < -1e9).count();
        assert!(filtered_count > 0, "Some tokens should be filtered");

        // Test identity case
        let mut identity_processor = TopPProcessor::new(1.0).unwrap();
        let mut identity_logits = original;
        let expected = identity_logits.clone();

        identity_processor
            .process_logits(&mut identity_logits, &context)
            .unwrap();

        // Should be unchanged
        for (actual, expected) in identity_logits.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_numerical_stability() {
        let mut processor = TopPProcessor::new(0.5).unwrap();
        let context = ProcessingContext::default();

        // Test with large logit values
        let mut large_logits = vec![100.0, 99.0, 98.0, 1.0, 0.0];
        assert!(
            processor
                .process_logits(&mut large_logits, &context)
                .is_ok()
        );

        // Test with small logit values
        let mut small_logits = vec![-100.0, -101.0, -102.0, -103.0, -104.0];
        assert!(
            processor
                .process_logits(&mut small_logits, &context)
                .is_ok()
        );
    }

    #[test]
    fn test_edge_cases() {
        let mut processor = TopPProcessor::new(0.9).unwrap();
        let context = ProcessingContext::default();

        // Test empty logits
        let mut empty: Vec<f32> = vec![];
        assert!(processor.process_logits(&mut empty, &context).is_ok());

        // Test single logit
        let mut single = vec![1.0];
        assert!(processor.process_logits(&mut single, &context).is_ok());

        // Test uniform distribution
        let mut uniform = vec![1.0, 1.0, 1.0, 1.0];
        assert!(processor.process_logits(&mut uniform, &context).is_ok());
    }

    #[test]
    fn test_configuration_updates() {
        let mut processor = TopPProcessor::new(0.9).unwrap();

        // Test valid update
        assert!(processor.set_p(0.5).is_ok());
        assert_eq!(processor.p(), 0.5);
        assert!(!processor.is_identity());

        // Test invalid updates
        assert!(processor.set_p(0.0).is_err());
        assert!(processor.set_p(1.1).is_err());

        // Processor should maintain previous valid state
        assert_eq!(processor.p(), 0.5);

        // Test identity update
        assert!(processor.set_p(1.0).is_ok());
        assert!(processor.is_identity());
    }

    #[test]
    fn test_validation() {
        let processor = TopPProcessor::new(0.9).unwrap();
        assert!(processor.validate().is_ok());

        let mut invalid_processor = TopPProcessor::new(0.9).unwrap();
        invalid_processor.p = 1.5; // Manually set invalid value
        assert!(invalid_processor.validate().is_err());
    }
}
