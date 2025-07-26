//! Repetition penalty processor with context-aware token tracking
//!
//! Applies exponential penalty to recently generated tokens to reduce repetitive outputs.
//! Maintains efficient token history with configurable context window.

use std::collections::HashMap;

use arrayvec::ArrayVec;
use candle_core::Tensor;

use super::SamplingError;
// Removed unused import: crate::processing::traits::LogitsProcessor

/// Maximum context size for repetition tracking (bounded for performance)
const MAX_REPETITION_CONTEXT: usize = 2048;

/// Repetition penalty processor with efficient token tracking
///
/// Repetition penalty works by:
/// 1. Track recently generated tokens within context window
/// 2. Apply exponential penalty: logit = logit / (penalty^count)
/// 3. Use frequency-based penalties for repeated tokens
/// 4. Maintain efficient lookups with bounded memory usage
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyProcessor {
    penalty: f64,
    context_size: usize,
    is_identity: bool, // Optimization for penalty = 1.0
    // Pre-computed penalty powers for efficiency
    penalty_powers: ArrayVec<f64, 16>, // Cache common penalty powers
}

impl RepetitionPenaltyProcessor {
    /// Create new repetition penalty processor with validation
    ///
    /// # Arguments
    /// * `penalty` - Penalty factor (must be >= 1.0, where 1.0 = no penalty)
    /// * `context_size` - Number of recent tokens to track (capped at MAX_REPETITION_CONTEXT)
    ///
    /// # Errors
    /// Returns `SamplingError::InvalidRepetitionPenalty` if penalty < 1.0 or not finite
    pub fn new(penalty: f64, context_size: usize) -> Result<Self, SamplingError> {
        if penalty < 1.0 || !penalty.is_finite() {
            return Err(SamplingError::InvalidRepetitionPenalty(penalty));
        }

        let is_identity = (penalty - 1.0).abs() < f64::EPSILON;
        let capped_context = context_size.min(MAX_REPETITION_CONTEXT);

        // Pre-compute common penalty powers for performance
        let mut penalty_powers = ArrayVec::new();
        if !is_identity {
            for i in 1..=16 {
                if penalty_powers.try_push(penalty.powi(i)).is_err() {
                    break;
                }
            }
        }

        Ok(Self {
            penalty,
            context_size: capped_context,
            is_identity,
            penalty_powers})
    }

    /// Get current penalty value
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// Get current context size
    pub fn context_size(&self) -> usize {
        self.context_size
    }

    /// Update penalty with validation
    pub fn set_penalty(&mut self, penalty: f64) -> Result<(), SamplingError> {
        if penalty < 1.0 || !penalty.is_finite() {
            return Err(SamplingError::InvalidRepetitionPenalty(penalty));
        }

        self.penalty = penalty;
        self.is_identity = (penalty - 1.0).abs() < f64::EPSILON;

        // Recompute penalty powers
        self.penalty_powers.clear();
        if !self.is_identity {
            for i in 1..=16 {
                if self.penalty_powers.try_push(penalty.powi(i)).is_err() {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Update context size
    pub fn set_context_size(&mut self, context_size: usize) {
        self.context_size = context_size.min(MAX_REPETITION_CONTEXT);
    }

    /// Apply repetition penalty based on token history
    ///
    /// Uses efficient algorithms for different scenarios:
    /// - Empty history: No-op
    /// - Small vocabulary: Direct logits modification
    /// - Large vocabulary: Sparse penalty application
    #[allow(dead_code)]
    fn apply_repetition_penalty(
        &self,
        logits: &mut Tensor,
        token_ids: &[u32],
        position: usize,
    ) -> Result<(), SamplingError> {
        if self.is_identity || token_ids.is_empty() {
            return Ok(()); // No penalty needed
        }

        // Get relevant context window
        let context_tokens = self.get_context_window(token_ids, position);
        if context_tokens.is_empty() {
            return Ok(());
        }

        // Count token frequencies in context
        let token_counts = self.count_token_frequencies(&context_tokens);
        if token_counts.is_empty() {
            return Ok(());
        }

        // Apply penalties to logits
        self.apply_penalty_to_logits(logits, &token_counts)?;

        Ok(())
    }

    /// Extract relevant context window from token history
    #[allow(dead_code)]
    #[inline(always)]
    fn get_context_window<'a>(&self, token_ids: &'a [u32], position: usize) -> &'a [u32] {
        if token_ids.is_empty() || self.context_size == 0 {
            return &[];
        }

        // Take the last context_size tokens, but don't exceed available history
        let available_history = position.min(token_ids.len());
        let context_start = available_history.saturating_sub(self.context_size);
        let context_end = available_history;

        &token_ids[context_start..context_end]
    }

    /// Count token frequencies in context window with efficient data structures
    #[allow(dead_code)]
    #[inline(always)]
    fn count_token_frequencies(&self, context_tokens: &[u32]) -> HashMap<u32, u32> {
        let mut counts = HashMap::with_capacity(context_tokens.len().min(256));

        for &token_id in context_tokens {
            *counts.entry(token_id).or_insert(0) += 1;
        }

        counts
    }

    /// Apply penalty to logits based on token frequencies
    #[allow(dead_code)]
    #[inline(always)]
    fn apply_penalty_to_logits(
        &self,
        logits: &mut Tensor,
        token_counts: &HashMap<u32, u32>,
    ) -> Result<(), SamplingError> {
        let mut logits_vec = logits
            .to_vec1::<f32>()
            .map_err(|e| SamplingError::TensorError(e.to_string()))?;

        // Apply penalties to each token that appears in context
        for (&token_id, &count) in token_counts {
            let token_idx = token_id as usize;

            // Bounds check for safety
            if token_idx >= logits_vec.len() {
                continue; // Skip invalid token IDs
            }

            // Calculate penalty factor efficiently
            let penalty_factor = self.calculate_penalty_factor(count);

            // Apply penalty: logit = logit / penalty_factor
            // In log space: logit = logit - log(penalty_factor)
            let log_penalty = penalty_factor.ln() as f32;
            logits_vec[token_idx] -= log_penalty;
        }

        // Create updated tensor
        let device = logits.device();
        *logits = Tensor::from_vec(logits_vec, logits.shape(), device)
            .map_err(|e| SamplingError::TensorError(e.to_string()))?;

        Ok(())
    }

    /// Calculate penalty factor for given token count with optimization
    #[allow(dead_code)]
    #[inline(always)]
    fn calculate_penalty_factor(&self, count: u32) -> f64 {
        if count == 0 {
            return 1.0;
        }

        // Use pre-computed powers for common cases
        let count_idx = (count as usize).saturating_sub(1);
        if count_idx < self.penalty_powers.len() {
            self.penalty_powers[count_idx]
        } else {
            // Fall back to computation for high counts
            self.penalty.powi(count as i32)
        }
    }

    /// Validate penalty application for debugging
    #[cfg(test)]
    fn validate_penalty_application(
        &self,
        original_logits: &[f32],
        penalized_logits: &[f32],
        token_counts: &HashMap<u32, u32>,
    ) -> bool {
        for (&token_id, &count) in token_counts {
            if count == 0 {
                continue;
            }

            let token_idx = token_id as usize;
            if token_idx >= original_logits.len() || token_idx >= penalized_logits.len() {
                continue;
            }

            let expected_penalty = self.calculate_penalty_factor(count).ln() as f32;
            let actual_change = original_logits[token_idx] - penalized_logits[token_idx];

            if (actual_change - expected_penalty).abs() > f32::EPSILON * 10.0 {
                return false;
            }
        }
        true
    }
}

// TODO: Update to new LogitsProcessor API that uses process_logits() instead of process()
// impl LogitsProcessor for RepetitionPenaltyProcessor {
//     fn process_logits(&mut self, logits: &mut [f32], context: &ProcessingContext) -> ProcessingResult<()> {
//         // Implementation needed for new API
//     }
//
//     fn validate(&self) -> ProcessingResult<()> {
//         // Implementation needed for new API
//     }
//
//     fn name(&self) -> &'static str {
//         "RepetitionPenaltyProcessor"
//     }
//
//     fn is_identity(&self) -> bool {
//         self.is_identity
//     }
// }

// Implement common traits for ergonomics
impl std::fmt::Display for RepetitionPenaltyProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RepetitionPenalty({}, ctx={})",
            self.penalty, self.context_size
        )
    }
}

impl PartialEq for RepetitionPenaltyProcessor {
    fn eq(&self, other: &Self) -> bool {
        (self.penalty - other.penalty).abs() < f64::EPSILON
            && self.context_size == other.context_size
    }
}
