//! Repetition Penalty Processor
//!
//! Implements sophisticated repetition penalty mechanisms to reduce repetitive outputs
//! in language model generation. Supports both frequency-based and presence-based penalties
//! with efficient token frequency tracking and context-aware penalty application.

use std::collections::HashMap;

use arrayvec::ArrayVec;

use crate::processing::traits::{LogitsProcessor, ProcessingResult};
use crate::processing::{ProcessingContext, ProcessingError};

/// Maximum number of unique tokens to track efficiently on the stack
const MAX_STACK_TOKENS: usize = 1024;

/// Repetition penalty processor with frequency-based and presence-based penalties
///
/// This processor reduces the likelihood of generating repetitive text by applying
/// penalties to tokens that have appeared in the generation context. It supports
/// multiple penalty mechanisms:
/// - Repetition penalty: Scales down probabilities of repeated tokens
/// - Frequency penalty: Applies linear penalty based on token frequency
/// - Presence penalty: Applies fixed penalty for any token that has appeared
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyProcessor {
    /// Base repetition penalty (1.0 = no penalty, > 1.0 = more penalty)
    pub repetition_penalty: f32,

    /// Frequency-based penalty coefficient (0.0 = disabled, > 0.0 = more penalty)
    pub frequency_penalty: f32,

    /// Presence-based penalty coefficient (0.0 = disabled, > 0.0 = more penalty)
    pub presence_penalty: f32,

    /// Context window size for penalty application (0 = use full context)
    pub context_window: usize,

    /// Fast token frequency map for stack-allocated common cases
    token_frequencies: ArrayVec<(u32, u32), MAX_STACK_TOKENS>,

    /// Overflow frequency map for large vocabularies
    overflow_frequencies: HashMap<u32, u32>,

    /// Cached identity state for optimization
    is_identity: bool,
}

impl RepetitionPenaltyProcessor {
    /// Create new repetition penalty processor
    ///
    /// # Arguments
    /// * `repetition_penalty` - Base repetition penalty (1.0 = no penalty, > 1.0 = penalty)
    /// * `frequency_penalty` - Frequency-based penalty coefficient (0.0 = disabled)
    /// * `presence_penalty` - Presence-based penalty coefficient (0.0 = disabled)
    /// * `context_window` - Context window size (0 = use full context)
    pub fn new(
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        context_window: usize,
    ) -> ProcessingResult<Self> {
        if repetition_penalty < 1.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Repetition penalty must be >= 1.0, got {}",
                repetition_penalty
            )));
        }

        if frequency_penalty < 0.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Frequency penalty must be >= 0.0, got {}",
                frequency_penalty
            )));
        }

        if presence_penalty < 0.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Presence penalty must be >= 0.0, got {}",
                presence_penalty
            )));
        }

        let is_identity = (repetition_penalty - 1.0).abs() < f32::EPSILON
            && frequency_penalty.abs() < f32::EPSILON
            && presence_penalty.abs() < f32::EPSILON;

        Ok(Self {
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            context_window,
            token_frequencies: ArrayVec::new(),
            overflow_frequencies: HashMap::new(),
            is_identity,
        })
    }

    /// Create processor with only repetition penalty
    pub fn with_repetition_penalty(penalty: f32) -> ProcessingResult<Self> {
        Self::new(penalty, 0.0, 0.0, 0)
    }

    /// Create processor with frequency-based penalty
    pub fn with_frequency_penalty(penalty: f32) -> ProcessingResult<Self> {
        Self::new(1.0, penalty, 0.0, 0)
    }

    /// Create processor with presence-based penalty
    pub fn with_presence_penalty(penalty: f32) -> ProcessingResult<Self> {
        Self::new(1.0, 0.0, penalty, 0)
    }

    /// Update token frequencies from context with efficient tracking
    #[inline]
    fn update_token_frequencies(&mut self, context: &ProcessingContext) {
        // Clear previous frequencies
        self.token_frequencies.clear();
        self.overflow_frequencies.clear();

        let token_history = context.token_history();
        let tokens = if self.context_window > 0 && self.context_window < token_history.len() {
            // Use only recent tokens from context window
            let start_idx = token_history.len() - self.context_window;
            &token_history[start_idx..]
        } else {
            // Use full token history
            token_history
        };

        // Count token frequencies with hybrid stack/heap storage
        for &token_id in tokens {
            self.increment_token_frequency(token_id);
        }
    }

    /// Efficiently increment token frequency using hybrid storage
    #[inline]
    fn increment_token_frequency(&mut self, token_id: u32) {
        // Try to find in stack-allocated buffer first (fast path)
        for (stored_id, count) in self.token_frequencies.iter_mut() {
            if *stored_id == token_id {
                *count += 1;
                return;
            }
        }

        // If not found and buffer has space, add to stack buffer
        if !self.token_frequencies.is_full() {
            self.token_frequencies.push((token_id, 1));
            return;
        }

        // Otherwise, use heap-allocated overflow map
        *self.overflow_frequencies.entry(token_id).or_insert(0) += 1;
    }

    /// Get token frequency with hybrid lookup
    #[inline]
    fn get_token_frequency(&self, token_id: u32) -> u32 {
        // Check stack buffer first (fast path)
        for (stored_id, count) in &self.token_frequencies {
            if *stored_id == token_id {
                return *count;
            }
        }

        // Check overflow map
        self.overflow_frequencies
            .get(&token_id)
            .copied()
            .unwrap_or(0)
    }

    /// Apply repetition penalties with numerical stability
    #[inline]
    fn apply_penalties(
        &mut self,
        logits: &mut [f32],
        context: &ProcessingContext,
    ) -> ProcessingResult<()> {
        if logits.is_empty() {
            return Ok(());
        }

        // Update token frequencies from context
        self.update_token_frequencies(context);

        // Apply penalties to each token
        for (token_id, logit) in logits.iter_mut().enumerate() {
            let token_id_u32 = token_id as u32;
            let frequency = self.get_token_frequency(token_id_u32);

            if frequency > 0 {
                let mut penalty_factor = 1.0;

                // Apply base repetition penalty (divide if logit > 0, multiply if logit < 0)
                if self.repetition_penalty > 1.0 {
                    if *logit > 0.0 {
                        penalty_factor /= self.repetition_penalty;
                    } else {
                        penalty_factor *= self.repetition_penalty;
                    }
                }

                // Apply frequency penalty (linear with frequency)
                if self.frequency_penalty > 0.0 {
                    let freq_penalty = self.frequency_penalty * frequency as f32;
                    penalty_factor *= 1.0 - freq_penalty.min(0.99); // Prevent total elimination
                }

                // Apply presence penalty (fixed penalty for any presence)
                if self.presence_penalty > 0.0 {
                    penalty_factor *= 1.0 - self.presence_penalty.min(0.99); // Prevent total elimination
                }

                // Apply combined penalty with numerical stability
                *logit *= penalty_factor.max(1e-8); // Prevent underflow
            }
        }

        Ok(())
    }

    /// Update repetition penalty with validation
    pub fn set_repetition_penalty(&mut self, penalty: f32) -> ProcessingResult<()> {
        if penalty < 1.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Repetition penalty must be >= 1.0, got {}",
                penalty
            )));
        }

        self.repetition_penalty = penalty;
        self.update_identity_cache();
        Ok(())
    }

    /// Update frequency penalty with validation
    pub fn set_frequency_penalty(&mut self, penalty: f32) -> ProcessingResult<()> {
        if penalty < 0.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Frequency penalty must be >= 0.0, got {}",
                penalty
            )));
        }

        self.frequency_penalty = penalty;
        self.update_identity_cache();
        Ok(())
    }

    /// Update presence penalty with validation
    pub fn set_presence_penalty(&mut self, penalty: f32) -> ProcessingResult<()> {
        if penalty < 0.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Presence penalty must be >= 0.0, got {}",
                penalty
            )));
        }

        self.presence_penalty = penalty;
        self.update_identity_cache();
        Ok(())
    }

    /// Update context window size
    pub fn set_context_window(&mut self, window_size: usize) {
        self.context_window = window_size;
    }

    /// Update cached identity state after parameter changes
    #[inline]
    fn update_identity_cache(&mut self) {
        self.is_identity = (self.repetition_penalty - 1.0).abs() < f32::EPSILON
            && self.frequency_penalty.abs() < f32::EPSILON
            && self.presence_penalty.abs() < f32::EPSILON;
    }

    /// Get current repetition penalty
    #[inline(always)]
    pub fn repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }

    /// Get current frequency penalty
    #[inline(always)]
    pub fn frequency_penalty(&self) -> f32 {
        self.frequency_penalty
    }

    /// Get current presence penalty
    #[inline(always)]
    pub fn presence_penalty(&self) -> f32 {
        self.presence_penalty
    }

    /// Get current context window size
    #[inline(always)]
    pub fn context_window(&self) -> usize {
        self.context_window
    }

    /// Get number of unique tokens currently tracked
    pub fn tracked_token_count(&self) -> usize {
        self.token_frequencies.len() + self.overflow_frequencies.len()
    }
}

impl LogitsProcessor for RepetitionPenaltyProcessor {
    #[inline]
    fn process_logits(
        &mut self,
        logits: &mut [f32],
        context: &ProcessingContext,
    ) -> ProcessingResult<()> {
        if self.is_identity {
            return Ok(()); // No-op when all penalties are disabled
        }

        self.apply_penalties(logits, context)
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "RepetitionPenaltyProcessor"
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        self.is_identity
    }

    fn validate(&self) -> ProcessingResult<()> {
        if self.repetition_penalty < 1.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Repetition penalty must be >= 1.0, got {}",
                self.repetition_penalty
            )));
        }

        if self.frequency_penalty < 0.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Frequency penalty must be >= 0.0, got {}",
                self.frequency_penalty
            )));
        }

        if self.presence_penalty < 0.0 {
            return Err(ProcessingError::InvalidConfiguration(format!(
                "Presence penalty must be >= 0.0, got {}",
                self.presence_penalty
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_penalty_creation() {
        // Valid configurations
        let processor = RepetitionPenaltyProcessor::new(1.2, 0.0, 0.0, 0);
        assert!(processor.is_ok());

        let processor = RepetitionPenaltyProcessor::with_repetition_penalty(1.5);
        assert!(processor.is_ok());
        assert_eq!(processor.unwrap().repetition_penalty(), 1.5);

        // Invalid configurations
        assert!(RepetitionPenaltyProcessor::new(0.8, 0.0, 0.0, 0).is_err()); // penalty < 1.0
        assert!(RepetitionPenaltyProcessor::new(1.0, -0.1, 0.0, 0).is_err()); // negative frequency
        assert!(RepetitionPenaltyProcessor::new(1.0, 0.0, -0.1, 0).is_err()); // negative presence

        // Identity case
        let identity = RepetitionPenaltyProcessor::new(1.0, 0.0, 0.0, 0).unwrap();
        assert!(identity.is_identity());
    }

    #[test]
    fn test_token_frequency_tracking() {
        let mut processor = RepetitionPenaltyProcessor::new(1.2, 0.0, 0.0, 0).unwrap();

        // Create context with repeated tokens
        let mut context = ProcessingContext::default();
        // Token 1 appears 3 times, token 2 twice, token 3 once
        context.add_token(1).unwrap();
        context.add_token(2).unwrap();
        context.add_token(3).unwrap();
        context.add_token(1).unwrap();
        context.add_token(2).unwrap();
        context.add_token(1).unwrap();

        processor.update_token_frequencies(&context);

        assert_eq!(processor.get_token_frequency(1), 3);
        assert_eq!(processor.get_token_frequency(2), 2);
        assert_eq!(processor.get_token_frequency(3), 1);
        assert_eq!(processor.get_token_frequency(99), 0); // Not present
    }

    #[test]
    fn test_context_window() {
        let mut processor = RepetitionPenaltyProcessor::new(1.2, 0.0, 0.0, 3).unwrap(); // Window of 3

        let mut context = ProcessingContext::default();
        // Only last 3 tokens: [4, 5, 1]
        context.add_token(1).unwrap();
        context.add_token(2).unwrap();
        context.add_token(3).unwrap();
        context.add_token(4).unwrap();
        context.add_token(5).unwrap();
        context.add_token(1).unwrap();

        processor.update_token_frequencies(&context);

        assert_eq!(processor.get_token_frequency(1), 1); // Only count the last occurrence
        assert_eq!(processor.get_token_frequency(4), 1);
        assert_eq!(processor.get_token_frequency(5), 1);
        assert_eq!(processor.get_token_frequency(2), 0); // Outside window
        assert_eq!(processor.get_token_frequency(3), 0); // Outside window
    }

    #[test]
    fn test_penalty_application() {
        let mut processor = RepetitionPenaltyProcessor::new(2.0, 0.1, 0.05, 0).unwrap();

        let mut context = ProcessingContext::default();
        // Token 0 appears twice, token 1 once
        context.add_token(0).unwrap();
        context.add_token(1).unwrap();
        context.add_token(0).unwrap();

        let mut logits = vec![1.0, 0.5, 2.0]; // 3 tokens
        let original = logits.clone();

        processor.process_logits(&mut logits, &context).unwrap();

        // Token 0 and 1 should have penalties applied
        assert!(logits[0] < original[0], "Token 0 should be penalized");
        assert!(logits[1] < original[1], "Token 1 should be penalized");
        assert_eq!(logits[2], original[2]); // Token 2 unchanged (not in history)
    }

    #[test]
    fn test_frequency_vs_presence_penalty() {
        // Test frequency penalty (scales with frequency)
        let mut freq_processor = RepetitionPenaltyProcessor::new(1.0, 0.1, 0.0, 0).unwrap();
        let mut context = ProcessingContext::default();
        // Token 0 appears 3 times, token 1 once
        context.add_token(0).unwrap();
        context.add_token(0).unwrap();
        context.add_token(0).unwrap();
        context.add_token(1).unwrap();

        let mut logits = vec![1.0, 1.0];
        freq_processor
            .process_logits(&mut logits, &context)
            .unwrap();

        // Token 0 should have larger penalty due to higher frequency
        assert!(
            logits[0] < logits[1],
            "Higher frequency should result in larger penalty"
        );

        // Test presence penalty (fixed regardless of frequency)
        let mut pres_processor = RepetitionPenaltyProcessor::new(1.0, 0.0, 0.1, 0).unwrap();
        let mut logits2 = vec![1.0, 1.0];
        pres_processor
            .process_logits(&mut logits2, &context)
            .unwrap();

        // Both tokens should have similar penalties (only presence matters)
        let diff = (logits2[0] - logits2[1]).abs();
        assert!(
            diff < 0.01,
            "Presence penalty should not depend on frequency"
        );
    }

    #[test]
    fn test_identity_optimization() {
        let mut processor = RepetitionPenaltyProcessor::new(1.0, 0.0, 0.0, 0).unwrap();
        assert!(processor.is_identity());

        let context = ProcessingContext::default();
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();

        processor.process_logits(&mut logits, &context).unwrap();

        // Should be unchanged
        assert_eq!(logits, original);
    }

    #[test]
    fn test_parameter_updates() {
        let mut processor = RepetitionPenaltyProcessor::new(1.0, 0.0, 0.0, 0).unwrap();

        // Test valid updates
        assert!(processor.set_repetition_penalty(1.5).is_ok());
        assert_eq!(processor.repetition_penalty(), 1.5);
        assert!(!processor.is_identity());

        assert!(processor.set_frequency_penalty(0.1).is_ok());
        assert_eq!(processor.frequency_penalty(), 0.1);

        assert!(processor.set_presence_penalty(0.05).is_ok());
        assert_eq!(processor.presence_penalty(), 0.05);

        // Test invalid updates
        assert!(processor.set_repetition_penalty(0.5).is_err());
        assert!(processor.set_frequency_penalty(-0.1).is_err());
        assert!(processor.set_presence_penalty(-0.1).is_err());
    }

    #[test]
    fn test_edge_cases() {
        let mut processor = RepetitionPenaltyProcessor::new(1.2, 0.0, 0.0, 0).unwrap();
        let context = ProcessingContext::default();

        // Empty logits
        let mut empty: Vec<f32> = vec![];
        assert!(processor.process_logits(&mut empty, &context).is_ok());

        // Single token
        let mut single = vec![1.0];
        assert!(processor.process_logits(&mut single, &context).is_ok());

        // Empty context
        let empty_context = ProcessingContext::default();
        let mut logits = vec![1.0, 2.0, 3.0];
        assert!(
            processor
                .process_logits(&mut logits, &empty_context)
                .is_ok()
        );
    }

    #[test]
    fn test_validation() {
        let processor = RepetitionPenaltyProcessor::new(1.2, 0.1, 0.05, 10).unwrap();
        assert!(processor.validate().is_ok());

        // Test invalid configurations
        let mut invalid = RepetitionPenaltyProcessor::new(1.0, 0.0, 0.0, 0).unwrap();
        invalid.repetition_penalty = 0.5; // Invalid
        assert!(invalid.validate().is_err());

        invalid.repetition_penalty = 1.2;
        invalid.frequency_penalty = -0.1; // Invalid
        assert!(invalid.validate().is_err());

        invalid.frequency_penalty = 0.1;
        invalid.presence_penalty = -0.1; // Invalid
        assert!(invalid.validate().is_err());
    }
}
