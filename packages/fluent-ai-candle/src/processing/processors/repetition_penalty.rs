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
    is_identity: bool}

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
            is_identity})
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
