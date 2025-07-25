//! Core processing context implementation
//!
//! Provides the main ProcessingContext struct with token history tracking,
//! frequency analysis, and zero-allocation patterns for sophisticated sampling.

use arrayvec::ArrayVec;
// Re-export the base ProcessingContext from fluent-ai-simd
pub use fluent_ai_simd::context::ProcessingContext as BaseProcessingContext;

use crate::processing::error::ProcessingError;
use crate::processing::traits::ProcessingResult;

use super::context_stats::{RepetitionPenaltyType, SequenceStats};

/// Maximum supported vocabulary size for bounded allocation
pub const MAX_VOCAB_SIZE: usize = 128_000;

/// Maximum context window size for token history
pub const MAX_CONTEXT_SIZE: usize = 8_192;

/// Default context window size for most applications
pub const DEFAULT_CONTEXT_SIZE: usize = 1_024;

/// Enhanced processing context for candle-specific operations
///
/// Extends the base ProcessingContext from fluent-ai-simd with additional
/// capabilities needed for sophisticated logits processing in candle.
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Base processing context from fluent-ai-simd
    base: BaseProcessingContext,

    /// Vocabulary size for bounds checking
    vocab_size: usize,

    /// Token frequency counts for repetition penalty (stack-allocated)
    token_frequencies: ArrayVec<u32, MAX_VOCAB_SIZE>,

    /// Recent token history with sliding window (stack-allocated)
    token_history: ArrayVec<u32, MAX_CONTEXT_SIZE>,

    /// Current sequence position
    position: usize,

    /// Context window size limit
    context_size: usize,

    /// Sequence statistics for advanced processing
    stats: SequenceStats,
}

impl ProcessingContext {
    /// Create new processing context
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of the vocabulary (must be ≤ MAX_VOCAB_SIZE)
    /// * `context_size` - Maximum context window size (must be ≤ MAX_CONTEXT_SIZE)
    ///
    /// # Returns
    ///
    /// * `Ok(ProcessingContext)` - Successfully created context
    /// * `Err(ProcessingError)` - Invalid parameters or initialization failure
    pub fn new(vocab_size: usize, context_size: usize) -> ProcessingResult<Self> {
        if vocab_size == 0 {
            return Err(ProcessingError::configuration(
                "Vocabulary size cannot be zero",
            ));
        }

        if vocab_size > MAX_VOCAB_SIZE {
            return Err(ProcessingError::configuration(format!(
                "Vocabulary size {} exceeds maximum {}",
                vocab_size, MAX_VOCAB_SIZE
            )));
        }

        if context_size == 0 {
            return Err(ProcessingError::configuration(
                "Context size cannot be zero",
            ));
        }

        if context_size > MAX_CONTEXT_SIZE {
            return Err(ProcessingError::configuration(format!(
                "Context size {} exceeds maximum {}",
                context_size, MAX_CONTEXT_SIZE
            )));
        }

        // Initialize base context
        let base = BaseProcessingContext::new();

        // Initialize token frequencies to zero
        let mut token_frequencies = ArrayVec::new();
        for _ in 0..vocab_size {
            if token_frequencies.try_push(0).is_err() {
                return Err(ProcessingError::resource(
                    "Failed to initialize token frequency array",
                ));
            }
        }

        Ok(Self {
            base,
            vocab_size,
            token_frequencies,
            token_history: ArrayVec::new(),
            position: 0,
            context_size,
            stats: SequenceStats::new(),
        })
    }

    /// Create context with default context size
    #[inline(always)]
    pub fn with_vocab_size(vocab_size: usize) -> ProcessingResult<Self> {
        Self::new(vocab_size, DEFAULT_CONTEXT_SIZE)
    }

    /// Get vocabulary size
    #[inline(always)]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get context window size
    #[inline(always)]
    pub fn context_size(&self) -> usize {
        self.context_size
    }

    /// Get current sequence position
    #[inline(always)]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get current sequence length
    #[inline(always)]
    pub fn sequence_length(&self) -> usize {
        self.token_history.len()
    }

    /// Get token history as slice
    #[inline(always)]
    pub fn token_history(&self) -> &[u32] {
        &self.token_history
    }

    /// Get token frequencies as slice
    #[inline(always)]
    pub fn token_frequencies(&self) -> &[u32] {
        &self.token_frequencies
    }

    /// Get frequency for specific token
    #[inline(always)]
    pub fn token_frequency(&self, token_id: u32) -> u32 {
        if (token_id as usize) < self.token_frequencies.len() {
            self.token_frequencies[token_id as usize]
        } else {
            0
        }
    }

    /// Check if token is present in history
    #[inline(always)]
    pub fn has_token(&self, token_id: u32) -> bool {
        self.token_frequency(token_id) > 0
    }

    /// Get reference to base processing context
    #[inline(always)]
    pub fn base_context(&self) -> &BaseProcessingContext {
        &self.base
    }

    /// Get mutable reference to base processing context
    #[inline(always)]
    pub fn base_context_mut(&mut self) -> &mut BaseProcessingContext {
        &mut self.base
    }

    /// Get sequence statistics
    #[inline(always)]
    pub fn stats(&self) -> &SequenceStats {
        &self.stats
    }

    /// Add token to context with frequency tracking
    ///
    /// Updates token history, frequencies, and statistics.
    /// Uses sliding window approach when context is full.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Token ID to add
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Token added successfully
    /// * `Err(ProcessingError)` - Invalid token ID or internal error
    pub fn add_token(&mut self, token_id: u32) -> ProcessingResult<()> {
        // Validate token ID
        if (token_id as usize) >= self.vocab_size {
            return Err(ProcessingError::validation(format!(
                "Token ID {} exceeds vocabulary size {}",
                token_id, self.vocab_size
            )));
        }

        // Handle sliding window when context is full
        if self.token_history.is_full() {
            if let Some(oldest_token) = self.token_history.first().copied() {
                // Decrease frequency of oldest token
                let oldest_idx = oldest_token as usize;
                if oldest_idx < self.token_frequencies.len()
                    && self.token_frequencies[oldest_idx] > 0
                {
                    self.token_frequencies[oldest_idx] -= 1;
                }

                // Remove oldest token from history
                self.token_history.remove(0);
            }
        }

        // Add new token to history
        if self.token_history.try_push(token_id).is_err() {
            return Err(ProcessingError::resource("Failed to add token to history"));
        }

        // Update token frequency
        let token_idx = token_id as usize;
        if token_idx < self.token_frequencies.len() {
            self.token_frequencies[token_idx] = self.token_frequencies[token_idx].saturating_add(1);
        }

        // Update base context (extend history)
        let token_slice = &[token_id];
        self.base.extend_history(token_slice);

        // Update position and statistics
        self.position += 1;
        self.stats.add_token(token_id);

        Ok(())
    }

    /// Add multiple tokens to context
    ///
    /// More efficient than adding tokens one by one.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Slice of token IDs to add
    pub fn add_tokens(&mut self, tokens: &[u32]) -> ProcessingResult<()> {
        for &token_id in tokens {
            self.add_token(token_id)?;
        }
        Ok(())
    }

    /// Reset context for new sequence
    ///
    /// Clears token history, frequencies, and statistics while
    /// maintaining vocabulary and context size configuration.
    pub fn reset(&mut self) {
        // Clear token history
        self.token_history.clear();

        // Reset token frequencies
        for freq in &mut self.token_frequencies {
            *freq = 0;
        }

        // Reset base context
        self.base = BaseProcessingContext::new();

        // Reset position and statistics
        self.position = 0;
        self.stats = SequenceStats::new();
    }

}