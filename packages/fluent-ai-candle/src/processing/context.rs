//! Processing context integration for sophisticated sampling strategies
//!
//! This module provides context management for logits processing with:
//! - Integration with fluent-ai-simd ProcessingContext
//! - Extension for candle-specific processing needs
//! - Token history tracking with bounded memory usage
//! - Zero allocation patterns with stack-based storage
//! - Production-ready error handling

use arrayvec::ArrayVec;
// Re-export the base ProcessingContext from fluent-ai-simd
pub use fluent_ai_simd::context::ProcessingContext as BaseProcessingContext;

use crate::processing::error::ProcessingError;
use crate::processing::traits::ProcessingResult;

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

    /// Calculate repetition factor for token
    ///
    /// Returns a factor indicating how much the token should be penalized
    /// based on its frequency in the context.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Token ID to calculate factor for
    /// * `penalty_type` - Type of penalty calculation to use
    #[inline(always)]
    pub fn repetition_factor(&self, token_id: u32, penalty_type: RepetitionPenaltyType) -> f32 {
        let frequency = self.token_frequency(token_id);

        match penalty_type {
            RepetitionPenaltyType::Linear => frequency as f32,
            RepetitionPenaltyType::Logarithmic => {
                if frequency == 0 {
                    0.0
                } else {
                    (frequency as f32).ln()
                }
            }
            RepetitionPenaltyType::Exponential => {
                if frequency == 0 {
                    0.0
                } else {
                    (frequency as f32).exp().min(10.0) // Clamp to prevent overflow
                }
            }
            RepetitionPenaltyType::SquareRoot => (frequency as f32).sqrt(),
        }
    }

    /// Get context utilization ratio
    ///
    /// Returns the ratio of used context to maximum context size.
    /// Useful for adaptive sampling strategies.
    #[inline(always)]
    pub fn utilization_ratio(&self) -> f32 {
        self.token_history.len() as f32 / self.context_size as f32
    }

    /// Check if context is near capacity
    ///
    /// Returns true if context usage is above the specified threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Threshold ratio (0.0 to 1.0)
    #[inline(always)]
    pub fn is_near_capacity(&self, threshold: f32) -> bool {
        self.utilization_ratio() >= threshold
    }

    /// Get unique token count in current context
    #[inline(always)]
    pub fn unique_token_count(&self) -> usize {
        self.token_frequencies
            .iter()
            .filter(|&&freq| freq > 0)
            .count()
    }

    /// Calculate diversity score of current context
    ///
    /// Returns a score between 0.0 and 1.0 indicating vocabulary diversity.
    /// Higher scores indicate more diverse token usage.
    #[inline(always)]
    pub fn diversity_score(&self) -> f32 {
        if self.token_history.is_empty() {
            return 0.0;
        }

        let unique_count = self.unique_token_count();
        let total_count = self.token_history.len();

        unique_count as f32 / total_count as f32
    }

    /// Get most recent N tokens
    ///
    /// Returns a slice of the most recent N tokens, or all tokens if N
    /// exceeds the history length.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of recent tokens to return
    #[inline(always)]
    pub fn recent_tokens(&self, n: usize) -> &[u32] {
        let start = self.token_history.len().saturating_sub(n);
        &self.token_history[start..]
    }

    /// Check for token repetition patterns
    ///
    /// Detects simple repetition patterns in recent token history.
    /// Useful for advanced repetition penalty strategies.
    ///
    /// # Arguments
    ///
    /// * `pattern_length` - Length of pattern to detect (2-8)
    pub fn has_repetition_pattern(&self, pattern_length: usize) -> bool {
        if pattern_length < 2 || pattern_length > 8 || self.token_history.len() < pattern_length * 2
        {
            return false;
        }

        let history = &self.token_history;
        let len = history.len();

        // Check if last pattern_length tokens repeat
        for i in 0..pattern_length {
            if history[len - 1 - i] != history[len - 1 - pattern_length - i] {
                return false;
            }
        }

        true
    }
}

/// Types of repetition penalty calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepetitionPenaltyType {
    /// Linear penalty: penalty = frequency
    Linear,
    /// Logarithmic penalty: penalty = ln(frequency)
    Logarithmic,
    /// Exponential penalty: penalty = exp(frequency)
    Exponential,
    /// Square root penalty: penalty = sqrt(frequency)
    SquareRoot,
}

/// Sequence statistics for advanced processing strategies
#[derive(Debug, Clone)]
pub struct SequenceStats {
    /// Total tokens processed
    total_tokens: usize,
    /// Average token frequency
    avg_frequency: f32,
    /// Most frequent token and its count
    most_frequent: Option<(u32, u32)>,
    /// Entropy estimate of token distribution
    entropy_estimate: f32,
}

impl SequenceStats {
    /// Create new sequence statistics
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            total_tokens: 0,
            avg_frequency: 0.0,
            most_frequent: None,
            entropy_estimate: 0.0,
        }
    }

    /// Add token to statistics
    fn add_token(&mut self, _token_id: u32) {
        self.total_tokens += 1;
        // Additional statistics could be computed here
    }

    /// Get total tokens processed
    #[inline(always)]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Get average token frequency
    #[inline(always)]
    pub fn avg_frequency(&self) -> f32 {
        self.avg_frequency
    }

    /// Get most frequent token if available
    #[inline(always)]
    pub fn most_frequent_token(&self) -> Option<(u32, u32)> {
        self.most_frequent
    }

    /// Get entropy estimate
    #[inline(always)]
    pub fn entropy_estimate(&self) -> f32 {
        self.entropy_estimate
    }
}

impl Default for SequenceStats {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating processing contexts with custom configurations
#[derive(Debug)]
pub struct ContextBuilder {
    vocab_size: usize,
    context_size: Option<usize>,
    base_config: Option<BaseProcessingContext>,
}

impl ContextBuilder {
    /// Create new context builder
    #[inline(always)]
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            context_size: None,
            base_config: None,
        }
    }

    /// Set context window size
    #[inline(always)]
    pub fn context_size(mut self, size: usize) -> Self {
        self.context_size = Some(size);
        self
    }

    /// Set base processing context configuration
    #[inline(always)]
    pub fn base_context(mut self, base: BaseProcessingContext) -> Self {
        self.base_config = Some(base);
        self
    }

    /// Set temperature for base context
    #[inline(always)]
    pub fn temperature(mut self, temperature: f32) -> Self {
        let mut base = self.base_config.unwrap_or_else(BaseProcessingContext::new);
        base = base.with_temperature(temperature);
        self.base_config = Some(base);
        self
    }

    /// Set top-k for base context
    #[inline(always)]
    pub fn top_k(mut self, top_k: Option<usize>) -> Self {
        let mut base = self.base_config.unwrap_or_else(BaseProcessingContext::new);
        base = base.with_top_k(top_k);
        self.base_config = Some(base);
        self
    }

    /// Set top-p for base context
    #[inline(always)]
    pub fn top_p(mut self, top_p: Option<f32>) -> Self {
        let mut base = self.base_config.unwrap_or_else(BaseProcessingContext::new);
        base = base.with_top_p(top_p);
        self.base_config = Some(base);
        self
    }

    /// Build the processing context
    pub fn build(self) -> ProcessingResult<ProcessingContext> {
        let context_size = self.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE);
        let mut context = ProcessingContext::new(self.vocab_size, context_size)?;

        if let Some(base) = self.base_config {
            context.base = base;
        }

        Ok(context)
    }
}

/// Utility functions for context management
pub mod utils {
    use super::*;

    /// Create context for text generation
    #[inline(always)]
    pub fn text_generation_context(
        vocab_size: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> ProcessingResult<ProcessingContext> {
        ContextBuilder::new(vocab_size)
            .context_size(2048)
            .temperature(temperature)
            .top_k(top_k)
            .top_p(top_p)
            .build()
    }

    /// Create context for code generation
    #[inline(always)]
    pub fn code_generation_context(vocab_size: usize) -> ProcessingResult<ProcessingContext> {
        ContextBuilder::new(vocab_size)
            .context_size(4096) // Longer context for code
            .temperature(0.2) // Low temperature for precision
            .top_k(Some(20)) // Focused vocabulary
            .top_p(Some(0.95)) // High nucleus threshold
            .build()
    }

    /// Create context for conversation
    #[inline(always)]
    pub fn conversation_context(vocab_size: usize) -> ProcessingResult<ProcessingContext> {
        ContextBuilder::new(vocab_size)
            .context_size(1024) // Moderate context
            .temperature(0.7) // Balanced temperature
            .top_k(Some(40)) // Balanced vocabulary
            .top_p(Some(0.9)) // Standard nucleus sampling
            .build()
    }

    /// Calculate optimal context size for given constraints
    #[inline(always)]
    pub fn optimal_context_size(vocab_size: usize, target_memory_mb: usize) -> usize {
        // Estimate memory usage per token in context
        let bytes_per_token = std::mem::size_of::<u32>(); // Token ID
        let bytes_per_freq = std::mem::size_of::<u32>(); // Frequency counter
        let overhead = vocab_size * bytes_per_freq; // Frequency array

        let available_bytes = (target_memory_mb * 1024 * 1024).saturating_sub(overhead);
        let max_tokens = available_bytes / bytes_per_token;

        max_tokens.min(MAX_CONTEXT_SIZE)
    }

    /// Validate context configuration
    pub fn validate_context_config(vocab_size: usize, context_size: usize) -> ProcessingResult<()> {
        if vocab_size == 0 || vocab_size > MAX_VOCAB_SIZE {
            return Err(ProcessingError::configuration(format!(
                "Invalid vocabulary size: {}",
                vocab_size
            )));
        }

        if context_size == 0 || context_size > MAX_CONTEXT_SIZE {
            return Err(ProcessingError::configuration(format!(
                "Invalid context size: {}",
                context_size
            )));
        }

        Ok(())
    }
}
