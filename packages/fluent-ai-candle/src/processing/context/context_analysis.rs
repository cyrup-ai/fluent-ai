//! Context analysis methods for advanced processing strategies
//!
//! Provides analysis capabilities for ProcessingContext including
//! repetition detection, diversity scoring, and pattern analysis.

use super::context_core::ProcessingContext;
use super::context_stats::RepetitionPenaltyType;

impl ProcessingContext {
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
            RepetitionPenaltyType::SquareRoot => (frequency as f32).sqrt()}
    }

    /// Get context utilization ratio
    ///
    /// Returns the ratio of used context to maximum context size.
    /// Useful for adaptive sampling strategies.
    #[inline(always)]
    pub fn utilization_ratio(&self) -> f32 {
        self.token_history().len() as f32 / self.context_size() as f32
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
        self.token_frequencies()
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
        if self.token_history().is_empty() {
            return 0.0;
        }

        let unique_count = self.unique_token_count();
        let total_count = self.token_history().len();

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
        let history = self.token_history();
        let start = history.len().saturating_sub(n);
        &history[start..]
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
        let history = self.token_history();
        
        if pattern_length < 2 || pattern_length > 8 || history.len() < pattern_length * 2 {
            return false;
        }

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