//! Sequence statistics and repetition penalty types
//!
//! Provides statistical analysis of token sequences and penalty calculation
//! strategies for advanced text generation control.

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
    pub(super) fn add_token(&mut self, _token_id: u32) {
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