//! Model-specific parameters for completion requests

use serde::{Deserialize, Serialize};

/// Model-specific parameters for completion requests with zero-allocation design
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelParams {
    /// RoPE theta parameter for positional encoding
    pub rope_theta: f32,
    /// RoPE frequency base for positional encoding  
    pub rope_freq_base: f32,
    /// Context window size
    pub context_length: u32,
    /// Vocabulary size
    pub vocab_size: u32,
}

impl Default for ModelParams {
    /// Default model parameters optimized for common use cases
    #[inline(always)]
    fn default() -> Self {
        Self {
            rope_theta: 10000.0,
            rope_freq_base: 1.0,
            context_length: 2048,
            vocab_size: 32000,
        }
    }
}

impl ModelParams {
    /// Create new model parameters with blazing-fast const initialization
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            rope_theta: 10000.0,
            rope_freq_base: 1.0,
            context_length: 2048,
            vocab_size: 32000,
        }
    }

    /// Create with custom rope theta for advanced positional encoding
    #[inline(always)]
    pub const fn with_rope_theta(mut self, theta: f32) -> Self {
        self.rope_theta = theta;
        self
    }

    /// Create with custom rope frequency base
    #[inline(always)]
    pub const fn with_rope_freq_base(mut self, freq_base: f32) -> Self {
        self.rope_freq_base = freq_base;
        self
    }

    /// Create with custom context length
    #[inline(always)]
    pub const fn with_context_length(mut self, length: u32) -> Self {
        self.context_length = length;
        self
    }

    /// Create with custom vocabulary size
    #[inline(always)]
    pub const fn with_vocab_size(mut self, size: u32) -> Self {
        self.vocab_size = size;
        self
    }

    /// Check if parameters are valid for inference
    #[inline(always)]
    pub const fn is_valid(&self) -> bool {
        self.rope_theta > 0.0
            && self.rope_freq_base > 0.0
            && self.context_length > 0
            && self.vocab_size > 0
    }
}
