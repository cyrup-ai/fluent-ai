//! Shared types and constants for completion functionality
//!
//! Contains common types, parameters, and constants used across completion modules.

use std::num::NonZeroU64;
use std::ops::RangeInclusive;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::validation::{ValidationError, ValidationResult};

/// Temperature range for generation (0.0 to 2.0)
const TEMPERATURE_RANGE: RangeInclusive<f64> = 0.0..=2.0;
/// Maximum tokens for a single completion
const MAX_TOKENS: u64 = 8192;
/// Maximum chunk size for streaming
const MAX_CHUNK_SIZE: usize = 4096;

/// Parameters for completion generation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompletionParams {
    /// Sampling temperature (0.0 to 2.0)
    pub temperature: f64,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<NonZeroU64>,
    /// Number of completions to generate
    pub n: std::num::NonZeroU8,
    /// Whether to stream the response
    pub stream: bool,
}

impl Default for CompletionParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            max_tokens: None,
            n: std::num::NonZeroU8::new(1).unwrap(),
            stream: false,
        }
    }
}

impl CompletionParams {
    /// Create new completion parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f64) -> ValidationResult<Self> {
        if !TEMPERATURE_RANGE.contains(&temperature) {
            return Err(ValidationError::InvalidRange {
                field: "temperature".into(),
                value: temperature.to_string(),
                expected: format!(
                    "between {:.1} and {:.1}",
                    TEMPERATURE_RANGE.start(),
                    TEMPERATURE_RANGE.end()
                ),
            });
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Set the maximum number of tokens
    pub fn with_max_tokens(mut self, max_tokens: Option<NonZeroU64>) -> Self {
        self.max_tokens = max_tokens.and_then(|t| NonZeroU64::new(t.get().min(MAX_TOKENS)));
        self
    }
}

/// Tool definition for completion requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Model-specific parameters for completion requests
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
