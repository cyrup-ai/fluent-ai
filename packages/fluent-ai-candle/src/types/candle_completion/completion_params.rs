//! Completion parameter types

use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use super::constants::{MAX_TOKENS, TEMPERATURE_RANGE};
use crate::model::{ValidationError, ValidationResult};

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
    #[inline(always)]
    fn default() -> Self {
        // Use const instead of unwrap() for zero-allocation, no-panic guarantee
        const ONE: std::num::NonZeroU8 = match std::num::NonZeroU8::new(1) {
            Some(val) => val,
            None => unreachable!(), // Compile-time guarantee that 1 is non-zero
        };

        Self {
            temperature: 1.0,
            max_tokens: None,
            n: ONE,
            stream: false,
        }
    }
}

impl CompletionParams {
    /// Create new completion parameters with blazing-fast inline optimization
    #[inline(always)]
    pub const fn new() -> Self {
        const ONE: std::num::NonZeroU8 = match std::num::NonZeroU8::new(1) {
            Some(val) => val,
            None => unreachable!(), // Compile-time guarantee
        };

        Self {
            temperature: 1.0,
            max_tokens: None,
            n: ONE,
            stream: false,
        }
    }

    /// Set the temperature with zero-allocation validation
    #[inline(always)]
    pub fn with_temperature(mut self, temperature: f64) -> ValidationResult<Self> {
        if !TEMPERATURE_RANGE.contains(&temperature) {
            return Err(ValidationError::InvalidRange {
                parameter: "temperature".into(),
                actual: temperature.to_string(),
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

    /// Set the maximum number of tokens with elegant bounds checking
    #[inline(always)]
    pub fn with_max_tokens(mut self, max_tokens: Option<NonZeroU64>) -> Self {
        // Zero-allocation token limiting using min without temporary allocation
        self.max_tokens = max_tokens.and_then(|t| {
            let clamped_value = t.get().min(MAX_TOKENS);
            NonZeroU64::new(clamped_value)
        });
        self
    }

    /// Set number of completions with validation
    #[inline(always)]
    pub fn with_n(mut self, n: std::num::NonZeroU8) -> Self {
        self.n = n;
        self
    }

    /// Set streaming mode
    #[inline(always)]
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }
}
