//! Token usage tracking for API operations
//!
//! Provides a zero-allocation, thread-safe way to track token usage across
//! different operations in the system.

use serde::{Deserialize, Serialize};

/// Tracks token usage statistics for API operations
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[must_use = "Token usage should be handled or logged"]
#[repr(C)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total tokens used (prompt + completion)
    pub total_tokens: u32,
}

impl Usage {
    /// Creates a new `Usage` instance with the given token counts
    #[inline]
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    /// Creates a new `Usage` instance with zero tokens
    #[inline]
    pub const fn zero() -> Self {
        Self::new(0, 0)
    }

    /// Adds another `Usage` to this one, returning a new instance
    #[inline]
    #[must_use = "Returns a new Usage instance"]
    pub const fn add(&self, other: &Self) -> Self {
        let prompt = self.prompt_tokens + other.prompt_tokens;
        let completion = self.completion_tokens + other.completion_tokens;
        Self::new(prompt, completion)
    }

    /// Returns whether no tokens have been used
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.prompt_tokens == 0 && self.completion_tokens == 0
    }
}

impl std::ops::Add for Usage {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let prompt = self.prompt_tokens + rhs.prompt_tokens;
        let completion = self.completion_tokens + rhs.completion_tokens;
        Self::new(prompt, completion)
    }
}

impl std::ops::AddAssign for Usage {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(&rhs);
    }
}