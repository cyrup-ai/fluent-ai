//! Completion response types and builders
//!
//! Contains response structures and builder patterns for completion functionality.

use std::borrow::Cow;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::domain::model::usage::Usage;

/// A response from a text completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse<'a> {
    /// The generated completion text
    pub text: Cow<'a, str>,
    /// The model that generated the completion
    pub model: Cow<'a, str>,
    /// The provider that generated the completion (optional)
    pub provider: Option<Cow<'a, str>>,
    /// Token usage information (optional)
    pub usage: Option<Usage>,
    /// The reason the completion finished (optional)
    pub finish_reason: Option<Cow<'a, str>>,
    /// Response time in milliseconds (optional)
    pub response_time_ms: Option<u64>,
    /// Generation time in milliseconds for performance tracking (optional)
    pub generation_time_ms: Option<u32>,
    /// Tokens per second throughput for performance tracking (optional)
    pub tokens_per_second: Option<f64>,
}

impl<'a> CompletionResponse<'a> {
    /// Get the completion text
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get the model name
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get the provider name if available
    pub fn provider(&self) -> Option<&str> {
        self.provider.as_deref()
    }

    /// Get the finish reason if available
    pub fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }

    /// Get the response time in milliseconds if available
    pub fn response_time_ms(&self) -> Option<u64> {
        self.response_time_ms
    }

    /// Get the token usage if available
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// Get the generation time in milliseconds if available
    pub fn generation_time_ms(&self) -> Option<u32> {
        self.generation_time_ms
    }

    /// Get the tokens per second throughput if available
    pub fn tokens_per_second(&self) -> Option<f64> {
        self.tokens_per_second
    }

    /// Set the generation time in milliseconds for performance tracking
    pub fn set_generation_time_ms(&mut self, ms: u32) {
        self.generation_time_ms = Some(ms);
    }

    /// Set the tokens per second throughput for performance tracking
    pub fn set_tokens_per_second(&mut self, tps: f64) {
        self.tokens_per_second = Some(tps);
    }

    /// Get the number of tokens generated (output tokens) if available
    pub fn tokens_generated(&self) -> Option<u32> {
        self.usage.as_ref().map(|u| u.output_tokens)
    }
}

/// A more compact representation of a completion response using Arcs for shared ownership
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactCompletionResponse {
    /// The generated completion text
    pub content: Arc<str>,
    /// The model that generated the completion
    pub model: Arc<str>,
    /// The provider that generated the completion
    pub provider: Arc<str>,
    /// Total tokens used in the completion
    pub tokens_used: u32,
    /// The reason the completion finished
    pub finish_reason: Arc<str>,
    /// Response time in milliseconds
    pub response_time_ms: u64,
}

impl CompactCompletionResponse {
    /// Convert back to a standard CompletionResponse
    pub fn into_standard(self) -> CompletionResponse<'static> {
        CompletionResponse {
            text: Cow::Owned((*self.content).to_owned()),
            model: Cow::Owned((*self.model).to_owned()),
            provider: Some(Cow::Owned((*self.provider).to_owned())),
            usage: Some(Usage {
                total_tokens: self.tokens_used,
                input_tokens: 0,  // Not available in compact form
                output_tokens: 0, // Not available in compact form
            }),
            finish_reason: Some(Cow::Owned((*self.finish_reason).to_owned())),
            response_time_ms: if self.response_time_ms > 0 {
                Some(self.response_time_ms)
            } else {
                None
            },
            generation_time_ms: None, // Not available in compact form
            tokens_per_second: None,  // Not available in compact form
        }
    }
}
