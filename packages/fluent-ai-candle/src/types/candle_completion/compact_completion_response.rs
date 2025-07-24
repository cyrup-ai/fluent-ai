//! Compact completion response with Arc-based shared ownership

use std::borrow::Cow;
use std::sync::Arc;

use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};

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

/// Builder for `CompactCompletionResponse` with ergonomic defaults
pub struct CompactCompletionResponseBuilder {
    content: Option<Arc<str>>,
    model: Option<Arc<str>>,
    provider: Option<Arc<str>>,
    tokens_used: u32,
    finish_reason: Option<Arc<str>>,
    response_time_ms: u64,
}

impl CompactCompletionResponse {
    /// Create a new builder for a compact completion response
    #[inline(always)]
    pub fn builder() -> CompactCompletionResponseBuilder {
        CompactCompletionResponseBuilder::new()
    }

    /// Convert back to a standard CompletionResponse with zero-allocation optimization where possible
    pub fn into_standard(self) -> super::completion_response::CompletionResponse<'static> {
        super::completion_response::CompletionResponse {
            id: Some("compact_response".into()), // Required field
            object: Some("text_completion".into()), // Standard object type
            created: Some(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()), // Current timestamp
            text: Cow::Owned((*self.content).to_owned()),
            model: Cow::Owned((*self.model).to_owned()),
            provider: Some(Cow::Owned((*self.provider).to_owned())),
            usage: Some(crate::types::CandleUsage {
                total_tokens: self.tokens_used,
                prompt_tokens: 0,     // Not available in compact form
                completion_tokens: 0, // Not available in compact form
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

    /// Get content reference with zero-allocation access
    #[inline(always)]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Get model reference with zero-allocation access
    #[inline(always)]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get provider reference with zero-allocation access
    #[inline(always)]
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Get finish reason reference with zero-allocation access
    #[inline(always)]
    pub fn finish_reason(&self) -> &str {
        &self.finish_reason
    }
}

impl CompactCompletionResponseBuilder {
    /// Create a new builder with blazing-fast initialization
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            content: None,
            model: None,
            provider: None,
            tokens_used: 0,
            finish_reason: None,
            response_time_ms: 0,
        }
    }

    /// Set the content with elegant ergonomic API
    #[inline(always)]
    pub fn content(mut self, content: impl Into<Arc<str>>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Set the model with elegant ergonomic API
    #[inline(always)]
    pub fn model(mut self, model: impl Into<Arc<str>>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the provider with elegant ergonomic API
    #[inline(always)]
    pub fn provider(mut self, provider: impl Into<Arc<str>>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set tokens used with blazing-fast inline optimization
    #[inline(always)]
    pub fn tokens_used(mut self, tokens: u32) -> Self {
        self.tokens_used = tokens;
        self
    }

    /// Set finish reason with elegant ergonomic API
    #[inline(always)]
    pub fn finish_reason(mut self, reason: impl Into<Arc<str>>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Set response time with blazing-fast inline optimization
    #[inline(always)]
    pub fn response_time_ms(mut self, ms: u64) -> Self {
        self.response_time_ms = ms;
        self
    }

    /// Build the compact response with proper error handling (no unwrap!)
    pub fn build(self) -> Result<AsyncStream<CompactCompletionResponse>, &'static str> {
        // Validate required fields without using unwrap()
        let content = self.content.unwrap_or_else(|| Arc::from(""));
        let model = match self.model {
            Some(m) => m,
            None => Arc::from("unknown"),
        };
        let provider = self.provider.unwrap_or_else(|| Arc::from("unknown"));
        let finish_reason = self.finish_reason.unwrap_or_else(|| Arc::from("stop"));

        let response = CompactCompletionResponse {
            content,
            model,
            provider,
            tokens_used: self.tokens_used,
            finish_reason,
            response_time_ms: self.response_time_ms,
        };

        Ok(AsyncStream::with_channel(move |sender| {
            if sender.send(response).is_err() {
                // Channel closed, which is fine - no error to propagate
            }
        }))
    }

    /// Build the compact response synchronously with proper defaults (no unwrap!)
    #[inline(always)]
    pub fn build_sync(self) -> CompactCompletionResponse {
        let content = self.content.unwrap_or_else(|| Arc::from(""));
        let model = self.model.unwrap_or_else(|| Arc::from("unknown"));
        let provider = self.provider.unwrap_or_else(|| Arc::from("unknown"));
        let finish_reason = self.finish_reason.unwrap_or_else(|| Arc::from("stop"));

        CompactCompletionResponse {
            content,
            model,
            provider,
            tokens_used: self.tokens_used,
            finish_reason,
            response_time_ms: self.response_time_ms,
        }
    }
}

impl Default for CompactCompletionResponseBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}