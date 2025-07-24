//! Completion response with lifetime-based zero-copy optimization

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// A response from a text completion request with zero-allocation design
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse<'a> {
    /// Unique identifier for this completion (optional, for compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Object type identifier (optional, for compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    /// Unix timestamp of when the completion was created (optional, for compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
    /// The generated completion text
    pub text: Cow<'a, str>,
    /// The model that generated the completion
    pub model: Cow<'a, str>,
    /// The provider that generated the completion (optional)
    pub provider: Option<Cow<'a, str>>,
    /// Token usage information (optional)
    pub usage: Option<crate::types::CandleUsage>,
    /// The reason the completion finished (optional)
    pub finish_reason: Option<Cow<'a, str>>,
    /// Response time in milliseconds (optional)
    pub response_time_ms: Option<u64>,
    /// Generation time in milliseconds for performance tracking (optional)
    pub generation_time_ms: Option<u32>,
    /// Tokens per second throughput for performance tracking (optional)
    pub tokens_per_second: Option<f64>,
}

impl<'a> Default for CompletionResponse<'a> {
    fn default() -> Self {
        Self {
            id: Some("default_response".into()),    // Required field
            object: Some("text_completion".into()), // Standard object type
            created: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ), // Current timestamp
            text: Cow::Borrowed(""),
            model: Cow::Borrowed(""),
            provider: None,
            usage: None,
            finish_reason: None,
            response_time_ms: None,
            generation_time_ms: None,
            tokens_per_second: None,
        }
    }
}

/// Builder for `CompletionResponse` with blazing-fast inline optimization
pub struct CompletionResponseBuilder<'a> {
    inner: CompletionResponse<'a>,
}

impl<'a> CompletionResponse<'a> {
    /// Create a new builder for a completion response
    #[inline(always)]
    pub fn builder() -> CompletionResponseBuilder<'static> {
        CompletionResponseBuilder::new()
    }

    /// Get the completion text with zero-allocation access
    #[inline(always)]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get the model name with zero-allocation access
    #[inline(always)]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get the provider name if available
    #[inline(always)]
    pub fn provider(&self) -> Option<&str> {
        self.provider.as_deref()
    }

    /// Get the finish reason if available
    #[inline(always)]
    pub fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }

    /// Get the response time in milliseconds if available
    #[inline(always)]
    pub fn response_time_ms(&self) -> Option<u64> {
        self.response_time_ms
    }

    /// Get the token usage if available
    #[inline(always)]
    pub fn usage(&self) -> Option<&crate::types::CandleUsage> {
        self.usage.as_ref()
    }

    /// Get the generation time in milliseconds if available
    #[inline(always)]
    pub fn generation_time_ms(&self) -> Option<u32> {
        self.generation_time_ms
    }

    /// Get the tokens per second throughput if available
    #[inline(always)]
    pub fn tokens_per_second(&self) -> Option<f64> {
        self.tokens_per_second
    }

    /// Set the generation time in milliseconds for performance tracking
    #[inline(always)]
    pub fn set_generation_time_ms(&mut self, ms: u32) {
        self.generation_time_ms = Some(ms);
    }

    /// Set the tokens per second throughput for performance tracking
    #[inline(always)]
    pub fn set_tokens_per_second(&mut self, tps: f64) {
        self.tokens_per_second = Some(tps);
    }

    /// Get the number of tokens generated (completion tokens) if available
    #[inline(always)]
    pub fn tokens_generated(&self) -> Option<u32> {
        self.usage.as_ref().map(|u| u.completion_tokens)
    }
}

impl<'a> CompletionResponseBuilder<'a> {
    /// Create a new builder with default values - blazing-fast initialization
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            inner: CompletionResponse {
                id: Some("builder_response".into()),
                object: Some("text_completion".into()),
                created: Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                ),
                text: Cow::Borrowed(""),
                model: Cow::Borrowed(""),
                provider: None,
                usage: None,
                finish_reason: None,
                response_time_ms: None,
                generation_time_ms: None,
                tokens_per_second: None,
            },
        }
    }

    /// Set the completion text with zero-allocation when possible
    #[inline(always)]
    pub fn text<S: Into<Cow<'a, str>>>(mut self, text: S) -> Self {
        self.inner.text = text.into();
        self
    }

    /// Set the model name with zero-allocation when possible
    #[inline(always)]
    pub fn model<S: Into<Cow<'a, str>>>(mut self, model: S) -> Self {
        self.inner.model = model.into();
        self
    }

    /// Set the provider name
    #[inline(always)]
    pub fn provider<S: Into<Cow<'a, str>>>(mut self, provider: S) -> Self {
        self.inner.provider = Some(provider.into());
        self
    }

    /// Set the token usage
    #[inline(always)]
    pub fn usage(mut self, usage: crate::types::CandleUsage) -> Self {
        self.inner.usage = Some(usage);
        self
    }

    /// Set the finish reason
    #[inline(always)]
    pub fn finish_reason<S: Into<Cow<'a, str>>>(mut self, reason: S) -> Self {
        self.inner.finish_reason = Some(reason.into());
        self
    }

    /// Set the response time in milliseconds
    #[inline(always)]
    pub fn response_time_ms(mut self, ms: u64) -> Self {
        self.inner.response_time_ms = Some(ms);
        self
    }

    /// Set the generation time in milliseconds
    #[inline(always)]
    pub fn generation_time_ms(mut self, ms: u32) -> Self {
        self.inner.generation_time_ms = Some(ms);
        self
    }

    /// Set the tokens per second throughput
    #[inline(always)]
    pub fn tokens_per_second(mut self, tps: f64) -> Self {
        self.inner.tokens_per_second = Some(tps);
        self
    }

    /// Set the number of tokens generated (completion tokens) with intelligent usage creation
    #[inline(always)]
    pub fn tokens_generated(mut self, tokens: u32) -> Self {
        match &mut self.inner.usage {
            Some(usage) => {
                usage.completion_tokens = tokens;
                usage.total_tokens = usage.prompt_tokens + tokens;
            }
            None => {
                self.inner.usage = Some(crate::types::CandleUsage {
                    prompt_tokens: 0,
                    completion_tokens: tokens,
                    total_tokens: tokens,
                });
            }
        }
        self
    }

    /// Build the completion response with zero-allocation optimization
    #[inline(always)]
    pub fn build(self) -> CompletionResponse<'a> {
        self.inner
    }
}

impl<'a> Default for CompletionResponseBuilder<'a> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}
