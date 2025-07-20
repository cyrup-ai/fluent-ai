//! Completion response types and builders
//!
//! Contains response structures and builder patterns for completion functionality.

use std::borrow::Cow;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::usage::Usage;

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
}

/// Builder for `CompletionResponse`
pub struct CompletionResponseBuilder<'a> {
    inner: CompletionResponse<'a>,
}

impl<'a> CompletionResponse<'a> {
    /// Create a new builder for a completion response
    pub fn builder() -> CompletionResponseBuilder<'static> {
        CompletionResponseBuilder::new()
    }

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
}

impl<'a> CompletionResponseBuilder<'a> {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            inner: CompletionResponse {
                text: Cow::Borrowed(""),
                model: Cow::Borrowed(""),
                provider: None,
                usage: None,
                finish_reason: None,
                response_time_ms: None,
            },
        }
    }

    /// Set the completion text
    pub fn text<S: Into<Cow<'a, str>>>(mut self, text: S) -> Self {
        self.inner.text = text.into();
        self
    }

    /// Set the model name
    pub fn model<S: Into<Cow<'a, str>>>(mut self, model: S) -> Self {
        self.inner.model = model.into();
        self
    }

    /// Set the provider name
    pub fn provider<S: Into<Cow<'a, str>>>(mut self, provider: S) -> Self {
        self.inner.provider = Some(provider.into());
        self
    }

    /// Set the token usage
    pub fn usage(mut self, usage: Usage) -> Self {
        self.inner.usage = Some(usage);
        self
    }

    /// Set the finish reason
    pub fn finish_reason<S: Into<Cow<'a, str>>>(mut self, reason: S) -> Self {
        self.inner.finish_reason = Some(reason.into());
        self
    }

    /// Set the response time in milliseconds
    pub fn response_time_ms(mut self, ms: u64) -> Self {
        self.inner.response_time_ms = Some(ms);
        self
    }

    /// Build the completion response
    pub fn build(self) -> CompletionResponse<'a> {
        self.inner
    }
}

/// A more compact representation of a completion response using Arcs for shared ownership
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactCompletionResponse<'a> {
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

    #[serde(skip)]
    _marker: std::marker::PhantomData<&'a ()>,
}

/// Builder for `CompactCompletionResponse`
pub struct CompactCompletionResponseBuilder<'a> {
    content: Option<Arc<str>>,
    model: Option<Arc<str>>,
    provider: Option<Arc<str>>,
    tokens_used: u32,
    finish_reason: Option<Arc<str>>,
    response_time_ms: u64,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> CompactCompletionResponse<'a> {
    /// Create a new builder for a compact completion response
    pub fn builder() -> CompactCompletionResponseBuilder<'static> {
        CompactCompletionResponseBuilder::new()
    }

    /// Convert back to a standard CompletionResponse
    pub fn into_standard(self) -> CompletionResponse<'static> {
        CompletionResponse {
            text: Cow::Owned((*self.content).to_owned()),
            model: Cow::Owned((*self.model).to_owned()),
            provider: Some(Cow::Owned((*self.provider).to_owned())),
            usage: Some(Usage {
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
        }
    }
}

impl<'a> CompactCompletionResponseBuilder<'a> {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            content: None,
            model: None,
            provider: None,
            tokens_used: 0,
            finish_reason: None,
            response_time_ms: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the content
    pub fn content(mut self, content: impl Into<Arc<str>>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Set the model
    pub fn model(mut self, model: impl Into<Arc<str>>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the provider
    pub fn provider(mut self, provider: impl Into<Arc<str>>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set tokens used
    pub fn tokens_used(mut self, tokens: u32) -> Self {
        self.tokens_used = tokens;
        self
    }

    /// Set finish reason
    pub fn finish_reason(mut self, reason: impl Into<Arc<str>>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Set response time
    pub fn response_time_ms(mut self, ms: u64) -> Self {
        self.response_time_ms = ms;
        self
    }

    /// Build the compact response
    pub fn build(self) -> Result<CompactCompletionResponse<'a>, &'static str> {
        Ok(CompactCompletionResponse {
            content: self.content.ok_or("content is required")?,
            model: self.model.ok_or("model is required")?,
            provider: self.provider.ok_or("provider is required")?,
            tokens_used: self.tokens_used,
            finish_reason: self.finish_reason.ok_or("finish_reason is required")?,
            response_time_ms: self.response_time_ms,
            _marker: std::marker::PhantomData,
        })
    }
}
