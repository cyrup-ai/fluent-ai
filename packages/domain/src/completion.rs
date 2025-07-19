//! Completion domain types
//!
//! Contains pure data structures and traits for completion.
//! Builder implementations are in fluent_ai package.

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::ops::RangeInclusive;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::chunk::CompletionChunk;
use crate::prompt::Prompt;
use crate::usage::Usage;
use crate::validation::{ValidationError, ValidationResult};
use crate::{Models, ZeroOneOrMany};

/// Temperature range for generation (0.0 to 2.0)
const TEMPERATURE_RANGE: RangeInclusive<f64> = 0.0..=2.0;
/// Maximum tokens for a single completion
const MAX_TOKENS: u64 = 8192;
/// Maximum chunk size for streaming
const MAX_CHUNK_SIZE: usize = 4096;

/// Core trait for completion models
pub trait CompletionModel: Send + Sync + 'static {
    /// Generate completion from prompt
    /// 
    /// # Arguments
    /// * `prompt` - The input prompt for generation
    /// * `params` - Generation parameters
    /// 
    /// # Returns
    /// Stream of completion chunks
    fn prompt<'a>(
        &'a self, 
        prompt: Prompt<'a>,
        params: &'a CompletionParams
    ) -> crate::async_task::AsyncStream<CompletionChunk<'a>>;
}

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
                expected: format!("between {:.1} and {:.1}", 
                    TEMPERATURE_RANGE.start(), 
                    TEMPERATURE_RANGE.end()),
            });
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Set the maximum number of tokens
    pub fn with_max_tokens(mut self, max_tokens: Option<NonZeroU64>) -> Self {
        self.max_tokens = max_tokens.and_then(|t| {
            NonZeroU64::new(t.get().min(MAX_TOKENS))
        });
        self
    }
}

/// Backend for completion processing
pub trait CompletionBackend: Send + Sync + 'static {
    /// Submit a completion request
    /// 
    /// # Arguments
    /// * `request` - The completion request
    /// 
    /// # Returns
    /// Async task that resolves to the completion result
    fn submit_completion<'a>(
        &'a self,
        request: CompletionRequest<'a>,
    ) -> crate::async_task::AsyncTask<CompletionResponse<'a>>;
}

/// A request for text completion
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'static"))]
pub struct CompletionRequest<'a> {
    /// System prompt providing instructions
    pub system_prompt: Cow<'a, str>,
    /// Conversation history
    pub chat_history: ZeroOneOrMany<crate::Message>,
    /// Documents to use as context
    pub documents: ZeroOneOrMany<crate::Document>,
    /// Tools available to the model
    pub tools: ZeroOneOrMany<ToolDefinition>,
    /// Sampling temperature (0.0 to 2.0)
    pub temperature: f64,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<NonZeroU64>,
    /// Size of chunks for streaming
    pub chunk_size: Option<usize>,
    /// Additional provider-specific parameters
    pub additional_params: Option<&'a Value>,
}

/// Builder for `CompletionRequest`
pub struct CompletionRequestBuilder<'a> {
    inner: CompletionRequest<'a>,
}

/// Error type for completion request validation
#[derive(Debug, Error)]
pub enum CompletionRequestError {
    /// Invalid parameter value
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// Validation error
    #[error(transparent)]
    Validation(#[from] ValidationError),
}

impl<'a> CompletionRequest<'a> {
    /// Create a new builder with required fields
    pub fn builder() -> CompletionRequestBuilder<'static> {
        CompletionRequestBuilder::new()
    }
    
    /// Validate the request parameters
    pub fn validate(&self) -> ValidationResult<()> {
        if !TEMPERATURE_RANGE.contains(&self.temperature) {
            return Err(ValidationError::InvalidRange {
                field: "temperature".into(),
                value: self.temperature.to_string(),
                expected: format!("between {:.1} and {:.1}", 
                    TEMPERATURE_RANGE.start(), 
                    TEMPERATURE_RANGE.end()),
            });
        }
        
        if let Some(max_tokens) = self.max_tokens {
            if max_tokens.get() > MAX_TOKENS {
                return Err(ValidationError::InvalidRange {
                    field: "max_tokens".into(),
                    value: max_tokens.get().to_string(),
                    expected: format!("up to {}", MAX_TOKENS),
                });
            }
        }
        
        if let Some(chunk_size) = self.chunk_size {
            if chunk_size > MAX_CHUNK_SIZE {
                return Err(ValidationError::InvalidRange {
                    field: "chunk_size".into(),
                    value: chunk_size.to_string(),
                    expected: format!("up to {}", MAX_CHUNK_SIZE),
                });
            }
        }
        
        Ok(())
    }
}

impl<'a> CompletionRequestBuilder<'a> {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            inner: CompletionRequest {
                system_prompt: Cow::Borrowed(""),
                chat_history: ZeroOneOrMany::None,
                documents: ZeroOneOrMany::None,
                tools: ZeroOneOrMany::None,
                temperature: 1.0,
                max_tokens: None,
                chunk_size: None,
                additional_params: None,
            },
        }
    }
    
    /// Set the system prompt
    pub fn system_prompt<S: Into<Cow<'a, str>>>(mut self, prompt: S) -> Self {
        self.inner.system_prompt = prompt.into();
        self
    }
    
    /// Set the chat history
    pub fn chat_history(mut self, history: ZeroOneOrMany<crate::Message>) -> Self {
        self.inner.chat_history = history;
        self
    }
    
    /// Set the documents
    pub fn documents(mut self, docs: ZeroOneOrMany<crate::Document>) -> Self {
        self.inner.documents = docs;
        self
    }
    
    /// Set the tools
    pub fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> Self {
        self.inner.tools = tools;
        self
    }
    
    /// Set the temperature
    pub fn temperature(mut self, temp: f64) -> ValidationResult<Self> {
        if !TEMPERATURE_RANGE.contains(&temp) {
            return Err(ValidationError::InvalidRange {
                field: "temperature".into(),
                value: temp.to_string(),
                expected: format!("between {:.1} and {:.1}", 
                    TEMPERATURE_RANGE.start(), 
                    TEMPERATURE_RANGE.end()),
            });
        }
        self.inner.temperature = temp;
        Ok(self)
    }
    
    /// Set the maximum number of tokens
    pub fn max_tokens(mut self, max_tokens: Option<NonZeroU64>) -> Self {
        self.inner.max_tokens = max_tokens.and_then(|t| {
            NonZeroU64::new(t.get().min(MAX_TOKENS))
        });
        self
    }
    
    /// Set the chunk size for streaming
    pub fn chunk_size(mut self, size: Option<usize>) -> ValidationResult<Self> {
        if let Some(size) = size {
            if size > MAX_CHUNK_SIZE {
                return Err(ValidationError::InvalidRange {
                    field: "chunk_size".into(),
                    value: size.to_string(),
                    expected: format!("up to {}", MAX_CHUNK_SIZE),
                });
            }
            self.inner.chunk_size = Some(size);
        } else {
            self.inner.chunk_size = None;
        }
        Ok(self)
    }
    
    /// Set additional parameters
    pub fn additional_params(mut self, params: Option<&'a Value>) -> Self {
        self.inner.additional_params = params;
        self
    }
    
    /// Build the request
    pub fn build(self) -> Result<CompletionRequest<'a>, CompletionRequestError> {
        self.inner.validate()
            .map_err(CompletionRequestError::from)?;
        Ok(self.inner)
    }
}

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

use std::sync::Arc;

/// Standard response format for completion operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse<'a> {
    /// The generated completion text
    pub text: Cow<'a, str>,
    /// The model that generated the completion
    pub model: Cow<'a, str>,
    /// The provider that generated the completion (e.g., "openai", "anthropic")
    pub provider: Option<Cow<'a, str>>,
    /// Token usage statistics, if available
    pub usage: Option<Usage>,
    /// The reason the completion finished (e.g., "stop", "length", "content_filter")
    pub finish_reason: Option<Cow<'a, str>>,
    /// Response time in milliseconds, if available
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
    
    /// Create a new completion response with minimal required fields
    pub fn new(
        text: impl Into<Cow<'a, str>>, 
        model: impl Into<Cow<'a, str>>
    ) -> Self {
        Self {
            text: text.into(),
            model: model.into(),
            provider: None,
            usage: None,
            finish_reason: None,
            response_time_ms: None,
        }
    }

    /// Create a new completion response with usage statistics
    pub fn with_usage(
        text: impl Into<Cow<'a, str>>,
        model: impl Into<Cow<'a, str>>,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> Self {
        Self {
            text: text.into(),
            model: model.into(),
            provider: None,
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
            finish_reason: None,
            response_time_ms: None,
        }
    }

    /// Convert to a more compact representation with Arcs for shared ownership
    pub fn into_compact(self) -> CompactCompletionResponse<'a> {
        CompactCompletionResponse {
            content: self.text.into_owned().into(),
            model: self.model.into_owned().into(),
            provider: self.provider
                .map(|p| p.into_owned().into())
                .unwrap_or_else(|| Arc::from("unknown")),
            tokens_used: self.usage.map(|u| u.total_tokens).unwrap_or(0),
            finish_reason: self.finish_reason
                .map(|r| r.into_owned().into())
                .unwrap_or_else(|| Arc::from("unknown")),
            response_time_ms: self.response_time_ms.unwrap_or(0),
            _marker: std::marker::PhantomData,
        }
    }
    
    /// Get the text content as a string slice
    pub fn text(&self) -> &str {
        &self.text
    }
    
    /// Get the model name as a string slice
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
                prompt_tokens: 0, // Not available in compact form
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
