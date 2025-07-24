//! Completion request types and builders
//!
//! Contains request structures and builder patterns for completion functionality.

// Removed unused import: std::borrow::Cow
use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use super::constants::{MAX_CHUNK_SIZE, MAX_TOKENS, TEMPERATURE_RANGE};
use super::tool_definition::ToolDefinition;
use crate::model::{ValidationError, ValidationResult};
use crate::types::CandleDocument;
use crate::types::ZeroOneOrMany;
use crate::types::candle_chat::message::CandleMessage;

/// A request for text completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleCompletionRequest {
    /// System prompt providing instructions
    pub system_prompt: String,
    /// Conversation history
    pub chat_history: ZeroOneOrMany<CandleMessage>,
    /// Documents to use as context
    pub documents: ZeroOneOrMany<CandleDocument>,
    /// Tools available to the model
    pub tools: ZeroOneOrMany<ToolDefinition>,
    /// Sampling temperature (0.0 to 2.0)
    pub temperature: f64,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<NonZeroU64>,
    /// Size of chunks for streaming
    pub chunk_size: Option<usize>,
    /// Additional provider-specific parameters
    pub additional_params: Option<Value>,
}

/// Builder for `CompletionRequest`
#[derive(Clone)]
pub struct CompletionRequestBuilder {
    system_prompt: String,
    chat_history: ZeroOneOrMany<CandleMessage>,
    documents: ZeroOneOrMany<CandleDocument>,
    tools: ZeroOneOrMany<ToolDefinition>,
    temperature: f64,
    max_tokens: Option<NonZeroU64>,
    chunk_size: Option<usize>,
    additional_params: Option<Value>,
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

impl CandleCompletionRequest {
    /// Create a new builder with required fields
    pub fn builder() -> CompletionRequestBuilder {
        CompletionRequestBuilder::new()
    }

    /// Validate the request parameters
    pub fn validate(&self) -> ValidationResult<()> {
        // Validate temperature
        if !TEMPERATURE_RANGE.contains(&self.temperature) {
            return Err(ValidationError::InvalidRange {
                parameter: "temperature".into(),
                actual: self.temperature.to_string(),
                expected: format!(
                    "between {:.1} and {:.1}",
                    TEMPERATURE_RANGE.start(),
                    TEMPERATURE_RANGE.end()
                ),
            });
        }

        // Validate max_tokens
        if let Some(max_tokens) = self.max_tokens {
            if max_tokens.get() > MAX_TOKENS {
                return Err(ValidationError::InvalidRange {
                    parameter: "max_tokens".into(),
                    actual: max_tokens.to_string(),
                    expected: format!("less than or equal to {}", MAX_TOKENS),
                });
            }
        }

        // Validate chunk_size
        if let Some(chunk_size) = self.chunk_size {
            if chunk_size == 0 || chunk_size > MAX_CHUNK_SIZE {
                return Err(ValidationError::InvalidRange {
                    parameter: "chunk_size".into(),
                    actual: chunk_size.to_string(),
                    expected: format!("between 1 and {}", MAX_CHUNK_SIZE),
                });
            }
        }

        Ok(())
    }

    /// Convert to a static lifetime version by making all borrowed data owned
    #[inline]
    pub fn into_static(self) -> CandleCompletionRequest {
        CandleCompletionRequest {
            system_prompt: self.system_prompt,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            additional_params: self.additional_params,
        }
    }
}

impl CompletionRequestBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            system_prompt: String::new(),
            chat_history: ZeroOneOrMany::None,
            documents: ZeroOneOrMany::None,
            tools: ZeroOneOrMany::None,
            temperature: 1.0,
            max_tokens: None,
            chunk_size: None,
            additional_params: None,
        }
    }

    /// Set the system prompt
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the chat history
    pub fn chat_history(mut self, history: ZeroOneOrMany<CandleMessage>) -> Self {
        self.chat_history = history;
        self
    }

    /// Set the documents
    pub fn documents(mut self, docs: ZeroOneOrMany<CandleDocument>) -> Self {
        self.documents = docs;
        self
    }

    /// Set the tools
    pub fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temp: f64) -> Self {
        // Only set if valid, otherwise keep current value
        if TEMPERATURE_RANGE.contains(&temp) {
            self.temperature = temp;
        }
        self
    }

    /// Set the maximum number of tokens
    pub fn max_tokens(mut self, max_tokens: Option<NonZeroU64>) -> Self {
        self.max_tokens = max_tokens.and_then(|t| NonZeroU64::new(t.get().min(MAX_TOKENS)));
        self
    }

    /// Set the chunk size for streaming
    pub fn chunk_size(mut self, size: Option<usize>) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set additional parameters
    pub fn additional_params(mut self, params: Option<Value>) -> Self {
        self.additional_params = params;
        self
    }

    /// Build the request
    pub fn build(self) -> Result<CandleCompletionRequest, CompletionRequestError> {
        let request = CandleCompletionRequest {
            system_prompt: self.system_prompt,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            additional_params: self.additional_params,
        };

        // Validate the request before returning
        request
            .validate()
            .map_err(CompletionRequestError::Validation)?;
        Ok(request)
    }
}
