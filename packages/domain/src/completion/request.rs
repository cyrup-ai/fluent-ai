//! Completion request types and builders
//!
//! Contains request structures and builder patterns for completion functionality.

// Removed unused import: std::borrow::Cow
use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use super::types::{MAX_CHUNK_SIZE, MAX_TOKENS, TEMPERATURE_RANGE, ToolDefinition};
use crate::ZeroOneOrMany;
use crate::chat::Message as ChatMessage;
use crate::context::Document;
use crate::model::ValidationError;

/// A request for text completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// System prompt providing instructions
    pub system_prompt: String,
    /// Conversation history
    pub chat_history: ZeroOneOrMany<ChatMessage>,
    /// Documents to use as context
    pub documents: ZeroOneOrMany<Document>,
    /// Tools available to the model
    pub tools: ZeroOneOrMany<ToolDefinition>,
    /// Sampling temperature (0.0 to 2.0)
    pub temperature: f64,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<NonZeroU64>,
    /// Size of chunks for streaming
    pub chunk_size: Option<usize>,
    /// Additional provider-specific parameters
    pub additional_params: Option<Value>}

/// Error type for completion request validation
#[derive(Debug, Error)]
pub enum CompletionRequestError {
    /// Invalid parameter value
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Validation error
    #[error(transparent)]
    Validation(#[from] ValidationError)}

/// Builder for creating CompletionRequest instances
#[derive(Debug, Clone)]
pub struct CompletionRequestBuilder {
    system_prompt: String,
    chat_history: ZeroOneOrMany<ChatMessage>,
    documents: ZeroOneOrMany<Document>,
    tools: ZeroOneOrMany<ToolDefinition>,
    temperature: f64,
    max_tokens: Option<NonZeroU64>,
    chunk_size: Option<usize>,
    additional_params: Option<Value>,
}

impl Default for CompletionRequestBuilder {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            chat_history: crate::ZeroOneOrMany::None,
            documents: crate::ZeroOneOrMany::None,
            tools: crate::ZeroOneOrMany::None,
            temperature: 1.0,
            max_tokens: None,
            chunk_size: None,
            additional_params: None,
        }
    }
}

impl CompletionRequestBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set system prompt
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set chat history
    pub fn chat_history(mut self, history: ZeroOneOrMany<ChatMessage>) -> Self {
        self.chat_history = history;
        self
    }

    /// Set documents
    pub fn documents(mut self, documents: ZeroOneOrMany<Document>) -> Self {
        self.documents = documents;
        self
    }

    /// Set tools
    pub fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, tokens: Option<NonZeroU64>) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set chunk size
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
    pub fn build(self) -> Result<CompletionRequest, CompletionRequestError> {
        let request = CompletionRequest {
            system_prompt: self.system_prompt,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            additional_params: self.additional_params,
        };

        request.validate().map_err(CompletionRequestError::Validation)?;
        Ok(request)
    }
}

impl CompletionRequest {
    /// Create a new builder
    pub fn builder() -> CompletionRequestBuilder {
        CompletionRequestBuilder::new()
    }

    /// Validate the request parameters
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Validate temperature
        if !TEMPERATURE_RANGE.contains(&self.temperature) {
            return Err(ValidationError::InvalidRange {
                field: "temperature".into(),
                value: self.temperature.to_string(),
                expected: format!(
                    "between {:.1} and {:.1}",
                    TEMPERATURE_RANGE.start(),
                    TEMPERATURE_RANGE.end()
                )});
        }

        // Validate max_tokens
        if let Some(max_tokens) = self.max_tokens {
            if max_tokens.get() > MAX_TOKENS {
                return Err(ValidationError::InvalidRange {
                    field: "max_tokens".into(),
                    value: max_tokens.to_string(),
                    expected: format!("less than or equal to {}", MAX_TOKENS)});
            }
        }

        // Validate chunk_size
        if let Some(chunk_size) = self.chunk_size {
            if chunk_size == 0 || chunk_size > MAX_CHUNK_SIZE {
                return Err(ValidationError::InvalidRange {
                    field: "chunk_size".into(),
                    value: chunk_size.to_string(),
                    expected: format!("between 1 and {}", MAX_CHUNK_SIZE)});
            }
        }

        Ok(())
    }

    /// Convert to a static lifetime version by making all borrowed data owned
    #[inline]
    pub fn into_static(self) -> CompletionRequest {
        CompletionRequest {
            system_prompt: self.system_prompt,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            additional_params: self.additional_params}
    }
}

