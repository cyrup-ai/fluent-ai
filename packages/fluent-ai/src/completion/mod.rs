pub mod builder;
pub mod message;
pub mod request_builder;

pub use builder::*;
pub use message::*;
pub use request_builder::*;

// Re-export the correct completion model trait
pub use crate::client::completion::CompletionModel;
pub use crate::streaming::streaming::StreamingCompletionResponse;

/// Completion request type
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub prompt: String,
    pub model: String,
    pub temperature: f64,
    pub max_tokens: Option<u32>,
}

/// Completion response type
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub text: String,
    pub model: String,
    pub usage: Option<Usage>,
}

/// Usage statistics
#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Document type for completion
#[derive(Debug, Clone)]
pub struct Document {
    pub content: String,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Tool definition for completion
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Prompt trait for completion
pub trait Prompt {
    type PromptedBuilder;
    fn prompt(self, prompt: impl ToString) -> Result<Self::PromptedBuilder, PromptError>;
}

/// Chat trait for completion
pub trait Chat {
    fn first_text(&self) -> Option<String>;
}

/// Completion request builder
pub struct CompletionRequestBuilder {
    request: CompletionRequest,
}

impl CompletionRequestBuilder {
    pub fn new(prompt: String) -> Self {
        Self {
            request: CompletionRequest {
                prompt,
                model: "default".to_string(),
                temperature: 0.7,
                max_tokens: None,
            },
        }
    }

    pub fn model(mut self, model: String) -> Self {
        self.request.model = model;
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.request.temperature = temperature;
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.request.max_tokens = Some(max_tokens);
        self
    }

    pub fn build(self) -> CompletionRequest {
        self.request
    }
}

// Core types that other developers added
use thiserror::Error;

use crate::runtime::AsyncTask;

/// Dynamic trait for completion models
pub trait CompletionModelDyn: Send + Sync {
    /// Complete a prompt
    fn complete(&self, prompt: &str) -> AsyncTask<Result<String, CompletionError>>;
}

#[derive(Debug, Error)]
pub enum CompletionError {
    #[error("Request failed: {0}")]
    RequestFailed(String),
    #[error("Model error: {0}")]
    ModelError(String),
}

#[derive(Debug, Error)]
pub enum PromptError {
    #[error("Prompt error: {0}")]
    PromptFailed(String),
    #[error("Max depth error - max_depth: {max_depth}")]
    MaxDepthError {
        max_depth: usize,
        chat_history: Vec<crate::domain::message::Message>,
    },
}

/// Trait for completion models with full functionality
pub trait CompletionModelTrait: Send + Sync {
    /// Strongly typed model from provider crate
    type Model: fluent_ai_provider::Model;
    /// Streaming response type
    type StreamingResponse: Send + Sync;
    /// Get the model instance
    fn model(&self) -> &Self::Model;
    /// Complete a prompt
    fn complete(&self, prompt: &str) -> AsyncTask<Result<String, CompletionError>>;
    /// Stream completion
    fn stream_completion(
        &self,
        prompt: &str,
    ) -> AsyncTask<Result<Self::StreamingResponse, CompletionError>>;
}

// Additional traits that other developers added
pub trait Completion<M: CompletionModelTrait> {
    fn completion(
        &self,
        prompt: impl Into<crate::domain::message::Message> + Send,
        chat_history: Vec<crate::domain::message::Message>,
    ) -> AsyncTask<Result<CompletionRequestBuilder<M>, CompletionError>>;
}

// Streaming traits
pub trait StreamingCompletion<M: CompletionModelTrait> {
    fn stream_completion(
        &self,
        prompt: impl Into<crate::domain::message::Message> + Send,
        chat_history: Vec<crate::domain::message::Message>,
    ) -> AsyncTask<Result<CompletionRequestBuilder<M>, CompletionError>>;
}

pub trait StreamingPrompt<T> {
    fn stream_prompt(
        &self,
        prompt: impl Into<crate::domain::message::Message> + Send,
    ) -> AsyncTask<Result<StreamingCompletionResponse<T>, CompletionError>>;
}

pub trait StreamingChat<T> {
    fn stream_chat(
        &self,
        prompt: impl Into<crate::domain::message::Message> + Send,
        chat_history: Vec<crate::domain::message::Message>,
    ) -> AsyncTask<Result<StreamingCompletionResponse<T>, CompletionError>>;
}

// Generic completion response for provider implementations
#[derive(Debug, Clone)]
pub struct CompletionResponseGeneric<T> {
    pub choice: T,
}

// Type alias to maintain backward compatibility
pub type CompletionResponseWithRaw<T> = CompletionResponseGeneric<T>;

// StreamingCompletionResponse is imported from crate::streaming::streaming
