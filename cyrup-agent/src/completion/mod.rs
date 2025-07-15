pub mod builder;
pub mod message;
pub mod request_builder;

pub use builder::*;
pub use message::*;
pub use request_builder::*;

// Core types that other developers added
use crate::runtime::AsyncTask;
use thiserror::Error;

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
        chat_history: Vec<crate::message::Message>,
    },
}

pub trait CompletionModel: Send + Sync + Clone {
    type Response: Send + 'static;
    type StreamingResponse: Send + 'static;
}

#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct Document {
    pub content: String,
}

pub trait Chat {
    fn first_text(&self) -> Option<String>;
}

// Additional traits that other developers added
pub trait Completion<M: CompletionModel> {
    fn completion(
        &self,
        prompt: impl Into<crate::message::Message> + Send,
        chat_history: Vec<crate::message::Message>,
    ) -> AsyncTask<Result<CompletionRequestBuilder<M>, CompletionError>>;
}

pub trait Prompt {
    fn prompt(&self, prompt: impl Into<crate::message::Message> + Send) -> PromptRequest<M>;
}

// Streaming traits
pub trait StreamingCompletion<M: CompletionModel> {
    fn stream_completion(
        &self,
        prompt: impl Into<crate::message::Message> + Send,
        chat_history: Vec<crate::message::Message>,
    ) -> AsyncTask<Result<CompletionRequestBuilder<M>, CompletionError>>;
}

pub trait StreamingPrompt<T> {
    fn stream_prompt(
        &self,
        prompt: impl Into<crate::message::Message> + Send,
    ) -> AsyncTask<Result<StreamingCompletionResponse<T>, CompletionError>>;
}

pub trait StreamingChat<T> {
    fn stream_chat(
        &self,
        prompt: impl Into<crate::message::Message> + Send,
        chat_history: Vec<crate::message::Message>,
    ) -> AsyncTask<Result<StreamingCompletionResponse<T>, CompletionError>>;
}

// Core request/response types
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub messages: Vec<crate::message::Message>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct CompletionResponse<T> {
    pub choice: T,
}

#[derive(Debug, Clone)]
pub struct StreamingCompletionResponse<T> {
    pub stream: T,
}
