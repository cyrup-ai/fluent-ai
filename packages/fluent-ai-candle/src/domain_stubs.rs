//! Temporary stub types to replace missing domain dependency
//! These are minimal implementations to allow compilation testing.

use std::borrow::Cow;

use fluent_ai_core::stream::AsyncStream;
use serde::{Deserialize, Serialize};

/// Temporary stub for completion request
#[derive(Debug, Clone)]
pub struct CompletionRequest<'a> {
    pub messages: Vec<Message<'a>>,
    pub model: Option<Cow<'a, str>>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub stream: bool,
}

/// Temporary stub for completion response
#[derive(Debug, Clone)]
pub struct CompletionResponse<'a> {
    pub text: Cow<'a, str>,
    pub model: Cow<'a, str>,
    pub provider: Option<Cow<'a, str>>,
    pub usage: Option<Usage>,
    pub finish_reason: Option<Cow<'a, str>>,
    pub response_time_ms: u64,
    pub generation_time_ms: u64,
    pub tokens_per_second: f32,
}

/// Temporary stub for completion core response (using owned strings for simplicity)
#[derive(Debug, Clone)]
pub struct CompletionCoreResponse {
    pub content: String,
    pub finish_reason: Option<FinishReason>,
    pub model: Option<String>,
}

impl<'a> CompletionResponse<'a> {
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn provider(&self) -> Option<&str> {
        self.provider.as_deref()
    }
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }
    pub fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }
    pub fn response_time_ms(&self) -> u64 {
        self.response_time_ms
    }
    pub fn generation_time_ms(&self) -> u64 {
        self.generation_time_ms
    }
    pub fn tokens_per_second(&self) -> f32 {
        self.tokens_per_second
    }
    pub fn tokens_generated(&self) -> Option<usize> {
        self.usage.as_ref().map(|u| u.completion_tokens)
    }
}

/// Temporary stub for streaming response
#[derive(Debug, Clone)]
pub struct StreamingResponse {
    // Minimal implementation
}

/// Temporary stub for message role
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
}

impl MessageRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::System => "system",
            MessageRole::Tool => "tool",
        }
    }
}

/// Temporary stub for message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message<'a> {
    pub role: Cow<'a, str>,
    pub content: Cow<'a, str>,
}

/// Temporary stub for chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage<'a> {
    pub role: MessageRole,
    pub content: Cow<'a, str>,
}

/// Temporary stub for tool definition
#[derive(Debug, Clone)]
pub struct ToolDefinition<'a> {
    pub name: Cow<'a, str>,
    pub description: Cow<'a, str>,
}

/// Temporary stub for document
#[derive(Debug, Clone)]
pub struct Document<'a> {
    pub content: Cow<'a, str>,
}

/// Temporary stub for usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Temporary stub for finish reason
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
}

/// Temporary stub for completion errors
#[derive(Debug, thiserror::Error)]
pub enum CompletionRequestError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}

#[derive(Debug, thiserror::Error)]
pub enum CompletionCoreError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Generation failed: {0}")]
    GenerationFailed(String),
}

pub type CompletionCoreResult<T> = Result<T, CompletionCoreError>;

/// Temporary stub for extraction error
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Temporary stub for completion client trait
pub trait CompletionCoreClient: Send + Sync {
    fn complete<'a>(
        &'a self,
        request: CompletionRequest<'a>,
    ) -> AsyncStream<CompletionCoreResult<CompletionResponse<'a>>>;
    fn complete_stream<'a>(
        &'a self,
        request: CompletionRequest<'a>,
    ) -> AsyncStream<CompletionCoreResult<StreamingResponse>>;
}
