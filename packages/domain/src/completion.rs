//! Completion domain types
//!
//! Contains pure data structures and traits for completion.
//! Builder implementations are in fluent_ai package.

use crate::chunk::CompletionChunk;
use crate::prompt::Prompt;
use crate::{ZeroOneOrMany, Models};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Core trait for completion models
pub trait CompletionModel: Send + Sync + Clone {
    /// Generate completion from prompt
    fn prompt(&self, prompt: Prompt) -> crate::async_task::AsyncStream<CompletionChunk>;
}

pub trait CompletionBackend {
    fn submit_completion(
        &self,
        prompt: &str,
        tools: &[String],
    ) -> crate::async_task::AsyncTask<String>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub system_prompt: String,
    pub chat_history: ZeroOneOrMany<crate::Message>,
    pub documents: ZeroOneOrMany<crate::Document>,
    pub tools: ZeroOneOrMany<ToolDefinition>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub chunk_size: Option<usize>,
    pub additional_params: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl CompletionRequest {
    /// Create a new completion request with basic configuration
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            chat_history: ZeroOneOrMany::None,
            documents: ZeroOneOrMany::None,
            tools: ZeroOneOrMany::None,
            temperature: None,
            max_tokens: None,
            chunk_size: None,
            additional_params: None,
        }
    }
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