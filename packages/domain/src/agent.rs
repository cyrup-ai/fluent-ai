//! Agent domain types
//!
//! Contains pure data structures for agents. 
//! Builder implementations are in fluent_ai package.

use crate::{ZeroOneOrMany, Models, Document, Memory};
use crate::mcp_tool::McpToolData;
use serde_json::Value;

/// Agent data structure - pure data only
#[derive(Debug, Clone)]
pub struct Agent {
    pub model: Models,
    pub system_prompt: String,
    pub context: ZeroOneOrMany<Document>,
    pub tools: ZeroOneOrMany<McpToolData>,
    pub memory: Option<Memory>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub additional_params: Option<Value>,
}

impl Agent {
    /// Create a new agent with minimal configuration
    pub fn new(model: Models, system_prompt: impl Into<String>) -> Self {
        Self {
            model,
            system_prompt: system_prompt.into(),
            context: ZeroOneOrMany::None,
            tools: ZeroOneOrMany::None,
            memory: None,
            temperature: None,
            max_tokens: None,
            additional_params: None,
        }
    }
}