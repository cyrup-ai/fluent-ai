use std::num::NonZeroU64;
use serde_json::Value;
use thiserror::Error;
use crate::domain::{
    CandleZeroOneOrMany as ZeroOneOrMany,
    chat::{CandleMessage as ChatMessage},
    context::{CandleDocument as Document},
    completion::{
        CandleCompletionRequest as CompletionRequest,
        types::{MAX_TOKENS, TEMPERATURE_RANGE}
    },
    tool::{CandleTool},
    model::{CandleValidationError as ValidationError},
    http::requests::completion::{ToolDefinition, FunctionDefinition, ToolType}
};

/// Builder for completion requests
pub struct CompletionRequestBuilder {
    system_prompt: String,
    chat_history: ZeroOneOrMany<ChatMessage>,
    documents: ZeroOneOrMany<Document>,
    tools: ZeroOneOrMany<Box<dyn CandleTool>>,
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
    pub fn chat_history(mut self, history: ZeroOneOrMany<ChatMessage>) -> Self {
        self.chat_history = history;
        self
    }

    /// Set the documents
    pub fn documents(mut self, docs: ZeroOneOrMany<Document>) -> Self {
        self.documents = docs;
        self
    }

    /// Set the tools
    pub fn tools(mut self, tools: ZeroOneOrMany<Box<dyn CandleTool>>) -> Self {
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
    pub fn build(self) -> Result<CompletionRequest, CompletionRequestError> {
        // Convert Box<dyn CandleTool> to ToolDefinition
        let converted_tools = match self.tools {
            ZeroOneOrMany::None => ZeroOneOrMany::None,
            ZeroOneOrMany::One(tool) => {
                let tool_def = ToolDefinition {
                    tool_type: ToolType::Function,
                    function: FunctionDefinition::new(
                        tool.name(),
                        tool.description(),
                        tool.parameters().clone()
                    ).map_err(|_| CompletionRequestError::InvalidParameter("Failed to create function definition".to_string()))?
                };
                ZeroOneOrMany::One(tool_def)
            },
            ZeroOneOrMany::Many(tools) => {
                let mut tool_defs = Vec::new();
                for tool in tools {
                    let tool_def = ToolDefinition {
                        tool_type: ToolType::Function,
                        function: FunctionDefinition::new(
                            tool.name(),
                            tool.description(),
                            tool.parameters().clone()
                        ).map_err(|_| CompletionRequestError::InvalidParameter("Failed to create function definition".to_string()))?
                    };
                    tool_defs.push(tool_def);
                }
                ZeroOneOrMany::Many(tool_defs)
            }
        };

        let request = CompletionRequest {
            system_prompt: self.system_prompt,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: converted_tools,
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

impl Default for CompletionRequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}