// ============================================================================
// File: src/agent/agent.rs
// ----------------------------------------------------------------------------
// Core Agent struct - only buildable through AgentBuilder
// ============================================================================

use std::collections::HashMap;

use super::{
    builder::{AgentBuilder, MissingCtx, MissingSys},
    prompt::{Prompt, PromptRequest},
};
use crate::{
    completion::{CompletionModel, CompletionRequest, CompletionRequestBuilder, Document, Message},
    domain::tool::ToolSet,
    runtime::{AsyncStream, AsyncTask},
    vector_store::VectorStoreIndexDyn,
};

// ============================================================================
// Configuration constants
// ============================================================================
pub mod cfg {
    pub const CHAT_CAPACITY: usize = 256;
    pub const TOOL_CAPACITY: usize = 64;
}

// ============================================================================
// Typestate markers for fluent API
// ============================================================================
pub struct MissingModel;
pub struct MissingPrompt;
pub struct Ready;

// ============================================================================
// Core Agent Provider - only constructible through builder
// ============================================================================
pub struct Agent<M: CompletionModel> {
    model: M,
    preamble: String,
    static_context: Vec<Document>,
    static_tools: Vec<String>,
    dynamic_context: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    tools: ToolSet,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<serde_json::Value>,
    extended_thinking: bool,
    prompt_cache: bool,
}

impl<M: CompletionModel> Agent<M> {
    /// Create a new AgentBuilder for the given provider model
    pub fn for_provider(model: M) -> AgentBuilder<M, MissingSys, MissingCtx> {
        AgentBuilder::new(model)
    }

    /// Internal constructor for building from AgentBuilder
    pub(crate) fn new(
        model: M,
        preamble: String,
        static_context: Vec<Document>,
        static_tools: Vec<String>,
        dynamic_context: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
        dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
        tools: ToolSet,
        temperature: Option<f64>,
        max_tokens: Option<u64>,
        additional_params: Option<serde_json::Value>,
        extended_thinking: bool,
        prompt_cache: bool,
    ) -> Self {
        Self {
            model,
            preamble,
            static_context,
            static_tools,
            dynamic_context,
            dynamic_tools,
            tools,
            temperature,
            max_tokens,
            additional_params,
            extended_thinking,
            prompt_cache,
        }
    }

    /// Start a prompt request - returns async stream
    pub fn prompt(
        &self,
        prompt: impl Prompt,
    ) -> AsyncStream<PromptRequest<M>, { cfg::CHAT_CAPACITY }> {
        let request = PromptRequest::new(self, prompt);
        AsyncStream::from_single(request)
    }

    /// Create a completion request - returns async stream
    pub fn completion(
        &self,
        prompt: Message,
        history: Vec<Message>,
    ) -> AsyncStream<CompletionRequestBuilder, { cfg::CHAT_CAPACITY }> {
        let builder_result = self.create_completion_builder(prompt, history);
        match builder_result {
            Ok(builder) => AsyncStream::from_single(builder),
            Err(_) => AsyncStream::empty(), // Handle error case
        }
    }

    /// Create completion builder internally
    fn create_completion_builder(
        &self,
        prompt: Message,
        history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder, crate::completion::PromptError> {
        let mut builder = CompletionRequestBuilder::new(
            "default-model".to_string(), // This should come from the model
        )?;

        builder = builder.preamble(&self.preamble);

        if !history.is_empty() {
            builder = builder.chat_history(history);
        }

        Ok(builder)
    }

    /// Get the underlying model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get the preamble
    pub fn preamble(&self) -> &str {
        &self.preamble
    }

    /// Get temperature
    pub fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    /// Get max tokens
    pub fn max_tokens(&self) -> Option<u64> {
        self.max_tokens
    }

    /// Get extended thinking
    pub fn extended_thinking(&self) -> bool {
        self.extended_thinking
    }
}

// ============================================================================
// Stream type aliases
// ============================================================================
pub type CompletionStream<T> = AsyncStream<T, { cfg::CHAT_CAPACITY }>;
pub type ToolStream<T> = AsyncStream<T, { cfg::TOOL_CAPACITY }>;
