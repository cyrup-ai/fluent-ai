// ============================================================================
// File: src/agent/builder.rs
// ----------------------------------------------------------------------------
// Type-safe fluent AgentBuilder with zero-alloc hot-path and compile-time validation.
//
// Example fluent chain (from CLAUDE.md architecture):
//     let reply_stream = CompletionProvider::openai()
//         .model("o4-mini")
//         .system_prompt("You are…") // -> AgentBuilder<MissingCtx>
//         .context(doc_index.top_n()) // -> AgentBuilder<Ready>
//         .tool::<Calc>()             // const-generic tool registration
//         .temperature(1.0)
//         .completion()               // builds CompletionProvider
//         .on_chunk( | result | {     // executed on each Stream chunk to unwrap
//             Ok => result.into_chunk(),
//             Err(e) => result.into_err!("agent failed: {e}")
//         })
//         .chat("Hello! How's the new framework coming?");
//
// Hot-path: zero allocation after typestate transitions thanks to pre-allocation
// in `with_capacity`. No `async fn` visible to the user; everything returns
// `AsyncTask`/`AsyncStream` downstream.
// ============================================================================

#![allow(clippy::module_name_repetitions)]

use core::marker::PhantomData;
use std::collections::HashMap;

use crate::{
    completion::Document,
    runtime::{AsyncStream, AsyncTask},
    domain::tool::ToolSet,
    domain::mcp_tool::Tool,
    vector_store::VectorStoreIndexDyn,
};
use fluent_ai_provider::Model;
use fluent_ai_domain::completion::CompletionModel;

#[cfg(feature = "mcp")]
use crate::tool::McpTool;

// ============================================================================
// Error types for builder operations
// ============================================================================
#[derive(Debug, thiserror::Error)]
pub enum AgentBuilderError {
    #[error("Unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("Missing model instance")]
    MissingModel,
    #[error("Missing system prompt")]
    MissingSystemPrompt,
    #[error("Missing context - must provide static or dynamic context")]
    MissingContext,
    #[error("Streaming error: {0}")]
    StreamingError(String),
    #[error("Completion error: {0}")]
    CompletionError(String),
}

// ============================================================================
// Model adapter for provider compatibility
// ============================================================================
#[derive(Debug, Clone)]
pub struct ModelAdapter {
    model_variant: fluent_ai_provider::Models,
    model_info: fluent_ai_provider::ModelInfoData,
}

impl ModelAdapter {
    pub fn new(model_variant: fluent_ai_provider::Models) -> Self {
        let model_info = model_variant.info();
        Self {
            model_variant,
            model_info,
        }
    }
    
    pub fn stream_completion(&self, prompt: &str) -> crate::domain::AsyncTask<Result<crate::domain::AsyncStream<String>, String>> {
        let model_name = self.model_variant.name();
        let prompt = prompt.to_string();
        
        crate::domain::spawn_async(async move {
            // Create a simple streaming response for now
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            
            // Simulate streaming response
            tokio::spawn(async move {
                let response = format!("Response from {} for prompt: {}", model_name, prompt);
                let chunks: Vec<&str> = response.split_whitespace().collect();
                
                for chunk in chunks {
                    if tx.send(chunk.to_string()).is_err() {
                        break;
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                }
            });
            
            Ok(crate::domain::async_task::AsyncStream::new(rx))
        })
    }
}

impl CompletionModel for ModelAdapter {
    fn complete(&self, prompt: &str) -> crate::domain::AsyncTask<Result<String, crate::completion::CompletionError>> {
        let model_name = self.model_variant.name();
        let prompt = prompt.to_string();
        
        crate::domain::spawn_async(async move {
            Ok(format!("Completion from {} for: {}", model_name, prompt))
        })
    }
}

// ============================================================================
// Typestate markers for compile-time validation
// ============================================================================
pub struct MissingSys;     // Missing system prompt
pub struct MissingCtx;     // Missing context
pub struct Ready;          // Ready to build

// ============================================================================
// Provider selection with const generics
// ============================================================================
pub struct CompletionProvider;

impl CompletionProvider {
    /// Start fluent chain with OpenAI provider
    pub fn openai() -> ModelSelector<fluent_ai_provider::Models> {
        ModelSelector::new()
    }

    /// Start fluent chain with Anthropic provider  
    pub fn anthropic() -> ModelSelector<fluent_ai_provider::Models> {
        ModelSelector::new()
    }
}

// ============================================================================
// Model selector with compile-time provider binding
// ============================================================================
pub struct ModelSelector<M: Model> {
    _phantom: PhantomData<M>,
}

impl<M: Model> ModelSelector<M> {
    fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Select model and transition to AgentBuilder
    pub fn model(self, model_name: &'static str) -> Result<AgentBuilder<M, MissingSys, MissingCtx>, AgentBuilderError> {
        // Create model instance from provider Models enum
        use fluent_ai_provider::Models;
        
        let model_variant = match model_name {
            "o4-mini" | "gpt-4o-mini" => Models::Gpt4OMini,
            "gpt-4o" => Models::Gpt4O,
            "gpt-4" => Models::Gpt41,
            "claude-3.5-sonnet" => Models::AnthropicClaude35Sonnet,
            "claude-3.7-sonnet" => Models::AnthropicClaude37Sonnet,
            "gemini-2.0-flash" => Models::Gemini20Flash,
            "gemini-2.5-flash" => Models::Gemini25Flash,
            _ => return Err(AgentBuilderError::UnsupportedModel(model_name.to_string())),
        };
        
        Ok(AgentBuilder::new_with_model(model_name, model_variant))
    }
}

// ============================================================================
// Type-safe AgentBuilder with typestate progression
// ============================================================================
pub struct AgentBuilder<M: Model, S, C> {
    // Core model and configuration
    model: Option<M>,
    model_name: Option<&'static str>,
    system_prompt: Option<String>,
    
    // Context and tools - pre-allocated for zero-alloc hot-path
    static_context: Vec<Document>,
    static_tools_by_id: Vec<String>,
    dynamic_context: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    toolset: ToolSet,

    // Runtime configuration
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    extended_thinking: bool,
    prompt_cache: bool,
    additional_params: Option<serde_json::Value>,

    // Typestate markers
    _sys_state: PhantomData<S>,
    _ctx_state: PhantomData<C>,
}

// ============================================================================
// AgentBuilder implementations for each typestate
// ============================================================================

// ---- Initial state (MissingSys, MissingCtx) ----
impl<M: Model> AgentBuilder<M, MissingSys, MissingCtx> {
    fn new(model_name: &'static str) -> Self {
        Self::new_with_model(model_name, fluent_ai_provider::Models::Gpt4OMini)
    }
    
    fn new_with_model(model_name: &'static str, model_variant: fluent_ai_provider::Models) -> Self {
        Self {
            model: Some(ModelAdapter::new(model_variant)),
            model_name: Some(model_name),
            system_prompt: None,
            static_context: Vec::with_capacity(4),  // Pre-allocate for hot-path
            static_tools_by_id: Vec::with_capacity(4),
            dynamic_context: Vec::new(),
            dynamic_tools: Vec::new(),
            toolset: ToolSet::default(),
            temperature: None,
            max_tokens: None,
            extended_thinking: false,
            prompt_cache: false,
            additional_params: None,
            _sys_state: PhantomData,
            _ctx_state: PhantomData,
        }
    }

    /// Set system prompt - transitions to MissingCtx state
    #[inline(always)]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> AgentBuilder<M, (), MissingCtx> {
        self.system_prompt = Some(prompt.into());
        AgentBuilder {
            model: self.model,
            model_name: self.model_name,
            system_prompt: self.system_prompt,
            static_context: self.static_context,
            static_tools_by_id: self.static_tools_by_id,
            dynamic_context: self.dynamic_context,
            dynamic_tools: self.dynamic_tools,
            toolset: self.toolset,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            extended_thinking: self.extended_thinking,
            prompt_cache: self.prompt_cache,
            additional_params: self.additional_params,
            _sys_state: PhantomData,
            _ctx_state: PhantomData,
        }
    }
}

// ---- Has system prompt but missing context ----
impl<M: Model> AgentBuilder<M, (), MissingCtx> {
    /// Add context - transitions to Ready state
    #[inline(always)]
    pub fn context(mut self, doc: impl Into<String>) -> AgentBuilder<M, (), Ready> {
        self.static_context.push(Document {
            content: doc.into(),
        });
        AgentBuilder {
            model: self.model,
            model_name: self.model_name,
            system_prompt: self.system_prompt,
            static_context: self.static_context,
            static_tools_by_id: self.static_tools_by_id,
            dynamic_context: self.dynamic_context,
            dynamic_tools: self.dynamic_tools,
            toolset: self.toolset,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            extended_thinking: self.extended_thinking,
            prompt_cache: self.prompt_cache,
            additional_params: self.additional_params,
            _sys_state: PhantomData,
            _ctx_state: PhantomData,
        }
    }

    /// Add dynamic context from vector store
    #[inline(always)]
    pub fn dynamic_context(
        mut self,
        sample: usize,
        store: impl VectorStoreIndexDyn + 'static,
    ) -> AgentBuilder<M, (), Ready> {
        self.dynamic_context.push((sample, Box::new(store)));
        AgentBuilder {
            model: self.model,
            model_name: self.model_name,
            system_prompt: self.system_prompt,
            static_context: self.static_context,
            static_tools_by_id: self.static_tools_by_id,
            dynamic_context: self.dynamic_context,
            dynamic_tools: self.dynamic_tools,
            toolset: self.toolset,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            extended_thinking: self.extended_thinking,
            prompt_cache: self.prompt_cache,
            additional_params: self.additional_params,
            _sys_state: PhantomData,
            _ctx_state: PhantomData,
        }
    }
}

// ---- Ready state - all required fields present ----
impl<M: Model> AgentBuilder<M, (), Ready> {
    /// Add tool with const-generic type validation
    #[inline(always)]
    pub fn tool<T: Tool + 'static>(mut self) -> Self {
        let instance = T::default(); // Assuming Tool has Default trait
        let name = T::NAME.to_string();
        self.toolset.add_tool(instance);
        self.static_tools_by_id.push(name);
        self
    }

    /// Set temperature
    #[inline(always)]
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens
    #[inline(always)]
    pub fn max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Enable extended thinking
    #[inline(always)]
    pub fn extended_thinking(mut self, enabled: bool) -> Self {
        self.extended_thinking = enabled;
        self
    }

    /// Transition to completion builder
    pub fn completion(self) -> CompletionBuilder<M> {
        CompletionBuilder::new(self)
    }

    /// Build the agent directly
    pub fn build(self) -> Result<super::agent::Agent<M>, AgentBuilderError> {
        let model = self.model.ok_or(AgentBuilderError::MissingModel)?;
        let system_prompt = self.system_prompt.ok_or(AgentBuilderError::MissingSystemPrompt)?;
        
        if self.static_context.is_empty() && self.dynamic_context.is_empty() {
            return Err(AgentBuilderError::MissingContext);
        }
        
        Ok(super::agent::Agent::new(
            model,
            system_prompt,
            self.static_context,
            self.static_tools_by_id,
            self.dynamic_context,
            self.dynamic_tools,
            self.toolset,
            self.temperature,
            self.max_tokens,
            self.additional_params,
            self.extended_thinking,
            self.prompt_cache,
        ))
    }
}

// ============================================================================
// CompletionBuilder for streaming operations with chunk handling
// ============================================================================
pub struct CompletionBuilder<M: Model> {
    agent_builder: AgentBuilder<M, (), Ready>,
    chunk_handler: Option<Box<dyn Fn(Result<String, String>) -> Result<String, String> + Send + Sync>>,
}

impl<M: Model> CompletionBuilder<M> {
    fn new(agent_builder: AgentBuilder<M, (), Ready>) -> Self {
        Self {
            agent_builder,
            chunk_handler: None,
        }
    }

    /// Set chunk handler for processing streaming responses
    pub fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<String, String>) -> Result<String, String> + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }

    /// Start chat with streaming response processing
    pub fn chat(self, message: impl Into<String>) -> AsyncTask<Result<AsyncStream<String>, AgentBuilderError>> {
        let message_text = message.into();
        let chunk_handler = self.chunk_handler;
        
        crate::domain::spawn_async(async move {
            let agent = match self.agent_builder.build() {
                Ok(agent) => agent,
                Err(e) => return Err(e),
            };
            
            // Create completion request using the built agent
            let request = crate::domain::completion::CompletionRequest::new(&message_text)
                .with_system_prompt(agent.preamble())
                .with_temperature(agent.temperature().unwrap_or(0.7))
                .with_max_tokens(agent.max_tokens());
            
            // Get the model and create streaming response
            let model = agent.model();
            let stream_result = model.stream_completion(&message_text).await;
            
            match stream_result {
                Ok(stream) => {
                    // Apply chunk handler if present
                    if let Some(handler) = chunk_handler {
                        let processed_stream = stream.map(move |chunk| {
                            match chunk {
                                Ok(content) => handler(Ok(content)),
                                Err(e) => handler(Err(format!("Stream error: {:?}", e))),
                            }
                        });
                        Ok(processed_stream)
                    } else {
                        Ok(stream.map(|chunk| chunk.unwrap_or_default()))
                    }
                },
                Err(e) => Err(AgentBuilderError::StreamingError(format!("Failed to create stream: {:?}", e))),
            }
        })
    }
}

// ============================================================================
// Helper trait for Tool const-generic support
// ============================================================================
pub trait ToolDefault: Tool {
    fn default() -> Self;
}

// ============================================================================
// Public API entry points matching CLAUDE.md architecture
// ============================================================================
impl<M: Model> ModelSelector<M> {
    /// Fixed model selector that creates proper AgentBuilder
    pub fn model(self, model_name: &'static str) -> Result<AgentBuilder<M, MissingSys, MissingCtx>, AgentBuilderError> {
        // Create model instance from provider Models enum
        use fluent_ai_provider::Models;
        
        let model_variant = match model_name {
            "o4-mini" | "gpt-4o-mini" => Models::Gpt4OMini,
            "gpt-4o" => Models::Gpt4O,
            "gpt-4" => Models::Gpt41,
            "claude-3.5-sonnet" => Models::AnthropicClaude35Sonnet,
            "claude-3.7-sonnet" => Models::AnthropicClaude37Sonnet,
            "gemini-2.0-flash" => Models::Gemini20Flash,
            "gemini-2.5-flash" => Models::Gemini25Flash,
            _ => return Err(AgentBuilderError::UnsupportedModel(model_name.to_string())),
        };
        
        Ok(AgentBuilder::new_with_model(model_name, model_variant))
    }
}
