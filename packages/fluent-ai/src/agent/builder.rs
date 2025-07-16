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

#[cfg(feature = "mcp")]
use crate::tool::McpTool;

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
    pub fn model(self, model_name: &'static str) -> AgentBuilder<M, MissingSys, MissingCtx> {
        // In real implementation, this would create the actual model instance
        todo!("Create model instance based on provider and model_name")
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
        Self {
            model: None,
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
    pub fn build(self) -> super::agent::Agent<M> {
        // Would need to implement Agent constructor that takes all these fields
        todo!("Build agent from builder state")
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
    pub fn chat(self, message: impl Into<String>) -> AsyncStream<String, { super::cfg::CHAT_CAPACITY }> {
        let agent = self.agent_builder.build();
        let message_text = message.into();

        // Create completion request and return processed stream
        AsyncStream::from_future(async move {
            // This would integrate with the actual completion system
            // and apply the chunk handler if present
            todo!("Implement streaming chat with chunk processing")
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
    pub fn model(self, model_name: &'static str) -> AgentBuilder<M, MissingSys, MissingCtx> {
        AgentBuilder::new(model_name)
    }
}
