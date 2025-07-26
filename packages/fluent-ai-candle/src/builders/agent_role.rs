//! Agent role builder implementation moved from domain
//! Builders are behavioral/construction logic, separate from core domain models

use std::fmt;
use std::marker::PhantomData;
use std::collections::HashMap;

use crate::domain::{
    // Agent types - now properly exported from agent module
    CandleAgentConversation, CandleAgentConversationMessage, CandleAgentRole, CandleAgentRoleAgent, CandleAgentRoleImpl,
    
    // Core domain types exported at top level
    CandleAgent, CandleMessage, CandleMessageChunk, CandleMessageRole, 
    CandleMemory, CandleConversationTrait as CandleConversation,
    CandleZeroOneOrMany, AsyncStream,
    
    // Module-specific types
    completion::{CandleCompletionModel},
    context::{CandleContext, CandleDocument},
    tool::{CandleTool},
};
use serde_json::Value;
use futures_util::StreamExt;

/// MCP Server type enum
#[derive(Debug, Clone)]
pub enum McpServerType {
    StdIo,
    Sse,
    Http,
}

/// Candle MCP Server configuration
#[derive(Debug, Clone)]
struct CandleMcpServerConfig {
    server_type: McpServerType,
    bin_path: Option<String>,
    init_command: Option<String>,
}

/// CandleFluentAi entry point for creating agent roles
pub struct CandleFluentAi;

impl CandleFluentAi {
    /// Create a new Candle agent role builder - main entry point
    pub fn agent_role(name: impl Into<String>) -> impl CandleAgentRoleBuilder {
        CandleAgentRoleBuilderImpl::new(name)
    }
}

/// Candle MCP Server builder - zero Box<dyn> usage
pub struct CandleMcpServerBuilder<T, F1 = fn(&mut String), F2 = fn(&CandleAgentConversation, &CandleAgentRoleAgent)>
where
    F1: FnMut(String) + Send + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{
    parent: CandleAgentRoleBuilderImpl<F1, F2>,
    server_type: PhantomData<T>,
    bin_path: Option<String>,
}

/// Placeholder for Stdio type
pub struct Stdio;

/// Candle agent role builder trait - elegant zero-allocation builder pattern
pub trait CandleAgentRoleBuilder: Sized {
    /// Create a new Candle agent role builder - EXACT syntax: CandleFluentAi::agent_role("name")
    fn new(name: impl Into<String>) -> impl CandleAgentRoleBuilder;
    
    /// Set the completion provider - EXACT syntax: .completion_provider(Mistral::MagistralSmall)
    /// Zero-allocation: uses compile-time type information instead of Any trait
    fn completion_provider<P>(self, provider: P) -> impl CandleAgentRoleBuilder
    where
        P: CandleCompletionModel + Send + Sync + 'static;
    
    /// Set model - EXACT syntax: .model(Models::Gpt4OMini)
    fn model(self, model: impl CandleCompletionModel) -> impl CandleAgentRoleBuilder;
    
    /// Set temperature - EXACT syntax: .temperature(1.0)
    fn temperature(self, temp: f64) -> impl CandleAgentRoleBuilder;
    
    /// Set max tokens - EXACT syntax: .max_tokens(8000)
    fn max_tokens(self, max: u64) -> impl CandleAgentRoleBuilder;
    
    /// Set system prompt - EXACT syntax: .system_prompt("...")
    fn system_prompt(self, prompt: impl Into<String>) -> impl CandleAgentRoleBuilder;
    
    /// Add context - EXACT syntax: .context(CandleContext<File>::of(...), CandleContext<Files>::glob(...), ...)
    fn context(self, contexts: impl CandleContextArgs) -> impl CandleAgentRoleBuilder;
    
    /// Add tools - EXACT syntax: .tools(CandleTool<Perplexity>::new(...), CandleTool::named("cargo")...)
    fn tools(self, tools: impl CandleToolArgs) -> impl CandleAgentRoleBuilder;
    
    /// Add MCP server - EXACT syntax: .mcp_server<Stdio>().bin(...).init(...)
    fn mcp_server<T>(self) -> CandleMcpServerBuilder<T>;
    
    /// Set additional parameters - EXACT syntax: .additional_params({"beta" => "true"})
    fn additional_params<P>(self, params: P) -> impl CandleAgentRoleBuilder
    where
        P: Into<hashbrown::HashMap<&'static str, &'static str>>;
    
    /// Set memory - EXACT syntax: .memory(Library::named("obsidian_vault"))
    fn memory<M>(self, memory: M) -> impl CandleAgentRoleBuilder
    where
        M: CandleMemory + Send + Sync + 'static;
    
    /// Set metadata - EXACT syntax: .metadata({"key" => "val", "foo" => "bar"})
    fn metadata<M>(self, metadata: M) -> impl CandleAgentRoleBuilder
    where
        M: Into<hashbrown::HashMap<&'static str, &'static str>>;
    
    /// Set tool result handler - EXACT syntax: .on_tool_result(|results| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_tool_result<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: FnMut(String) + Send + 'static;
    
    /// Set conversation turn handler - EXACT syntax: .on_conversation_turn(|conversation, agent| { ... })
    fn on_conversation_turn<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static;
        
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    fn on_error<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: FnMut(String) + Send + 'static;
        
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    fn on_chunk<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static;
        
    /// Convert to agent - EXACT syntax: .into_agent()
    fn into_agent(self) -> impl CandleAgentBuilder;
}

/// Hidden Candle implementation struct - zero-allocation builder state with zero Box<dyn> usage
struct CandleAgentRoleBuilderImpl<F1 = fn(String), F2 = fn(&CandleAgentConversation, &CandleAgentRoleAgent)>
where
    F1: FnMut(String) + Send + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{
    name: String,
    completion_provider: CandleZeroOneOrMany<CandleCompletionProvider>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    system_prompt: Option<String>,
    contexts: CandleZeroOneOrMany<CandleContext>,
    tools: CandleZeroOneOrMany<CandleTool>,
    mcp_servers: CandleZeroOneOrMany<CandleMcpServer>,
    additional_params: CandleZeroOneOrMany<CandleAdditionalParams>,
    memory: CandleZeroOneOrMany<CandleMemory>,
    metadata: CandleZeroOneOrMany<CandleMetadata>,
    on_tool_result_handler: Option<F1>,
    on_conversation_turn_handler: Option<F2>,
}

impl<F1, F2> CandleAgentRoleBuilderImpl<F1, F2>
where
    F1: FnMut(String) + Send + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{
    /// Create a new Candle agent role builder with default function handlers
    pub fn new(name: impl Into<String>) -> CandleAgentRoleBuilderImpl {
        CandleAgentRoleBuilderImpl {
            name: name.into(),
            completion_provider: CandleZeroOneOrMany::Zero,
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            contexts: CandleZeroOneOrMany::Zero,
            tools: CandleZeroOneOrMany::Zero,
            mcp_servers: CandleZeroOneOrMany::Zero,
            additional_params: CandleZeroOneOrMany::Zero,
            memory: CandleZeroOneOrMany::Zero,
            metadata: CandleZeroOneOrMany::Zero,
            on_tool_result_handler: None,
            on_conversation_turn_handler: None,
        }
    }
}

impl<F1, F2> CandleAgentRoleBuilder for CandleAgentRoleBuilderImpl<F1, F2>
where
    F1: FnMut(String) + Send + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{
    /// Create a new Candle agent role builder - EXACT syntax: CandleFluentAi::agent_role("name")
    fn new(name: impl Into<String>) -> impl CandleAgentRoleBuilder {
        Self::new(name)
    }

    /// Set the completion provider - EXACT syntax: .completion_provider(Mistral::MagistralSmall)
    /// Zero-allocation: uses compile-time type information instead of Any trait
    fn completion_provider<P>(mut self, provider: P) -> impl CandleAgentRoleBuilder
    where
        P: CandleCompletionModel + Send + Sync + 'static,
    {
        // Store actual completion provider domain object - zero allocation with static dispatch
        self.completion_provider = self.completion_provider.with_pushed(provider);
        self
    }

    /// Set model - EXACT syntax: .model(Models::Gpt4OMini)
    fn model<M>(mut self, model: M) -> impl CandleAgentRoleBuilder
    where
        M: CandleCompletionModel + Send + Sync + 'static,
    {
        // Store model configuration using actual domain object - zero allocation at build time
        self.completion_provider = self.completion_provider.with_pushed(model);
        self
    }

    /// Set temperature - EXACT syntax: .temperature(1.0)
    fn temperature(mut self, temp: f64) -> impl CandleAgentRoleBuilder {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens - EXACT syntax: .max_tokens(8000)
    fn max_tokens(mut self, max: u64) -> impl CandleAgentRoleBuilder {
        self.max_tokens = Some(max);
        self
    }

    /// Set system prompt - EXACT syntax: .system_prompt("...")
    fn system_prompt(mut self, prompt: impl Into<String>) -> impl CandleAgentRoleBuilder {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add context - EXACT syntax: .context(Context<File>::of(...), Context<Files>::glob(...), ...)
    fn context(mut self, contexts: impl CandleContextArgs) -> impl CandleAgentRoleBuilder {
        contexts.add_to(&mut self.contexts);
        self
    }

    /// Add MCP server - EXACT syntax: .mcp_server<Stdio>().bin(...).init(...)
    fn mcp_server<T>(self) -> CandleMcpServerBuilder<T, F1, F2> {
        CandleMcpServerBuilder {
            parent: self,
            server_type: PhantomData,
            bin_path: None,
        }
    }

    /// Add tools - EXACT syntax: .tools(Tool<Perplexity>::new(...), Tool::named(...).bin(...).description(...))
    fn tools(mut self, tools: impl CandleToolArgs) -> impl CandleAgentRoleBuilder {
        tools.add_to(&mut self.tools);
        self
    }

    /// Set additional parameters - EXACT syntax: .additional_params({"beta" => "true"})
    fn additional_params<P>(mut self, params: P) -> impl CandleAgentRoleBuilder
    where
        P: Into<hashbrown::HashMap<&'static str, &'static str>>,
    {
        let config_map = params.into();
        let mut param_map = HashMap::new();
        for (k, v) in config_map {
            param_map.insert(k.to_string(), Value::String(v.to_string()));
        }
        
        // Create CandleAdditionalParams domain object and store in CandleZeroOneOrMany
        let additional_param = CandleAdditionalParams::new(param_map);
        self.additional_params = self.additional_params.with_pushed(additional_param);
        self
    }

    /// Set memory - EXACT syntax: .memory(Library::named("obsidian_vault"))
    fn memory<M>(mut self, memory: M) -> impl CandleAgentRoleBuilder
    where
        M: CandleMemory + Send + Sync + 'static,
    {
        // Store actual memory domain object - zero allocation with static dispatch
        self.memory = self.memory.with_pushed(memory);
        self
    }

    /// Set metadata - EXACT syntax: .metadata({"key" => "val", "foo" => "bar"})
    fn metadata<M>(mut self, metadata: M) -> impl CandleAgentRoleBuilder
    where
        M: Into<hashbrown::HashMap<&'static str, &'static str>>,
    {
        let config_map = metadata.into();
        let mut meta_map = HashMap::new();
        for (k, v) in config_map {
            meta_map.insert(k.to_string(), Value::String(v.to_string()));
        }
        
        // Create CandleMetadata domain object and store in CandleZeroOneOrMany
        let metadata_obj = CandleMetadata::new(meta_map);
        self.metadata = self.metadata.with_pushed(metadata_obj);
        self
    }

    /// Set tool result handler - EXACT syntax: .on_tool_result(|results| { ... })
    /// Zero-allocation: uses generic parameter instead of Box<dyn>
    fn on_tool_result<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: FnMut(String) + Send + 'static,
    {
        // Return new builder with handler - zero allocation, zero Box<dyn>
        CandleAgentRoleBuilderImpl {
            name: self.name,
            completion_provider: self.completion_provider,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            system_prompt: self.system_prompt,
            contexts: self.contexts,
            tools: self.tools,
            mcp_servers: self.mcp_servers,
            additional_params: self.additional_params,
            memory: self.memory,
            metadata: self.metadata,
            on_tool_result_handler: Some(handler),
            on_conversation_turn_handler: self.on_conversation_turn_handler,
        }
    }

    /// Set on_conversation_turn handler - EXACT syntax: .on_conversation_turn(|conversation, agent| { ... })
    fn on_conversation_turn<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
    {
        // Return new builder with handler - zero allocation, zero Box<dyn>
        CandleAgentRoleBuilderImpl {
            name: self.name,
            completion_provider: self.completion_provider,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            system_prompt: self.system_prompt,
            contexts: self.contexts,
            tools: self.tools,
            mcp_servers: self.mcp_servers,
            additional_params: self.additional_params,
            memory: self.memory,
            metadata: self.metadata,
            on_tool_result_handler: self.on_tool_result_handler,
            on_conversation_turn_handler: Some(handler),
        }
    }

    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic parameter instead of Box<dyn>
    fn on_error<F>(self, error_handler: F) -> impl CandleAgentRoleBuilder
    where
        F: FnMut(String) + Send + 'static,
    {
        // Return specialized builder with error handler - zero allocation, zero Box<dyn>
        CandleAgentRoleBuilderWithHandler {
            inner: self,
            error_handler,
        }
    }

    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: uses generic parameter instead of Box<dyn>
    fn on_chunk<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
    {
        // Return specialized builder with chunk handler - zero allocation, zero Box<dyn>
        CandleAgentRoleBuilderWithChunkHandler {
            inner: self,
            chunk_handler: handler,
        }
    }

    /// Convert to agent - EXACT syntax: .into_agent()
    /// Returns CandleAgentBuilder that supports conversation_history() and chat() methods
    fn into_agent(self) -> impl CandleAgentBuilder {
        CandleAgentBuilderImpl {
            inner: self,
            conversation_history: CandleZeroOneOrMany::Zero,
        }
    }
}

impl<T, F1, F2> CandleMcpServerBuilder<T, F1, F2>
where
    F1: FnMut(String) + Send + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{
    /// Set binary path - EXACT syntax: .bin("/path/to/bin")
    pub fn bin(mut self, path: impl Into<String>) -> Self {
        self.bin_path = Some(path.into());
        self
    }

    /// Initialize - EXACT syntax: .init("cargo run -- --stdio")
    pub fn init(self, command: impl Into<String>) -> impl CandleAgentRoleBuilder {
        let mut parent = self.parent;
        
        // Create CandleMcpServer domain object
        let mcp_server = CandleMcpServer::new(
            std::any::type_name::<T>().to_string(),
            self.bin_path,
            Some(command.into()),
        );
        
        // Store actual CandleMcpServer domain object in CandleZeroOneOrMany
        parent.mcp_servers = parent.mcp_servers.with_pushed(mcp_server);
        parent
    }
}

/// Builder with general error handler - zero-allocation with static dispatch
pub struct CandleAgentRoleBuilderWithHandler<F = fn(String)>
where
    F: FnMut(String) + Send + 'static,
{
    #[allow(dead_code)] // TODO: Use for accessing agent role configuration and building
    inner: CandleAgentRoleBuilderImpl,
    #[allow(dead_code)] // TODO: Use for polymorphic error handling during agent role creation
    error_handler: F,
}

impl<F> CandleAgentRoleBuilder for CandleAgentRoleBuilderWithHandler<F>
where
    F: FnMut(String) + Send + 'static,
{
    fn new(name: impl Into<String>) -> impl CandleAgentRoleBuilder {
        CandleAgentRoleBuilderImpl::new(name)
    }

    fn completion_provider<P>(self, provider: P) -> impl CandleAgentRoleBuilder
    where
        P: CandleCompletionModel + Send + Sync + 'static,
    {
        let mut inner = self.inner;
        inner.completion_provider = inner.completion_provider.with_pushed(provider);
        CandleAgentRoleBuilderWithHandler {
            inner,
            error_handler: self.error_handler,
        }
    }

    fn model(self, model: impl CandleCompletionModel) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        inner.completion_provider = inner.completion_provider.with_pushed(model);
        CandleAgentRoleBuilderWithHandler {
            inner,
            error_handler: self.error_handler,
        }
    }

    fn temperature(self, temp: f64) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        inner.temperature = Some(temp);
        CandleAgentRoleBuilderWithHandler {
            inner,
            error_handler: self.error_handler,
        }
    }

    fn max_tokens(self, max: u64) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        inner.max_tokens = Some(max);
        CandleAgentRoleBuilderWithHandler {
            inner,
            error_handler: self.error_handler,
        }
    }

    fn system_prompt(self, prompt: impl Into<String>) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        inner.system_prompt = Some(prompt.into());
        CandleAgentRoleBuilderWithHandler {
            inner,
            error_handler: self.error_handler,
        }
    }

    fn context(self, contexts: impl CandleContextArgs) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        contexts.add_to(&mut inner.contexts);
        CandleAgentRoleBuilderWithHandler {
            inner,
            error_handler: self.error_handler,
        }
    }

    fn tools(self, tools: impl CandleToolArgs) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        tools.add_to(&mut inner.tools);
        CandleAgentRoleBuilderWithHandler {
            inner,
            error_handler: self.error_handler,
        }
    }

    fn mcp_server<T>(self) -> CandleMcpServerBuilder<T> {
        // Note: This transitions back to the base builder pattern
        self.inner.mcp_server()
    }

    fn additional_params<P>(self, params: P) -> impl CandleAgentRoleBuilder
    where
        P: Into<CandleHashMap<&'static str, &'static str>>,
    {
        let mut inner = self.inner;
        let config_map = params.into();
        let mut param_map = CandleHashMap::new();
        for (k, v) in config_map {
            param_map.insert(k.to_string(), Value::String(v.to_string()));
        }
        let additional_param = CandleAdditionalParams::new(param_map);
        inner.additional_params = inner.additional_params.with_pushed(additional_param);
        CandleAgentRoleBuilderWithHandler {
            inner,
            error_handler: self.error_handler,
        }
    }

    fn memory<M>(self, memory: M) -> impl CandleAgentRoleBuilder
    where
        M: CandleMemory + Send + Sync + 'static,
    {
        let mut inner = self.inner;
        inner.memory = inner.memory.with_pushed(memory);
        CandleAgentRoleBuilderWithHandler {
            inner,
            error_handler: self.error_handler,
        }
    }

    fn metadata<M>(self, metadata: M) -> impl CandleAgentRoleBuilder
    where
        M: Into<CandleHashMap<&'static str, &'static str>>,
    {
        let mut inner = self.inner;
        let config_map = metadata.into();
        let mut meta_map = CandleHashMap::new();
        for (k, v) in config_map {
            meta_map.insert(k.to_string(), Value::String(v.to_string()));
        }
        let metadata_obj = CandleMetadata::new(meta_map);
        inner.metadata = inner.metadata.with_pushed(metadata_obj);
        CandleAgentRoleBuilderWithHandler {
            inner,
            error_handler: self.error_handler,
        }
    }

    fn on_tool_result<G>(self, handler: G) -> impl CandleAgentRoleBuilder
    where
        G: FnMut(String) + Send + 'static,
    {
        CandleAgentRoleBuilderImpl {
            name: self.inner.name,
            completion_provider: self.inner.completion_provider,
            temperature: self.inner.temperature,
            max_tokens: self.inner.max_tokens,
            system_prompt: self.inner.system_prompt,
            contexts: self.inner.contexts,
            tools: self.inner.tools,
            mcp_servers: self.inner.mcp_servers,
            additional_params: self.inner.additional_params,
            memory: self.inner.memory,
            metadata: self.inner.metadata,
            on_tool_result_handler: Some(handler),
            on_conversation_turn_handler: self.inner.on_conversation_turn_handler,
        }
    }

    fn on_conversation_turn<G>(self, handler: G) -> impl CandleAgentRoleBuilder
    where
        G: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
    {
        CandleAgentRoleBuilderImpl {
            name: self.inner.name,
            completion_provider: self.inner.completion_provider,
            temperature: self.inner.temperature,
            max_tokens: self.inner.max_tokens,
            system_prompt: self.inner.system_prompt,
            contexts: self.inner.contexts,
            tools: self.inner.tools,
            mcp_servers: self.inner.mcp_servers,
            additional_params: self.inner.additional_params,
            memory: self.inner.memory,
            metadata: self.inner.metadata,
            on_tool_result_handler: self.inner.on_tool_result_handler,
            on_conversation_turn_handler: Some(handler),
        }
    }

    fn on_error<G>(self, error_handler: G) -> impl CandleAgentRoleBuilder
    where
        G: FnMut(String) + Send + 'static,
    {
        // Replace the current error handler with the new one
        CandleAgentRoleBuilderWithHandler {
            inner: self.inner,
            error_handler,
        }
    }

    fn on_chunk<G>(self, handler: G) -> impl CandleAgentRoleBuilder
    where
        G: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
    {
        CandleAgentRoleBuilderWithChunkHandler {
            inner: self.inner,
            chunk_handler: handler,
        }
    }

    fn into_agent(self) -> impl CandleAgentBuilder {
        CandleAgentBuilderImpl {
            inner: self.inner,
            conversation_history: CandleZeroOneOrMany::Zero,
        }
    }
}

impl CandleAgentRoleBuilderWithHandler {
    /// Add chunk handler after error handler - EXACT syntax: .on_chunk(|chunk| { ... })
    fn on_chunk<F>(self, handler: F) -> CandleAgentRoleBuilderWithChunkHandler
    where
        F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
    {
        CandleAgentRoleBuilderWithChunkHandler {
            inner: self.inner,
            chunk_handler: handler}
    }

    /// Build an agent role directly - EXACT syntax: .build()
    /// Returns a configured agent role ready for use
    pub fn build(self) -> impl CandleAgentRole {
        CandleAgentRoleImpl {
            name: self.inner.name,
            completion_provider: self.inner.completion_provider,
            temperature: self.inner.temperature,
            max_tokens: self.inner.max_tokens,
            system_prompt: self.inner.system_prompt,
            contexts: self.inner.contexts,
            tools: self.inner.tools,
            mcp_servers: self.inner.mcp_servers,
            additional_params: self.inner.additional_params,
            memory: self.inner.memory,
            metadata: self.inner.metadata,
            on_tool_result_handler: self.inner.on_tool_result_handler,
            on_conversation_turn_handler: self.inner.on_conversation_turn_handler}
    }
}

/// Builder with chunk handler - zero-allocation with static dispatch
pub struct CandleAgentRoleBuilderWithChunkHandler<F = fn(CandleMessageChunk) -> CandleMessageChunk>
where
    F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
{
    inner: CandleAgentRoleBuilderImpl,
    chunk_handler: F,
}

impl<F> CandleAgentRoleBuilder for CandleAgentRoleBuilderWithChunkHandler<F>
where
    F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
{
    fn new(name: impl Into<String>) -> impl CandleAgentRoleBuilder {
        CandleAgentRoleBuilderImpl::new(name)
    }

    fn completion_provider<P>(self, provider: P) -> impl CandleAgentRoleBuilder
    where
        P: CandleCompletionModel + Send + Sync + 'static,
    {
        let mut inner = self.inner;
        inner.completion_provider = inner.completion_provider.with_pushed(provider);
        CandleAgentRoleBuilderWithChunkHandler {
            inner,
            chunk_handler: self.chunk_handler,
        }
    }

    fn model(self, model: impl CandleCompletionModel) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        inner.completion_provider = inner.completion_provider.with_pushed(model);
        CandleAgentRoleBuilderWithChunkHandler {
            inner,
            chunk_handler: self.chunk_handler,
        }
    }

    fn temperature(self, temp: f64) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        inner.temperature = Some(temp);
        CandleAgentRoleBuilderWithChunkHandler {
            inner,
            chunk_handler: self.chunk_handler,
        }
    }

    fn max_tokens(self, max: u64) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        inner.max_tokens = Some(max);
        CandleAgentRoleBuilderWithChunkHandler {
            inner,
            chunk_handler: self.chunk_handler,
        }
    }

    fn system_prompt(self, prompt: impl Into<String>) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        inner.system_prompt = Some(prompt.into());
        CandleAgentRoleBuilderWithChunkHandler {
            inner,
            chunk_handler: self.chunk_handler,
        }
    }

    fn context(self, contexts: impl CandleContextArgs) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        contexts.add_to(&mut inner.contexts);
        CandleAgentRoleBuilderWithChunkHandler {
            inner,
            chunk_handler: self.chunk_handler,
        }
    }

    fn tools(self, tools: impl CandleToolArgs) -> impl CandleAgentRoleBuilder {
        let mut inner = self.inner;
        tools.add_to(&mut inner.tools);
        CandleAgentRoleBuilderWithChunkHandler {
            inner,
            chunk_handler: self.chunk_handler,
        }
    }

    fn mcp_server<T>(self) -> CandleMcpServerBuilder<T> {
        // Note: This transitions back to the base builder pattern
        self.inner.mcp_server()
    }

    fn additional_params<P>(self, params: P) -> impl CandleAgentRoleBuilder
    where
        P: Into<CandleHashMap<&'static str, &'static str>>,
    {
        let mut inner = self.inner;
        let config_map = params.into();
        let mut param_map = CandleHashMap::new();
        for (k, v) in config_map {
            param_map.insert(k.to_string(), Value::String(v.to_string()));
        }
        let additional_param = CandleAdditionalParams::new(param_map);
        inner.additional_params = inner.additional_params.with_pushed(additional_param);
        CandleAgentRoleBuilderWithChunkHandler {
            inner,
            chunk_handler: self.chunk_handler,
        }
    }

    fn memory<M>(self, memory: M) -> impl CandleAgentRoleBuilder
    where
        M: CandleMemory + Send + Sync + 'static,
    {
        let mut inner = self.inner;
        inner.memory = inner.memory.with_pushed(memory);
        CandleAgentRoleBuilderWithChunkHandler {
            inner,
            chunk_handler: self.chunk_handler,
        }
    }

    fn metadata<M>(self, metadata: M) -> impl CandleAgentRoleBuilder
    where
        M: Into<CandleHashMap<&'static str, &'static str>>,
    {
        let mut inner = self.inner;
        let config_map = metadata.into();
        let mut meta_map = CandleHashMap::new();
        for (k, v) in config_map {
            meta_map.insert(k.to_string(), Value::String(v.to_string()));
        }
        let metadata_obj = CandleMetadata::new(meta_map);
        inner.metadata = inner.metadata.with_pushed(metadata_obj);
        CandleAgentRoleBuilderWithChunkHandler {
            inner,
            chunk_handler: self.chunk_handler,
        }
    }

    fn on_tool_result<G>(self, handler: G) -> impl CandleAgentRoleBuilder
    where
        G: FnMut(String) + Send + 'static,
    {
        CandleAgentRoleBuilderImpl {
            name: self.inner.name,
            completion_provider: self.inner.completion_provider,
            temperature: self.inner.temperature,
            max_tokens: self.inner.max_tokens,
            system_prompt: self.inner.system_prompt,
            contexts: self.inner.contexts,
            tools: self.inner.tools,
            mcp_servers: self.inner.mcp_servers,
            additional_params: self.inner.additional_params,
            memory: self.inner.memory,
            metadata: self.inner.metadata,
            on_tool_result_handler: Some(handler),
            on_conversation_turn_handler: self.inner.on_conversation_turn_handler,
        }
    }

    fn on_conversation_turn<G>(self, handler: G) -> impl CandleAgentRoleBuilder
    where
        G: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
    {
        CandleAgentRoleBuilderImpl {
            name: self.inner.name,
            completion_provider: self.inner.completion_provider,
            temperature: self.inner.temperature,
            max_tokens: self.inner.max_tokens,
            system_prompt: self.inner.system_prompt,
            contexts: self.inner.contexts,
            tools: self.inner.tools,
            mcp_servers: self.inner.mcp_servers,
            additional_params: self.inner.additional_params,
            memory: self.inner.memory,
            metadata: self.inner.metadata,
            on_tool_result_handler: self.inner.on_tool_result_handler,
            on_conversation_turn_handler: Some(handler),
        }
    }

    fn on_error<G>(self, error_handler: G) -> impl CandleAgentRoleBuilder
    where
        G: FnMut(String) + Send + 'static,
    {
        CandleAgentRoleBuilderWithHandler {
            inner: self.inner,
            error_handler,
        }
    }

    fn on_chunk<G>(self, chunk_handler: G) -> impl CandleAgentRoleBuilder
    where
        G: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
    {
        // Replace the current chunk handler with the new one
        CandleAgentRoleBuilderWithChunkHandler {
            inner: self.inner,
            chunk_handler,
        }
    }

    fn into_agent(self) -> impl CandleAgentBuilder {
        CandleAgentBuilderImpl {
            inner: self.inner,
            conversation_history: CandleZeroOneOrMany::Zero,
        }
    }
}

impl CandleAgentRoleBuilderWithChunkHandler {
    /// Convert to agent - EXACT syntax: .into_agent()
    /// Returns CandleAgentBuilder that supports conversation_history() and chat() methods
    pub fn into_agent(self) -> impl CandleAgentBuilder {
        CandleAgentBuilderImpl {
            inner: self.inner,
            conversation_history: CandleZeroOneOrMany::Zero,
        }
    }
}

/// Candle Agent builder trait - elegant zero-allocation agent construction
pub trait CandleAgentBuilder: Sized {
    /// Set conversation history - EXACT syntax from ARCHITECTURE.md
    /// Supports: .conversation_history(CandleMessageRole::User => "content", CandleMessageRole::System => "content", ...)
    fn conversation_history<H>(self, history: H) -> impl CandleAgentBuilder
    where
        H: CandleConversationHistoryArgs;
    
    /// Simple chat method - EXACT syntax: .chat("Hello")
    fn chat(&self, message: impl Into<String>) -> AsyncStream<CandleMessageChunk>;
    
    /// Closure-based chat loop - EXACT syntax: .chat(|conversation| CandleChatLoop)  
    fn chat_with_closure<F>(&self, closure: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static;
}

/// Hidden CandleAgentBuilder implementation - zero-allocation agent state with static dispatch
struct CandleAgentBuilderImpl {
    inner: CandleAgentRoleBuilderImpl,
    conversation_history: CandleZeroOneOrMany<(CandleMessageRole, String)>,
}

impl CandleAgentBuilder for CandleAgentBuilderImpl {
    /// Set conversation history - EXACT syntax from ARCHITECTURE.md
    /// Supports: .conversation_history(CandleMessageRole::User => "content", CandleMessageRole::System => "content", ...)
    fn conversation_history<H>(mut self, history: H) -> impl CandleAgentBuilder
    where
        H: CandleConversationHistoryArgs,
    {
        self.conversation_history = match history.into_history() {
            Some(h) => h,
            None => CandleZeroOneOrMany::Zero,
        };
        self
    }

    /// Simple chat method - EXACT syntax: .chat("Hello")
    fn chat(&self, message: impl Into<String>) -> AsyncStream<CandleMessageChunk> {
        let message_text = message.into();
        
        // Clone required data for async closure - zero allocation patterns
        let completion_provider = self.inner.completion_provider.clone();
        let conversation_history = self.conversation_history.clone();
        let temperature = self.inner.temperature;
        let max_tokens = self.inner.max_tokens;
        let system_prompt = self.inner.system_prompt.clone();
        
        AsyncStream::with_channel(move |sender| {
            Box::pin(async move {
                // Build conversation messages from history + new user message
                let mut messages = Vec::new();
                
                // Add system prompt if configured
                if let Some(system_msg) = system_prompt {
                    messages.push(CandleMessage {
                        role: CandleMessageRole::System,
                        content: system_msg,
                    });
                }
                
                // Add conversation history
                match &conversation_history {
                    CandleZeroOneOrMany::Zero => {},
                    CandleZeroOneOrMany::One((role, content)) => {
                        messages.push(CandleMessage {
                            role: *role,
                            content: content.clone(),
                        });
                    },
                    CandleZeroOneOrMany::Many(history_items) => {
                        for (role, content) in history_items {
                            messages.push(CandleMessage {
                                role: *role,
                                content: content.clone(),
                            });
                        }
                    }
                }
                
                // Add new user message
                messages.push(CandleMessage {
                    role: CandleMessageRole::User,
                    content: message_text,
                });
                
                // Get the first available completion provider
                let provider = match &completion_provider {
                    CandleZeroOneOrMany::Zero => {
                        // Send error chunk if no provider configured
                        let _ = sender.send(CandleMessageChunk {
                            content: "Error: No completion provider configured".to_string(),
                            done: true,
                        }).await;
                        return Ok(());
                    },
                    CandleZeroOneOrMany::One(provider) => provider,
                    CandleZeroOneOrMany::Many(providers) => {
                        // Use the first provider
                        if let Some(provider) = providers.first() {
                            provider
                        } else {
                            let _ = sender.send(CandleMessageChunk {
                                content: "Error: No completion provider available".to_string(),
                                done: true,
                            }).await;
                            return Ok(());
                        }
                    }
                };
                
                // Create completion request
                let completion_request = CandleCompletionRequest {
                    messages,
                    temperature,
                    max_tokens,
                    stream: true,
                    additional_params: None,
                };
                
                // Stream completion response
                let mut completion_stream = provider.stream_completion(completion_request);
                
                // Forward chunks from provider to our sender
                use futures_util::StreamExt;
                let mut completion_stream = std::pin::Pin::new(&mut completion_stream);
                
                while let Some(chunk) = completion_stream.next().await {
                    // Transform provider chunk to our chunk format
                    let candle_chunk = CandleMessageChunk {
                        content: chunk.content,
                        done: chunk.done,
                    };
                    
                    let send_result = sender.send(candle_chunk).await;
                    if send_result.is_err() {
                        // Receiver dropped, stop streaming
                        break;
                    }
                    
                    if chunk.done {
                        break;
                    }
                }
                
                Ok(())
            })
        })
    }

    /// Closure-based chat loop - EXACT syntax: .chat(|conversation| CandleChatLoop)  
    fn chat_with_closure<F>(&self, closure: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static,
    {
        // AsyncStream-only architecture - no Result wrapping
        AsyncStream::with_channel(move |sender| {
            // Create conversation from current history
            let conversation = CandleAgentConversation {
                messages: match &self.conversation_history {
                    CandleZeroOneOrMany::Zero => None,
                    _ => Some(self.conversation_history.clone()),
                }
            };
            
            // Execute closure to get ChatLoop decision
            let chat_result = closure(&conversation);
            
            match chat_result {
                CandleChatLoop::Break => {
                    // Send final chunk and close stream
                    let _ = sender.send(CandleMessageChunk { text: String::new(), done: true });
                }
                CandleChatLoop::Reprompt(response) => {
                    // Send response as chunk
                    let _ = sender.send(CandleMessageChunk { text: response, done: true });
                }
                CandleChatLoop::UserPrompt(prompt) => {
                    // Send prompt as chunk
                    let prompt_text = prompt.unwrap_or_else(|| "Waiting for user input...".to_string());
                    let _ = sender.send(CandleMessageChunk { text: prompt_text, done: true });
                }
            }
        })
    }
}

// Trait implementations for transparent ARCHITECTURE.md syntax
// All macros work INSIDE closures buried in builders and are NEVER EXPOSED in public API surface

// Candle trait definitions for variadic builder syntax
pub trait CandleContextArgs {
    fn add_to(self, contexts: &mut CandleZeroOneOrMany<CandleContext>);
}

pub trait CandleToolArgs {
    fn add_to(self, tools: &mut CandleZeroOneOrMany<CandleTool>);
}

pub trait CandleConversationHistoryArgs {
    fn into_history(self) -> Option<CandleZeroOneOrMany<(CandleMessageRole, String)>>;
}

// ContextArgs implementations for variadic context syntax
// Enables: .context(Context<File>::of(...), Context<Files>::glob(...), Context<Directory>::of(...), Context<Github>::glob(...))

impl<T> CandleContextArgs for T
where
    T: CandleContext + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut CandleZeroOneOrMany<CandleContext>) {
        // Store actual context domain object - zero allocation with static dispatch
        *contexts = contexts.with_pushed(self);
    }
}

impl<T1, T2> CandleContextArgs for (T1, T2)
where
    T1: CandleContext + Send + Sync + 'static,
    T2: CandleContext + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut CandleZeroOneOrMany<CandleContext>) {
        self.0.add_to(contexts);
        self.1.add_to(contexts);
    }
}

impl<T1, T2, T3> CandleContextArgs for (T1, T2, T3)
where
    T1: CandleContext + Send + Sync + 'static,
    T2: CandleContext + Send + Sync + 'static,
    T3: CandleContext + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut CandleZeroOneOrMany<CandleContext>) {
        self.0.add_to(contexts);
        self.1.add_to(contexts);
        self.2.add_to(contexts);
    }
}

impl<T1, T2, T3, T4> CandleContextArgs for (T1, T2, T3, T4)
where
    T1: CandleContext + Send + Sync + 'static,
    T2: CandleContext + Send + Sync + 'static,
    T3: CandleContext + Send + Sync + 'static,
    T4: CandleContext + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut CandleZeroOneOrMany<CandleContext>) {
        self.0.add_to(contexts);
        self.1.add_to(contexts);
        self.2.add_to(contexts);
        self.3.add_to(contexts);
    }
}

// ToolArgs implementations for variadic tool syntax  
// Enables: .tools(Tool<Perplexity>::new({"citations" => "true"}), Tool::named("cargo").bin("~/.cargo/bin").description(...))

impl<T> CandleToolArgs for T
where
    T: CandleTool + Send + Sync + 'static,
{
    fn add_to(self, tools: &mut CandleZeroOneOrMany<CandleTool>) {
        // Store actual tool domain object - zero allocation with static dispatch
        *tools = tools.with_pushed(self);
    }
}

impl<T1, T2> CandleToolArgs for (T1, T2)
where
    T1: CandleTool + Send + Sync + 'static,
    T2: CandleTool + Send + Sync + 'static,
{
    fn add_to(self, tools: &mut CandleZeroOneOrMany<CandleTool>) {
        self.0.add_to(tools);
        self.1.add_to(tools);
    }
}

impl<T1, T2, T3> CandleToolArgs for (T1, T2, T3)
where
    T1: CandleTool + Send + Sync + 'static,
    T2: CandleTool + Send + Sync + 'static,
    T3: CandleTool + Send + Sync + 'static,
{
    fn add_to(self, tools: &mut CandleZeroOneOrMany<CandleTool>) {
        self.0.add_to(tools);
        self.1.add_to(tools);
        self.2.add_to(tools);
    }
}

// Newtype wrapper for hashbrown::HashMap to avoid orphan rule violations
#[derive(Debug, Default, Clone)]
pub struct CandleHashMap<K, V>(pub hashbrown::HashMap<K, V>);

impl<K: std::hash::Hash + Eq, V> std::ops::Deref for CandleHashMap<K, V> {
    type Target = hashbrown::HashMap<K, V>;
    
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K: std::hash::Hash + Eq, V> std::ops::DerefMut for CandleHashMap<K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Implement From for our newtype wrapper
// This enables the same syntax: .additional_params({"beta" => "true"}) and .metadata({"key" => "val"})

impl From<[(&'static str, &'static str); 1]> for CandleHashMap<&'static str, &'static str> {
    fn from(arr: [(&'static str, &'static str); 1]) -> Self {
        let mut map = hashbrown::HashMap::new();
        let [(k, v)] = arr;
        map.insert(k, v);
        CandleHashMap(map)
    }
}

impl From<[(&'static str, &'static str); 2]> for CandleHashMap<&'static str, &'static str> {
    fn from(arr: [(&'static str, &'static str); 2]) -> Self {
        let mut map = hashbrown::HashMap::new();
        for (k, v) in arr {
            map.insert(k, v);
        }
        CandleHashMap(map)
    }
}

impl From<[(&'static str, &'static str); 3]> for CandleHashMap<&'static str, &'static str> {
    fn from(arr: [(&'static str, &'static str); 3]) -> Self {
        let mut map = hashbrown::HashMap::new();
        for (k, v) in arr {
            map.insert(k, v);
        }
        CandleHashMap(map)
    }
}

impl From<[(&'static str, &'static str); 4]> for CandleHashMap<&'static str, &'static str> {
    fn from(arr: [(&'static str, &'static str); 4]) -> Self {
        let mut map = hashbrown::HashMap::new();
        for (k, v) in arr {
            map.insert(k, v);
        }
        CandleHashMap(map)
    }
}

// Add a convenience method to convert to the inner HashMap if needed
impl<K, V> From<CandleHashMap<K, V>> for hashbrown::HashMap<K, V> {
    fn from(wrapper: CandleHashMap<K, V>) -> Self {
        wrapper.0
    }
}

// ConversationHistoryArgs implementations for => syntax
// Enables: .conversation_history(MessageRole::User => "What time is it in Paris, France", MessageRole::System => "...", MessageRole::Assistant => "...")

impl CandleConversationHistoryArgs for (CandleMessageRole, &str) {
    fn into_history(self) -> Option<CandleZeroOneOrMany<(CandleMessageRole, String)>> {
        Some(CandleZeroOneOrMany::one((self.0, self.1.to_string())))
    }
}

impl CandleConversationHistoryArgs for (CandleMessageRole, String) {
    fn into_history(self) -> Option<CandleZeroOneOrMany<(CandleMessageRole, String)>> {
        Some(CandleZeroOneOrMany::one(self))
    }
}

impl<T1, T2> CandleConversationHistoryArgs for (T1, T2)
where
    T1: CandleConversationHistoryArgs,
    T2: CandleConversationHistoryArgs,
{
    fn into_history(self) -> Option<CandleZeroOneOrMany<(CandleMessageRole, String)>> {
        match (self.0.into_history(), self.1.into_history()) {
            (Some(h1), Some(h2)) => {
                let mut combined = h1;
                for item in h2.into_iter() {
                    combined = combined.with_pushed(item);
                }
                Some(combined)
            }
            (Some(h), None) | (None, Some(h)) => Some(h),
            (None, None) => None,
        }
    }
}

impl<T1, T2, T3> CandleConversationHistoryArgs for (T1, T2, T3)
where
    T1: CandleConversationHistoryArgs,
    T2: CandleConversationHistoryArgs,
    T3: CandleConversationHistoryArgs,
{
    fn into_history(self) -> Option<CandleZeroOneOrMany<(CandleMessageRole, String)>> {
        let h12 = (self.0, self.1).into_history();
        let h3 = self.2.into_history();
        match (h12, h3) {
            (Some(mut combined), Some(h3)) => {
                for item in h3.into_iter() {
                    combined = combined.with_pushed(item);
                }
                Some(combined)
            }
            (Some(h), None) | (None, Some(h)) => Some(h),
            (None, None) => None,
        }
    }
}
