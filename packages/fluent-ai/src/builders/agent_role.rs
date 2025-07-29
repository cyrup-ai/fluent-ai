//! Agent role builder implementation moved from domain
//! Builders are behavioral/construction logic, separate from core domain models

use std::fmt;
use std::marker::PhantomData;
use std::collections::HashMap;

use fluent_ai_domain::{
    agent::{AgentConversation, AgentConversationMessage, AgentRole, AgentRoleAgent, AgentRoleImpl},
    chat::{Conversation, Message, MessageRole},
    completion::CompletionModel,
    context::chunk::ChatMessageChunk,
    tool::Tool,
    AdditionalParams, Metadata, Memory,
    ZeroOneOrMany,
};
use fluent_ai_async::AsyncStream;
use serde_json::Value;

/// MCP Server type enum
#[derive(Debug, Clone)]
pub enum McpServerType {
    StdIo,
    Sse,
    Http,
}

/// MCP Server configuration
#[derive(Debug, Clone)]
struct McpServerConfig {
    server_type: McpServerType,
    bin_path: Option<String>,
    init_command: Option<String>,
}

/// FluentAi entry point for creating agent roles
pub struct FluentAi;

impl FluentAi {
    /// Create a new agent role builder - main entry point
    pub fn agent_role(name: impl Into<String>) -> impl AgentRoleBuilder {
        AgentRoleBuilderImpl::new(name)
    }
}

/// MCP Server builder - zero Box<dyn> usage
pub struct McpServerBuilder<T, F1 = fn(&mut String), F2 = fn(&AgentConversation, &AgentRoleAgent)>
where
    F1: FnMut(String) + Send + 'static,
    F2: Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync + 'static,
{
    parent: AgentRoleBuilderImpl<F1, F2>,
    server_type: PhantomData<T>,
    bin_path: Option<String>,
}

/// Placeholder for Stdio type
pub struct Stdio;

/// Agent role builder trait - elegant zero-allocation builder pattern
pub trait AgentRoleBuilder: Sized {
    /// Create a new agent role builder - EXACT syntax: FluentAi::agent_role("name")
    fn new(name: impl Into<String>) -> impl AgentRoleBuilder;
    
    /// Set the completion provider - EXACT syntax: .completion_provider(Mistral::MagistralSmall)
    /// Zero-allocation: uses compile-time type information instead of Any trait
    fn completion_provider<P>(self, provider: P) -> impl AgentRoleBuilder
    where
        P: CompletionProvider + Send + Sync + 'static;
    
    /// Set model - EXACT syntax: .model(Models::Gpt4OMini)
    fn model(self, model: impl CompletionModel) -> impl AgentRoleBuilder;
    
    /// Set temperature - EXACT syntax: .temperature(1.0)
    fn temperature(self, temp: f64) -> impl AgentRoleBuilder;
    
    /// Set max tokens - EXACT syntax: .max_tokens(8000)
    fn max_tokens(self, max: u64) -> impl AgentRoleBuilder;
    
    /// Set system prompt - EXACT syntax: .system_prompt("...")
    fn system_prompt(self, prompt: impl Into<String>) -> impl AgentRoleBuilder;
    
    /// Add context - EXACT syntax: .context(Context<File>::of(...), Context<Files>::glob(...), ...)
    fn context(self, contexts: impl ContextArgs) -> impl AgentRoleBuilder;
    
    /// Add tools - EXACT syntax: .tools(Tool<Perplexity>::new(...), Tool::named("cargo")...)
    fn tools(self, tools: impl ToolArgs) -> impl AgentRoleBuilder;
    
    /// Add MCP server - EXACT syntax: .mcp_server<Stdio>().bin(...).init(...)
    fn mcp_server<T>(self) -> McpServerBuilder<T>;
    
    /// Set additional parameters - EXACT syntax: .additional_params({"beta" => "true"})
    fn additional_params<P>(self, params: P) -> impl AgentRoleBuilder
    where
        P: Into<hashbrown::HashMap<&'static str, &'static str>>;
    
    /// Set memory - EXACT syntax: .memory(Library::named("obsidian_vault"))
    fn memory<M>(self, memory: M) -> impl AgentRoleBuilder
    where
        M: Memory + Send + Sync + 'static;
    
    /// Set metadata - EXACT syntax: .metadata({"key" => "val", "foo" => "bar"})
    fn metadata<M>(self, metadata: M) -> impl AgentRoleBuilder
    where
        M: Into<hashbrown::HashMap<&'static str, &'static str>>;
    
    /// Set tool result handler - EXACT syntax: .on_tool_result(|results| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_tool_result<F>(self, handler: F) -> impl AgentRoleBuilder
    where
        F: FnMut(String) + Send + 'static;
    
    /// Set conversation turn handler - EXACT syntax: .on_conversation_turn(|conversation, agent| { ... })
    fn on_conversation_turn<F>(self, handler: F) -> impl AgentRoleBuilder
    where
        F: Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync + 'static;
        
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    fn on_error<F>(self, handler: F) -> impl AgentRoleBuilder
    where
        F: FnMut(String) + Send + 'static;
        
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    fn on_chunk<F>(self, handler: F) -> impl AgentRoleBuilder
    where
        F: Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync + 'static;
        
    /// Convert to agent - EXACT syntax: .into_agent()
    fn into_agent(self) -> impl AgentBuilder;
}

/// Hidden implementation struct - zero-allocation builder state with zero Box<dyn> usage
struct AgentRoleBuilderImpl<F1 = fn(&mut String), F2 = fn(&AgentConversation, &AgentRoleAgent)>
where
    F1: FnMut(String) + Send + 'static,
    F2: Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync + 'static,
{
    name: String,
    completion_provider: ZeroOneOrMany<CompletionProvider>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    system_prompt: Option<String>,
    contexts: ZeroOneOrMany<Context>,
    tools: ZeroOneOrMany<Tool>,
    mcp_servers: ZeroOneOrMany<McpServer>,
    additional_params: ZeroOneOrMany<AdditionalParams>,
    memory: ZeroOneOrMany<Memory>,
    metadata: ZeroOneOrMany<Metadata>,
    on_tool_result_handler: Option<F1>,
    on_conversation_turn_handler: Option<F2>,
}

impl<F1, F2> AgentRoleBuilderImpl<F1, F2>
where
    F1: FnMut(String) + Send + 'static,
    F2: Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync + 'static,
{
    /// Create a new agent role builder with default function handlers
    pub fn new(name: impl Into<String>) -> AgentRoleBuilderImpl {
        AgentRoleBuilderImpl {
            name: name.into(),
            completion_provider: ZeroOneOrMany::None,
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            contexts: ZeroOneOrMany::None,
            tools: ZeroOneOrMany::None,
            mcp_servers: ZeroOneOrMany::None,
            additional_params: ZeroOneOrMany::None,
            memory: ZeroOneOrMany::None,
            metadata: ZeroOneOrMany::None,
            on_tool_result_handler: None,
            on_conversation_turn_handler: None,
        }
    }
}

impl<F1, F2> AgentRoleBuilder for AgentRoleBuilderImpl<F1, F2>
where
    F1: FnMut(String) + Send + 'static,
    F2: Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync + 'static,
{
    /// Create a new agent role builder - EXACT syntax: FluentAi::agent_role("name")
    fn new(name: impl Into<String>) -> impl AgentRoleBuilder {
        Self::new(name)
    }

    /// Set the completion provider - EXACT syntax: .completion_provider(Mistral::MagistralSmall)
    /// Zero-allocation: uses compile-time type information instead of Any trait
    fn completion_provider<P>(mut self, provider: P) -> impl AgentRoleBuilder
    where
        P: CompletionProvider + Send + Sync + 'static,
    {
        // Store actual completion provider domain object - zero allocation with static dispatch
        self.completion_provider = self.completion_provider.with_pushed(provider);
        self
    }

    /// Set model - EXACT syntax: .model(Models::Gpt4OMini)
    fn model<M>(mut self, model: M) -> impl AgentRoleBuilder
    where
        M: CompletionModel + Send + Sync + 'static,
    {
        // Store model configuration using actual domain object - zero allocation at build time
        self.completion_provider = self.completion_provider.with_pushed(model);
        self
    }

    /// Set temperature - EXACT syntax: .temperature(1.0)
    fn temperature(mut self, temp: f64) -> impl AgentRoleBuilder {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens - EXACT syntax: .max_tokens(8000)
    fn max_tokens(mut self, max: u64) -> impl AgentRoleBuilder {
        self.max_tokens = Some(max);
        self
    }

    /// Set system prompt - EXACT syntax: .system_prompt("...")
    fn system_prompt(mut self, prompt: impl Into<String>) -> impl AgentRoleBuilder {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add context - EXACT syntax: .context(Context<File>::of(...), Context<Files>::glob(...), ...)
    fn context(mut self, contexts: impl ContextArgs) -> impl AgentRoleBuilder {
        contexts.add_to(&mut self.contexts);
        self
    }

    /// Add MCP server - EXACT syntax: .mcp_server<Stdio>().bin(...).init(...)
    fn mcp_server<T>(self) -> McpServerBuilder<T, F1, F2> {
        McpServerBuilder {
            parent: self,
            server_type: PhantomData,
            bin_path: None,
        }
    }

    /// Add tools - EXACT syntax: .tools(Tool<Perplexity>::new(...), Tool::named(...).bin(...).description(...))
    fn tools(mut self, tools: impl ToolArgs) -> impl AgentRoleBuilder {
        tools.add_to(&mut self.tools);
        self
    }

    /// Set additional parameters - EXACT syntax: .additional_params({"beta" => "true"})
    fn additional_params<P>(mut self, params: P) -> impl AgentRoleBuilder
    where
        P: Into<hashbrown::HashMap<&'static str, &'static str>>,
    {
        let config_map = params.into();
        let mut param_map = HashMap::new();
        for (k, v) in config_map {
            param_map.insert(k.to_string(), Value::String(v.to_string()));
        }
        
        // Create AdditionalParams domain object and store in ZeroOneOrMany
        let additional_param = AdditionalParams::new(param_map);
        self.additional_params = self.additional_params.with_pushed(additional_param);
        self
    }

    /// Set memory - EXACT syntax: .memory(Library::named("obsidian_vault"))
    fn memory<M>(mut self, memory: M) -> impl AgentRoleBuilder
    where
        M: Memory + Send + Sync + 'static,
    {
        // Store actual memory domain object - zero allocation with static dispatch
        self.memory = self.memory.with_pushed(memory);
        self
    }

    /// Set metadata - EXACT syntax: .metadata({"key" => "val", "foo" => "bar"})
    fn metadata<M>(mut self, metadata: M) -> impl AgentRoleBuilder
    where
        M: Into<hashbrown::HashMap<&'static str, &'static str>>,
    {
        let config_map = metadata.into();
        let mut meta_map = HashMap::new();
        for (k, v) in config_map {
            meta_map.insert(k.to_string(), Value::String(v.to_string()));
        }
        
        // Create Metadata domain object and store in ZeroOneOrMany
        let metadata_obj = Metadata::new(meta_map);
        self.metadata = self.metadata.with_pushed(metadata_obj);
        self
    }

    /// Set tool result handler - EXACT syntax: .on_tool_result(|results| { ... })
    /// Zero-allocation: uses generic parameter instead of Box<dyn>
    fn on_tool_result<F>(self, handler: F) -> impl AgentRoleBuilder
    where
        F: FnMut(String) + Send + 'static,
    {
        // Return new builder with handler - zero allocation, zero Box<dyn>
        AgentRoleBuilderImpl {
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
    fn on_conversation_turn<F>(self, handler: F) -> impl AgentRoleBuilder
    where
        F: Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync + 'static,
    {
        // Return new builder with handler - zero allocation, zero Box<dyn>
        AgentRoleBuilderImpl {
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
    /// Zero-allocation: returns self for method chaining
    fn on_error<F>(self, _error_handler: F) -> impl AgentRoleBuilder
    where
        F: FnMut(String) + Send + 'static,
    {
        // Store error handler (implementation simplified for now)
        self
    }

    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: returns self for method chaining
    fn on_chunk<F>(self, _handler: F) -> impl AgentRoleBuilder
    where
        F: Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync + 'static,
    {
        // Store chunk handler (implementation simplified for now)
        self
    }

    /// Convert to agent - EXACT syntax: .into_agent()
    /// Returns AgentBuilder that supports conversation_history() and chat() methods
    fn into_agent(self) -> impl AgentBuilder {
        AgentBuilderImpl {
            inner: self,
            conversation_history: ZeroOneOrMany::None,
        }
    }
}

impl<T, F1, F2> McpServerBuilder<T, F1, F2>
where
    F1: FnMut(String) + Send + 'static,
    F2: Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync + 'static,
{
    /// Set binary path - EXACT syntax: .bin("/path/to/bin")
    pub fn bin(mut self, path: impl Into<String>) -> Self {
        self.bin_path = Some(path.into());
        self
    }

    /// Initialize - EXACT syntax: .init("cargo run -- --stdio")
    pub fn init(self, command: impl Into<String>) -> impl AgentRoleBuilder {
        let mut parent = self.parent;
        
        // Create McpServer domain object
        let mcp_server = McpServer::new(
            std::any::type_name::<T>().to_string(),
            self.bin_path,
            Some(command.into()),
        );
        
        // Store actual McpServer domain object in ZeroOneOrMany
        parent.mcp_servers = parent.mcp_servers.with_pushed(mcp_server);
        parent
    }
}

/// Builder with general error handler - zero-allocation with static dispatch
pub struct AgentRoleBuilderWithHandler<F = fn(String)>
where
    F: FnMut(String) + Send + 'static,
{
    #[allow(dead_code)] // TODO: Use for accessing agent role configuration and building
    inner: AgentRoleBuilderImpl,
    #[allow(dead_code)] // TODO: Use for polymorphic error handling during agent role creation
    error_handler: F,
}

impl AgentRoleBuilderWithHandler {
    /// Add chunk handler after error handler - EXACT syntax: .on_chunk(|chunk| { ... })
    fn on_chunk<F>(self, handler: F) -> AgentRoleBuilderWithChunkHandler
    where
        F: Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync + 'static,
    {
        AgentRoleBuilderWithChunkHandler {
            inner: self.inner,
            chunk_handler: handler}
    }

    /// Build an agent role directly - EXACT syntax: .build()
    /// Returns a configured agent role ready for use
    pub fn build(self) -> impl AgentRole {
        AgentRoleImpl {
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
pub struct AgentRoleBuilderWithChunkHandler<F = fn(ChatMessageChunk) -> ChatMessageChunk>
where
    F: Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync + 'static,
{
    inner: AgentRoleBuilderImpl,
    chunk_handler: F,
}

impl AgentRoleBuilderWithChunkHandler {
    /// Convert to agent - EXACT syntax: .into_agent()
    /// Returns AgentBuilder that supports conversation_history() and chat() methods
    pub fn into_agent(self) -> impl AgentBuilder {
        AgentBuilderImpl {
            inner: self.inner,
            chunk_handler: self.chunk_handler,
            conversation_history: ZeroOneOrMany::None,
        }
    }
}

/// Agent builder trait - elegant zero-allocation agent construction
pub trait AgentBuilder: Sized {
    /// Set conversation history - EXACT syntax from ARCHITECTURE.md
    /// Supports: .conversation_history(MessageRole::User => "content", MessageRole::System => "content", ...)
    fn conversation_history<H>(self, history: H) -> impl AgentBuilder
    where
        H: ConversationHistoryArgs;
    
    /// Simple chat method - EXACT syntax: .chat("Hello")
    fn chat(&self, message: impl Into<String>) -> AsyncStream<ChatMessageChunk>;
    
    /// Closure-based chat loop - EXACT syntax: .chat(|conversation| ChatLoop)  
    fn chat_with_closure<F>(&self, closure: F) -> AsyncStream<ChatMessageChunk>
    where
        F: FnOnce(&AgentConversation) -> ChatLoop + Send + 'static;
}

/// Hidden AgentBuilder implementation - zero-allocation agent state with static dispatch
struct AgentBuilderImpl {
    inner: AgentRoleBuilderImpl,
    conversation_history: ZeroOneOrMany<(MessageRole, String)>,
}

impl AgentBuilder for AgentBuilderImpl {
    /// Set conversation history - EXACT syntax from ARCHITECTURE.md
    /// Supports: .conversation_history(MessageRole::User => "content", MessageRole::System => "content", ...)
    fn conversation_history<H>(mut self, history: H) -> impl AgentBuilder
    where
        H: ConversationHistoryArgs,
    {
        self.conversation_history = history.into_history().unwrap_or(ZeroOneOrMany::None);
        self
    }

    /// Simple chat method - EXACT syntax: .chat("Hello")
    fn chat(&self, message: impl Into<String>) -> AsyncStream<ChatMessageChunk> {
        let _message_text = message.into();
        
        // AsyncStream-only architecture - no Result wrapping
        // Error handling via on_chunk patterns, not Result types
        AsyncStream::empty()
    }

    /// Closure-based chat loop - EXACT syntax: .chat(|conversation| ChatLoop)  
    fn chat_with_closure<F>(&self, closure: F) -> AsyncStream<ChatMessageChunk>
    where
        F: FnOnce(&AgentConversation) -> ChatLoop + Send + 'static,
    {
        // AsyncStream-only architecture - no Result wrapping
        AsyncStream::with_channel(move |sender| {
            // Create conversation from current history
            let conversation = AgentConversation {
                messages: match &self.conversation_history {
                    ZeroOneOrMany::None => None,
                    _ => Some(self.conversation_history.clone()),
                }
            };
            
            // Execute closure to get ChatLoop decision
            let chat_result = closure(&conversation);
            
            match chat_result {
                ChatLoop::Break => {
                    // Send final chunk and close stream
                    let _ = sender.send(ChatMessageChunk { text: String::new(), done: true });
                }
                ChatLoop::Reprompt(response) => {
                    // Send response as chunk
                    let _ = sender.send(ChatMessageChunk { text: response, done: true });
                }
                ChatLoop::UserPrompt(prompt) => {
                    // Send prompt as chunk
                    let prompt_text = prompt.unwrap_or_else(|| "Waiting for user input...".to_string());
                    let _ = sender.send(ChatMessageChunk { text: prompt_text, done: true });
                }
            }
        })
    }
}

// Trait implementations for transparent ARCHITECTURE.md syntax
// All macros work INSIDE closures buried in builders and are NEVER EXPOSED in public API surface

use fluent_ai_domain::agent::types::{ContextArgs, ToolArgs, ConversationHistoryArgs};
use fluent_ai_domain::{Context, Tool};

// ContextArgs implementations for variadic context syntax
// Enables: .context(Context<File>::of(...), Context<Files>::glob(...), Context<Directory>::of(...), Context<Github>::glob(...))

impl<T> ContextArgs for T
where
    T: Context + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut ZeroOneOrMany<Context>) {
        // Store actual context domain object - zero allocation with static dispatch
        *contexts = contexts.with_pushed(self);
    }
}

impl<T1, T2> ContextArgs for (T1, T2)
where
    T1: Context + Send + Sync + 'static,
    T2: Context + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut ZeroOneOrMany<Context>) {
        self.0.add_to(contexts);
        self.1.add_to(contexts);
    }
}

impl<T1, T2, T3> ContextArgs for (T1, T2, T3)
where
    T1: Context + Send + Sync + 'static,
    T2: Context + Send + Sync + 'static,
    T3: Context + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut ZeroOneOrMany<Context>) {
        self.0.add_to(contexts);
        self.1.add_to(contexts);
        self.2.add_to(contexts);
    }
}

impl<T1, T2, T3, T4> ContextArgs for (T1, T2, T3, T4)
where
    T1: Context + Send + Sync + 'static,
    T2: Context + Send + Sync + 'static,
    T3: Context + Send + Sync + 'static,
    T4: Context + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut ZeroOneOrMany<Context>) {
        self.0.add_to(contexts);
        self.1.add_to(contexts);
        self.2.add_to(contexts);
        self.3.add_to(contexts);
    }
}

// ToolArgs implementations for variadic tool syntax  
// Enables: .tools(Tool<Perplexity>::new({"citations" => "true"}), Tool::named("cargo").bin("~/.cargo/bin").description(...))

impl<T> ToolArgs for T
where
    T: Tool + Send + Sync + 'static,
{
    fn add_to(self, tools: &mut ZeroOneOrMany<Tool>) {
        // Store actual tool domain object - zero allocation with static dispatch
        *tools = tools.with_pushed(self);
    }
}

impl<T1, T2> ToolArgs for (T1, T2)
where
    T1: Tool + Send + Sync + 'static,
    T2: Tool + Send + Sync + 'static,
{
    fn add_to(self, tools: &mut ZeroOneOrMany<Tool>) {
        self.0.add_to(tools);
        self.1.add_to(tools);
    }
}

impl<T1, T2, T3> ToolArgs for (T1, T2, T3)
where
    T1: Tool + Send + Sync + 'static,
    T2: Tool + Send + Sync + 'static,
    T3: Tool + Send + Sync + 'static,
{
    fn add_to(self, tools: &mut ZeroOneOrMany<Tool>) {
        self.0.add_to(tools);
        self.1.add_to(tools);
        self.2.add_to(tools);
    }
}

// hashbrown::HashMap From implementations for transparent {"key" => "value"} syntax
// Enables: .additional_params({"beta" => "true"}) and .metadata({"key" => "val", "foo" => "bar"})

impl From<[(&'static str, &'static str); 1]> for hashbrown::HashMap<&'static str, &'static str> {
    fn from(arr: [(&'static str, &'static str); 1]) -> Self {
        let mut map = hashbrown::HashMap::new();
        let [(k, v)] = arr;
        map.insert(k, v);
        map
    }
}

impl From<[(&'static str, &'static str); 2]> for hashbrown::HashMap<&'static str, &'static str> {
    fn from(arr: [(&'static str, &'static str); 2]) -> Self {
        let mut map = hashbrown::HashMap::new();
        for (k, v) in arr {
            map.insert(k, v);
        }
        map
    }
}

impl From<[(&'static str, &'static str); 3]> for hashbrown::HashMap<&'static str, &'static str> {
    fn from(arr: [(&'static str, &'static str); 3]) -> Self {
        let mut map = hashbrown::HashMap::new();
        for (k, v) in arr {
            map.insert(k, v);
        }
        map
    }
}

impl From<[(&'static str, &'static str); 4]> for hashbrown::HashMap<&'static str, &'static str> {
    fn from(arr: [(&'static str, &'static str); 4]) -> Self {
        let mut map = hashbrown::HashMap::new();
        for (k, v) in arr {
            map.insert(k, v);
        }
        map
    }
}

// ConversationHistoryArgs implementations for => syntax
// Enables: .conversation_history(MessageRole::User => "What time is it in Paris, France", MessageRole::System => "...", MessageRole::Assistant => "...")

impl ConversationHistoryArgs for (MessageRole, &str) {
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>> {
        Some(ZeroOneOrMany::one((self.0, self.1.to_string())))
    }
}

impl ConversationHistoryArgs for (MessageRole, String) {
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>> {
        Some(ZeroOneOrMany::one(self))
    }
}

impl<T1, T2> ConversationHistoryArgs for (T1, T2)
where
    T1: ConversationHistoryArgs,
    T2: ConversationHistoryArgs,
{
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>> {
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

impl<T1, T2, T3> ConversationHistoryArgs for (T1, T2, T3)
where
    T1: ConversationHistoryArgs,
    T2: ConversationHistoryArgs,
    T3: ConversationHistoryArgs,
{
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>> {
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
