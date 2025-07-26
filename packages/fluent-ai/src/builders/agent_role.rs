//! Agent role builder implementation moved from domain
//! Builders are behavioral/construction logic, separate from core domain models

use std::fmt;
use std::marker::PhantomData;

use fluent_ai_domain::{
    AgentConversation, AgentConversationMessage, AgentRole, AgentRoleAgent, AgentRoleImpl,
    ChatMessageChunk, CompletionProvider, Context, Tool, McpServer, Memory, 
    AdditionalParams, Metadata, Conversation, MessageRole,
    ZeroOneOrMany, AsyncStream};
use crate::agent::Agent;
use crate::completion::{Message, CompletionModel};
use crate::chat::ChatLoop;
use serde_json::Value;
use std::collections::HashMap;

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
    pub fn agent_role(name: impl Into<String>) -> AgentRoleBuilder {
        AgentRoleBuilder::new(name)
    }
}

/// MCP Server builder
pub struct McpServerBuilder<T> {
    parent: AgentRoleBuilder,
    server_type: PhantomData<T>,
    bin_path: Option<String>}

/// Placeholder for Stdio type
pub struct Stdio;

impl AgentRoleBuilder {
    /// Create a new agent role builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            completion_provider: None,
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            contexts: None,
            tools: None,
            mcp_servers: None,
            additional_params: None,
            memory: None,
            metadata: None,
            on_tool_result_handler: None,
            on_conversation_turn_handler: None}
    }

    /// Set the completion provider - EXACT syntax: .completion_provider(Mistral::MagistralSmall)
    pub fn completion_provider(
        mut self,
        provider: impl std::any::Any + Send + Sync + 'static,
    ) -> Self {
        self.completion_provider = Some(Box::new(provider));
        self
    }

    /// Set model - EXACT syntax: .model(Models::Gpt4OMini)
    pub fn model(mut self, model: impl CompletionModel) -> Self {
        // Store model in the same field as completion_provider for compatibility
        self.completion_provider = Some(Box::new(model));
        self
    }

    /// Set temperature - EXACT syntax: .temperature(1.0)
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens - EXACT syntax: .max_tokens(8000)
    pub fn max_tokens(mut self, max: u64) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set system prompt - EXACT syntax: .system_prompt("...")
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add context - EXACT syntax: .context(Context<File>::of(...), Context<Files>::glob(...), ...)
    pub fn context(mut self, contexts: impl ContextArgs) -> Self {
        contexts.add_to(&mut self.contexts);
        self
    }

    /// Add MCP server - EXACT syntax: .mcp_server<Stdio>::bin(...).init(...)
    pub fn mcp_server<T>(self) -> McpServerBuilder<T> {
        McpServerBuilder {
            parent: self,
            server_type: PhantomData,
            bin_path: None}
    }

    /// Add tools - EXACT syntax: .tools(Tool<Perplexity>::new(...), Tool::named(...).bin(...).description(...))
    pub fn tools(mut self, tools: impl ToolArgs) -> Self {
        tools.add_to(&mut self.tools);
        self
    }

    /// Set additional params - EXACT syntax: .additional_params({"beta" => "true"})
    pub fn additional_params<P>(mut self, params: P) -> Self
    where
        P: Into<hashbrown::HashMap<&'static str, &'static str>>,
    {
        let config_map = params.into();
        let mut map = HashMap::new();
        for (k, v) in config_map {
            map.insert(k.to_string(), Value::String(v.to_string()));
        }
        self.additional_params = Some(map);
        self
    }

    /// Set memory - EXACT syntax: .memory(Library::named("obsidian_vault"))
    pub fn memory(mut self, memory: impl std::any::Any + Send + Sync + 'static) -> Self {
        self.memory = Some(Box::new(memory));
        self
    }

    /// Set metadata - EXACT syntax: .metadata({"key" => "val", "foo" => "bar"})
    pub fn metadata<P>(mut self, metadata: P) -> Self
    where
        P: Into<hashbrown::HashMap<&'static str, &'static str>>,
    {
        let config_map = metadata.into();
        let mut map = HashMap::new();
        for (k, v) in config_map {
            map.insert(k.to_string(), Value::String(v.to_string()));
        }
        self.metadata = Some(map);
        self
    }

    /// Set on_tool_result handler - EXACT syntax: .on_tool_result(|results| { ... })
    pub fn on_tool_result<F>(mut self, handler: F) -> Self
    where
        F: Fn(ZeroOneOrMany<Value>) + Send + Sync + 'static,
    {
        self.on_tool_result_handler = Some(Box::new(handler));
        self
    }

    /// Set on_conversation_turn handler - EXACT syntax: .on_conversation_turn(|conversation, agent| { ... })
    pub fn on_conversation_turn<F>(mut self, handler: F) -> Self
    where
        F: Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync + 'static,
    {
        self.on_conversation_turn_handler = Some(Box::new(handler));
        self
    }

    /// Add general error handler - EXACT syntax: .on_error(|error| { ... })
    /// Enables terminal methods for agent role creation
    pub fn on_error<F>(self, error_handler: F) -> AgentRoleBuilderWithHandler
    where
        F: FnMut(String) + Send + 'static,
    {
        AgentRoleBuilderWithHandler {
            inner: self,
            error_handler: Box::new(error_handler)}
    }

    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// MUST precede .into_agent()
    pub fn on_chunk<F>(self, handler: F) -> AgentBuilder
    where
        F: Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync + 'static,
    {
        AgentBuilder {
            inner: self,
            chunk_handler: Box::new(handler),
            conversation_history: ZeroOneOrMany::None,
        }
    }

    /// Convert to agent - EXACT syntax: .into_agent()
    /// Returns AgentBuilder that supports conversation_history() and chat() methods
    pub fn into_agent(self) -> AgentBuilder {
        AgentBuilder {
            inner: self,
            chunk_handler: Box::new(|chunk| chunk), // Default passthrough handler
            conversation_history: ZeroOneOrMany::None,
        }
    }
}

impl<T> McpServerBuilder<T> {
    /// Set binary path - EXACT syntax: .bin("/path/to/bin")
    pub fn bin(mut self, path: impl Into<String>) -> Self {
        self.bin_path = Some(path.into());
        self
    }

    /// Initialize - EXACT syntax: .init("cargo run -- --stdio")
    pub fn init(self, command: impl Into<String>) -> AgentRoleBuilder {
        let mut parent = self.parent;
        let new_config = McpServerConfig {
            server_type: std::any::type_name::<T>().to_string(),
            bin_path: self.bin_path,
            init_command: Some(command.into())};
        parent.mcp_servers = match parent.mcp_servers {
            Some(servers) => Some(servers.with_pushed(new_config)),
            None => Some(ZeroOneOrMany::one(new_config))};
        parent
    }
}

/// Builder with general error handler - has access to terminal methods for agent role creation
pub struct AgentRoleBuilderWithHandler {
    #[allow(dead_code)] // TODO: Use for accessing agent role configuration and building
    inner: AgentRoleBuilder,
    #[allow(dead_code)] // TODO: Use for polymorphic error handling during agent role creation
    error_handler: Box<dyn FnMut(String) + Send>}

impl AgentRoleBuilderWithHandler {
    /// Add chunk handler after error handler - EXACT syntax: .on_chunk(|chunk| { ... })
    pub fn on_chunk<F>(self, handler: F) -> AgentRoleBuilderWithChunkHandler
    where
        F: Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync + 'static,
    {
        AgentRoleBuilderWithChunkHandler {
            inner: self.inner,
            chunk_handler: Box::new(handler)}
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

/// Builder with chunk handler - has access to terminal methods
pub struct AgentRoleBuilderWithChunkHandler {
    inner: AgentRoleBuilder,
    chunk_handler: Box<dyn Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync>}

impl AgentRoleBuilderWithChunkHandler {
    /// Convert to agent - EXACT syntax: .into_agent()
    /// Returns AgentBuilder that supports conversation_history() and chat() methods
    pub fn into_agent(self) -> AgentBuilder {
        AgentBuilder {
            inner: self.inner,
            chunk_handler: self.chunk_handler,
            conversation_history: ZeroOneOrMany::None,
        }
    }
}

/// Unified AgentBuilder that handles the complete builder flow
/// This is what .into_agent() returns - supports conversation_history() and chat()
pub struct AgentBuilder {
    inner: AgentRoleBuilder,
    chunk_handler: Box<dyn Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync>,
    conversation_history: ZeroOneOrMany<(MessageRole, String)>,
}

impl AgentBuilder {
    /// Set conversation history - EXACT syntax from ARCHITECTURE.md
    /// Supports: .conversation_history(MessageRole::User => "content", MessageRole::System => "content", ...)
    pub fn conversation_history<H>(mut self, history: H) -> Self
    where
        H: ConversationHistoryArgs,
    {
        self.conversation_history = history.into_history().unwrap_or(ZeroOneOrMany::None);
        self
    }

    /// Simple chat method - EXACT syntax: .chat("Hello")
    pub fn chat(&self, message: impl Into<String>) -> AsyncStream<ChatMessageChunk> {
        let _message_text = message.into();
        
        // AsyncStream-only architecture - no Result wrapping
        // Error handling via on_chunk patterns, not Result types
        AsyncStream::empty()
    }

    /// Closure-based chat loop - EXACT syntax: .chat(|conversation| ChatLoop)  
    pub fn chat<F>(&self, closure: F) -> AsyncStream<ChatMessageChunk>
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
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
        let boxed_context = Box::new(self) as Box<dyn std::any::Any + Send + Sync>;
        *contexts = match contexts.take() {
            Some(existing) => Some(existing.with_pushed(boxed_context)),
            None => Some(ZeroOneOrMany::one(boxed_context)),
        };
    }
}

impl<T1, T2> ContextArgs for (T1, T2)
where
    T1: Context + Send + Sync + 'static,
    T2: Context + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
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
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
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
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
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
    fn add_to(self, tools: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
        let boxed_tool = Box::new(self) as Box<dyn std::any::Any + Send + Sync>;
        *tools = match tools.take() {
            Some(existing) => Some(existing.with_pushed(boxed_tool)),
            None => Some(ZeroOneOrMany::one(boxed_tool)),
        };
    }
}

impl<T1, T2> ToolArgs for (T1, T2)
where
    T1: Tool + Send + Sync + 'static,
    T2: Tool + Send + Sync + 'static,
{
    fn add_to(self, tools: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
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
    fn add_to(self, tools: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
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
