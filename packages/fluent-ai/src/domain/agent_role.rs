//! Agent role builder implementation following ARCHITECTURE.md exactly

use crate::domain::MessageRole;
use crate::domain::chunk::ChatMessageChunk;
use crate::{AsyncStream, HashMap, ZeroOneOrMany};
use serde_json::Value;
use std::fmt;
use std::marker::PhantomData;

/// MCP Server configuration
#[derive(Debug, Clone)]
struct McpServerConfig {
    #[allow(dead_code)] // TODO: Use for MCP server type identification (stdio, socket, etc.)
    server_type: String,
    #[allow(dead_code)] // TODO: Use for MCP server binary executable path
    bin_path: Option<String>,
    #[allow(dead_code)] // TODO: Use for MCP server initialization command
    init_command: Option<String>,
}

/// Core agent role trait defining all operations and properties
pub trait AgentRole: Send + Sync + fmt::Debug + Clone {
    /// Get the name of the agent role
    fn name(&self) -> &str;
    
    /// Get the temperature setting
    fn temperature(&self) -> Option<f64>;
    
    /// Get the max tokens setting
    fn max_tokens(&self) -> Option<u64>;
    
    /// Get the system prompt
    fn system_prompt(&self) -> Option<&str>;
    
    /// Create a new agent role with the given name
    fn new(name: impl Into<String>) -> Self;
}

/// Default implementation of the AgentRole trait
pub struct AgentRoleImpl {
    name: String,
    #[allow(dead_code)] // TODO: Use for completion provider integration (OpenAI, Anthropic, etc.)
    completion_provider: Option<Box<dyn std::any::Any + Send + Sync>>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    system_prompt: Option<String>,
    #[allow(dead_code)] // TODO: Use for document context loading and management
    contexts: Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>,
    #[allow(dead_code)] // TODO: Use for tool integration and function calling
    tools: Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>,
    #[allow(dead_code)] // TODO: Use for MCP server configuration and management
    mcp_servers: Option<ZeroOneOrMany<McpServerConfig>>,
    #[allow(dead_code)] // TODO: Use for provider-specific parameters (beta features, custom options)
    additional_params: Option<HashMap<String, Value>>,
    #[allow(dead_code)] // TODO: Use for persistent memory and conversation storage
    memory: Option<Box<dyn std::any::Any + Send + Sync>>,
    #[allow(dead_code)] // TODO: Use for agent metadata and custom attributes
    metadata: Option<HashMap<String, Value>>,
    #[allow(dead_code)] // TODO: Use for tool result processing and callback handling
    on_tool_result_handler: Option<Box<dyn Fn(ZeroOneOrMany<Value>) + Send + Sync>>,
    #[allow(dead_code)] // TODO: Use for conversation turn event handling and logging
    on_conversation_turn_handler:
        Option<Box<dyn Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync>>,
}

impl std::fmt::Debug for AgentRoleImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentRoleImpl")
            .field("name", &self.name)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("system_prompt", &self.system_prompt)
            .field("additional_params", &self.additional_params)
            .field("completion_provider", &"<opaque>")
            .field("contexts", &"<opaque>")
            .field("tools", &"<opaque>")
            .field("mcp_servers", &self.mcp_servers)
            .field("memory", &"<opaque>")
            .field("metadata", &self.metadata)
            .field("on_tool_result_handler", &"<function>")
            .field("on_conversation_turn_handler", &"<function>")
            .finish()
    }
}

impl Clone for AgentRoleImpl {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            completion_provider: None, // Can't clone trait objects
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            system_prompt: self.system_prompt.clone(),
            contexts: None, // Can't clone trait objects
            tools: None, // Can't clone trait objects
            mcp_servers: self.mcp_servers.clone(),
            additional_params: self.additional_params.clone(),
            memory: None, // Can't clone trait objects
            metadata: self.metadata.clone(),
            on_tool_result_handler: None, // Can't clone function pointers
            on_conversation_turn_handler: None, // Can't clone function pointers
        }
    }
}

impl AgentRole for AgentRoleImpl {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn temperature(&self) -> Option<f64> {
        self.temperature
    }
    
    fn max_tokens(&self) -> Option<u64> {
        self.max_tokens
    }
    
    fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }
    
    fn new(name: impl Into<String>) -> Self {
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
            on_conversation_turn_handler: None,
        }
    }
}

/// Builder for creating agent roles - EXACT API from ARCHITECTURE.md
pub struct AgentRoleBuilder {
    name: String,
    completion_provider: Option<Box<dyn std::any::Any + Send + Sync>>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    system_prompt: Option<String>,
    contexts: Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>,
    tools: Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>,
    mcp_servers: Option<ZeroOneOrMany<McpServerConfig>>,
    additional_params: Option<HashMap<String, Value>>,
    memory: Option<Box<dyn std::any::Any + Send + Sync>>,
    metadata: Option<HashMap<String, Value>>,
    on_tool_result_handler: Option<Box<dyn Fn(ZeroOneOrMany<Value>) + Send + Sync>>,
    on_conversation_turn_handler:
        Option<Box<dyn Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync>>,
}

/// MCP Server builder
pub struct McpServerBuilder<T> {
    parent: AgentRoleBuilder,
    server_type: PhantomData<T>,
    bin_path: Option<String>,
}

/// Placeholder for Stdio type
pub struct Stdio;

/// Agent type placeholder for agent role
pub struct AgentRoleAgent;

/// Agent conversation type
pub struct AgentConversation {
    messages: Option<ZeroOneOrMany<(MessageRole, String)>>,
}

impl AgentConversation {
    pub fn last(&self) -> AgentConversationMessage {
        AgentConversationMessage {
            content: self
                .messages
                .as_ref()
                .and_then(|msgs| {
                    // Get the last element from ZeroOneOrMany
                    let all: Vec<_> = msgs.clone().into_iter().collect();
                    all.last().map(|(_, m)| m.clone())
                })
                .unwrap_or_default(),
        }
    }
}

pub struct AgentConversationMessage {
    content: String,
}

impl AgentConversationMessage {
    pub fn message(&self) -> &str {
        &self.content
    }
}

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
            on_conversation_turn_handler: None,
        }
    }

    /// Set the completion provider - EXACT syntax: .completion_provider(Mistral::MagistralSmall)
    pub fn completion_provider(
        mut self,
        provider: impl std::any::Any + Send + Sync + 'static,
    ) -> Self {
        self.completion_provider = Some(Box::new(provider));
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
            bin_path: None,
        }
    }

    /// Add tools - EXACT syntax: .tools(Tool<Perplexity>::new(...), Tool::named(...).bin(...).description(...))
    pub fn tools(mut self, tools: impl ToolArgs) -> Self {
        tools.add_to(&mut self.tools);
        self
    }

    /// Set additional params - EXACT syntax: .additional_params(hash_map_fn!({"beta" => "true"}))
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

    /// Set metadata - EXACT syntax: .metadata(hash_map_fn!({"key" => "val", "foo" => "bar"}))
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
            error_handler: Box::new(error_handler),
        }
    }

    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// MUST precede .chat()
    pub fn on_chunk<F>(self, handler: F) -> AgentRoleBuilderWithChunkHandler
    where
        F: Fn(ChatMessageChunk) -> ChatMessageChunk
            + Send
            + Sync
            + 'static,
    {
        AgentRoleBuilderWithChunkHandler {
            inner: self,
            chunk_handler: Box::new(handler),
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
            init_command: Some(command.into()),
        };
        parent.mcp_servers = match parent.mcp_servers {
            Some(mut servers) => {
                servers.push(new_config);
                Some(servers)
            }
            None => Some(ZeroOneOrMany::one(new_config)),
        };
        parent
    }
}

/// Builder with general error handler - has access to terminal methods for agent role creation
pub struct AgentRoleBuilderWithHandler {
    #[allow(dead_code)] // TODO: Use for accessing agent role configuration and building
    inner: AgentRoleBuilder,
    #[allow(dead_code)] // TODO: Use for polymorphic error handling during agent role creation
    error_handler: Box<dyn FnMut(String) + Send>,
}

impl AgentRoleBuilderWithHandler {
    /// Add chunk handler after error handler - EXACT syntax: .on_chunk(|chunk| { ... })
    pub fn on_chunk<F>(self, handler: F) -> AgentRoleBuilderWithChunkHandler
    where
        F: Fn(ChatMessageChunk) -> ChatMessageChunk
            + Send
            + Sync
            + 'static,
    {
        AgentRoleBuilderWithChunkHandler {
            inner: self.inner,
            chunk_handler: Box::new(handler),
        }
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
            on_conversation_turn_handler: self.inner.on_conversation_turn_handler,
        }
    }
}

/// Builder with chunk handler - has access to terminal methods
pub struct AgentRoleBuilderWithChunkHandler {
    inner: AgentRoleBuilder,
    chunk_handler: Box<
        dyn Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync,
    >,
}

impl AgentRoleBuilderWithChunkHandler {
    /// Convert to agent - EXACT syntax: .into_agent()
    pub fn into_agent(self) -> AgentWithHistory {
        AgentWithHistory {
            inner: self.inner,
            chunk_handler: self.chunk_handler,
            conversation_history: None,
        }
    }
}

/// Agent with conversation history
pub struct AgentWithHistory {
    #[allow(dead_code)] // TODO: Use for accessing agent role configuration during chat
    inner: AgentRoleBuilder,
    chunk_handler: Box<
        dyn Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync,
    >,
    #[allow(dead_code)] // TODO: Use for loading previous conversation context during chat
    conversation_history: Option<ZeroOneOrMany<(MessageRole, String)>>,
}

impl AgentWithHistory {
    /// Set conversation history - EXACT syntax: .conversation_history(MessageRole::User => "...", MessageRole::System => "...", MessageRole::Assistant => "...")
    pub fn conversation_history(mut self, history: impl ConversationHistoryArgs) -> Self {
        self.conversation_history = history.into_history();
        self
    }

    /// Start chat - EXACT syntax: .chat("Hello")
    pub fn chat(self, message: impl Into<String>) -> AsyncStream<ChatMessageChunk> {
        let message = message.into();
        let handler = self.chunk_handler;

        // Create channel for streaming chunks
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Spawn task to handle chat
        tokio::spawn(async move {
            // Send conversation history first
            if let Some(history) = self.conversation_history {
                for (role, content) in history.into_iter() {
                    let chunk = ChatMessageChunk::new(content, role);
                    let processed_chunk = handler(chunk);
                    let _ = tx.send(processed_chunk);
                }
            }

            // Send the new user message
            let user_chunk = ChatMessageChunk::new(message.clone(), MessageRole::User);
            let processed_chunk = handler(user_chunk);
            let _ = tx.send(processed_chunk);

            // TODO: Actual implementation will delegate to completion provider
            let response_chunk =
                ChatMessageChunk::new("Response placeholder", MessageRole::Assistant);
            let processed_chunk = handler(response_chunk);
            let _ = tx.send(processed_chunk);
        });

        AsyncStream::new(rx)
    }
}

/// Trait for context arguments
pub trait ContextArgs {
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>);
}

/// Trait for tool arguments
pub trait ToolArgs {
    fn add_to(self, tools: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>);
}

/// Trait for conversation history arguments
pub trait ConversationHistoryArgs {
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>>;
}

// Implement ContextArgs for tuples to support multiple arguments
// Note: We can't have a blanket impl for T1 because it conflicts with tuple impls
// Users must use tuples for multiple arguments

// Implement for single Context<T> items
impl<T> ContextArgs for crate::domain::context::Context<T>
where
    T: Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
        match contexts {
            Some(list) => list.push(Box::new(self)),
            None => *contexts = Some(ZeroOneOrMany::one(Box::new(self))),
        }
    }
}

impl<T1, T2> ContextArgs for (T1, T2)
where
    T1: std::any::Any + Send + Sync + 'static,
    T2: std::any::Any + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
        match contexts {
            Some(list) => {
                list.push(Box::new(self.0));
                list.push(Box::new(self.1));
            }
            None => {
                *contexts = Some(ZeroOneOrMany::many(vec![
                    Box::new(self.0),
                    Box::new(self.1),
                ]))
            }
        }
    }
}

impl<T1, T2, T3> ContextArgs for (T1, T2, T3)
where
    T1: std::any::Any + Send + Sync + 'static,
    T2: std::any::Any + Send + Sync + 'static,
    T3: std::any::Any + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
        match contexts {
            Some(list) => {
                list.push(Box::new(self.0));
                list.push(Box::new(self.1));
                list.push(Box::new(self.2));
            }
            None => {
                *contexts = Some(ZeroOneOrMany::many(vec![
                    Box::new(self.0),
                    Box::new(self.1),
                    Box::new(self.2),
                ]))
            }
        }
    }
}

impl<T1, T2, T3, T4> ContextArgs for (T1, T2, T3, T4)
where
    T1: std::any::Any + Send + Sync + 'static,
    T2: std::any::Any + Send + Sync + 'static,
    T3: std::any::Any + Send + Sync + 'static,
    T4: std::any::Any + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
        match contexts {
            Some(list) => {
                list.push(Box::new(self.0));
                list.push(Box::new(self.1));
                list.push(Box::new(self.2));
                list.push(Box::new(self.3));
            }
            None => {
                *contexts = Some(ZeroOneOrMany::many(vec![
                    Box::new(self.0),
                    Box::new(self.1),
                    Box::new(self.2),
                    Box::new(self.3),
                ]))
            }
        }
    }
}

// Implement ToolArgs for tuples
// Implement for single Tool<T> items
impl<T> ToolArgs for crate::domain::tool_v2::Tool<T>
where
    T: Send + Sync + 'static,
{
    fn add_to(self, tools: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
        match tools {
            Some(existing) => existing.push(Box::new(self)),
            None => *tools = Some(ZeroOneOrMany::one(Box::new(self))),
        }
    }
}

// Implement for NamedTool
impl ToolArgs for crate::domain::tool_v2::NamedTool {
    fn add_to(self, tools: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
        match tools {
            Some(existing) => existing.push(Box::new(self)),
            None => *tools = Some(ZeroOneOrMany::one(Box::new(self))),
        }
    }
}

impl<T1, T2> ToolArgs for (T1, T2)
where
    T1: std::any::Any + Send + Sync + 'static,
    T2: std::any::Any + Send + Sync + 'static,
{
    fn add_to(self, tools: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>) {
        match tools {
            Some(existing) => {
                existing.push(Box::new(self.0));
                existing.push(Box::new(self.1));
            }
            None => {
                *tools = Some(ZeroOneOrMany::many(vec![
                    Box::new(self.0),
                    Box::new(self.1),
                ]))
            }
        }
    }
}


// Support for conversation history with tuple syntax
impl ConversationHistoryArgs for ZeroOneOrMany<(MessageRole, String)> {
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>> {
        Some(self)
    }
}

// Support for single tuple
impl ConversationHistoryArgs for (MessageRole, &str) {
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>> {
        Some(ZeroOneOrMany::one((self.0, self.1.to_string())))
    }
}

// Support for multiple tuples
impl ConversationHistoryArgs for ((MessageRole, &str), (MessageRole, &str)) {
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>> {
        Some(ZeroOneOrMany::many(vec![
            (self.0.0, self.0.1.to_string()),
            (self.1.0, self.1.1.to_string()),
        ]))
    }
}

impl ConversationHistoryArgs for ((MessageRole, &str), (MessageRole, &str), (MessageRole, &str)) {
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>> {
        Some(ZeroOneOrMany::many(vec![
            (self.0.0, self.0.1.to_string()),
            (self.1.0, self.1.1.to_string()),
            (self.2.0, self.2.1.to_string()),
        ]))
    }
}

