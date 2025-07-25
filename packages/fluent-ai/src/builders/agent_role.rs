//! Agent role builder implementation moved from domain
//! Builders are behavioral/construction logic, separate from core domain models

use std::fmt;
use std::marker::PhantomData;

use fluent_ai_domain::{
    AgentConversation, AgentConversationMessage, AgentRole, AgentRoleAgent, AgentRoleImpl,
    AgentWithHistory, ChatMessageChunk, ContextArgs, ConversationHistoryArgs, MessageRole,
    ToolArgs, ZeroOneOrMany};
use serde_json::Value;

/// MCP Server configuration
#[derive(Debug, Clone)]
struct McpServerConfig {
    #[allow(dead_code)] // TODO: Use for MCP server type identification (stdio, socket, etc.)
    server_type: String,
    #[allow(dead_code)] // TODO: Use for MCP server binary executable path
    bin_path: Option<String>,
    #[allow(dead_code)] // TODO: Use for MCP server initialization command
    init_command: Option<String>}

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
        Option<Box<dyn Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync>>}

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
    /// MUST precede .chat()
    pub fn on_chunk<F>(self, handler: F) -> AgentRoleBuilderWithChunkHandler
    where
        F: Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync + 'static,
    {
        AgentRoleBuilderWithChunkHandler {
            inner: self,
            chunk_handler: Box::new(handler)}
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
    pub fn into_agent(self) -> AgentWithHistory {
        AgentWithHistory {
            inner: self.inner,
            chunk_handler: self.chunk_handler,
            conversation_history: None}
    }
}
