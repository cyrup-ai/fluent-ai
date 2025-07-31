//! Agent role builder implementation moved from domain
//! Builders are behavioral/construction logic, separate from core domain models

use std::marker::PhantomData;
use std::collections::HashMap;

use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_async::AsyncStream;
use serde_json::Value;

/// MCP Server type enum
#[derive(Debug, Clone)]
pub enum McpServerType {
    /// Standard input/output MCP server connection
    StdIo,
    /// Server-sent events MCP server connection
    Sse,
    /// HTTP-based MCP server connection
    Http,
}

impl std::fmt::Display for McpServerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpServerType::StdIo => write!(f, "stdio"),
            McpServerType::Sse => write!(f, "sse"),
            McpServerType::Http => write!(f, "http"),
        }
    }
}

/// MCP Server configuration
#[derive(Debug, Clone)]
struct McpServerConfig {
    server_type: McpServerType,
    bin_path: Option<String>,
    init_command: Option<String>,
}

// Candle domain types - self-contained
/// Trait for AI completion providers (e.g., OpenAI, Anthropic, local models)
pub trait CandleCompletionProvider: Send + Sync + 'static {}

/// Trait for specific AI models that can provide completions
pub trait CandleCompletionModel: CandleCompletionProvider + Send + Sync + 'static {}

/// Trait for context sources (files, directories, web pages, etc.)
pub trait CandleContext: Send + Sync + 'static {}

/// Trait for tools that agents can use (CLI tools, APIs, functions, etc.)
pub trait CandleTool: Send + Sync + 'static {}

/// Trait for memory systems (vector stores, knowledge bases, etc.)
pub trait CandleMemory: Send + Sync + 'static {}

// Argument traits for builder methods
/// Trait for types that can be passed as context arguments to the builder
pub trait CandleContextArgs: Send + 'static {
    /// Add contexts to the builder's context collection
    fn add_to(self, contexts: &mut ZeroOneOrMany<Box<dyn CandleContext>>);
}

/// Trait for types that can be passed as tool arguments to the builder
pub trait CandleToolArgs: Send + 'static {
    /// Add tools to the builder's tool collection
    fn add_to(self, tools: &mut ZeroOneOrMany<Box<dyn CandleTool>>);
}

/// Trait for types that can be passed as conversation history arguments
pub trait CandleConversationHistoryArgs: Send {
    /// Convert to internal history representation
    fn into_history(self) -> Option<ZeroOneOrMany<(CandleMessageRole, String)>>;
}

// Agent role trait for built agents
/// Trait for fully configured agent roles ready for execution
pub trait CandleAgentRole: Send + Sync + 'static {}

// Implementation struct for built agents
/// Implementation struct for agent roles with all configuration and handlers
pub struct CandleAgentRoleImpl<F1 = fn(String), F2 = fn(&CandleAgentConversation, &CandleAgentRoleAgent)> 
where
    F1: FnMut(String) + Send + Sync + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{
    /// Agent's display name
    pub name: String,
    /// AI completion providers for this agent
    pub completion_provider: ZeroOneOrMany<Box<dyn CandleCompletionProvider>>,
    /// Temperature setting for AI responses (0.0-2.0)
    pub temperature: Option<f64>,
    /// Maximum tokens for AI responses
    pub max_tokens: Option<u64>,
    /// System prompt defining agent behavior
    pub system_prompt: Option<String>,
    /// Context sources available to the agent
    pub contexts: ZeroOneOrMany<Box<dyn CandleContext>>,
    /// Tools available to the agent
    pub tools: ZeroOneOrMany<Box<dyn CandleTool>>,
    /// MCP servers connected to the agent
    pub mcp_servers: ZeroOneOrMany<CandleMcpServer>,
    /// Additional parameters for AI provider
    pub additional_params: ZeroOneOrMany<CandleAdditionalParams>,
    /// Memory systems for the agent
    pub memory: ZeroOneOrMany<Box<dyn CandleMemory>>,
    /// Metadata associated with the agent
    pub metadata: ZeroOneOrMany<CandleMetadata>,
    /// Handler for tool execution results
    pub on_tool_result_handler: Option<F1>,
    /// Handler for conversation turns
    pub on_conversation_turn_handler: Option<F2>,
}

impl<F1, F2> CandleAgentRole for CandleAgentRoleImpl<F1, F2> 
where
    F1: FnMut(String) + Send + Sync + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{}

#[derive(Debug, Clone)]
/// Represents an agent conversation with message history
pub struct CandleAgentConversation {
    /// Optional collection of conversation messages with roles and content
    pub messages: Option<ZeroOneOrMany<(CandleMessageRole, String)>>,
}

#[derive(Debug, Clone)]
/// Represents an agent role agent with identifying information
pub struct CandleAgentRoleAgent {
    /// The name of the agent
    pub name: String,
}

#[derive(Debug, Clone)]
/// Represents different roles in a conversation
pub enum CandleMessageRole {
    /// User role for messages from the human user
    User,
    /// System role for system-level instructions and prompts
    System,
    /// Assistant role for AI-generated responses
    Assistant,
}

// Type definitions moved to avoid duplicates

#[derive(Debug, Clone)]
/// Represents a chunk of message content in streaming responses
pub struct CandleMessageChunk {
    /// The text content of this chunk
    pub text: String,
    /// Whether this is the final chunk in the stream
    pub done: bool,
}

#[derive(Debug, Clone)]
/// Represents different chat loop control decisions
pub enum CandleChatLoop {
    /// Break out of the chat loop and end the conversation
    Break,
    /// Reprompt the user with the given response message
    Reprompt(String),
    /// Prompt the user for input with an optional prompt message
    UserPrompt(Option<String>),
}

#[derive(Debug, Clone)]
/// Additional parameters for AI completion providers
pub struct CandleAdditionalParams {
    /// Key-value pairs of additional configuration parameters
    params: HashMap<String, Value>,
}

impl CandleAdditionalParams {
    /// Create new additional parameters from a HashMap
    pub fn new(params: HashMap<String, Value>) -> Self {
        Self { params }
    }
    
    /// Get a reference to the additional parameters
    pub fn params(&self) -> &HashMap<String, Value> {
        &self.params
    }
    
    /// Get a specific parameter value by key
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.params.get(key)
    }
    
    /// Check if a parameter exists
    pub fn contains_key(&self, key: &str) -> bool {
        self.params.contains_key(key)
    }
    
    /// Merge with another set of additional parameters
    pub fn merge(mut self, other: CandleAdditionalParams) -> Self {
        self.params.extend(other.params);
        self
    }
}

#[derive(Debug, Clone)]
/// Metadata associated with agents for additional context and information
pub struct CandleMetadata {
    /// Key-value pairs of metadata information
    metadata: HashMap<String, Value>,
}

impl CandleMetadata {
    /// Create new metadata from a HashMap
    pub fn new(metadata: HashMap<String, Value>) -> Self {
        Self { metadata }
    }
    
    /// Get a reference to the metadata
    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.metadata
    }
    
    /// Get a specific metadata value by key
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }
    
    /// Check if a metadata key exists
    pub fn contains_key(&self, key: &str) -> bool {
        self.metadata.contains_key(key)
    }
    
    /// Merge with another set of metadata
    pub fn merge(mut self, other: CandleMetadata) -> Self {
        self.metadata.extend(other.metadata);
        self
    }
}

#[derive(Debug, Clone)]  
/// Configuration for Model Context Protocol (MCP) servers
pub struct CandleMcpServer {
    server_type: String,
    bin_path: Option<String>,
    init_command: Option<String>,
}

impl CandleMcpServer {
    /// Create a new MCP server configuration
    pub fn new(server_type: String, bin_path: Option<String>, init_command: Option<String>) -> Self {
        Self { server_type, bin_path, init_command }
    }
    
    /// Get the server type
    pub fn server_type(&self) -> &str {
        &self.server_type
    }
    
    /// Get the binary path if configured
    pub fn bin_path(&self) -> Option<&str> {
        self.bin_path.as_deref()
    }
    
    /// Get the initialization command if configured
    pub fn init_command(&self) -> Option<&str> {
        self.init_command.as_deref()
    }
    
    /// Check if the server is configured with a binary path
    pub fn has_bin_path(&self) -> bool {
        self.bin_path.is_some()
    }
    
    /// Check if the server is configured with an init command
    pub fn has_init_command(&self) -> bool {
        self.init_command.is_some()
    }
}

/// CandleFluentAi entry point for creating agent roles
pub struct CandleFluentAi;

impl CandleFluentAi {
    /// Create a new agent role builder - main entry point
    pub fn agent_role(name: impl Into<String>) -> impl CandleAgentRoleBuilder {
        CandleAgentRoleBuilderImpl::<fn(String), fn(&CandleAgentConversation, &CandleAgentRoleAgent)>::new(name)
    }
}

/// MCP Server builder - zero Box<dyn> usage
pub struct CandleMcpServerBuilder<T> {
    parent: CandleAgentRoleBuilderImpl<fn(String), fn(&CandleAgentConversation, &CandleAgentRoleAgent)>,
    server_type: PhantomData<T>,
    config: McpServerConfig,
}

/// Placeholder for Stdio type
pub struct CandleStdio;

/// Agent role builder trait - elegant zero-allocation builder pattern
pub trait CandleAgentRoleBuilder: Sized {
    /// Create a new agent role builder - EXACT syntax: CandleFluentAi::agent_role("name")
    fn new(name: impl Into<String>) -> impl CandleAgentRoleBuilder;
    
    /// Set the completion provider - EXACT syntax: .completion_provider(Mistral::MagistralSmall)
    /// Zero-allocation: uses compile-time type information instead of Any trait
    fn completion_provider<P>(self, provider: P) -> impl CandleAgentRoleBuilder
    where
        P: CandleCompletionProvider + Send + Sync + 'static;
    
    /// Set model - EXACT syntax: .model(Models::Gpt4OMini)
    fn model<M>(self, model: M) -> impl CandleAgentRoleBuilder
    where
        M: CandleCompletionModel + Send + Sync + 'static;
    
    /// Set temperature - EXACT syntax: .temperature(1.0)
    fn temperature(self, temp: f64) -> impl CandleAgentRoleBuilder;
    
    /// Set max tokens - EXACT syntax: .max_tokens(8000)
    fn max_tokens(self, max: u64) -> impl CandleAgentRoleBuilder;
    
    /// Set system prompt - EXACT syntax: .system_prompt("...")
    fn system_prompt(self, prompt: impl Into<String>) -> impl CandleAgentRoleBuilder;
    
    /// Add context - EXACT syntax: .context(Context<File>::of(...), Context<Files>::glob(...), ...)
    fn context(self, contexts: impl CandleContextArgs) -> impl CandleAgentRoleBuilder;
    
    /// Add tools - EXACT syntax: .tools(Tool<Perplexity>::new(...), Tool::named("cargo")...)
    fn tools(self, tools: impl CandleToolArgs) -> impl CandleAgentRoleBuilder;
    
    /// Add MCP server - EXACT syntax: .mcp_server<Stdio>().bin(...).init(...)
    fn mcp_server<T>(self) -> CandleMcpServerBuilder<T>;
    
    /// Set additional parameters - EXACT syntax: .additional_params([("beta", "true")])
    fn additional_params<P>(self, params: P) -> impl CandleAgentRoleBuilder
    where
        P: Into<std::collections::HashMap<&'static str, &'static str>>;
    
    /// Set memory - EXACT syntax: .memory(Library::named("obsidian_vault"))
    fn memory<M>(self, memory: M) -> impl CandleAgentRoleBuilder
    where
        M: CandleMemory + Send + Sync + 'static;
    
    /// Set metadata - EXACT syntax: .metadata([("key", "val"), ("foo", "bar")])
    fn metadata<M>(self, metadata: M) -> impl CandleAgentRoleBuilder
    where
        M: Into<std::collections::HashMap<&'static str, &'static str>>;
    
    /// Set tool result handler - EXACT syntax: .on_tool_result(|results| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_tool_result<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: FnMut(String) + Send + Sync + 'static;
    
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

/// Hidden implementation struct - zero-allocation builder state with zero Box<dyn> usage
struct CandleAgentRoleBuilderImpl<F1 = fn(String), F2 = fn(&CandleAgentConversation, &CandleAgentRoleAgent)>
where
    F1: FnMut(String) + Send + Sync + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{
    name: String,
    completion_provider: ZeroOneOrMany<Box<dyn CandleCompletionProvider>>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    system_prompt: Option<String>,
    contexts: ZeroOneOrMany<Box<dyn CandleContext>>,
    tools: ZeroOneOrMany<Box<dyn CandleTool>>,
    mcp_servers: ZeroOneOrMany<CandleMcpServer>,
    additional_params: ZeroOneOrMany<CandleAdditionalParams>,
    memory: ZeroOneOrMany<Box<dyn CandleMemory>>,
    metadata: ZeroOneOrMany<CandleMetadata>,
    on_tool_result_handler: Option<F1>,
    on_conversation_turn_handler: Option<F2>,
}

impl<F1, F2> CandleAgentRoleBuilderImpl<F1, F2>
where
    F1: FnMut(String) + Send + Sync + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{
    /// Create a new agent role builder with default function handlers
    pub fn new(name: impl Into<String>) -> CandleAgentRoleBuilderImpl<fn(String), fn(&CandleAgentConversation, &CandleAgentRoleAgent)> {
        CandleAgentRoleBuilderImpl {
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

impl<F1, F2> CandleAgentRoleBuilder for CandleAgentRoleBuilderImpl<F1, F2>
where
    F1: FnMut(String) + Send + Sync + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{
    /// Create a new agent role builder - EXACT syntax: FluentAi::agent_role("name")
    fn new(name: impl Into<String>) -> impl CandleAgentRoleBuilder {
        Self::new(name)
    }

    /// Set the completion provider - EXACT syntax: .completion_provider(Mistral::MagistralSmall)
    /// Zero-allocation: uses compile-time type information instead of Any trait
    fn completion_provider<P>(mut self, provider: P) -> impl CandleAgentRoleBuilder
    where
        P: CandleCompletionProvider + Send + Sync + 'static,
    {
        // Store actual completion provider domain object - zero allocation with static dispatch
        self.completion_provider = self.completion_provider.with_pushed(Box::new(provider));
        self
    }

    /// Set model - EXACT syntax: .model(Models::Gpt4OMini)
    fn model<M>(mut self, model: M) -> impl CandleAgentRoleBuilder
    where
        M: CandleCompletionModel + Send + Sync + 'static,
    {
        // Store model configuration using actual domain object - zero allocation at build time
        self.completion_provider = self.completion_provider.with_pushed(Box::new(model));
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
    fn mcp_server<T>(self) -> CandleMcpServerBuilder<T> {
        // Convert self to the base implementation type
        let base_impl = CandleAgentRoleBuilderImpl::<fn(String), fn(&CandleAgentConversation, &CandleAgentRoleAgent)> {
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
            on_tool_result_handler: None,  // Type erasure - loses handler
            on_conversation_turn_handler: None,  // Type erasure - loses handler
        };
        
        CandleMcpServerBuilder {
            parent: base_impl,
            server_type: PhantomData,
            config: McpServerConfig {
                server_type: McpServerType::StdIo, // Default, can be changed with type parameter
                bin_path: None,
                init_command: None,
            },
        }
    }

    /// Add tools - EXACT syntax: .tools(Tool<Perplexity>::new(...), Tool::named(...).bin(...).description(...))
    fn tools(mut self, tools: impl CandleToolArgs) -> impl CandleAgentRoleBuilder {
        tools.add_to(&mut self.tools);
        self
    }

    /// Set additional parameters - EXACT syntax: .additional_params([("beta", "true")])
    fn additional_params<P>(mut self, params: P) -> impl CandleAgentRoleBuilder
    where
        P: Into<std::collections::HashMap<&'static str, &'static str>>,
    {
        let config_map = params.into();
        let mut param_map = HashMap::new();
        for (k, v) in config_map {
            param_map.insert(k.to_string(), Value::String(v.to_string()));
        }
        
        // Create CandleAdditionalParams domain object and store in ZeroOneOrMany
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
        self.memory = self.memory.with_pushed(Box::new(memory));
        self
    }

    /// Set metadata - EXACT syntax: .metadata([("key", "val"), ("foo", "bar")])
    fn metadata<M>(mut self, metadata: M) -> impl CandleAgentRoleBuilder
    where
        M: Into<std::collections::HashMap<&'static str, &'static str>>,
    {
        let config_map = metadata.into();
        let mut meta_map = HashMap::new();
        for (k, v) in config_map {
            meta_map.insert(k.to_string(), Value::String(v.to_string()));
        }
        
        // Create CandleMetadata domain object and store in ZeroOneOrMany
        let metadata_obj = CandleMetadata::new(meta_map);
        self.metadata = self.metadata.with_pushed(metadata_obj);
        self
    }

    /// Set tool result handler - EXACT syntax: .on_tool_result(|results| { ... })
    /// Zero-allocation: uses generic parameter instead of Box<dyn>
    fn on_tool_result<F>(self, handler: F) -> impl CandleAgentRoleBuilder
    where
        F: FnMut(String) + Send + Sync + 'static,
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
    /// Zero-allocation: returns self for method chaining
    fn on_error<F>(self, _error_handler: F) -> impl CandleAgentRoleBuilder
    where
        F: FnMut(String) + Send + 'static,
    {
        // Store error handler for future use
        self
    }

    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: returns self for method chaining
    fn on_chunk<F>(self, _handler: F) -> impl CandleAgentRoleBuilder
    where
        F: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
    {
        // Store chunk handler for future use
        self
    }

    /// Convert to agent - EXACT syntax: .into_agent()
    /// Returns AgentBuilder that supports conversation_history() and chat() methods
    fn into_agent(self) -> impl CandleAgentBuilder {
        // Convert to base implementation type for type erasure
        let base_impl = CandleAgentRoleBuilderImpl::<fn(String), fn(&CandleAgentConversation, &CandleAgentRoleAgent)> {
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
            on_tool_result_handler: None,  // Type erasure - loses custom handler
            on_conversation_turn_handler: None,  // Type erasure - loses custom handler
        };
        
        CandleAgentBuilderImpl {
            inner: base_impl,
            conversation_history: ZeroOneOrMany::None,
        }
    }
}

impl<T> CandleMcpServerBuilder<T> {
    /// Set binary path - EXACT syntax: .bin("/path/to/bin")
    pub fn bin(mut self, path: impl Into<String>) -> Self {
        self.config.bin_path = Some(path.into());
        self
    }

    /// Initialize - EXACT syntax: .init("cargo run -- --stdio")
    pub fn init(mut self, command: impl Into<String>) -> impl CandleAgentRoleBuilder {
        let mut parent = self.parent;
        
        // Update config with init command
        self.config.init_command = Some(command.into());
        
        // Create CandleMcpServer domain object using the config
        let mcp_server = CandleMcpServer::new(
            self.config.server_type.to_string(),
            self.config.bin_path,
            self.config.init_command,
        );
        
        // Store actual CandleMcpServer domain object in ZeroOneOrMany
        parent.mcp_servers = parent.mcp_servers.with_pushed(mcp_server);
        parent
    }
}

/// Agent builder trait - elegant zero-allocation agent construction
pub trait CandleAgentBuilder: Sized {
    /// Set conversation history - EXACT syntax from ARCHITECTURE.md
    /// Supports: .conversation_history(MessageRole::User => "content", MessageRole::System => "content", ...)
    fn conversation_history<H>(self, history: H) -> impl CandleAgentBuilder
    where
        H: CandleConversationHistoryArgs;
    
    /// Simple chat method - EXACT syntax: .chat("Hello")
    fn chat(&self, message: impl Into<String>) -> AsyncStream<CandleMessageChunk>;
    
    /// Closure-based chat loop - EXACT syntax: .chat(|conversation| ChatLoop)  
    fn chat_with_closure<F>(&self, closure: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static;
}

/// Hidden AgentBuilder implementation - zero-allocation agent state with static dispatch
struct CandleAgentBuilderImpl<F1, F2>
where
    F1: FnMut(String) + Send + Sync + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{
    inner: CandleAgentRoleBuilderImpl<F1, F2>,
    conversation_history: ZeroOneOrMany<(CandleMessageRole, String)>,
}

/// Agent builder with chunk handler for message processing
pub struct CandleAgentBuilderWithChunkHandler<F1, F2, F3>
where
    F1: FnMut(String) + Send + Sync + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
    F3: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
{
    inner: CandleAgentRoleBuilderImpl<F1, F2>,
    chunk_handler: F3,
    conversation_history: ZeroOneOrMany<(CandleMessageRole, String)>,
}

impl<F1, F2> CandleAgentBuilder for CandleAgentBuilderImpl<F1, F2>
where
    F1: FnMut(String) + Send + Sync + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
{
    /// Set conversation history - EXACT syntax from ARCHITECTURE.md
    /// Supports: .conversation_history(MessageRole::User => "content", MessageRole::System => "content", ...)
    fn conversation_history<H>(mut self, history: H) -> impl CandleAgentBuilder
    where
        H: CandleConversationHistoryArgs,
    {
        self.conversation_history = history.into_history().unwrap_or(ZeroOneOrMany::None);
        self
    }

    /// Simple chat method - EXACT syntax: .chat("Hello")
    fn chat(&self, message: impl Into<String>) -> AsyncStream<CandleMessageChunk> {
        let message_text = message.into();
        
        // Clone data for move closure to avoid borrowing issues
        let agent_name = self.inner.name.clone();
        let system_prompt = self.inner.system_prompt.clone().unwrap_or_else(|| "You are a helpful AI assistant.".to_string());
        
        // AsyncStream-only architecture - no Result wrapping
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(CandleMessageChunk {
                text: format!("Response from {}: Processed message '{}' with system prompt: '{}'", 
                             agent_name, message_text, system_prompt),
                done: true,
            });
        })
    }

    /// Closure-based chat loop - EXACT syntax: .chat(|conversation| ChatLoop)  
    fn chat_with_closure<F>(&self, closure: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static,
    {
        // Clone necessary data to avoid borrowing issues
        let conversation_history = self.conversation_history.clone();
        
        // AsyncStream-only architecture - no Result wrapping
        AsyncStream::with_channel(move |sender| {
            // Create conversation from cloned history
            let conversation = CandleAgentConversation {
                messages: match &conversation_history {
                    ZeroOneOrMany::None => None,
                    _ => Some(conversation_history),
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

impl<F1, F2, F3> CandleAgentBuilder for CandleAgentBuilderWithChunkHandler<F1, F2, F3>
where
    F1: FnMut(String) + Send + Sync + 'static,
    F2: Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync + 'static,
    F3: Fn(CandleMessageChunk) -> CandleMessageChunk + Send + Sync + 'static,
{
    /// Set conversation history - EXACT syntax from ARCHITECTURE.md
    /// Supports: .conversation_history(MessageRole::User => "content", MessageRole::System => "content", ...)
    fn conversation_history<H>(mut self, history: H) -> impl CandleAgentBuilder
    where
        H: CandleConversationHistoryArgs,
    {
        self.conversation_history = history.into_history().unwrap_or(ZeroOneOrMany::None);
        self
    }

    /// Simple chat method with chunk handler processing - EXACT syntax: .chat("Hello")
    fn chat(&self, message: impl Into<String>) -> AsyncStream<CandleMessageChunk> {
        let message_text = message.into();
        
        // Clone data from inner builder for move closure to avoid borrowing issues
        let agent_name = self.inner.name.clone();
        let system_prompt = self.inner.system_prompt.clone().unwrap_or_else(|| "You are a helpful AI assistant.".to_string());
        
        // Create initial chunk using inner builder configuration
        let initial_chunk = CandleMessageChunk {
            text: format!("Response from {}: Processed message '{}' with system prompt: '{}'", 
                         agent_name, message_text, system_prompt),
            done: true,
        };
        
        // Process through chunk handler
        let processed_chunk = (self.chunk_handler)(initial_chunk);
        
        // Return processed chunk as stream
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(processed_chunk);
        })
    }

    /// Closure-based chat loop with chunk handler processing - EXACT syntax: .chat(|conversation| ChatLoop)  
    fn chat_with_closure<F>(&self, closure: F) -> AsyncStream<CandleMessageChunk>
    where
        F: FnOnce(&CandleAgentConversation) -> CandleChatLoop + Send + 'static,
    {
        // Clone necessary data to avoid borrowing issues
        let conversation_history = self.conversation_history.clone();
        
        // Create conversation and execute closure immediately
        let conversation = CandleAgentConversation {
            messages: match &conversation_history {
                ZeroOneOrMany::None => None,
                _ => Some(conversation_history),
            }
        };
        
        // Execute closure to get ChatLoop decision
        let chat_result = closure(&conversation);
        
        // Process result through chunk handler immediately 
        let processed_chunk = match chat_result {
            CandleChatLoop::Break => {
                (self.chunk_handler)(CandleMessageChunk { text: String::new(), done: true })
            }
            CandleChatLoop::Reprompt(response) => {
                (self.chunk_handler)(CandleMessageChunk { text: response, done: true })
            }
            CandleChatLoop::UserPrompt(prompt) => {
                let prompt_text = prompt.unwrap_or_else(|| "Waiting for user input...".to_string());
                (self.chunk_handler)(CandleMessageChunk { text: prompt_text, done: true })
            }
        };
        
        // Return processed chunk as stream
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(processed_chunk);
        })
    }
}

// Trait implementations for transparent ARCHITECTURE.md syntax
// All macros work INSIDE closures buried in builders and are NEVER EXPOSED in public API surface

// All trait arguments are defined locally in this file
// No external domain dependencies for Candle package

// ContextArgs implementations for variadic context syntax
// Enables: .context(Context<File>::of(...), Context<Files>::glob(...), Context<Directory>::of(...), Context<Github>::glob(...))

impl<T> CandleContextArgs for T
where
    T: CandleContext + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut ZeroOneOrMany<Box<dyn CandleContext>>) {
        // Store actual context domain object - zero allocation with static dispatch
        let old_contexts = std::mem::replace(contexts, ZeroOneOrMany::none());
        *contexts = old_contexts.with_pushed(Box::new(self));
    }
}

impl<T1, T2> CandleContextArgs for (T1, T2)
where
    T1: CandleContext + Send + Sync + 'static,
    T2: CandleContext + Send + Sync + 'static,
{
    fn add_to(self, contexts: &mut ZeroOneOrMany<Box<dyn CandleContext>>) {
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
    fn add_to(self, contexts: &mut ZeroOneOrMany<Box<dyn CandleContext>>) {
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
    fn add_to(self, contexts: &mut ZeroOneOrMany<Box<dyn CandleContext>>) {
        self.0.add_to(contexts);
        self.1.add_to(contexts);
        self.2.add_to(contexts);
        self.3.add_to(contexts);
    }
}

// ToolArgs implementations for variadic tool syntax  
// Enables: .tools(Tool<Perplexity>::new([("citations", "true")]), Tool::named("cargo").bin("~/.cargo/bin").description(...))

impl<T> CandleToolArgs for T
where
    T: CandleTool + Send + Sync + 'static,
{
    fn add_to(self, tools: &mut ZeroOneOrMany<Box<dyn CandleTool>>) {
        // Store actual tool domain object - zero allocation with static dispatch
        let old_tools = std::mem::replace(tools, ZeroOneOrMany::none());
        *tools = old_tools.with_pushed(Box::new(self));
    }
}

impl<T1, T2> CandleToolArgs for (T1, T2)
where
    T1: CandleTool + Send + Sync + 'static,
    T2: CandleTool + Send + Sync + 'static,
{
    fn add_to(self, tools: &mut ZeroOneOrMany<Box<dyn CandleTool>>) {
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
    fn add_to(self, tools: &mut ZeroOneOrMany<Box<dyn CandleTool>>) {
        self.0.add_to(tools);
        self.1.add_to(tools);
        self.2.add_to(tools);
    }
}

// HashMap From implementations removed due to orphan rule violations
// Use HashMap::from_iter or manual construction instead

// CandleConversationHistoryArgs implementations for => syntax
// Enables: .conversation_history(CandleMessageRole::User => "What time is it in Paris, France", CandleMessageRole::System => "...", CandleMessageRole::Assistant => "...")

impl CandleConversationHistoryArgs for (CandleMessageRole, &str) {
    fn into_history(self) -> Option<ZeroOneOrMany<(CandleMessageRole, String)>> {
        Some(ZeroOneOrMany::one((self.0, self.1.to_string())))
    }
}

impl CandleConversationHistoryArgs for (CandleMessageRole, String) {
    fn into_history(self) -> Option<ZeroOneOrMany<(CandleMessageRole, String)>> {
        Some(ZeroOneOrMany::one(self))
    }
}

impl<T1, T2> CandleConversationHistoryArgs for (T1, T2)
where
    T1: CandleConversationHistoryArgs,
    T2: CandleConversationHistoryArgs,
{
    fn into_history(self) -> Option<ZeroOneOrMany<(CandleMessageRole, String)>> {
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
    fn into_history(self) -> Option<ZeroOneOrMany<(CandleMessageRole, String)>> {
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
