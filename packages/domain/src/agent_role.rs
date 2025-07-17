//! Agent role builder implementation following ARCHITECTURE.md exactly

use crate::MessageRole;
use crate::chunk::ChatMessageChunk;
use crate::HashMap;
use crate::ZeroOneOrMany;
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

// Builder types moved to fluent-ai/src/builders/agent_role.rs
// Only keeping core domain types here

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

// AgentRoleBuilder implementation moved to fluent-ai/src/builders/agent_role.rs

// All builder implementations moved to fluent-ai/src/builders/agent_role.rs

/// Agent with conversation history - domain data structure
pub struct AgentWithHistory {
    #[allow(dead_code)] // TODO: Use for accessing agent role configuration during chat
    inner: Box<dyn std::any::Any + Send + Sync>,
    chunk_handler: Box<
        dyn Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync,
    >,
    #[allow(dead_code)] // TODO: Use for loading previous conversation context during chat
    conversation_history: Option<ZeroOneOrMany<(MessageRole, String)>>,
}

/// Trait for context arguments - moved to fluent-ai/src/builders/
pub trait ContextArgs {
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>);
}

/// Trait for tool arguments - moved to fluent-ai/src/builders/
pub trait ToolArgs {
    fn add_to(self, tools: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>);
}

/// Trait for conversation history arguments - moved to fluent-ai/src/builders/
pub trait ConversationHistoryArgs {
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>>;
}

