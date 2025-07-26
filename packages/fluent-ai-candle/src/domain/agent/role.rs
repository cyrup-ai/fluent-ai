//! Agent role trait and implementation

use std::fmt;
use std::sync::atomic::AtomicUsize;

// Ultra-high-performance zero-allocation imports
use atomic_counter::RelaxedCounter;
use crossbeam_utils::CachePadded;
use once_cell::sync::Lazy;
use serde_json::Value;

use hashbrown::HashMap;
use crate::domain::{
    chat::message::CandleMessageRole as MessageRole, CandleZeroOneOrMany as ZeroOneOrMany};
use crate::domain::completion::CandleCompletionModel;
use crate::domain::context::CandleContext;
use crate::domain::tool::CandleTool;
use crate::domain::memory::CandleMemory;
// Note: These types may need to be defined or imported from their specific modules
// use crate::domain::CandleAdditionalParams;
// use crate::domain::CandleMetadata;
// use crate::domain::CandleAgentConversation;
// use crate::domain::CandleAgentRoleAgent;
// Unused imports cleaned up

/// Maximum number of relevant memories for context injection
#[allow(dead_code)] // TODO: Implement in memory context system
const MAX_RELEVANT_MEMORIES: usize = 10;

/// Global atomic counter for memory node creation
#[allow(dead_code)] // TODO: Implement in memory node creation system
static MEMORY_NODE_COUNTER: Lazy<CachePadded<RelaxedCounter>> =
    Lazy::new(|| CachePadded::new(RelaxedCounter::new(0)));

/// Global atomic counter for attention scoring operations
#[allow(dead_code)] // TODO: Implement in attention scoring system
static ATTENTION_SCORE_COUNTER: Lazy<CachePadded<AtomicUsize>> =
    Lazy::new(|| CachePadded::new(AtomicUsize::new(0)));

/// MCP Server configuration
#[derive(Debug, Clone)]
struct McpServerConfig {
    #[allow(dead_code)] // TODO: Use for MCP server type identification (stdio, socket, etc.)
    server_type: String,
    #[allow(dead_code)] // TODO: Use for MCP server binary executable path
    bin_path: Option<String>,
    #[allow(dead_code)] // TODO: Use for MCP server initialization command
    init_command: Option<String>}

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

/// Default implementation of the AgentRole trait with zero-allocation typed fields
pub struct AgentRoleImpl {
    name: String,
    /// Completion provider with proper static dispatch and zero Box<dyn> usage
    completion_provider: ZeroOneOrMany<CandleCompletionModel>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    system_prompt: Option<String>,
    /// OpenAI API key for completions (reads from OPENAI_API_KEY environment variable if not set)
    api_key: Option<String>,
    /// Document contexts with proper static dispatch and zero Box<dyn> usage
    contexts: ZeroOneOrMany<CandleContext>,
    /// Tools with proper static dispatch and zero Box<dyn> usage
    tools: ZeroOneOrMany<CandleTool>,
    /// MCP server configuration and management
    mcp_servers: ZeroOneOrMany<McpServerConfig>,
    /// Provider-specific parameters (beta features, custom options)
    additional_params: ZeroOneOrMany<CandleAdditionalParams>,
    /// Persistent memory and conversation storage with proper static dispatch
    memory: ZeroOneOrMany<CandleMemory>,
    /// Agent metadata and custom attributes
    metadata: ZeroOneOrMany<CandleMetadata>,
    /// Tool result processing handler with generic type parameter for zero allocation
    on_tool_result_handler: Option<fn(ZeroOneOrMany<Value>)>,
    /// Conversation turn event handler with generic type parameter for zero allocation
    on_conversation_turn_handler: Option<fn(&CandleAgentConversation, &CandleAgentRoleAgent)>}

impl std::fmt::Debug for AgentRoleImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentRoleImpl")
            .field("name", &self.name)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("system_prompt", &self.system_prompt)
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .finish()
    }
}

impl Clone for AgentRoleImpl {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            completion_provider: self.completion_provider.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            system_prompt: self.system_prompt.clone(),
            api_key: self.api_key.clone(),
            contexts: self.contexts.clone(),
            tools: self.tools.clone(),
            mcp_servers: self.mcp_servers.clone(),
            additional_params: self.additional_params.clone(),
            memory: self.memory.clone(),
            metadata: self.metadata.clone(),
            on_tool_result_handler: self.on_tool_result_handler,
            on_conversation_turn_handler: self.on_conversation_turn_handler,
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
            completion_provider: ZeroOneOrMany::None,
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            api_key: None,
            contexts: ZeroOneOrMany::None,
            tools: ZeroOneOrMany::None,
            mcp_servers: ZeroOneOrMany::None,
            additional_params: ZeroOneOrMany::None,
            memory: ZeroOneOrMany::None,
            metadata: ZeroOneOrMany::None,
            on_tool_result_handler: None,
            on_conversation_turn_handler: None}
    }
}

impl AgentRoleImpl {
    /// Get memory tools if available
    ///
    /// # Returns
    /// Reference to memory tools collection
    ///
    /// # Performance
    /// Zero cost abstraction with direct memory access
    #[inline]
    pub fn get_memory_tools(&self) -> &ZeroOneOrMany<CandleMemory> {
        &self.memory
    }

    /// Set memory tools for agent role
    ///
    /// # Arguments
    /// * `memory_tools` - Memory tools to set
    ///
    /// # Returns
    /// Updated agent role instance
    ///
    /// # Performance
    /// Zero allocation with direct field assignment
    #[inline]
    pub fn with_memory_tools(mut self, memory_tools: ZeroOneOrMany<CandleMemory>) -> Self {
        self.memory = memory_tools;
        self
    }

    /// Set the API key for OpenAI completions
    /// Zero allocation with direct field assignment
    #[inline]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Get the API key, falling back to environment variable if not set
    /// Zero allocation with efficient environment variable access
    #[allow(dead_code)] // TODO: Implement in API authentication system
    #[inline]
    fn get_api_key(&self) -> Result<String, ChatError> {
        if let Some(ref api_key) = self.api_key {
            Ok(api_key.clone())
        } else {
            std::env::var("OPENAI_API_KEY")
                .map_err(|_| ChatError::System(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable or use with_api_key()".to_string()
                ))
        }
    }
}

/// Placeholder for Stdio type
pub struct Stdio;

/// Agent type placeholder for agent role
pub struct AgentRoleAgent;

/// Agent conversation type
pub struct AgentConversation {
    messages: Option<ZeroOneOrMany<(MessageRole, String)>>}

impl AgentConversation {
    /// Get the last message from the conversation
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
                .unwrap_or_default()}
    }
}

/// A single message in an agent conversation
pub struct AgentConversationMessage {
    content: String}

impl AgentConversationMessage {
    /// Get the message content as a string slice
    pub fn message(&self) -> &str {
        &self.content
    }
}


/// Trait for context arguments - moved to fluent-ai/src/builders/
pub trait ContextArgs {
    /// Add this context to the collection of contexts
    fn add_to(self, contexts: &mut ZeroOneOrMany<CandleContext>);
}

/// Trait for tool arguments - moved to fluent-ai/src/builders/
pub trait ToolArgs {
    /// Add this tool to the collection of tools
    fn add_to(self, tools: &mut ZeroOneOrMany<CandleTool>);
}

/// Trait for conversation history arguments - moved to fluent-ai/src/builders/
pub trait ConversationHistoryArgs {
    /// Convert this into conversation history format
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>>;
}

// Forward declaration for ChatError - will be defined in chat.rs
use crate::domain::agent::chat::CandleChatError as ChatError;
