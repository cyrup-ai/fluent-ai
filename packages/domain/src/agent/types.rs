//! Additional agent-related types and utilities

// Removed unused Arc import

use crate::ZeroOneOrMany;
use crate::chat::MessageRole;
use crate::context::chunk::ChatMessageChunk;

/// Placeholder for Stdio type
pub struct Stdio;

/// Agent type placeholder for agent role
pub struct AgentRoleAgent;

/// Agent conversation type
pub struct AgentConversation {
    /// Optional collection of conversation messages with their roles
    pub messages: Option<ZeroOneOrMany<(MessageRole, String)>>}

impl AgentConversation {
    /// Create a new empty agent conversation
    pub fn new() -> Self {
        Self { messages: None }
    }

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

impl Default for AgentConversation {
    fn default() -> Self {
        Self::new()
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

/// Agent with conversation history - domain data structure
pub struct AgentWithHistory {
    #[allow(dead_code)] // TODO: Use for accessing agent role configuration during chat
    inner: Box<dyn std::any::Any + Send + Sync>,
    /// Handler function for processing chat message chunks during streaming
    pub chunk_handler: Box<dyn Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync>,
    #[allow(dead_code)] // TODO: Use for loading previous conversation context during chat
    conversation_history: Option<ZeroOneOrMany<(MessageRole, String)>>}

/// Trait for context arguments - moved to fluent-ai/src/builders/
pub trait ContextArgs {
    /// Add this context to the collection of contexts
    fn add_to(self, contexts: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>);
}

/// Trait for tool arguments - moved to fluent-ai/src/builders/
pub trait ToolArgs {
    /// Add this tool to the collection of tools
    fn add_to(self, tools: &mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>);
}

/// Trait for conversation history arguments - moved to fluent-ai/src/builders/
pub trait ConversationHistoryArgs {
    /// Convert this into conversation history format
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>>;
}
