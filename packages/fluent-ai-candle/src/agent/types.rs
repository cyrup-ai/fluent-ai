//! Additional agent-related types and utilities

// Removed unused Arc import

use cyrup_sugars::ZeroOneOrMany;
use crate::domain::chat::message::types::CandleMessageRole as MessageRole;
use crate::context::Context;
use crate::tool::Tool;

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


/// Trait for context arguments - zero-allocation with static dispatch
pub trait ContextArgs {
    /// Add this context to the collection of contexts
    fn add_to(self, contexts: &mut ZeroOneOrMany<crate::context::Context<String>>);
}

/// Trait for tool arguments - zero-allocation with static dispatch
pub trait ToolArgs {
    /// Add this tool to the collection of tools
    fn add_to(self, tools: &mut ZeroOneOrMany<crate::tool::Tool<String>>);
}

/// Trait for conversation history arguments - moved to fluent-ai/src/builders/
pub trait ConversationHistoryArgs {
    /// Convert this into conversation history format
    fn into_history(self) -> Option<ZeroOneOrMany<(MessageRole, String)>>;
}
