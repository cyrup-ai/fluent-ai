//! Additional agent-related types and utilities

// Removed unused Arc import

use crate::MessageRole;
use crate::ZeroOneOrMany;
use crate::chunk::ChatMessageChunk;

/// Placeholder for Stdio type
pub struct Stdio;

/// Agent type placeholder for agent role
pub struct AgentRoleAgent;

/// Agent conversation type
pub struct AgentConversation {
    pub messages: Option<ZeroOneOrMany<(MessageRole, String)>>,
}

impl AgentConversation {
    pub fn new() -> Self {
        Self { messages: None }
    }

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

impl Default for AgentConversation {
    fn default() -> Self {
        Self::new()
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

/// Agent with conversation history - domain data structure
pub struct AgentWithHistory {
    #[allow(dead_code)] // TODO: Use for accessing agent role configuration during chat
    inner: Box<dyn std::any::Any + Send + Sync>,
    pub chunk_handler: Box<dyn Fn(ChatMessageChunk) -> ChatMessageChunk + Send + Sync>,
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
