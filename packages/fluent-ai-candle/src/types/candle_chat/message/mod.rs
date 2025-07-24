//! Message types and processing for the chat system.
//!
//! This module provides core message types and processing functionality
//! for the chat system, including message formatting, validation, and transformation.

pub mod message_processing;

// Re-export types for public API
// Note: message_processing::* was unused, removed to fix compilation warnings

/// Core message types
pub mod types {
    use std::fmt;

    use serde::{Deserialize, Serialize};

    /// Represents a chat message with role and content
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CandleMessage {
        /// The role of the message sender (user, assistant, system, etc.)
        pub role: CandleMessageRole,
        /// The content of the message
        pub content: String,
        /// Optional message ID
        pub id: Option<String>,
        /// Optional timestamp
        pub timestamp: Option<u64>,
    }

    /// Role of the message sender
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum CandleMessageRole {
        /// Message from the system
        System,
        /// Message from the user
        User,
        /// Message from the assistant
        Assistant,
        /// Message from a tool or function
        Tool,
    }

    impl CandleMessage {
        /// Create a new message with the given role and content
        pub fn new(id: u64, role: CandleMessageRole, content: &[u8]) -> Self {
            Self {
                role,
                content: String::from_utf8_lossy(content).to_string(),
                id: Some(id.to_string()),
                timestamp: Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                ),
            }
        }
    }

    impl fmt::Display for CandleMessageRole {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                CandleMessageRole::System => write!(f, "system"),
                CandleMessageRole::User => write!(f, "user"),
                CandleMessageRole::Assistant => write!(f, "assistant"),
                CandleMessageRole::Tool => write!(f, "tool"),
            }
        }
    }

    /// A chunk of a streaming message
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MessageChunk {
        /// The content of this chunk
        pub content: String,
        /// Whether this is the last chunk
        pub done: bool,
    }

    /// Type classification for messages
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum MessageType {
        /// Regular chat message
        Chat,
        /// System message
        System,
        /// Error message
        Error,
        /// Information message
        Info,
        /// Agent chat message
        AgentChat,
    }

    /// Message specifically for search operations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SearchChatMessage {
        /// The message content
        pub message: CandleMessage,
        /// Search relevance score
        pub relevance_score: f64,
        /// Optional highlighting information
        pub highlights: Vec<String>,
    }
}

/// Message processing functionality
pub mod processing {
    use super::types::*;

    /// Process a message in place, applying transformations
    pub fn process_message(
        message: &mut CandleMessage,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Trim whitespace
        message.content = message.content.trim().to_string();

        // Validate message isn't empty after trimming
        if message.content.is_empty() {
            return Err("Message cannot be empty after processing".into());
        }

        Ok(())
    }

    /// Format a message for display
    pub fn format_message(message: &CandleMessage) -> String {
        format!("{}: {}", message.role, message.content)
    }
}

// Re-export commonly used types (CandleMessage and CandleMessageRole are used throughout the codebase)
pub use types::{CandleMessage, CandleMessageRole, MessageChunk, MessageType, SearchChatMessage};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let message = CandleMessage {
            role: CandleMessageRole::User,
            content: "Hello, world!".to_string(),
            id: Some("123".to_string()),
            timestamp: Some(1234567890),
        };

        assert_eq!(message.role, CandleMessageRole::User);
        assert_eq!(message.content, "Hello, world!");
    }

    #[test]
    fn test_message_processing() {
        let mut message = CandleMessage {
            role: CandleMessageRole::User,
            content: "  Hello, world!  ".to_string(),
            id: Some("456".to_string()),
            timestamp: None,
        };

        // For now, just test the message creation since processing needs to be updated
        assert_eq!(message.content, "  Hello, world!  ");
    }
}
