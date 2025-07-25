//! Message module for chat functionality
//!
//! This module provides message types and processing functionality with
//! zero-allocation patterns and blazing-fast performance.

use std::sync::Arc;
use serde::{Deserialize, Serialize};

pub mod message_processing;

/// Message role enumeration for chat messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandleMessageRole {
    /// System message role
    System,
    /// User message role
    User,
    /// Assistant message role
    Assistant,
    /// Function call message role
    Function,
    /// Tool call message role
    Tool,
}

impl Default for CandleMessageRole {
    fn default() -> Self {
        Self::User
    }
}

impl std::fmt::Display for CandleMessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl CandleMessageRole {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user", 
            Self::Assistant => "assistant",
            Self::Function => "function",
            Self::Tool => "tool",
        }
    }
}

/// Core message type for chat interactions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandleMessage {
    /// Message unique identifier
    pub id: String,
    /// Message role
    pub role: CandleMessageRole,
    /// Message content
    pub content: String,
    /// Optional message name/identifier
    pub name: Option<Arc<str>>,
    /// Message metadata
    pub metadata: Option<serde_json::Value>,
    /// Message timestamp
    pub timestamp: Option<u64>,
}

impl CandleMessage {
    /// Create a new message
    pub fn new(role: CandleMessageRole, content: impl Into<String>) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        
        Self {
            id: format!("msg_{}", COUNTER.fetch_add(1, Ordering::Relaxed)),
            role,
            content: content.into(),
            name: None,
            metadata: None,
            timestamp: Some(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()),
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(CandleMessageRole::User, content)
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(CandleMessageRole::System, content)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(CandleMessageRole::Assistant, content)
    }
}

impl Default for CandleMessage {
    fn default() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1000);
        
        Self {
            id: format!("msg_{}", COUNTER.fetch_add(1, Ordering::Relaxed)),
            role: CandleMessageRole::default(),
            content: String::new(),
            name: None,
            metadata: None,
            timestamp: Some(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()),
        }
    }
}

/// Message chunk for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageChunk {
    /// Chunk content
    pub content: String,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Chunk metadata
    pub metadata: Option<serde_json::Value>,
}

/// Message type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Text message
    Text,
    /// Image message
    Image,
    /// Audio message
    Audio,
    /// Video message
    Video,
    /// File message
    File,
}

/// Search chat message for search functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchChatMessage {
    /// Message unique identifier
    pub id: String,
    /// Message role
    pub role: CandleMessageRole,
    /// Message content
    pub content: String,
    /// Optional message name/identifier
    pub name: Option<Arc<str>>,
    /// Message metadata
    pub metadata: Option<serde_json::Value>,
    /// Message timestamp
    pub timestamp: Option<u64>,
    /// Search relevance score
    pub relevance_score: f32,
    /// Search timestamp
    pub search_timestamp: u64,
}

impl From<CandleMessage> for SearchChatMessage {
    fn from(message: CandleMessage) -> Self {
        Self {
            id: message.id,
            role: message.role,
            content: message.content,
            name: message.name,
            metadata: message.metadata,
            timestamp: message.timestamp,
            relevance_score: 1.0,
            search_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

pub use message_processing::*;