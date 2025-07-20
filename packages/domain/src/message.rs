use std::collections::HashMap;
use std::time::Instant;

use arrayvec::ArrayVec;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use smallvec::SmallVec;

use crate::ZeroOneOrMany;

/// Error type for message operations
#[derive(Debug, thiserror::Error)]
pub enum MessageError {
    #[error("Invalid message format: {0}")]
    InvalidFormat(String),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Tool function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

/// MIME type for message content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MimeType(pub String);

impl MimeType {
    pub fn new(mime_type: &str) -> Self {
        Self(mime_type.to_string())
    }
}

/// Tool call in messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function: ToolFunction,
}

/// Tool result content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultContent {
    pub tool_call_id: String,
    pub content: String,
}

/// Text content wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Text {
    pub content: String,
}

/// User content types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserContent {
    Text(Text),
    Image { url: String, detail: Option<String> },
}

/// Assistant content types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssistantContent {
    Text(Text),
    ToolCall(ToolCall),
}

// Message role for context building
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
}

// Core message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyMessage {
    pub role: MessageRole,
    pub content: String,
    pub name: Option<String>,
    pub chunk: Option<MessageChunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageChunk {
    pub content: String,
}

// Trait for message content
pub trait Content {}

// Conversation container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub messages: Vec<LegacyMessage>,
}

impl Conversation {
    pub fn new() -> Self {
        Self { messages: vec![] }
    }

    pub fn add_message(&mut self, message: LegacyMessage) {
        self.messages.push(message);
    }

    pub fn to_string(&self) -> String {
        let mut text = String::new();
        for message in &self.messages {
            match message.role {
                MessageRole::User => {
                    text.push_str("User: ");
                    text.push_str(&message.content);
                }
                MessageRole::Assistant => {
                    text.push_str("Assistant: ");
                    text.push_str(&message.content);
                }
                MessageRole::System => {
                    text.push_str("System: ");
                    text.push_str(&message.content);
                }
                MessageRole::Tool => {
                    text.push_str("Tool: ");
                    text.push_str(&message.content);
                }
            }
            text.push('\n');
        }
        text
    }
}

// Content trait implementations
impl Content for UserContent {}
impl Content for AssistantContent {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub result: Value,
    pub error: Option<String>,
}

// Direct factory methods - no new(), no build()
impl LegacyMessage {
    pub fn user(content: impl Into<String>) -> Self {
        LegacyMessage {
            role: MessageRole::User,
            content: content.into(),
            name: None,
            chunk: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        LegacyMessage {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
            chunk: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        LegacyMessage {
            role: MessageRole::System,
            content: content.into(),
            name: None,
            chunk: None,
        }
    }

    pub fn tool(content: impl Into<String>) -> Self {
        LegacyMessage {
            role: MessageRole::Tool,
            content: content.into(),
            name: None,
            chunk: None,
        }
    }
}

// Message builder traits moved to fluent_ai/src/builders/message.rs

// Now let's also define a content container trait
pub trait ContentContainer: Content {
    type Item: Content;
    fn items(&self) -> &[Self::Item];
}

// Implementation for ZeroOneOrMany
impl<T: Content + Clone> Content for ZeroOneOrMany<T> {}

// High-performance, zero-allocation message types

/// Zero-allocation message with const generics for stack allocation
#[derive(Debug, Clone)]
pub struct Message<const N: usize = 256> {
    pub id: u64,
    pub message_type: MessageType,
    pub content: ArrayVec<u8, N>,
    pub metadata: SmallVec<u8, 32>,
    pub timestamp: Instant,
    pub priority: MessagePriority,
    pub retry_count: u8,
}

impl<const N: usize> Default for Message<N> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            id: 0,
            message_type: MessageType::AgentChat,
            content: ArrayVec::new(),
            metadata: SmallVec::new(),
            timestamp: Instant::now(),
            priority: MessagePriority::Normal,
            retry_count: 0,
        }
    }
}

impl<const N: usize> Message<N> {
    /// Create new message with zero allocation
    #[inline(always)]
    pub fn new(id: u64, message_type: MessageType, content: &[u8]) -> Result<Self, &'static str> {
        if content.len() > N {
            return Err("Content exceeds message capacity");
        }
        let mut msg = Self {
            id,
            message_type,
            ..Default::default()
        };
        msg.content.try_extend_from_slice(content).unwrap();
        Ok(msg)
    }
}

/// Message type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    AgentChat,
    SystemCommand,
    ToolRequest,
    ToolResponse,
    HealthCheck,
    ControlSignal,
    MetricsUpdate,
}

/// Message priority for QoS
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    High,
    Normal,
    Low,
}
