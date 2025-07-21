// Removed unused import: std::collections::HashMap
use std::time::Instant;

use arrayvec::ArrayVec;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use smallvec::SmallVec;

use crate::ZeroOneOrMany;

/// Custom serialization module for Instant
mod instant_serde {
    use std::time::{Instant, SystemTime, UNIX_EPOCH, Duration};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use once_cell::sync::Lazy;

    static START_TIME: Lazy<(Instant, SystemTime)> = Lazy::new(|| {
        (Instant::now(), SystemTime::now())
    });

    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let elapsed = instant.duration_since(START_TIME.0);
        let system_time = START_TIME.1 + elapsed;
        let timestamp = system_time.duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_secs();
        timestamp.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let timestamp_secs = u64::deserialize(deserializer)?;
        let system_time = UNIX_EPOCH + Duration::from_secs(timestamp_secs);
        let elapsed_since_start = system_time.duration_since(START_TIME.1)
            .unwrap_or(Duration::from_secs(0));
        Ok(START_TIME.0 + elapsed_since_start)
    }
}

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

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::User => write!(f, "User"),
            MessageRole::Assistant => write!(f, "Assistant"),
            MessageRole::System => write!(f, "System"),
            MessageRole::Tool => write!(f, "Tool"),
        }
    }
}

// Core message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub name: Option<String>,
    pub chunk: Option<MessageChunk>,
    #[serde(default = "default_timestamp")]
    pub timestamp: std::time::SystemTime,
}

fn default_timestamp() -> std::time::SystemTime {
    std::time::SystemTime::now()
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
    pub messages: Vec<ChatMessage>,
}

impl Conversation {
    pub fn new() -> Self {
        Self { messages: vec![] }
    }

    pub fn add_message(&mut self, message: ChatMessage) {
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
impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage {
            role: MessageRole::User,
            content: content.into(),
            name: None,
            chunk: None,
            timestamp: default_timestamp(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        ChatMessage {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
            chunk: None,
            timestamp: default_timestamp(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        ChatMessage {
            role: MessageRole::System,
            content: content.into(),
            name: None,
            chunk: None,
            timestamp: default_timestamp(),
        }
    }

    pub fn tool(content: impl Into<String>) -> Self {
        ChatMessage {
            role: MessageRole::Tool,
            content: content.into(),
            name: None,
            chunk: None,
            timestamp: default_timestamp(),
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

/// Chat message type for search and history functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchChatMessage {
    pub role: String,
    pub content: String,
    pub timestamp: u64,
    pub tokens: usize,
    pub metadata: Option<String>,
}

impl SearchChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            tokens: 0,
            metadata: None,
        }
    }
}


/// Zero-allocation message with const generics for stack allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message<const N: usize = 256> {
    pub id: u64,
    pub message_type: MessageType,
    pub content: ArrayVec<u8, N>,
    pub metadata: SmallVec<u8, 32>,
    #[serde(with = "instant_serde")]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    High,
    Normal,
    Low,
}
