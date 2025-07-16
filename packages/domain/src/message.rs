use crate::ZeroOneOrMany;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

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
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    pub name: Option<String>,
    pub chunk: Option<MessageChunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageChunk {
    pub index: usize,
    pub total: Option<usize>,
    pub content: String,
}

/// Extension trait for UserContent collections
pub trait UserContentExt {
    /// Extract text content from user content, concatenating multiple items
    fn as_text(&self) -> String;
}

impl UserContentExt for ZeroOneOrMany<UserContent> {
    fn as_text(&self) -> String {
        self.iter()
            .map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Extension trait for AssistantContent collections
pub trait AssistantContentExt {
    /// Extract text content from assistant content, concatenating multiple items
    fn as_text(&self) -> String;
}

impl AssistantContentExt for ZeroOneOrMany<AssistantContent> {
    fn as_text(&self) -> String {
        self.iter()
            .map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl UserContent {
    /// Extract text representation of user content
    pub fn as_text(&self) -> String {
        match self {
            UserContent::Text(text) => text.content.clone(),
            UserContent::Image { .. } => "[Image]".to_string(),
        }
    }
}

impl AssistantContent {
    /// Extract text representation of assistant content
    pub fn as_text(&self) -> String {
        match self {
            AssistantContent::Text(text) => text.content.clone(),
            AssistantContent::ToolCall(call) => {
                format!("Tool: {} ({})", call.function.name, call.function.parameters)
            }
        }
    }

    /// Create text content
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(Text { content: content.into() })
    }
}

/// Conversation trait for message conversations
pub trait Conversation {
    /// Convert the conversation to a text representation
    fn as_text(&self) -> String;
}

/// Content trait for message content types
pub trait Content: Serialize {
    /// Convert content to JSON string representation
    fn to_content_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Conversation wrapper around HashMap
#[derive(Debug, Clone)]
pub struct ConversationMap(HashMap<MessageRole, Message>);

impl std::ops::Deref for ConversationMap {
    type Target = HashMap<MessageRole, Message>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<HashMap<MessageRole, Message>> for ConversationMap {
    fn from(map: HashMap<MessageRole, Message>) -> Self {
        ConversationMap(map)
    }
}

impl Conversation for ConversationMap {
    fn as_text(&self) -> String {
        let mut text = String::new();
        for (role, message) in self.0.iter() {
            match role {
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

impl Conversation for HashMap<MessageRole, Message> {
    fn as_text(&self) -> String {
        let mut text = String::new();
        for (role, message) in self.iter() {
            match role {
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
impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::User,
            content: content.into(),
            name: None,
            chunk: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
            chunk: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::System,
            content: content.into(),
            name: None,
            chunk: None,
        }
    }
    
    pub fn tool(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::Tool,
            content: content.into(),
            name: None,
            chunk: None,
        }
    }
}

// Message builder traits
pub trait MessageBuilder {
    type Content: Content;
    fn add_content(self, content: Self::Content) -> Self;
    fn build(self) -> Message;
}

pub trait UserMessageBuilderTrait: MessageBuilder<Content = UserContent> {
    fn text(self, text: impl Into<String>) -> Self;
    fn image(self, image: crate::Image) -> Self;
    fn audio(self, audio: crate::Audio) -> Self;
    fn document(self, document: crate::Document) -> Self;
    fn say(self) -> Message;
}

pub trait AssistantMessageBuilderTrait: MessageBuilder<Content = AssistantContent> {
    fn text(self, text: impl Into<String>) -> Self;
    fn tool_call(self, id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self;
    fn tool_result(self, tool_call_id: impl Into<String>, result: Value) -> Self;
    fn tool_error(self, tool_call_id: impl Into<String>, error: impl Into<String>) -> Self;
    fn respond(self) -> Message;
}

// Trait for creating message builders
pub trait MessageFactory {
    fn user_message() -> impl UserMessageBuilderTrait;
    fn assistant_message() -> impl AssistantMessageBuilderTrait;
}

// Now let's also define a content container trait
pub trait ContentContainer: Content {
    type Item: Content;
    fn items(&self) -> &[Self::Item];
}

// Implementation for ZeroOneOrMany
impl<T: Content + Clone> Content for ZeroOneOrMany<T> {}
