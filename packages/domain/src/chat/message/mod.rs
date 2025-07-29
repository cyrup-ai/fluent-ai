//! Message types and processing for the chat system.
//!
//! This module provides core message types and processing functionality
//! for the chat system, including message formatting, validation, and transformation.

pub mod message_processing;

// Re-export types for public API
// Note: message_processing::* was unused, removed to fix compilation warnings

// Re-export MessagePriority from realtime module
pub use crate::chat::realtime::MessagePriority;

/// Core message types
pub mod types {
    use std::fmt;

    use serde::{Deserialize, Serialize};

    /// Represents a chat message with role and content
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Message {
        /// The role of the message sender (user, assistant, system, etc.)
        pub role: MessageRole,
        /// The content of the message
        pub content: String,
        /// Optional message ID
        pub id: Option<String>,
        /// Optional timestamp
        pub timestamp: Option<u64>}

    /// Role of the message sender
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum MessageRole {
        /// Message from the system
        System,
        /// Message from the user
        User,
        /// Message from the assistant
        Assistant,
        /// Message from a tool or function
        Tool}

    impl Message {
        /// Create a new message with the given role and content
        pub fn new(id: u64, role: MessageRole, content: &[u8]) -> Self {
            Self {
                role,
                content: String::from_utf8_lossy(content).to_string(),
                id: Some(id.to_string()),
                timestamp: Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                )}
        }
    }

    impl fmt::Display for MessageRole {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                MessageRole::System => write!(f, "system"),
                MessageRole::User => write!(f, "user"),
                MessageRole::Assistant => write!(f, "assistant"),
                MessageRole::Tool => write!(f, "tool")}
        }
    }

    /// A chunk of a streaming message
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MessageChunk {
        /// The content of this chunk
        pub content: String,
        /// Whether this is the last chunk
        pub done: bool}

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
        AgentChat}

    /// Message specifically for search operations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SearchChatMessage {
        /// The message content
        pub message: Message,
        /// Search relevance score
        pub relevance_score: f64,
        /// Optional highlighting information
        pub highlights: Vec<String>}
}

/// Media type handling for messages
pub mod media {
    use serde::{Deserialize, Serialize};

    /// Media type for message content
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum MediaType {
        /// Image content with specific image format
        Image(ImageMediaType),
        /// Document content with specific document format
        Document(DocumentMediaType),
        /// Audio content with specific audio format
        Audio(AudioMediaType),
        /// Video content with specific video format
        Video(VideoMediaType),
    }

    /// Image media types
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum ImageMediaType {
        /// JPEG image
        Jpeg,
        /// PNG image
        Png,
        /// GIF image
        Gif,
        /// WebP image
        WebP,
        /// Other image type
        Other(String),
    }

    /// Document media types
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum DocumentMediaType {
        /// PDF document
        Pdf,
        /// Plain text
        Text,
        /// HTML document
        Html,
        /// Markdown document
        Markdown,
        /// Other document type
        Other(String),
    }

    /// Audio media types
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum AudioMediaType {
        /// MP3 audio
        Mp3,
        /// WAV audio
        Wav,
        /// OGG audio
        Ogg,
        /// Other audio type
        Other(String),
    }

    /// Video media types
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum VideoMediaType {
        /// MP4 video
        Mp4,
        /// WebM video
        WebM,
        /// Other video type
        Other(String),
    }

    impl MediaType {
        /// Create a MediaType from a MIME type string
        pub fn from_mime_type(mime_type: &str) -> Option<Self> {
            match mime_type {
                // Image types
                "image/jpeg" => Some(MediaType::Image(ImageMediaType::Jpeg)),
                "image/png" => Some(MediaType::Image(ImageMediaType::Png)),
                "image/gif" => Some(MediaType::Image(ImageMediaType::Gif)),
                "image/webp" => Some(MediaType::Image(ImageMediaType::WebP)),
                
                // Document types
                "application/pdf" => Some(MediaType::Document(DocumentMediaType::Pdf)),
                "text/plain" => Some(MediaType::Document(DocumentMediaType::Text)),
                "text/html" => Some(MediaType::Document(DocumentMediaType::Html)),
                "text/markdown" => Some(MediaType::Document(DocumentMediaType::Markdown)),
                
                // Audio types
                "audio/mpeg" => Some(MediaType::Audio(AudioMediaType::Mp3)),
                "audio/wav" => Some(MediaType::Audio(AudioMediaType::Wav)),
                "audio/ogg" => Some(MediaType::Audio(AudioMediaType::Ogg)),
                
                // Video types
                "video/mp4" => Some(MediaType::Video(VideoMediaType::Mp4)),
                "video/webm" => Some(MediaType::Video(VideoMediaType::WebM)),
                
                // Handle unknown types
                mime_type if mime_type.starts_with("image/") => {
                    Some(MediaType::Image(ImageMediaType::Other(mime_type.to_string())))
                }
                mime_type if mime_type.starts_with("audio/") => {
                    Some(MediaType::Audio(AudioMediaType::Other(mime_type.to_string())))
                }
                mime_type if mime_type.starts_with("video/") => {
                    Some(MediaType::Video(VideoMediaType::Other(mime_type.to_string())))
                }
                _ => Some(MediaType::Document(DocumentMediaType::Other(mime_type.to_string()))),
            }
        }

        /// Convert to MIME type string
        pub fn to_mime_type(&self) -> String {
            match self {
                MediaType::Image(img) => match img {
                    ImageMediaType::Jpeg => "image/jpeg".to_string(),
                    ImageMediaType::Png => "image/png".to_string(),
                    ImageMediaType::Gif => "image/gif".to_string(),
                    ImageMediaType::WebP => "image/webp".to_string(),
                    ImageMediaType::Other(mime) => mime.clone(),
                },
                MediaType::Document(doc) => match doc {
                    DocumentMediaType::Pdf => "application/pdf".to_string(),
                    DocumentMediaType::Text => "text/plain".to_string(),
                    DocumentMediaType::Html => "text/html".to_string(),
                    DocumentMediaType::Markdown => "text/markdown".to_string(),
                    DocumentMediaType::Other(mime) => mime.clone(),
                },
                MediaType::Audio(audio) => match audio {
                    AudioMediaType::Mp3 => "audio/mpeg".to_string(),
                    AudioMediaType::Wav => "audio/wav".to_string(),
                    AudioMediaType::Ogg => "audio/ogg".to_string(),
                    AudioMediaType::Other(mime) => mime.clone(),
                },
                MediaType::Video(video) => match video {
                    VideoMediaType::Mp4 => "video/mp4".to_string(),
                    VideoMediaType::WebM => "video/webm".to_string(),
                    VideoMediaType::Other(mime) => mime.clone(),
                },
            }
        }
    }
}

/// Error types for message operations
pub mod error {
    use std::fmt;

    /// Errors that can occur during message processing
    #[derive(Debug, Clone)]
    pub enum MessageError {
        /// Error during message conversion
        ConversionError(String),
        /// Invalid message format
        InvalidFormat(String),
        /// Unsupported message type
        UnsupportedType(String),
    }

    impl fmt::Display for MessageError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                MessageError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
                MessageError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
                MessageError::UnsupportedType(msg) => write!(f, "Unsupported type: {}", msg),
            }
        }
    }

    impl std::error::Error for MessageError {}
}

/// Message processing functionality
pub mod processing {
    use super::types::*;

    /// Process a message in place, applying transformations
    pub fn process_message(
        message: &mut Message,
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
    pub fn format_message(message: &Message) -> String {
        format!("{}: {}", message.role, message.content)
    }
}

// Re-export commonly used types (MessageChunk and MessageRole are used throughout the codebase)
pub use types::{Message, MessageChunk, MessageRole, MessageType, SearchChatMessage};
pub use error::MessageError;
pub use media::{MediaType, ImageMediaType, DocumentMediaType, AudioMediaType, VideoMediaType};

// Import ToolCall from HTTP module for message processing
pub use crate::http::{ToolCall, ToolCallType, FunctionCall};

// Alias for backward compatibility - some code expects MimeType instead of MediaType
pub use media::MediaType as MimeType;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let message = Message {
            role: MessageRole::User,
            content: "Hello, world!".to_string(),
            id: Some("123".to_string()),
            timestamp: Some(1234567890)};

        assert_eq!(message.role, MessageRole::User);
        assert_eq!(message.content, "Hello, world!");
    }

    #[test]
    fn test_message_processing() {
        let mut message = Message {
            role: MessageRole::User,
            content: "  Hello, world!  ".to_string(),
            id: None,
            timestamp: None};

        processing::process_message(&mut message).unwrap();
        assert_eq!(message.content, "Hello, world!");
    }
}
