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

    /// Represents a Candle chat message with role and content
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CandleMessage {
        /// The role of the message sender (user, assistant, system, etc.)
        pub role: CandleMessageRole,
        /// The content of the message
        pub content: String,
        /// Optional message ID
        pub id: Option<String>,
        /// Optional timestamp
        pub timestamp: Option<u64>}

    /// Role of the Candle message sender
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
        Tool}

    impl CandleMessage {
        /// Create a new Candle message with the given role and content
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
                )}
        }
    }

    impl fmt::Display for CandleMessageRole {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                CandleMessageRole::System => write!(f, "system"),
                CandleMessageRole::User => write!(f, "user"),
                CandleMessageRole::Assistant => write!(f, "assistant"),
                CandleMessageRole::Tool => write!(f, "tool")}
        }
    }

    /// A chunk of a streaming Candle message
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CandleMessageChunk {
        /// The content of this chunk
        pub content: String,
        /// Whether this is the last chunk
        pub done: bool}

    /// Type classification for Candle messages
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum CandleMessageType {
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

    /// Candle message specifically for search operations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CandleSearchChatMessage {
        /// The message content
        pub message: CandleMessage,
        /// Search relevance score
        pub relevance_score: f64,
        /// Optional highlighting information
        pub highlights: Vec<String>}
}

/// Candle media type handling for messages
pub mod media {
    use serde::{Deserialize, Serialize};

    /// Candle media type for message content
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum CandleMediaType {
        /// Image content with specific image format
        Image(CandleImageMediaType),
        /// Document content with specific document format
        Document(CandleDocumentMediaType),
        /// Audio content with specific audio format
        Audio(CandleAudioMediaType),
        /// Video content with specific video format
        Video(CandleVideoMediaType),
    }

    /// Candle image media types
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum CandleImageMediaType {
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

    /// Candle document media types
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum CandleDocumentMediaType {
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

    /// Candle audio media types
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum CandleAudioMediaType {
        /// MP3 audio
        Mp3,
        /// WAV audio
        Wav,
        /// OGG audio
        Ogg,
        /// Other audio type
        Other(String),
    }

    /// Candle video media types
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum CandleVideoMediaType {
        /// MP4 video
        Mp4,
        /// WebM video
        WebM,
        /// Other video type
        Other(String),
    }

    impl CandleMediaType {
        /// Create a CandleMediaType from a MIME type string
        pub fn from_mime_type(mime_type: &str) -> Option<Self> {
            match mime_type {
                // Image types
                "image/jpeg" => Some(CandleMediaType::Image(CandleImageMediaType::Jpeg)),
                "image/png" => Some(CandleMediaType::Image(CandleImageMediaType::Png)),
                "image/gif" => Some(CandleMediaType::Image(CandleImageMediaType::Gif)),
                "image/webp" => Some(CandleMediaType::Image(CandleImageMediaType::WebP)),
                
                // Document types
                "application/pdf" => Some(CandleMediaType::Document(CandleDocumentMediaType::Pdf)),
                "text/plain" => Some(CandleMediaType::Document(CandleDocumentMediaType::Text)),
                "text/html" => Some(CandleMediaType::Document(CandleDocumentMediaType::Html)),
                "text/markdown" => Some(CandleMediaType::Document(CandleDocumentMediaType::Markdown)),
                
                // Audio types
                "audio/mpeg" => Some(CandleMediaType::Audio(CandleAudioMediaType::Mp3)),
                "audio/wav" => Some(CandleMediaType::Audio(CandleAudioMediaType::Wav)),
                "audio/ogg" => Some(CandleMediaType::Audio(CandleAudioMediaType::Ogg)),
                
                // Video types
                "video/mp4" => Some(CandleMediaType::Video(CandleVideoMediaType::Mp4)),
                "video/webm" => Some(CandleMediaType::Video(CandleVideoMediaType::WebM)),
                
                // Handle unknown types
                mime_type if mime_type.starts_with("image/") => {
                    Some(CandleMediaType::Image(CandleImageMediaType::Other(mime_type.to_string())))
                }
                mime_type if mime_type.starts_with("audio/") => {
                    Some(CandleMediaType::Audio(CandleAudioMediaType::Other(mime_type.to_string())))
                }
                mime_type if mime_type.starts_with("video/") => {
                    Some(CandleMediaType::Video(CandleVideoMediaType::Other(mime_type.to_string())))
                }
                _ => Some(CandleMediaType::Document(CandleDocumentMediaType::Other(mime_type.to_string()))),
            }
        }

        /// Convert to MIME type string
        pub fn to_mime_type(&self) -> String {
            match self {
                CandleMediaType::Image(img) => match img {
                    CandleImageMediaType::Jpeg => "image/jpeg".to_string(),
                    CandleImageMediaType::Png => "image/png".to_string(),
                    CandleImageMediaType::Gif => "image/gif".to_string(),
                    CandleImageMediaType::WebP => "image/webp".to_string(),
                    CandleImageMediaType::Other(mime) => mime.clone(),
                },
                CandleMediaType::Document(doc) => match doc {
                    CandleDocumentMediaType::Pdf => "application/pdf".to_string(),
                    CandleDocumentMediaType::Text => "text/plain".to_string(),
                    CandleDocumentMediaType::Html => "text/html".to_string(),
                    CandleDocumentMediaType::Markdown => "text/markdown".to_string(),
                    CandleDocumentMediaType::Other(mime) => mime.clone(),
                },
                CandleMediaType::Audio(audio) => match audio {
                    CandleAudioMediaType::Mp3 => "audio/mpeg".to_string(),
                    CandleAudioMediaType::Wav => "audio/wav".to_string(),
                    CandleAudioMediaType::Ogg => "audio/ogg".to_string(),
                    CandleAudioMediaType::Other(mime) => mime.clone(),
                },
                CandleMediaType::Video(video) => match video {
                    CandleVideoMediaType::Mp4 => "video/mp4".to_string(),
                    CandleVideoMediaType::WebM => "video/webm".to_string(),
                    CandleVideoMediaType::Other(mime) => mime.clone(),
                },
            }
        }
    }
}

/// Error types for Candle message operations
pub mod error {
    use std::fmt;

    /// Errors that can occur during Candle message processing
    #[derive(Debug, Clone)]
    pub enum CandleMessageError {
        /// Error during message conversion
        ConversionError(String),
        /// Invalid message format
        InvalidFormat(String),
        /// Unsupported message type
        UnsupportedType(String),
    }

    impl fmt::Display for CandleMessageError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                CandleMessageError::ConversionError(msg) => write!(f, "Candle conversion error: {}", msg),
                CandleMessageError::InvalidFormat(msg) => write!(f, "Candle invalid format: {}", msg),
                CandleMessageError::UnsupportedType(msg) => write!(f, "Candle unsupported type: {}", msg),
            }
        }
    }

    impl std::error::Error for CandleMessageError {}
}

/// Candle message processing functionality
pub mod processing {
    use super::types::*;
    use super::error::CandleMessageError;

    /// Process a Candle message in place, applying transformations
    pub fn candle_process_message(
        message: &mut CandleMessage,
    ) -> Result<(), CandleMessageError> {
        // Trim whitespace
        message.content = message.content.trim().to_string();

        // Validate message isn't empty after trimming
        if message.content.is_empty() {
            return Err(CandleMessageError::InvalidFormat("Message cannot be empty after processing".to_string()));
        }

        Ok(())
    }

    /// Format a Candle message for display
    pub fn candle_format_message(message: &CandleMessage) -> String {
        format!("{}: {}", message.role, message.content)
    }
}

// Re-export commonly used Candle types (CandleMessageChunk and CandleMessageRole are used throughout the codebase)
pub use types::{CandleMessage, CandleMessageChunk, CandleMessageRole, CandleMessageType, CandleSearchChatMessage};
pub use error::CandleMessageError;
pub use media::{CandleMediaType, CandleImageMediaType, CandleDocumentMediaType, CandleAudioMediaType, CandleVideoMediaType};

// Import ToolCall from HTTP module for message processing - will be updated to Candle prefixes later
pub use crate::http::{ToolCall, ToolCallType, FunctionCall};

// Alias for backward compatibility - some code expects CandleMimeType instead of CandleMediaType  
pub use media::CandleMediaType as CandleMimeType;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_message_creation() {
        let message = CandleMessage {
            role: CandleMessageRole::User,
            content: "Hello, world!".to_string(),
            id: Some("123".to_string()),
            timestamp: Some(1234567890)};

        assert_eq!(message.role, CandleMessageRole::User);
        assert_eq!(message.content, "Hello, world!");
    }

    #[test]
    fn test_candle_message_processing() {
        let mut message = CandleMessage {
            role: CandleMessageRole::User,
            content: "  Hello, world!  ".to_string(),
            id: None,
            timestamp: None};

        processing::candle_process_message(&mut message).unwrap();
        assert_eq!(message.content, "Hello, world!");
    }
}
