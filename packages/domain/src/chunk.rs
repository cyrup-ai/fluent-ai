//! Chunk types for streaming operations
//!
//! These types represent partial data that flows through AsyncStream<T>
//! and are designed to work with the NotResult constraint.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::usage::Usage;

/// Chunk of document content for streaming file operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// Optional path to the source file
    pub path: Option<PathBuf>,

    /// The content of this chunk
    pub content: String,

    /// Byte range in the original file
    pub byte_range: Option<(usize, usize)>,

    /// Additional metadata
    #[serde(flatten)]
    pub metadata: HashMap<String, Value>,
}

/// Image format types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImageFormat {
    PNG,
    JPEG,
    GIF,
    WebP,
    BMP,
    TIFF,
}

/// Chunk of image data for streaming image operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageChunk {
    /// Raw image data
    pub data: Vec<u8>,

    /// Image format
    pub format: ImageFormat,

    /// Optional dimensions (width, height)
    pub dimensions: Option<(u32, u32)>,

    /// Additional metadata
    #[serde(flatten)]
    pub metadata: HashMap<String, Value>,
}

/// Audio format types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AudioFormat {
    MP3,
    WAV,
    FLAC,
    OGG,
    M4A,
    OPUS,
}

/// Chunk of audio/voice data for streaming audio operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceChunk {
    /// Raw audio data
    pub audio_data: Vec<u8>,

    /// Audio format
    pub format: AudioFormat,

    /// Duration in milliseconds
    pub duration_ms: Option<u64>,

    /// Sample rate in Hz
    pub sample_rate: Option<u32>,

    /// Additional metadata
    #[serde(flatten)]
    pub metadata: HashMap<String, Value>,
}

/// Chunk of chat message for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageChunk {
    /// Partial message content
    pub content: String,

    /// Role of the message sender
    pub role: crate::message::MessageRole,

    /// Whether this is the final chunk
    pub is_final: bool,

    /// Additional metadata
    #[serde(flatten)]
    pub metadata: HashMap<String, Value>,
}

/// Reason why a completion finished
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Error,
}

/// Comprehensive completion chunk supporting all streaming features
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CompletionChunk {
    /// Text content chunk
    Text(String),

    /// Tool call started
    ToolCallStart { id: String, name: String },

    /// Partial tool call with streaming input
    ToolCall {
        id: String,
        name: String,
        partial_input: String,
    },

    /// Tool call completed
    ToolCallComplete {
        id: String,
        name: String,
        input: String,
    },

    /// Completion finished with final information
    Complete {
        text: String,
        finish_reason: Option<FinishReason>,
        usage: Option<Usage>,
    },

    /// Error occurred during streaming
    Error(String),
}

/// Legacy completion chunk for backward compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyCompletionChunk {
    /// The text content
    pub text: String,

    /// Reason for finishing (if this is the last chunk)
    pub finish_reason: Option<FinishReason>,

    /// Token usage information (if available)
    pub usage: Option<Usage>,
}

/// Chunk of embedding data for streaming embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingChunk {
    /// The embedding vector
    pub embeddings: crate::ZeroOneOrMany<f32>,

    /// Index in the batch
    pub index: usize,

    /// Additional metadata
    #[serde(flatten)]
    pub metadata: HashMap<String, Value>,
}

/// Chunk of transcribed text from speech-to-text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionChunk {
    /// The transcribed text for this chunk
    pub text: String,

    /// Confidence score (0.0 to 1.0)
    pub confidence: Option<f32>,

    /// Start time in milliseconds
    pub start_time_ms: Option<u64>,

    /// End time in milliseconds  
    pub end_time_ms: Option<u64>,

    /// Whether this is the final chunk
    pub is_final: bool,

    /// Additional metadata
    #[serde(flatten)]
    pub metadata: HashMap<String, Value>,
}

/// Chunk of synthesized speech for text-to-speech
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechChunk {
    /// Raw audio data
    pub audio_data: Vec<u8>,

    /// Audio format
    pub format: AudioFormat,

    /// Duration in milliseconds
    pub duration_ms: Option<u64>,

    /// Sample rate in Hz
    pub sample_rate: Option<u32>,

    /// Whether this is the final chunk
    pub is_final: bool,

    /// Additional metadata
    #[serde(flatten)]
    pub metadata: HashMap<String, Value>,
}

// Convenience constructors
impl DocumentChunk {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            path: None,
            content: content.into(),
            byte_range: None,
            metadata: HashMap::new(),
        }
    }
}

impl ChatMessageChunk {
    pub fn new(content: impl Into<String>, role: crate::message::MessageRole) -> Self {
        Self {
            content: content.into(),
            role,
            is_final: false,
            metadata: HashMap::new(),
        }
    }
}

impl CompletionChunk {
    /// Create a simple text chunk
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Create a tool call start chunk
    pub fn tool_start(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::ToolCallStart {
            id: id.into(),
            name: name.into(),
        }
    }

    /// Create a partial tool call chunk
    pub fn tool_partial(
        id: impl Into<String>,
        name: impl Into<String>,
        partial_input: impl Into<String>,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            partial_input: partial_input.into(),
        }
    }

    /// Create a completed tool call chunk
    pub fn tool_complete(
        id: impl Into<String>,
        name: impl Into<String>,
        input: impl Into<String>,
    ) -> Self {
        Self::ToolCallComplete {
            id: id.into(),
            name: name.into(),
            input: input.into(),
        }
    }

    /// Create a completion finished chunk
    pub fn complete(
        text: impl Into<String>,
        finish_reason: Option<FinishReason>,
        usage: Option<Usage>,
    ) -> Self {
        Self::Complete {
            text: text.into(),
            finish_reason,
            usage,
        }
    }

    /// Create an error chunk
    pub fn error(error: impl Into<String>) -> Self {
        Self::Error(error.into())
    }

    /// Check if this chunk contains text content
    pub fn has_text(&self) -> bool {
        matches!(self, Self::Text(_) | Self::Complete { .. })
    }

    /// Extract text content if available
    pub fn text_content(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Complete { text, .. } => Some(text),
            _ => None,
        }
    }

    /// Check if this is a completion chunk
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete { .. })
    }

    /// Check if this is an error chunk
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

impl LegacyCompletionChunk {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            finish_reason: None,
            usage: None,
        }
    }

    /// Convert to new CompletionChunk format
    pub fn into_completion_chunk(self) -> CompletionChunk {
        CompletionChunk::Complete {
            text: self.text,
            finish_reason: self.finish_reason,
            usage: self.usage,
        }
    }
}
