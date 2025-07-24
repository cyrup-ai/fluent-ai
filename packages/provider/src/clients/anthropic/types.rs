//! Anthropic API request/response structures
//!
//! Comprehensive, zero-allocation structures for Anthropic Claude API:
//! - Messages API (Chat Completions)
//! - Streaming responses with server-sent events
//! - Tool calling and function execution
//! - Content blocks (text, images, tool use/results)
//! - Cache control for prompt caching
//!
//! All structures are optimized for performance with ArrayVec for bounded collections,
//! proper lifetime annotations, and efficient serialization patterns.

use std::collections::HashMap;

use arrayvec::ArrayVec;
use serde::{Deserialize, Serialize};

use crate::{MAX_MESSAGES, MAX_TOOLS};

// =============================================================================
// Embedding API
// =============================================================================

/// Anthropic embedding request structure (placeholder)
#[derive(Debug, Serialize)]
pub struct AnthropicEmbeddingRequest<'a> {
    /// The model to use for embedding.
    pub model: &'a str,
    /// The input text to embed.
    pub input: &'a str,
}

// =============================================================================
// Messages API (Chat Completions)
// =============================================================================

/// Anthropic messages request with zero-allocation design
#[derive(Debug, Serialize)]
pub struct AnthropicCompletionRequest<'a> {
    /// Model identifier (e.g., "claude-3-5-sonnet-20241022")
    pub model: &'a str,
    /// Array of messages in the conversation
    pub messages: ArrayVec<AnthropicMessage<'a>, MAX_MESSAGES>,
    /// System message/prompt (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<AnthropicSystemMessage<'a>>,
    /// Maximum number of tokens to generate
    pub max_tokens: u32,
    /// Sampling temperature between 0 and 1
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Available tools for function calling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<serde_json::Value, MAX_TOOLS>>,
    /// Whether to stream responses
    pub stream: bool,
    /// Stop sequences to halt generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<ArrayVec<&'a str, 4>>,
    /// Metadata for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata<'a>>,
}

/// System message with optional cache control
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum AnthropicSystemMessage<'a> {
    /// Simple text system message
    Text(&'a str),
    /// Structured system message with cache control
    Structured {
        /// System message type
        #[serde(rename = "type")]
        message_type: &'static str, // "text"
        /// System message text
        text: &'a str,
        /// Cache control configuration
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
}

/// Message in a conversation
#[derive(Debug, Serialize)]
pub struct AnthropicMessage<'a> {
    /// Role of the message sender ("user" or "assistant")
    pub role: &'a str,
    /// Content of the message
    pub content: AnthropicContent<'a>,
}

/// Message content (text or structured blocks)
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum AnthropicContent<'a> {
    /// Simple text content
    Text(&'a str),
    /// Structured content blocks
    Blocks(ArrayVec<AnthropicContentBlock<'a>, 16>),
}

/// Content block for multimodal and tool messages
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock<'a> {
    /// Text content block
    #[serde(rename = "text")]
    Text {
        /// Text content
        text: &'a str,
        /// Cache control configuration
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    /// Image content block
    #[serde(rename = "image")]
    Image {
        /// Image source data
        source: AnthropicImageSource<'a>,
        /// Cache control configuration
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    /// Tool use block (function call)
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Unique identifier for the tool call
        id: &'a str,
        /// Name of the tool/function
        name: &'a str,
        /// Input parameters for the tool
        input: serde_json::Value,
        /// Cache control configuration
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    /// Tool result block (function response)
    #[serde(rename = "tool_result")]
    ToolResult {
        /// Tool call ID this result corresponds to
        tool_use_id: &'a str,
        /// Result content
        content: Box<AnthropicToolResultContent<'a>>,
        /// Whether this represents an error
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        /// Cache control configuration
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
}

/// Tool result content (can be text or structured)
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum AnthropicToolResultContent<'a> {
    /// Simple text result
    Text(&'a str),
    /// Structured result blocks
    Blocks(ArrayVec<AnthropicContentBlock<'a>, 8>),
}

/// Image source configuration
#[derive(Debug, Serialize)]
pub struct AnthropicImageSource<'a> {
    /// Source type (always "base64")
    #[serde(rename = "type")]
    pub source_type: &'static str,
    /// Media type (e.g., "image/jpeg", "image/png")
    pub media_type: &'a str,
    /// Base64 encoded image data
    pub data: &'a str,
}

/// Cache control configuration for prompt caching
#[derive(Debug, Serialize)]
pub struct AnthropicCacheControl {
    /// Cache type (always "ephemeral")
    #[serde(rename = "type")]
    pub cache_type: &'static str,
}

/// Request metadata
#[derive(Debug, Serialize)]
pub struct AnthropicMetadata<'a> {
    /// User identifier for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<&'a str>,
    /// Additional metadata fields
    #[serde(flatten)]
    pub extra: HashMap<&'a str, serde_json::Value>,
}

// =============================================================================
// Streaming Response Types
// =============================================================================

/// Streaming response chunk from Anthropic
#[derive(Debug, Deserialize)]
pub struct AnthropicStreamChunk {
    /// Type of chunk
    #[serde(rename = "type")]
    pub chunk_type: String,
    /// Message data (for message_start chunks)
    #[serde(default)]
    pub message: Option<AnthropicStreamMessage>,
    /// Content block index (for content_block_start chunks)
    #[serde(default)]
    pub index: Option<u32>,
    /// Content block data (for content_block_start chunks)
    #[serde(default)]
    pub content_block: Option<AnthropicContentBlockDelta>,
    /// Delta data (for delta chunks)
    #[serde(default)]
    pub delta: Option<AnthropicDelta>,
    /// Usage information (for message_stop chunks)
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
    /// Error information (for error chunks)
    #[serde(default)]
    pub error: Option<AnthropicStreamError>,
}

/// Message data in streaming response
#[derive(Debug, Deserialize)]
pub struct AnthropicStreamMessage {
    /// Unique identifier for the message
    pub id: String,
    /// Object type (always "message")
    #[serde(rename = "type")]
    pub message_type: String,
    /// Role (always "assistant")
    pub role: String,
    /// Content blocks array
    pub content: Vec<serde_json::Value>,
    /// Model used for generation
    pub model: String,
    /// Stop reason (if completed)
    #[serde(default)]
    pub stop_reason: Option<String>,
    /// Stop sequence that triggered completion
    #[serde(default)]
    pub stop_sequence: Option<String>,
    /// Token usage statistics
    pub usage: AnthropicUsage,
}

/// Content block delta for streaming
#[derive(Debug, Deserialize)]
pub struct AnthropicContentBlockDelta {
    /// Block type
    #[serde(rename = "type")]
    pub block_type: String,
    /// Text content (for text blocks)
    #[serde(default)]
    pub text: Option<String>,
    /// Tool use ID (for tool_use blocks)
    #[serde(default)]
    pub id: Option<String>,
    /// Tool name (for tool_use blocks)
    #[serde(default)]
    pub name: Option<String>,
    /// Tool input (for tool_use blocks)
    #[serde(default)]
    pub input: Option<serde_json::Value>,
}

/// Delta content in streaming chunks
#[derive(Debug, Deserialize)]
pub struct AnthropicDelta {
    /// Delta type
    #[serde(rename = "type")]
    pub delta_type: String,
    /// Text delta (for text deltas)
    #[serde(default)]
    pub text: Option<String>,
    /// Partial JSON for tool input (for input_json deltas)
    #[serde(default)]
    pub partial_json: Option<String>,
    /// Stop reason (for message deltas)
    #[serde(default)]
    pub stop_reason: Option<String>,
    /// Stop sequence (for message deltas)
    #[serde(default)]
    pub stop_sequence: Option<String>,
    /// Usage information (for message deltas)
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

/// Token usage statistics
#[derive(Debug, Deserialize)]
pub struct AnthropicUsage {
    /// Number of input tokens
    pub input_tokens: u32,
    /// Number of output tokens
    pub output_tokens: u32,
    /// Number of cached tokens
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    /// Number of cache read tokens
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

/// Error information in streaming response
#[derive(Debug, Deserialize)]
pub struct AnthropicStreamError {
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error message
    pub message: String,
}

// =============================================================================
// Non-streaming Response Types
// =============================================================================

/// Complete message response (non-streaming)
#[derive(Debug, Deserialize)]
pub struct AnthropicResponse {
    /// Unique identifier for the response
    pub id: String,
    /// Object type (always "message")
    #[serde(rename = "type")]
    pub response_type: String,
    /// Role (always "assistant")
    pub role: String,
    /// Content blocks
    pub content: Vec<AnthropicResponseContent>,
    /// Model used for generation
    pub model: String,
    /// Stop reason
    #[serde(default)]
    pub stop_reason: Option<String>,
    /// Stop sequence that triggered completion
    #[serde(default)]
    pub stop_sequence: Option<String>,
    /// Token usage statistics
    pub usage: AnthropicUsage,
}

/// Content block in complete response
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicResponseContent {
    /// Text content block
    #[serde(rename = "text")]
    Text {
        /// Text content
        text: String,
    },
    /// Tool use block
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Tool call identifier
        id: String,
        /// Tool name
        name: String,
        /// Tool input parameters
        input: serde_json::Value,
    },
}

// =============================================================================
// Utility Types and Implementations
// =============================================================================

impl<'a> Default for AnthropicCompletionRequest<'a> {
    fn default() -> Self {
        Self {
            model: "claude-3-5-sonnet-20241022",
            messages: ArrayVec::new(),
            system: None,
            max_tokens: 1024,
            temperature: None,
            top_p: None,
            top_k: None,
            tools: None,
            stream: false,
            stop_sequences: None,
            metadata: None,
        }
    }
}

impl<'a> AnthropicCompletionRequest<'a> {
    /// Create a new completion request with the specified model
    #[inline]
    pub fn new(model: &'a str, max_tokens: u32) -> Self {
        Self {
            model,
            max_tokens,
            ..Default::default()
        }
    }

    /// Add a user message to the conversation
    pub fn add_user_message(&mut self, content: &'a str) -> Result<(), &'static str> {
        let message = AnthropicMessage {
            role: "user",
            content: AnthropicContent::Text(content),
        };

        self.messages
            .try_push(message)
            .map_err(|_| "Maximum messages exceeded")
    }

    /// Add an assistant message to the conversation
    pub fn add_assistant_message(&mut self, content: &'a str) -> Result<(), &'static str> {
        let message = AnthropicMessage {
            role: "assistant",
            content: AnthropicContent::Text(content),
        };

        self.messages
            .try_push(message)
            .map_err(|_| "Maximum messages exceeded")
    }

    /// Add a system message
    #[inline]
    pub fn with_system(mut self, system: &'a str) -> Self {
        self.system = Some(AnthropicSystemMessage::Text(system));
        self
    }

    /// Set streaming mode
    #[inline]
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set temperature
    #[inline]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set top-p
    #[inline]
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k
    #[inline]
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Add a tool for function calling
    pub fn add_tool(&mut self, tool: serde_json::Value) -> Result<(), &'static str> {
        if self.tools.is_none() {
            self.tools = Some(ArrayVec::new());
        }

        if let Some(ref mut tools) = self.tools {
            tools.try_push(tool).map_err(|_| "Maximum tools exceeded")
        } else {
            Err("Failed to initialize tools array")
        }
    }
}

impl<'a> AnthropicContent<'a> {
    /// Create text content
    #[inline]
    pub fn text(text: &'a str) -> Self {
        Self::Text(text)
    }

    /// Create content with text and image
    pub fn with_image(text: &'a str, image_data: &'a str, media_type: &'a str) -> Self {
        let mut blocks = ArrayVec::new();

        // Add text block
        let _ = blocks.try_push(AnthropicContentBlock::Text {
            text,
            cache_control: None,
        });

        // Add image block
        let _ = blocks.try_push(AnthropicContentBlock::Image {
            source: AnthropicImageSource {
                source_type: "base64",
                media_type,
                data: image_data,
            },
            cache_control: None,
        });

        Self::Blocks(blocks)
    }
}

impl AnthropicCacheControl {
    /// Create ephemeral cache control
    #[inline]
    pub fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral",
        }
    }
}

/// Helper functions for streaming response processing
impl AnthropicStreamChunk {
    /// Check if this is a message start chunk
    #[inline]
    pub fn is_message_start(&self) -> bool {
        self.chunk_type == "message_start"
    }

    /// Check if this is a content block start chunk
    #[inline]
    pub fn is_content_block_start(&self) -> bool {
        self.chunk_type == "content_block_start"
    }

    /// Check if this is a content block delta chunk
    #[inline]
    pub fn is_content_block_delta(&self) -> bool {
        self.chunk_type == "content_block_delta"
    }

    /// Check if this is a content block stop chunk
    #[inline]
    pub fn is_content_block_stop(&self) -> bool {
        self.chunk_type == "content_block_stop"
    }

    /// Check if this is a message delta chunk
    #[inline]
    pub fn is_message_delta(&self) -> bool {
        self.chunk_type == "message_delta"
    }

    /// Check if this is a message stop chunk (final chunk)
    #[inline]
    pub fn is_message_stop(&self) -> bool {
        self.chunk_type == "message_stop"
    }

    /// Check if this is an error chunk
    #[inline]
    pub fn is_error(&self) -> bool {
        self.chunk_type == "error"
    }

    /// Get text content from delta if available
    #[inline]
    pub fn text(&self) -> Option<&str> {
        self.delta.as_ref()?.text.as_deref()
    }

    /// Get stop reason if available
    #[inline]
    pub fn stop_reason(&self) -> Option<&str> {
        if let Some(ref delta) = self.delta {
            delta.stop_reason.as_deref()
        } else if let Some(ref message) = self.message {
            message.stop_reason.as_deref()
        } else {
            None
        }
    }

    /// Get usage information if available
    #[inline]
    pub fn usage(&self) -> Option<&AnthropicUsage> {
        self.usage
            .as_ref()
            .or_else(|| self.delta.as_ref()?.usage.as_ref())
    }
}

impl AnthropicUsage {
    /// Calculate total tokens used
    #[inline]
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    /// Calculate total cached tokens
    #[inline]
    pub fn total_cached_tokens(&self) -> u32 {
        self.cache_creation_input_tokens.unwrap_or(0) + self.cache_read_input_tokens.unwrap_or(0)
    }
}

/// Convenience constructors for common content blocks
impl<'a> AnthropicContentBlock<'a> {
    /// Create a text block
    #[inline]
    pub fn text(text: &'a str) -> Self {
        Self::Text {
            text,
            cache_control: None,
        }
    }

    /// Create a cached text block
    #[inline]
    pub fn cached_text(text: &'a str) -> Self {
        Self::Text {
            text,
            cache_control: Some(AnthropicCacheControl::ephemeral()),
        }
    }

    /// Create an image block
    #[inline]
    pub fn image(data: &'a str, media_type: &'a str) -> Self {
        Self::Image {
            source: AnthropicImageSource {
                source_type: "base64",
                media_type,
                data,
            },
            cache_control: None,
        }
    }

    /// Create a tool use block
    #[inline]
    pub fn tool_use(id: &'a str, name: &'a str, input: serde_json::Value) -> Self {
        Self::ToolUse {
            id,
            name,
            input,
            cache_control: None,
        }
    }

    /// Create a tool result block
    #[inline]
    pub fn tool_result(tool_use_id: &'a str, content: &'a str) -> Self {
        Self::ToolResult {
            tool_use_id,
            content: Box::new(AnthropicToolResultContent::Text(content)),
            is_error: None,
            cache_control: None,
        }
    }

    /// Create a tool error result block
    #[inline]
    pub fn tool_error(tool_use_id: &'a str, error_message: &'a str) -> Self {
        Self::ToolResult {
            tool_use_id,
            content: Box::new(AnthropicToolResultContent::Text(error_message)),
            is_error: Some(true),
            cache_control: None,
        }
    }
}
