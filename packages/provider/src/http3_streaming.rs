//! HTTP3 Streaming Trait Architecture
//!
//! Zero-allocation, blazing-fast trait-based streaming for HTTP3 operations.
//! All HTTP methods return trait objects directly with unwrapped values.
//! Error handling is managed internally, consumers receive clean data streams.

use std::fmt;
use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

/// Base trait for successful HTTP3 streaming operations
///
/// Provides core functionality for HTTP streaming chunks with zero allocation
/// and lock-free access patterns. All implementations must be object-safe.
pub trait HttpChunk: Send + Sync {
    /// Get the chunk data as bytes slice
    /// Zero-allocation access to underlying data
    #[inline(always)]
    fn data(&self) -> &[u8];

    /// Get current progress as percentage (0.0-100.0)
    /// Returns None if progress is not applicable/available
    #[inline(always)]
    fn progress(&self) -> Option<f32>;

    /// Check if this is the final chunk in the stream
    #[inline(always)]
    fn is_final(&self) -> bool;

    /// Get chunk metadata as JSON value
    /// Returns None if no metadata is available
    fn metadata(&self) -> Option<&Value>;

    /// Get chunk size in bytes
    #[inline(always)]
    fn size(&self) -> usize {
        self.data().len()
    }

    /// Check if chunk has data
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.data().is_empty()
    }
}

/// Base trait for HTTP3 error conditions
///
/// Represents failed HTTP operations with structured error information.
/// All error chunks provide diagnostic information without Result wrapping.
pub trait BadHttpChunk: Send + Sync + fmt::Display + fmt::Debug {
    /// Get the error kind/category
    fn error_kind(&self) -> HttpErrorKind;

    /// Get human-readable error message
    fn message(&self) -> &str;

    /// Get error code if available
    fn code(&self) -> Option<u32>;

    /// Check if error is retryable
    fn is_retryable(&self) -> bool;

    /// Get structured error details as JSON
    fn details(&self) -> Option<&Value>;
}

/// HTTP error categories for structured error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HttpErrorKind {
    /// Network connectivity issues
    Network = 0,
    /// HTTP protocol errors (4xx, 5xx)
    Http = 1,
    /// Authentication/authorization failures
    Auth = 2,
    /// Request timeout
    Timeout = 3,
    /// Rate limiting
    RateLimit = 4,
    /// Data parsing/serialization errors
    Parsing = 5,
    /// Configuration errors
    Config = 6,
    /// Unknown/other errors
    Unknown = 7}

impl fmt::Display for HttpErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Network => write!(f, "Network"),
            Self::Http => write!(f, "HTTP"),
            Self::Auth => write!(f, "Authentication"),
            Self::Timeout => write!(f, "Timeout"),
            Self::RateLimit => write!(f, "Rate Limit"),
            Self::Parsing => write!(f, "Parsing"),
            Self::Config => write!(f, "Configuration"),
            Self::Unknown => write!(f, "Unknown")}
    }
}

/// File/model download operations with progress tracking
///
/// Specialized chunk type for downloads with progress reporting,
/// metadata access, and caching integration.
pub trait DownloadChunk: HttpChunk {
    /// Get the filename being downloaded
    fn filename(&self) -> &str;

    /// Get downloaded bytes so far
    fn downloaded_bytes(&self) -> u64;

    /// Get total bytes to download (if known)
    fn total_bytes(&self) -> Option<u64>;

    /// Get download speed in bytes/second
    fn speed_bps(&self) -> Option<u64>;

    /// Get estimated time remaining in seconds
    fn eta_seconds(&self) -> Option<u64>;

    /// Get file metadata/headers
    fn file_metadata(&self) -> Option<&Value>;
}

/// Bad download chunk for download errors
pub trait BadDownloadChunk: BadHttpChunk {
    /// Get the filename that failed to download
    fn filename(&self) -> &str;

    /// Get partially downloaded bytes (if any)
    fn partial_bytes(&self) -> u64;
}

/// Audio transcription operations with streaming results
///
/// Specialized for transcription operations with partial results,
/// confidence scores, and speaker information.
pub trait TranscriptionChunk: HttpChunk {
    /// Get transcribed text content
    fn text(&self) -> &str;

    /// Get confidence score (0.0-1.0)
    fn confidence(&self) -> Option<f32>;

    /// Get speaker ID if available
    fn speaker_id(&self) -> Option<&str>;

    /// Get timestamp information (start, end in seconds)
    fn timestamp(&self) -> Option<(f32, f32)>;

    /// Check if this is a partial/interim result
    fn is_partial(&self) -> bool;
}

/// Bad transcription chunk for transcription errors  
pub trait BadTranscriptionChunk: BadHttpChunk {
    /// Get the audio format that caused the error
    fn audio_format(&self) -> Option<&str>;

    /// Get audio duration if known
    fn audio_duration(&self) -> Option<f32>;
}

/// Text completion operations with streaming tokens
///
/// Specialized for LLM completions with token streaming,
/// finish reasons, and usage tracking.
pub trait CompletionChunk: HttpChunk {
    /// Get the completion text/tokens
    fn text(&self) -> &str;

    /// Get finish reason if completion is done
    fn finish_reason(&self) -> Option<&str>;

    /// Get token usage information
    fn token_usage(&self) -> Option<TokenUsage>;

    /// Get model name used for completion
    fn model(&self) -> Option<&str>;

    /// Check if this chunk contains tool calls
    fn has_tool_calls(&self) -> bool;

    /// Get tool call information if available
    fn tool_calls(&self) -> Option<&[ToolCall]>;
}

/// Bad completion chunk for completion errors
pub trait BadCompletionChunk: BadHttpChunk {
    /// Get the model that caused the error
    fn model(&self) -> Option<&str>;

    /// Get partial completion if any
    fn partial_text(&self) -> Option<&str>;
}

/// Embedding operations with vector results
///
/// Specialized for embedding operations with vector data,
/// dimensions, and similarity metrics.
pub trait EmbeddingChunk: HttpChunk {
    /// Get embedding vector as f32 slice
    fn vector(&self) -> &[f32];

    /// Get embedding dimensions
    #[inline(always)]
    fn dimensions(&self) -> usize {
        self.vector().len()
    }

    /// Get input text that was embedded
    fn input_text(&self) -> Option<&str>;

    /// Get model used for embedding
    fn model(&self) -> Option<&str>;

    /// Calculate cosine similarity with another vector
    fn cosine_similarity(&self, other: &[f32]) -> f32 {
        let a = self.vector();
        let b = other;

        if a.len() != b.len() {
            return 0.0;
        }

        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        // SIMD-friendly loop for auto-vectorization
        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a.sqrt() * norm_b.sqrt())
        }
    }
}

/// Bad embedding chunk for embedding errors
pub trait BadEmbeddingChunk: BadHttpChunk {
    /// Get the input text that failed to embed
    fn input_text(&self) -> Option<&str>;

    /// Get model that caused the error
    fn model(&self) -> Option<&str>;

    /// Get expected dimensions if known
    fn expected_dimensions(&self) -> Option<usize>;
}

/// Token usage information for LLM operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenUsage {
    /// Tokens in the prompt/input
    pub prompt_tokens: u32,
    /// Tokens in the completion/output
    pub completion_tokens: u32,
    /// Total tokens used
    pub total_tokens: u32}

impl TokenUsage {
    /// Create new token usage
    #[inline(always)]
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens}
    }

    /// Check if usage is empty
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.total_tokens == 0
    }
}

/// Tool call information for function calling
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Tool/function name
    pub name: String,
    /// Tool arguments as JSON
    pub arguments: Value,
    /// Tool call ID
    pub id: Option<String>}

impl ToolCall {
    /// Create new tool call
    #[inline(always)]
    pub fn new(name: String, arguments: Value) -> Self {
        Self {
            name,
            arguments,
            id: None}
    }

    /// Set tool call ID
    #[inline(always)]
    pub fn with_id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Concrete Trait Implementations
// ═══════════════════════════════════════════════════════════════════════════════

/// Concrete implementation of TranscriptionChunk
pub struct TranscriptionChunkImpl {
    transcribed_text: Option<String>,
    language: Option<String>,
    confidence: Option<f32>,
    data: Vec<u8>,
    progress: Option<f32>,
    is_final: bool,
    metadata: Option<Value>}

impl TranscriptionChunkImpl {
    pub fn new(
        transcribed_text: Option<String>,
        language: Option<String>,
        confidence: Option<f32>,
        data: Vec<u8>,
    ) -> Self {
        Self {
            transcribed_text,
            language,
            confidence,
            data,
            progress: None,
            is_final: false,
            metadata: None}
    }

    pub fn with_progress(mut self, progress: f32) -> Self {
        self.progress = Some(progress);
        self
    }

    pub fn with_final(mut self, is_final: bool) -> Self {
        self.is_final = is_final;
        self
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

impl TranscriptionChunk for TranscriptionChunkImpl {
    fn transcribed_text(&self) -> Option<&str> {
        self.transcribed_text.as_deref()
    }

    fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    fn confidence(&self) -> Option<f32> {
        self.confidence
    }
}

impl HttpChunk for TranscriptionChunkImpl {
    fn data(&self) -> &[u8] {
        &self.data
    }

    fn progress(&self) -> Option<f32> {
        self.progress
    }

    fn is_final(&self) -> bool {
        self.is_final
    }

    fn metadata(&self) -> Option<&Value> {
        self.metadata.as_ref()
    }
}

/// Concrete implementation of DownloadChunk  
pub struct DownloadChunkImpl {
    data: Vec<u8>,
    filename: Option<String>,
    mime_type: Option<String>,
    progress: Option<f32>,
    is_final: bool,
    metadata: Option<Value>}

impl DownloadChunkImpl {
    pub fn new(data: Vec<u8>, filename: Option<String>, mime_type: Option<String>) -> Self {
        Self {
            data,
            filename,
            mime_type,
            progress: None,
            is_final: false,
            metadata: None}
    }

    pub fn with_progress(mut self, progress: f32) -> Self {
        self.progress = Some(progress);
        self
    }

    pub fn with_final(mut self, is_final: bool) -> Self {
        self.is_final = is_final;
        self
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

impl DownloadChunk for DownloadChunkImpl {
    fn filename(&self) -> Option<&str> {
        self.filename.as_deref()
    }

    fn mime_type(&self) -> Option<&str> {
        self.mime_type.as_deref()
    }

    fn file_size(&self) -> Option<u64> {
        Some(self.data.len() as u64)
    }
}

impl HttpChunk for DownloadChunkImpl {
    fn data(&self) -> &[u8] {
        &self.data
    }

    fn progress(&self) -> Option<f32> {
        self.progress
    }

    fn is_final(&self) -> bool {
        self.is_final
    }

    fn metadata(&self) -> Option<&Value> {
        self.metadata.as_ref()
    }
}
