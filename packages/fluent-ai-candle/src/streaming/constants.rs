//! Streaming constants and configuration limits
//!
//! Defines performance-critical constants for zero-allocation streaming
//! with blazing-fast token transmission and bounded memory usage.

/// Maximum text length per token chunk for bounded memory usage
pub const MAX_CHUNK_TEXT_SIZE: usize = 512;

/// Default buffer size for token transmission queue
pub const DEFAULT_BUFFER_SIZE: usize = 64;

/// Maximum buffer size to prevent unbounded memory growth
pub const MAX_BUFFER_SIZE: usize = 256;

/// Target token transmission latency in microseconds
pub const TARGET_LATENCY_MICROS: u64 = 100;

/// Memory usage budget for streaming (32KB)
pub const MEMORY_BUDGET_BYTES: usize = 32 * 1024;

/// Error types for streaming operations
#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    /// Buffer overflow error
    #[error("Buffer overflow: {0}")]
    BufferOverflow(String),

    /// Encoding error
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Timeout error
    #[error("Timeout after {seconds} seconds")]
    Timeout { seconds: u64 },

    /// UTF-8 encoding/decoding error
    #[error("UTF-8 error: {0}")]
    Utf8Error(String),

    /// Backpressure handling error
    #[error("Backpressure error: {0}")]
    BackpressureError(String),

    /// Flow control error
    #[error("Flow control error: {0}")]
    FlowControlError(String),

    /// Format conversion error
    #[error("Format error: {0}")]
    FormatError(String)}

/// Response for streaming token operations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamingTokenResponse {
    /// Token text
    pub text: String,
    /// Sequence ID
    pub sequence_id: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Is final token
    pub is_final: bool}
