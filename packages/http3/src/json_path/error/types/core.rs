//! Core error types and result type for JSONPath operations

use std::fmt;

/// Internal Result type for JSONPath operations
///
/// Used internally for error propagation within the JSONPath evaluation engine.
/// External stream APIs use direct T values with emit!/handle_error! patterns.
pub type JsonPathResult<T> = Result<T, JsonPathError>;

/// Comprehensive error enumeration for JSONPath streaming operations
#[derive(Debug, Clone)]
pub enum JsonPathError {
    /// JSONPath expression syntax or semantic errors
    InvalidExpression {
        /// The invalid JSONPath expression
        expression: String,
        /// Specific parsing error details
        reason: String,
        /// Character position where error occurred (if available)
        position: Option<usize>,
    },

    /// JSON parsing and deserialization errors
    JsonParseError {
        /// Description of the JSON parsing failure
        message: String,
        /// Byte offset in stream where error occurred
        offset: usize,
        /// Context around the error location
        context: String,
    },

    /// Serde deserialization errors for target types
    DeserializationError {
        /// Serde error details
        message: String,
        /// JSON value that failed to deserialize
        json_fragment: String,
        /// Target type information
        target_type: &'static str,
    },

    /// Stream processing and buffer management errors
    StreamError {
        /// Stream processing error description
        message: String,
        /// Current stream state when error occurred
        state: String,
        /// Whether the error is recoverable
        recoverable: bool,
    },

    /// Memory allocation and capacity errors
    BufferError {
        /// Buffer operation that failed
        operation: String,
        /// Requested size or capacity
        requested_size: usize,
        /// Available capacity
        available_capacity: usize,
    },

    /// JSONPath feature not yet implemented
    UnsupportedFeature {
        /// JSONPath feature description
        feature: String,
        /// Suggested alternative or workaround
        alternative: Option<String>,
    },

    /// General deserialization error for compatibility
    Deserialization(String),
}
