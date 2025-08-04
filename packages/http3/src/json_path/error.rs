//! Comprehensive error handling for JSONPath streaming operations
//!
//! This module provides detailed error types for all failure modes in JSONPath
//! expression parsing, JSON deserialization, and stream processing operations.

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

impl fmt::Display for JsonPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonPathError::InvalidExpression {
                expression,
                reason,
                position,
            } => {
                write!(
                    f,
                    "Invalid JSONPath expression '{}': {}",
                    expression, reason
                )?;
                if let Some(pos) = position {
                    write!(f, " at position {}", pos)?;
                }
                Ok(())
            }

            JsonPathError::JsonParseError {
                message,
                offset,
                context,
            } => {
                write!(
                    f,
                    "JSON parsing error at byte {}: {} (context: '{}')",
                    offset, message, context
                )
            }

            JsonPathError::DeserializationError {
                message,
                json_fragment,
                target_type,
            } => {
                write!(
                    f,
                    "Failed to deserialize JSON '{}' to type {}: {}",
                    json_fragment, target_type, message
                )
            }

            JsonPathError::StreamError {
                message,
                state,
                recoverable,
            } => {
                write!(
                    f,
                    "Stream processing error in state '{}': {} (recoverable: {})",
                    state, message, recoverable
                )
            }

            JsonPathError::BufferError {
                operation,
                requested_size,
                available_capacity,
            } => {
                write!(
                    f,
                    "Buffer {} failed: requested {} bytes, available {} bytes",
                    operation, requested_size, available_capacity
                )
            }

            JsonPathError::UnsupportedFeature {
                feature,
                alternative,
            } => {
                write!(f, "Unsupported JSONPath feature: {}", feature)?;
                if let Some(alt) = alternative {
                    write!(f, " (try: {})", alt)?;
                }
                Ok(())
            }

            JsonPathError::Deserialization(message) => {
                write!(f, "Deserialization error: {}", message)
            }
        }
    }
}

impl std::error::Error for JsonPathError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // No nested errors in our current implementation
        None
    }
}

// Conversions from third-party error types

impl From<serde_json::Error> for JsonPathError {
    fn from(err: serde_json::Error) -> Self {
        let message = err.to_string();
        let offset = err.line().saturating_sub(1) * 80 + err.column(); // Rough byte estimate

        // Extract context around error if available
        let context = if err.is_data() {
            "data error".to_string()
        } else if err.is_eof() {
            "unexpected end of JSON".to_string()
        } else if err.is_syntax() {
            "syntax error".to_string()
        } else {
            "unknown error".to_string()
        };

        JsonPathError::JsonParseError {
            message,
            offset,
            context,
        }
    }
}

/// Extension trait for JsonPathResult error handling
///
/// Provides convenient error handling methods while maintaining compatibility
/// with the async-stream architecture for external APIs.
pub trait JsonPathResultExt<T> {
    /// Convert to stream-compatible error handling
    fn handle_or_default(self, default: T) -> T;

    /// Convert to stream-compatible error handling with logging
    fn handle_or_log(self, context: &str, default: T) -> T;

    /// Add stream context to errors
    fn with_stream_context(self, context: &str) -> JsonPathResult<T>;
}

impl<T> JsonPathResultExt<T> for JsonPathResult<T> {
    fn handle_or_default(self, default: T) -> T {
        self.unwrap_or_else(|_| {
            log::warn!("JsonPathResult defaulted due to error: using fallback value");
            default
        })
    }

    fn handle_or_log(self, context: &str, default: T) -> T {
        match self {
            Ok(value) => value,
            Err(err) => {
                log::error!("JSONPath error in {}: {}", context, err);
                default
            }
        }
    }

    fn with_stream_context(self, context: &str) -> JsonPathResult<T> {
        self.map_err(|err| match err {
            JsonPathError::StreamError { message, recoverable, .. } => {
                JsonPathError::StreamError {
                    message,
                    state: context.to_string(),
                    recoverable,
                }
            }
            other => other,
        })
    }
}

/// Create InvalidExpression error with detailed context
pub fn invalid_expression_error(
    expression: &str,
    reason: &str,
    position: Option<usize>,
) -> JsonPathError {
    JsonPathError::InvalidExpression {
        expression: expression.to_string(),
        reason: reason.to_string(),
        position,
    }
}

/// Create StreamError with recovery information
pub fn stream_error(message: &str, state: &str, recoverable: bool) -> JsonPathError {
    JsonPathError::StreamError {
        message: message.to_string(),
        state: state.to_string(),
        recoverable,
    }
}

/// Create BufferError for capacity issues
pub fn buffer_error(operation: &str, requested: usize, available: usize) -> JsonPathError {
    JsonPathError::BufferError {
        operation: operation.to_string(),
        requested_size: requested,
        available_capacity: available,
    }
}

/// Create UnsupportedFeature error with alternative suggestion
pub fn unsupported_feature_error(feature: &str, alternative: Option<&str>) -> JsonPathError {
    JsonPathError::UnsupportedFeature {
        feature: feature.to_string(),
        alternative: alternative.map(|s| s.to_string()),
    }
}

/// Create JsonParseError with detailed context
pub fn json_parse_error(message: String, offset: usize, context: String) -> JsonPathError {
    JsonPathError::JsonParseError {
        message,
        offset,
        context,
    }
}

/// Create DeserializationError with type information
pub fn deserialization_error(
    message: String,
    json_fragment: String,
    target_type: &'static str,
) -> JsonPathError {
    JsonPathError::DeserializationError {
        message,
        json_fragment,
        target_type,
    }
}

#[cfg(test)]
mod tests {
    // Tests for error module will be implemented here
}
