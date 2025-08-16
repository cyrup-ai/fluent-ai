//! Display implementations for JsonPathError variants

use std::fmt;

use super::core::JsonPathError;

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
