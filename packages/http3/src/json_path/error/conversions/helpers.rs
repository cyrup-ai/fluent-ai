//! Helper traits and utilities for error conversion
//!
//! Provides convenient traits and specialized methods for converting errors with context.

use super::super::types::JsonPathError;

/// Helper trait for converting Results to JsonPathError
pub trait IntoJsonPathError<T> {
    /// Converts a Result into JsonPathError with context
    fn into_jsonpath_error(self, context: &str) -> Result<T, JsonPathError>;

    /// Converts a Result into JsonPathError with custom error mapping
    fn map_jsonpath_error<F>(self, f: F) -> Result<T, JsonPathError>
    where
        F: FnOnce() -> JsonPathError;
}

impl<T, E> IntoJsonPathError<T> for Result<T, E>
where
    E: Into<JsonPathError>,
{
    fn into_jsonpath_error(self, context: &str) -> Result<T, JsonPathError> {
        self.map_err(|e| {
            let mut error = e.into();

            // Add context information if it's a stream error
            if let JsonPathError::StreamError { state, .. } = &mut error {
                if state == "io_operation" {
                    *state = context.to_string();
                }
            }

            error
        })
    }

    fn map_jsonpath_error<F>(self, f: F) -> Result<T, JsonPathError>
    where
        F: FnOnce() -> JsonPathError,
    {
        self.map_err(|_| f())
    }
}

/// Specialized conversions for common scenarios
impl JsonPathError {
    /// Creates a JsonPathError from a serde_json::Error with additional context
    pub fn from_serde_with_context(
        error: serde_json::Error,
        json_fragment: &str,
        target_type: &'static str,
    ) -> Self {
        JsonPathError::DeserializationError {
            message: error.to_string(),
            json_fragment: json_fragment.to_string(),
            target_type,
        }
    }

    /// Creates a JsonPathError from an IO error with stream context
    pub fn from_io_with_context(error: std::io::Error, state: &str) -> Self {
        JsonPathError::StreamError {
            message: error.to_string(),
            state: state.to_string(),
            recoverable: matches!(
                error.kind(),
                std::io::ErrorKind::Interrupted | std::io::ErrorKind::WouldBlock
            ),
        }
    }
}
