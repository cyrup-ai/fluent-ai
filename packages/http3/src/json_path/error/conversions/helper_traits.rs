//! Helper traits for error conversion and context management
//!
//! Provides utility traits for converting Results to JsonPathError with additional context.

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
