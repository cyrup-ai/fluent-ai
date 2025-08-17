//! Standard library type conversions to JsonPathError
//!
//! From trait implementations for converting common std types into JsonPathError variants.

use super::super::types::JsonPathError;

/// Conversion from serde_json::Error to JsonPathError
impl From<serde_json::Error> for JsonPathError {
    fn from(error: serde_json::Error) -> Self {
        // Extract useful information from serde_json::Error
        let message = error.to_string();

        // Try to extract line/column information if available
        if let Some(line) = error.line() {
            let offset = line.saturating_sub(1) * 80; // Rough estimate
            let context = format!("line {}, column {}", line, error.column());

            JsonPathError::JsonParseError {
                message,
                offset,
                context,
            }
        } else {
            // Fallback to simple deserialization error
            JsonPathError::Deserialization(message)
        }
    }
}

/// Conversion from std::io::Error to JsonPathError
impl From<std::io::Error> for JsonPathError {
    fn from(error: std::io::Error) -> Self {
        JsonPathError::StreamError {
            message: error.to_string(),
            state: "io_operation".to_string(),
            recoverable: matches!(
                error.kind(),
                std::io::ErrorKind::Interrupted | std::io::ErrorKind::WouldBlock
            ),
        }
    }
}

/// Conversion from std::fmt::Error to JsonPathError
impl From<std::fmt::Error> for JsonPathError {
    fn from(error: std::fmt::Error) -> Self {
        JsonPathError::StreamError {
            message: error.to_string(),
            state: "formatting".to_string(),
            recoverable: false,
        }
    }
}

/// Conversion from std::str::Utf8Error to JsonPathError
impl From<std::str::Utf8Error> for JsonPathError {
    fn from(error: std::str::Utf8Error) -> Self {
        JsonPathError::JsonParseError {
            message: format!("invalid UTF-8 sequence: {}", error),
            offset: error.valid_up_to(),
            context: "UTF-8 validation".to_string(),
        }
    }
}

/// Conversion from std::string::FromUtf8Error to JsonPathError
impl From<std::string::FromUtf8Error> for JsonPathError {
    fn from(error: std::string::FromUtf8Error) -> Self {
        let utf8_error = error.utf8_error();
        JsonPathError::JsonParseError {
            message: format!("invalid UTF-8 in string conversion: {}", utf8_error),
            offset: utf8_error.valid_up_to(),
            context: "string conversion".to_string(),
        }
    }
}

/// Conversion from std::num::ParseIntError to JsonPathError
impl From<std::num::ParseIntError> for JsonPathError {
    fn from(error: std::num::ParseIntError) -> Self {
        JsonPathError::DeserializationError {
            message: error.to_string(),
            json_fragment: "number".to_string(),
            target_type: "integer",
        }
    }
}

/// Conversion from std::num::ParseFloatError to JsonPathError
impl From<std::num::ParseFloatError> for JsonPathError {
    fn from(error: std::num::ParseFloatError) -> Self {
        JsonPathError::DeserializationError {
            message: error.to_string(),
            json_fragment: "number".to_string(),
            target_type: "float",
        }
    }
}
