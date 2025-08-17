//! Core error constructor functions
//!
//! Primary factory methods for creating JSONPath error types with proper context.

use super::super::types::JsonPathError;

impl JsonPathError {
    /// Creates an invalid JSONPath expression error
    ///
    /// # Arguments
    /// * `expression` - The invalid JSONPath expression
    /// * `reason` - Specific reason why the expression is invalid
    /// * `position` - Optional character position where error occurred
    ///
    /// # Examples
    /// ```
    /// use http3::json_path::error::JsonPathError;
    ///
    /// let error = JsonPathError::invalid_expression(
    ///     "$.users[name=",
    ///     "unclosed bracket in filter expression",
    ///     Some(12)
    /// );
    /// ```
    pub fn invalid_expression(
        expression: impl Into<String>,
        reason: impl Into<String>,
        position: Option<usize>,
    ) -> Self {
        JsonPathError::InvalidExpression {
            expression: expression.into(),
            reason: reason.into(),
            position,
        }
    }

    /// Creates a JSON parsing error with context
    ///
    /// # Arguments
    /// * `message` - Description of the parsing failure
    /// * `offset` - Byte offset where error occurred
    /// * `context` - Context information about the parsing state
    ///
    /// # Examples
    /// ```
    /// use http3::json_path::error::JsonPathError;
    ///
    /// let error = JsonPathError::json_parse_error(
    ///     "expected comma after object member",
    ///     142,
    ///     "parsing object in array element"
    /// );
    /// ```
    pub fn json_parse_error(
        message: impl Into<String>,
        offset: usize,
        context: impl Into<String>,
    ) -> Self {
        JsonPathError::JsonParseError {
            message: message.into(),
            offset,
            context: context.into(),
        }
    }

    /// Creates a deserialization error for target types
    ///
    /// # Arguments
    /// * `message` - Serde error details
    /// * `json_fragment` - JSON value that failed to deserialize
    /// * `target_type` - Target type information
    ///
    /// # Examples
    /// ```
    /// use http3::json_path::error::JsonPathError;
    ///
    /// let error = JsonPathError::deserialization_error(
    ///     "invalid type: string, expected i32",
    ///     r#""not_a_number""#,
    ///     "i32"
    /// );
    /// ```
    pub fn deserialization_error(
        message: impl Into<String>,
        json_fragment: impl Into<String>,
        target_type: &'static str,
    ) -> Self {
        JsonPathError::DeserializationError {
            message: message.into(),
            json_fragment: json_fragment.into(),
            target_type,
        }
    }

    /// Creates a stream processing error
    ///
    /// # Arguments
    /// * `message` - Error description
    /// * `state` - Current stream state when error occurred
    /// * `recoverable` - Whether the error is recoverable
    ///
    /// # Examples
    /// ```
    /// use http3::json_path::error::JsonPathError;
    ///
    /// let error = JsonPathError::stream_error(
    ///     "buffer capacity exceeded",
    ///     "processing_array",
    ///     false
    /// );
    /// ```
    pub fn stream_error(
        message: impl Into<String>,
        state: impl Into<String>,
        recoverable: bool,
    ) -> Self {
        JsonPathError::StreamError {
            message: message.into(),
            state: state.into(),
            recoverable,
        }
    }

    /// Creates a buffer management error
    ///
    /// # Arguments
    /// * `operation` - Buffer operation that failed
    /// * `requested_size` - Requested size or capacity
    /// * `available_capacity` - Available capacity
    ///
    /// # Examples
    /// ```
    /// use http3::json_path::error::JsonPathError;
    ///
    /// let error = JsonPathError::buffer_error(
    ///     "allocation",
    ///     2048,
    ///     1024
    /// );
    /// ```
    pub fn buffer_error(
        operation: impl Into<String>,
        requested_size: usize,
        available_capacity: usize,
    ) -> Self {
        JsonPathError::BufferError {
            operation: operation.into(),
            requested_size,
            available_capacity,
        }
    }

    /// Creates an unsupported feature error
    ///
    /// # Arguments
    /// * `feature` - JSONPath feature description
    /// * `alternative` - Optional suggested alternative or workaround
    ///
    /// # Examples
    /// ```
    /// use http3::json_path::error::JsonPathError;
    ///
    /// let error = JsonPathError::unsupported_feature(
    ///     "recursive descent operator (..)",
    ///     Some("use explicit path traversal")
    /// );
    /// ```
    pub fn unsupported_feature(
        feature: impl Into<String>,
        alternative: Option<impl Into<String>>,
    ) -> Self {
        JsonPathError::UnsupportedFeature {
            feature: feature.into(),
            alternative: alternative.map(|a| a.into()),
        }
    }

    /// Creates a simple deserialization error for compatibility
    ///
    /// # Arguments
    /// * `message` - Error message
    ///
    /// # Examples
    /// ```
    /// use http3::json_path::error::JsonPathError;
    ///
    /// let error = JsonPathError::deserialization("type mismatch");
    /// ```
    pub fn deserialization(message: impl Into<String>) -> Self {
        JsonPathError::Deserialization(message.into())
    }
}
