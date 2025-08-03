//! JSON Path Error Tests
//!
//! Tests for the JSONPath error handling, moved from src/json_path/error.rs

use fluent_ai_http3::json_path::error::{
    JsonPathError, JsonPathErrorExt, JsonPathResult, invalid_expression_error,
};

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_error_display_formatting() {
        let err = invalid_expression_error("$.invalid[", "unclosed bracket", Some(10));
        let display = format!("{}", err);
        assert!(display.contains("Invalid JSONPath expression"));
        assert!(display.contains("$.invalid["));
        assert!(display.contains("position 10"));
    }

    #[test]
    fn test_error_context_chaining() {
        let result: JsonPathResult<()> = Err(JsonPathError::StreamError {
            message: "test error".to_string(),
            state: "initial".to_string(),
            recoverable: true,
        });

        let with_context = result.with_stream_context("parsing");
        assert!(
            matches!(with_context, Err(JsonPathError::StreamError { state, .. }) if state == "parsing")
        );
    }

    #[test]
    fn test_serde_json_error_conversion() {
        let json_err = serde_json::from_str::<i32>("invalid json")
            .err()
            .expect("Should error");
        let path_err: JsonPathError = json_err.into();

        assert!(matches!(path_err, JsonPathError::JsonParseError { .. }));
    }
}
