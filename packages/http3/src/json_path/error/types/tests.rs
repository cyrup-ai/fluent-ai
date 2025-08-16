//! Comprehensive tests for JsonPathError types and implementations

use super::core::*;

#[test]
fn test_invalid_expression_display() {
    let error = JsonPathError::InvalidExpression {
        expression: "$.invalid[".to_string(),
        reason: "unclosed bracket".to_string(),
        position: Some(9),
    };
    let display = format!("{}", error);
    assert!(display.contains("Invalid JSONPath expression"));
    assert!(display.contains("$.invalid["));
    assert!(display.contains("unclosed bracket"));
    assert!(display.contains("position 9"));
}

#[test]
fn test_invalid_expression_display_no_position() {
    let error = JsonPathError::InvalidExpression {
        expression: "$.test".to_string(),
        reason: "semantic error".to_string(),
        position: None,
    };
    let display = format!("{}", error);
    assert!(display.contains("Invalid JSONPath expression"));
    assert!(!display.contains("position"));
}

#[test]
fn test_json_parse_error_display() {
    let error = JsonPathError::JsonParseError {
        message: "expected comma".to_string(),
        offset: 42,
        context: "parsing array".to_string(),
    };
    let display = format!("{}", error);
    assert!(display.contains("JSON parsing error at byte 42"));
    assert!(display.contains("expected comma"));
    assert!(display.contains("parsing array"));
}
#[test]
fn test_deserialization_error_display() {
    let error = JsonPathError::DeserializationError {
        message: "invalid type".to_string(),
        json_fragment: r#"{"key": "value"}"#.to_string(),
        target_type: "i32",
    };
    let display = format!("{}", error);
    assert!(display.contains("Failed to deserialize JSON"));
    assert!(display.contains(r#"{"key": "value"}"#));
    assert!(display.contains("i32"));
    assert!(display.contains("invalid type"));
}

#[test]
fn test_stream_error_display() {
    let error = JsonPathError::StreamError {
        message: "buffer overflow".to_string(),
        state: "processing".to_string(),
        recoverable: true,
    };
    let display = format!("{}", error);
    assert!(display.contains("Stream processing error"));
    assert!(display.contains("processing"));
    assert!(display.contains("buffer overflow"));
    assert!(display.contains("recoverable: true"));
}

#[test]
fn test_buffer_error_display() {
    let error = JsonPathError::BufferError {
        operation: "allocation".to_string(),
        requested_size: 1024,
        available_capacity: 512,
    };
    let display = format!("{}", error);
    assert!(display.contains("Buffer allocation failed"));
    assert!(display.contains("1024 bytes"));
    assert!(display.contains("512 bytes"));
}

#[test]
fn test_unsupported_feature_display_with_alternative() {
    let error = JsonPathError::UnsupportedFeature {
        feature: "recursive descent".to_string(),
        alternative: Some("use explicit path".to_string()),
    };
    let display = format!("{}", error);
    assert!(display.contains("Unsupported JSONPath feature"));
    assert!(display.contains("recursive descent"));
    assert!(display.contains("try: use explicit path"));
}
#[test]
fn test_unsupported_feature_display_no_alternative() {
    let error = JsonPathError::UnsupportedFeature {
        feature: "advanced filter".to_string(),
        alternative: None,
    };
    let display = format!("{}", error);
    assert!(display.contains("Unsupported JSONPath feature"));
    assert!(display.contains("advanced filter"));
    assert!(!display.contains("try:"));
}

#[test]
fn test_deserialization_simple_display() {
    let error = JsonPathError::Deserialization("type mismatch".to_string());
    let display = format!("{}", error);
    assert!(display.contains("Deserialization error"));
    assert!(display.contains("type mismatch"));
}

#[test]
fn test_error_trait_implementation() {
    let error = JsonPathError::InvalidExpression {
        expression: "test".to_string(),
        reason: "test reason".to_string(),
        position: None,
    };

    // Test that it implements std::error::Error
    let _: &dyn std::error::Error = &error;
    assert!(error.source().is_none());
}

#[test]
fn test_error_clone() {
    let error = JsonPathError::StreamError {
        message: "test".to_string(),
        state: "test_state".to_string(),
        recoverable: false,
    };
    let cloned = error.clone();
    assert_eq!(format!("{}", error), format!("{}", cloned));
}

#[test]
fn test_error_debug() {
    let error = JsonPathError::BufferError {
        operation: "test_op".to_string(),
        requested_size: 100,
        available_capacity: 50,
    };
    let debug_str = format!("{:?}", error);
    assert!(debug_str.contains("BufferError"));
    assert!(debug_str.contains("test_op"));
    assert!(debug_str.contains("100"));
    assert!(debug_str.contains("50"));
}
