//! Tests for the completion module

use serde_json::json;

use crate::completion::*;
use crate::prompt::Prompt;
use crate::validation::ValidationResult;

#[test]
fn test_completion_request_validation() {
    // Test valid temperature
    let valid = CompletionParams::new().with_temperature(0.7).unwrap();
    assert!((0.7 - valid.temperature).abs() < f64::EPSILON);

    // Test invalid temperature (too low)
    assert!(CompletionParams::new().with_temperature(-0.1).is_err());

    // Test invalid temperature (too high)
    assert!(CompletionParams::new().with_temperature(2.1).is_err());
}

#[test]
fn test_completion_response_builder() {
    let response = CompletionResponse::builder()
        .text("Test response")
        .model("test-model")
        .build();

    assert_eq!(response.text(), "Test response");
    assert_eq!(response.model(), "test-model");
}

#[test]
fn test_compact_completion_response() {
    let response = CompletionResponse::new("Test response", "test-model");
    let compact = response.into_compact();
    let back = compact.into_standard();

    assert_eq!(back.text(), "Test response");
    assert_eq!(back.model(), "test-model");
}
