//! Error handling integration tests
//!
//! Tests that verify proper error handling for unknown functions and invalid arguments

use serde_json::json;
use fluent_ai_http3_client::jsonpath::functions::FunctionEvaluator;

// Mock evaluator for testing
fn mock_evaluator(
    _context: &serde_json::Value,
    _expr: &fluent_ai_http3_client::jsonpath::parser::FilterExpression,
) -> fluent_ai_http3_client::jsonpath::error::JsonPathResult<fluent_ai_http3_client::jsonpath::parser::FilterValue> {
    Ok(fluent_ai_http3_client::jsonpath::parser::FilterValue::String("mock".to_string()))
}

#[test]
fn test_integration_function_error_handling() {
    let context = json!({});

    // Test unknown function
    let result =
        FunctionEvaluator::evaluate_function_value(&context, "unknown", &[], &mock_evaluator);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("unknown function"));

    // Test wrong argument count
    let result =
        FunctionEvaluator::evaluate_function_value(&context, "length", &[], &mock_evaluator);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("exactly one argument")
    );
}