//! Error handling integration tests
//!
//! Tests that verify proper error handling for unknown functions and invalid arguments

use serde_json::json;
use super::super::FunctionEvaluator;
use super::mock_evaluator::mock_evaluator;

#[cfg(test)]
mod tests {
    use super::*;

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
}