//! Edge case tests for value() function
//!
//! Tests that verify proper handling of Unicode, special characters, and edge cases

use serde_json::json;
use crate::json_path::parser::{FilterExpression, FilterValue};
use super::super::core::evaluate_value_function;
use super::mock_evaluator::mock_evaluator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_function_unicode_handling() {
        let context = json!({"message": "Hello 世界 🌍"});
        let args = vec![FilterExpression::Property {
            path: vec!["message".to_string()],
        }];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert_eq!(
            result.unwrap(),
            FilterValue::String("Hello 世界 🌍".to_string())
        );
    }
}