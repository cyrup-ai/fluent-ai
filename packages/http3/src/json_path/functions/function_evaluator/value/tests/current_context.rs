//! Current context tests for value() function
//!
//! Tests that verify current context (@) expression handling

use serde_json::json;
use crate::json_path::parser::{FilterExpression, FilterValue};
use super::super::core::evaluate_value_function;
use super::mock_evaluator::mock_evaluator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_function_current_context() {
        let context = json!({"test": "value"});
        let args = vec![FilterExpression::Current];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null); // Objects convert to null
    }

    #[test]
    fn test_value_function_current_context_primitive() {
        let context = json!("string value");
        let args = vec![FilterExpression::Current];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert_eq!(
            result.unwrap(),
            FilterValue::String("string value".to_string())
        );

        let context = json!(42);
        let args = vec![FilterExpression::Current];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(42));
    }
}