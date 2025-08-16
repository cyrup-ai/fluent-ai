//! Argument validation tests for value() function
//!
//! Tests that verify proper argument count validation

use serde_json::json;
use crate::json_path::parser::{FilterExpression, FilterValue};
use super::super::core::evaluate_value_function;
use super::mock_evaluator::mock_evaluator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_function_wrong_arg_count() {
        let context = json!({});
        let args = vec![];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exactly one argument")
        );

        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("test".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String("extra".to_string()),
            },
        ];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exactly one argument")
        );
    }
}