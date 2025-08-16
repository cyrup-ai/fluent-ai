//! Type conversion tests for value() function
//!
//! Tests that verify proper type conversion for different JSON value types

use serde_json::json;
use crate::json_path::parser::{FilterExpression, FilterValue};
use super::super::core::evaluate_value_function;
use super::mock_evaluator::mock_evaluator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_function_property_array_object_conversion() {
        let context = json!({"items": [1, 2, 3], "obj": {"a": 1}});

        let args = vec![FilterExpression::Property {
            path: vec!["items".to_string()],
        }];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null); // Arrays convert to null

        let args = vec![FilterExpression::Property {
            path: vec!["obj".to_string()],
        }];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null); // Objects convert to null
    }

    #[test]
    fn test_value_function_property_boolean() {
        let context = json!({"flag": true, "disabled": false});

        let args = vec![FilterExpression::Property {
            path: vec!["flag".to_string()],
        }];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(true));

        let args = vec![FilterExpression::Property {
            path: vec!["disabled".to_string()],
        }];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(false));
    }

    #[test]
    fn test_value_function_property_number() {
        let context = json!({"pi": 3.14, "count": 42});

        let args = vec![FilterExpression::Property {
            path: vec!["pi".to_string()],
        }];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Number(3.14));

        let args = vec![FilterExpression::Property {
            path: vec!["count".to_string()],
        }];
        let result = evaluate_value_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(42));
    }
}