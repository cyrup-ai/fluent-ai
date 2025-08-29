//! Edge case integration tests
//!
//! Tests that verify proper handling of edge cases like empty values, null values, and boundary conditions

use super::mock_evaluator::mock_evaluator;
use crate::jsonpath::parser::{FilterExpression, FilterValue};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_edge_cases() {
        let context = json!({
            "empty_string": "",
            "empty_array": [],
            "empty_object": {},
            "zero": 0,
            "false_value": false,
            "null_value": null
        });

        // Test functions with empty values
        let test_cases = vec![
            ("length", "empty_string", FilterValue::Integer(0)),
            ("length", "empty_array", FilterValue::Integer(0)),
            ("length", "empty_object", FilterValue::Integer(0)),
            ("count", "empty_array", FilterValue::Integer(0)),
            ("count", "empty_object", FilterValue::Integer(0)),
            ("count", "zero", FilterValue::Integer(1)),
            ("count", "false_value", FilterValue::Integer(1)),
            ("count", "null_value", FilterValue::Integer(0)),
        ];

        for (func_name, prop_name, expected) in test_cases {
            let result = FunctionEvaluator::evaluate_function_value(
                &context,
                func_name,
                &[FilterExpression::Property {
                    path: vec![prop_name.to_string()],
                }],
                &mock_evaluator,
            );
            assert_eq!(
                result.unwrap(),
                expected,
                "Failed for {}({}) - expected {:?}",
                func_name,
                prop_name,
                expected
            );
        }
    }
}
