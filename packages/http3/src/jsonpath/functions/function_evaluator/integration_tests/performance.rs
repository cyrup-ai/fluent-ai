//! Performance integration tests
//!
//! Tests that verify function performance with large data sets and stress conditions

use serde_json::json;

use super::super::FunctionEvaluator;
use super::mock_evaluator::mock_evaluator;
use crate::jsonpath::parser::{FilterExpression, FilterValue};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_performance_large_data() {
        // Create a large array for performance testing
        let large_array: Vec<i32> = (0..10000).collect();
        let context = json!({"large_data": large_array});

        // Test length function performance
        let result = FunctionEvaluator::evaluate_function_value(
            &context,
            "length",
            &[FilterExpression::Property {
                path: vec!["large_data".to_string()],
            }],
            &mock_evaluator,
        );
        assert_eq!(result.unwrap(), FilterValue::Integer(10000));

        // Test count function performance
        let result = FunctionEvaluator::evaluate_function_value(
            &context,
            "count",
            &[FilterExpression::Property {
                path: vec!["large_data".to_string()],
            }],
            &mock_evaluator,
        );
        assert_eq!(result.unwrap(), FilterValue::Integer(10000));
    }
}
