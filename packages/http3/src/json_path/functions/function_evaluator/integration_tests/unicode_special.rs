//! Unicode and special character integration tests
//!
//! Tests that verify proper handling of Unicode characters, emojis, and special characters

use serde_json::json;
use super::super::FunctionEvaluator;
use crate::json_path::parser::{FilterExpression, FilterValue};
use super::mock_evaluator::mock_evaluator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_unicode_and_special_characters() {
        let context = json!({
            "emoji": "🌍🌎🌏",
            "chinese": "你好世界",
            "mixed": "Hello 世界 🌍",
            "special": "line1\nline2\ttab"
        });

        // Test length with Unicode characters
        let result = FunctionEvaluator::evaluate_function_value(
            &context,
            "length",
            &[FilterExpression::Property {
                path: vec!["emoji".to_string()],
            }],
            &mock_evaluator,
        );
        assert_eq!(result.unwrap(), FilterValue::Integer(3)); // 3 emoji characters

        let result = FunctionEvaluator::evaluate_function_value(
            &context,
            "length",
            &[FilterExpression::Property {
                path: vec!["chinese".to_string()],
            }],
            &mock_evaluator,
        );
        assert_eq!(result.unwrap(), FilterValue::Integer(4)); // 4 Chinese characters

        // Test regex with Unicode
        let result = FunctionEvaluator::evaluate_function_value(
            &context,
            "search",
            &[
                FilterExpression::Property {
                    path: vec!["mixed".to_string()],
                },
                FilterExpression::Literal {
                    value: FilterValue::String("世界".to_string()),
                },
            ],
            &mock_evaluator,
        );
        assert_eq!(result.unwrap(), FilterValue::Boolean(true));
    }
}