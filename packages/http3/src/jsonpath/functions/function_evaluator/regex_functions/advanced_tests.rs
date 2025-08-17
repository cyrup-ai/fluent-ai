//! Advanced tests for regex functions
//!
//! Edge cases, unicode support, type handling, and caching tests

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::core::{evaluate_match_function, evaluate_search_function};
    use super::super::test_utils::mock_evaluator;
    use crate::jsonpath::parser::{FilterExpression, FilterValue};

    #[test]
    fn test_match_function_non_string_args() {
        let context = json!({});
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::Integer(42),
            },
            FilterExpression::Literal {
                value: FilterValue::String("\\d+".to_string()),
            },
        ];
        let result = evaluate_match_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(false));

        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("test".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::Integer(123),
            },
        ];
        let result = evaluate_match_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(false));
    }

    #[test]
    fn test_search_function_non_string_args() {
        let context = json!({});
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::Boolean(true),
            },
            FilterExpression::Literal {
                value: FilterValue::String("true".to_string()),
            },
        ];
        let result = evaluate_search_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(false));

        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("test".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::Null,
            },
        ];
        let result = evaluate_search_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(false));
    }

    #[test]
    fn test_match_function_unicode_strings() {
        let context = json!({});
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("Hello 世界".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String("世界".to_string()),
            },
        ];
        let result = evaluate_match_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(true));
    }

    #[test]
    fn test_search_function_unicode_strings() {
        let context = json!({});
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("Hello 世界 🌍".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String("🌍".to_string()),
            },
        ];
        let result = evaluate_search_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(true));
    }

    #[test]
    fn test_regex_cache_usage() {
        let context = json!({});
        let pattern = "test\\d+".to_string();

        // First call should compile and cache the regex
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("test123".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String(pattern.clone()),
            },
        ];
        let result1 = evaluate_match_function(&context, &args, &mock_evaluator);
        assert_eq!(result1.unwrap(), FilterValue::Boolean(true));

        // Second call should use cached regex
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("test456".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String(pattern),
            },
        ];
        let result2 = evaluate_match_function(&context, &args, &mock_evaluator);
        assert_eq!(result2.unwrap(), FilterValue::Boolean(true));
    }
}
