//! Basic tests for regex functions
//!
//! Argument validation and simple functionality tests

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::core::{evaluate_match_function, evaluate_search_function};
    use super::super::test_utils::mock_evaluator;
    use crate::jsonpath::parser::{FilterExpression, FilterValue};

    #[test]
    fn test_match_function_wrong_arg_count() {
        let context = json!({});
        let args = vec![];
        let result = evaluate_match_function(&context, &args, &mock_evaluator);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exactly two arguments")
        );

        let args = vec![FilterExpression::Literal {
            value: FilterValue::String("test".to_string()),
        }];
        let result = evaluate_match_function(&context, &args, &mock_evaluator);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exactly two arguments")
        );
    }

    #[test]
    fn test_search_function_wrong_arg_count() {
        let context = json!({});
        let args = vec![];
        let result = evaluate_search_function(&context, &args, &mock_evaluator);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exactly two arguments")
        );

        let args = vec![FilterExpression::Literal {
            value: FilterValue::String("test".to_string()),
        }];
        let result = evaluate_search_function(&context, &args, &mock_evaluator);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exactly two arguments")
        );
    }

    #[test]
    fn test_match_function_valid_pattern() {
        let context = json!({});
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("hello world".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String("^hello".to_string()),
            },
        ];
        let result = evaluate_match_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(true));
    }

    #[test]
    fn test_match_function_no_match() {
        let context = json!({});
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("hello world".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String("^world".to_string()),
            },
        ];
        let result = evaluate_match_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(false));
    }

    #[test]
    fn test_search_function_valid_pattern() {
        let context = json!({});
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("hello world".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String("world".to_string()),
            },
        ];
        let result = evaluate_search_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(true));
    }

    #[test]
    fn test_search_function_no_match() {
        let context = json!({});
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("hello world".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String("xyz".to_string()),
            },
        ];
        let result = evaluate_search_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Boolean(false));
    }

    #[test]
    fn test_match_function_invalid_regex() {
        let context = json!({});
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("test".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String("[".to_string()), // Invalid regex
            },
        ];
        let result = evaluate_match_function(&context, &args, &mock_evaluator);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid regex pattern")
        );
    }

    #[test]
    fn test_search_function_invalid_regex() {
        let context = json!({});
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("test".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String("*".to_string()), // Invalid regex
            },
        ];
        let result = evaluate_search_function(&context, &args, &mock_evaluator);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid regex pattern")
        );
    }

    #[test]
    fn test_match_vs_search_difference() {
        let context = json!({});

        // match() requires full match from start
        let args = vec![
            FilterExpression::Literal {
                value: FilterValue::String("hello world".to_string()),
            },
            FilterExpression::Literal {
                value: FilterValue::String("world".to_string()),
            },
        ];
        let match_result = evaluate_match_function(&context, &args, &mock_evaluator);
        let search_result = evaluate_search_function(&context, &args, &mock_evaluator);

        assert_eq!(match_result.unwrap(), FilterValue::Boolean(false)); // No full match
        assert_eq!(search_result.unwrap(), FilterValue::Boolean(true)); // Contains match
    }
}
