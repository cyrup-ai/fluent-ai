//! RFC 9535 Section 2.4.4: length() function implementation
//!
//! Returns number of characters in string, elements in array, or members in object

use crate::jsonpath::error::{JsonPathResult, invalid_expression_error};
use crate::jsonpath::parser::{FilterExpression, FilterValue};

/// RFC 9535 Section 2.4.4: length() function
/// Returns number of characters in string, elements in array, or members in object
#[inline]
pub fn evaluate_length_function(
    context: &serde_json::Value,
    args: &[FilterExpression],
    expression_evaluator: &dyn Fn(
        &serde_json::Value,
        &FilterExpression,
    ) -> JsonPathResult<FilterValue>,
) -> JsonPathResult<FilterValue> {
    if args.len() != 1 {
        return Err(invalid_expression_error(
            "",
            "length() function requires exactly one argument",
            None,
        ));
    }

    match &args[0] {
        FilterExpression::Property { path } => {
            let mut current = context;
            for segment in path {
                match current {
                    serde_json::Value::Object(obj) => {
                        current = obj.get(segment).map_or(&serde_json::Value::Null, |v| v);
                    }
                    _ => return Ok(FilterValue::Null),
                }
            }

            let len = match current {
                serde_json::Value::Array(arr) => arr.len() as i64,
                serde_json::Value::Object(obj) => obj.len() as i64,
                serde_json::Value::String(s) => s.chars().count() as i64, // Unicode-aware
                serde_json::Value::Null => return Ok(FilterValue::Null),
                _ => return Ok(FilterValue::Null), // Primitives return null per RFC
            };
            Ok(FilterValue::Integer(len))
        }
        _ => {
            let value = expression_evaluator(context, &args[0])?;
            match value {
                FilterValue::String(s) => Ok(FilterValue::Integer(s.chars().count() as i64)),
                FilterValue::Integer(_) | FilterValue::Number(_) | FilterValue::Boolean(_) => {
                    Ok(FilterValue::Null) // Primitives return null per RFC
                }
                FilterValue::Null => Ok(FilterValue::Null),
                FilterValue::Missing => Ok(FilterValue::Null), /* Missing properties have no length */
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn mock_evaluator(
        _context: &serde_json::Value,
        expr: &FilterExpression,
    ) -> JsonPathResult<FilterValue> {
        match expr {
            FilterExpression::Literal { value } => Ok(value.clone()),
            _ => Ok(FilterValue::Null),
        }
    }

    #[test]
    fn test_length_function_wrong_arg_count() {
        let context = json!({});
        let args = vec![];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
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
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exactly one argument")
        );
    }

    #[test]
    fn test_length_function_property_array() {
        let context = json!({"items": [1, 2, 3, 4, 5]});
        let args = vec![FilterExpression::Property {
            path: vec!["items".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(5));
    }

    #[test]
    fn test_length_function_property_object() {
        let context = json!({"user": {"name": "John", "age": 30, "city": "NYC"}});
        let args = vec![FilterExpression::Property {
            path: vec!["user".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(3));
    }

    #[test]
    fn test_length_function_property_string() {
        let context = json!({"message": "Hello World"});
        let args = vec![FilterExpression::Property {
            path: vec!["message".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(11));
    }

    #[test]
    fn test_length_function_property_unicode_string() {
        let context = json!({"text": "Hello 世界 🌍"});
        let args = vec![FilterExpression::Property {
            path: vec!["text".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(10)); // Unicode-aware counting
    }

    #[test]
    fn test_length_function_property_null() {
        let context = json!({"value": null});
        let args = vec![FilterExpression::Property {
            path: vec!["value".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null);
    }

    #[test]
    fn test_length_function_property_primitive() {
        let context = json!({"number": 42});
        let args = vec![FilterExpression::Property {
            path: vec!["number".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null); // Primitives return null per RFC

        let context = json!({"flag": true});
        let args = vec![FilterExpression::Property {
            path: vec!["flag".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null);
    }

    #[test]
    fn test_length_function_property_missing() {
        let context = json!({"other": "value"});
        let args = vec![FilterExpression::Property {
            path: vec!["missing".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null);
    }

    #[test]
    fn test_length_function_property_nested() {
        let context = json!({"data": {"items": [1, 2, 3]}});
        let args = vec![FilterExpression::Property {
            path: vec!["data".to_string(), "items".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(3));
    }

    #[test]
    fn test_length_function_property_nested_missing() {
        let context = json!({"data": "not an object"});
        let args = vec![FilterExpression::Property {
            path: vec!["data".to_string(), "items".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null);
    }

    #[test]
    fn test_length_function_literal_string() {
        let context = json!({});
        let args = vec![FilterExpression::Literal {
            value: FilterValue::String("test string".to_string()),
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(11));
    }

    #[test]
    fn test_length_function_literal_primitives() {
        let context = json!({});

        let args = vec![FilterExpression::Literal {
            value: FilterValue::Integer(42),
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null);

        let args = vec![FilterExpression::Literal {
            value: FilterValue::Number(3.14),
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null);

        let args = vec![FilterExpression::Literal {
            value: FilterValue::Boolean(true),
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null);
    }

    #[test]
    fn test_length_function_literal_null_and_missing() {
        let context = json!({});

        let args = vec![FilterExpression::Literal {
            value: FilterValue::Null,
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null);

        let args = vec![FilterExpression::Literal {
            value: FilterValue::Missing,
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Null);
    }

    #[test]
    fn test_length_function_empty_collections() {
        let context = json!({"empty_array": [], "empty_object": {}, "empty_string": ""});

        let args = vec![FilterExpression::Property {
            path: vec!["empty_array".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(0));

        let args = vec![FilterExpression::Property {
            path: vec!["empty_object".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(0));

        let args = vec![FilterExpression::Property {
            path: vec!["empty_string".to_string()],
        }];
        let result = evaluate_length_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(0));
    }
}
