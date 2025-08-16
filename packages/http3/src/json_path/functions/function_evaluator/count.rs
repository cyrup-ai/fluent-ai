//! RFC 9535 Section 2.4.5: count() function implementation
//!
//! Returns number of nodes in nodelist produced by argument expression

use crate::json_path::error::{JsonPathResult, invalid_expression_error};
use crate::json_path::parser::{FilterExpression, FilterValue};

/// RFC 9535 Section 2.4.5: count() function  
/// Returns number of nodes in nodelist produced by argument expression
#[inline]
pub fn evaluate_count_function(
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
            "count() function requires exactly one argument",
            None,
        ));
    }

    let count = match &args[0] {
        FilterExpression::Property { path } => {
            let mut current = context;
            for segment in path {
                match current {
                    serde_json::Value::Object(obj) => {
                        current = obj.get(segment).map_or(&serde_json::Value::Null, |v| v);
                    }
                    _ => return Ok(FilterValue::Integer(0)),
                }
            }

            match current {
                serde_json::Value::Array(arr) => arr.len() as i64,
                serde_json::Value::Object(obj) => obj.len() as i64,
                serde_json::Value::Null => 0,
                _ => 1, // Single value counts as 1
            }
        }
        _ => match expression_evaluator(context, &args[0])? {
            FilterValue::Null => 0,
            _ => 1,
        },
    };
    Ok(FilterValue::Integer(count))
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
    fn test_count_function_wrong_arg_count() {
        let context = json!({});
        let args = vec![];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
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
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exactly one argument")
        );
    }

    #[test]
    fn test_count_function_property_array() {
        let context = json!({"items": [1, 2, 3, 4, 5]});
        let args = vec![FilterExpression::Property {
            path: vec!["items".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(5));
    }

    #[test]
    fn test_count_function_property_object() {
        let context = json!({"user": {"name": "John", "age": 30, "city": "NYC"}});
        let args = vec![FilterExpression::Property {
            path: vec!["user".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(3));
    }

    #[test]
    fn test_count_function_property_null() {
        let context = json!({"value": null});
        let args = vec![FilterExpression::Property {
            path: vec!["value".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(0));
    }

    #[test]
    fn test_count_function_property_primitive() {
        let context = json!({"number": 42});
        let args = vec![FilterExpression::Property {
            path: vec!["number".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(1)); // Single value counts as 1

        let context = json!({"flag": true});
        let args = vec![FilterExpression::Property {
            path: vec!["flag".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(1));

        let context = json!({"text": "hello"});
        let args = vec![FilterExpression::Property {
            path: vec!["text".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(1));
    }

    #[test]
    fn test_count_function_property_missing() {
        let context = json!({"other": "value"});
        let args = vec![FilterExpression::Property {
            path: vec!["missing".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(0));
    }

    #[test]
    fn test_count_function_property_nested() {
        let context = json!({"data": {"items": [1, 2, 3]}});
        let args = vec![FilterExpression::Property {
            path: vec!["data".to_string(), "items".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(3));
    }

    #[test]
    fn test_count_function_property_nested_missing() {
        let context = json!({"data": "not an object"});
        let args = vec![FilterExpression::Property {
            path: vec!["data".to_string(), "items".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(0));
    }

    #[test]
    fn test_count_function_literal_values() {
        let context = json!({});

        let args = vec![FilterExpression::Literal {
            value: FilterValue::String("test".to_string()),
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(1));

        let args = vec![FilterExpression::Literal {
            value: FilterValue::Integer(42),
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(1));

        let args = vec![FilterExpression::Literal {
            value: FilterValue::Boolean(true),
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(1));
    }

    #[test]
    fn test_count_function_literal_null() {
        let context = json!({});
        let args = vec![FilterExpression::Literal {
            value: FilterValue::Null,
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(0));
    }

    #[test]
    fn test_count_function_empty_collections() {
        let context = json!({"empty_array": [], "empty_object": {}});

        let args = vec![FilterExpression::Property {
            path: vec!["empty_array".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(0));

        let args = vec![FilterExpression::Property {
            path: vec!["empty_object".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(0));
    }

    #[test]
    fn test_count_function_large_collections() {
        let large_array: Vec<i32> = (0..1000).collect();
        let context = json!({"large": large_array});
        let args = vec![FilterExpression::Property {
            path: vec!["large".to_string()],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(1000));
    }

    #[test]
    fn test_count_function_nested_structures() {
        let context = json!({
            "level1": {
                "level2": {
                    "items": [1, 2, 3, 4]
                }
            }
        });
        let args = vec![FilterExpression::Property {
            path: vec![
                "level1".to_string(),
                "level2".to_string(),
                "items".to_string(),
            ],
        }];
        let result = evaluate_count_function(&context, &args, &mock_evaluator);
        assert_eq!(result.unwrap(), FilterValue::Integer(4));
    }
}
