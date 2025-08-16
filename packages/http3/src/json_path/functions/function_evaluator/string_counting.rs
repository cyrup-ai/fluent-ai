//! String and counting functions
//!
//! This module implements the length() and count() functions from RFC 9535
//! for measuring string lengths and counting nodes in nodelists.

use crate::json_path::error::{JsonPathResult, invalid_expression_error};
use crate::json_path::parser::{FilterExpression, FilterValue};

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