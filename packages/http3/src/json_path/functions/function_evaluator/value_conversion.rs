//! Value conversion functions
//!
//! This module implements the value() function from RFC 9535 for converting
//! single-node nodelists to values with proper validation.

use super::super::jsonpath_nodelist::JsonPathNodelistEvaluator;
use super::core::FunctionEvaluator;
use crate::json_path::error::{JsonPathResult, invalid_expression_error};
use crate::json_path::parser::{FilterExpression, FilterValue};

/// RFC 9535 Section 2.4.8: value() function
/// Converts single-node nodelist to value (errors on multi-node or empty)
#[inline]
pub fn evaluate_value_function(
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
            "value() function requires exactly one argument",
            None,
        ));
    }

    match &args[0] {
        FilterExpression::JsonPath { selectors } => {
            // Evaluate JSONPath expression and validate nodelist size
            let nodelist =
                JsonPathNodelistEvaluator::evaluate_jsonpath_nodelist(context, selectors)?;

            if nodelist.is_empty() {
                return Err(invalid_expression_error(
                    "",
                    "value() function requires non-empty nodelist",
                    None,
                ));
            }

            if nodelist.len() > 1 {
                return Err(invalid_expression_error(
                    "",
                    "value() function requires single-node nodelist",
                    None,
                ));
            }

            // Safe to unwrap since we verified length == 1
            Ok(FunctionEvaluator::json_value_to_filter_value(&nodelist[0]))
        }
        FilterExpression::Property { path } => {
            // Property access produces exactly one node or null
            let mut current = context;
            for segment in path {
                match current {
                    serde_json::Value::Object(obj) => {
                        current = obj.get(segment).map_or(&serde_json::Value::Null, |v| v);
                    }
                    _ => return Ok(FilterValue::Null),
                }
            }
            Ok(FunctionEvaluator::json_value_to_filter_value(current))
        }
        FilterExpression::Current => {
            // Current context produces exactly one node
            Ok(FunctionEvaluator::json_value_to_filter_value(context))
        }
        FilterExpression::Literal { value } => {
            // Literal produces exactly one value
            Ok(value.clone())
        }
        _ => {
            // For other expressions, evaluate directly (they produce single values)
            expression_evaluator(context, &args[0])
        }
    }
}