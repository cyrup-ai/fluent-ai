//! Mock evaluator for value() function tests
//!
//! Provides a simple mock implementation for testing value() function behavior

use serde_json::json;
use crate::json_path::error::JsonPathResult;
use crate::json_path::parser::{FilterExpression, FilterValue};

/// Mock evaluator that returns literals as-is and null for other expressions
pub fn mock_evaluator(
    _context: &serde_json::Value,
    expr: &FilterExpression,
) -> JsonPathResult<FilterValue> {
    match expr {
        FilterExpression::Literal { value } => Ok(value.clone()),
        _ => Ok(FilterValue::Null),
    }
}