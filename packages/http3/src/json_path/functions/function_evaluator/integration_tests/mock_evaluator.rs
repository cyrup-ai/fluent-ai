//! Mock evaluator for integration tests
//!
//! Provides a simple mock implementation for testing function evaluator behavior

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