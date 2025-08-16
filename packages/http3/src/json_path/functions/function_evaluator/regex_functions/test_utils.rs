//! Test utilities for regex function testing
//!
//! Common mock evaluator and test setup utilities

#[cfg(test)]
use crate::json_path::error::JsonPathResult;
#[cfg(test)]
use crate::json_path::parser::{FilterExpression, FilterValue};

#[cfg(test)]
pub fn mock_evaluator(
    _context: &serde_json::Value,
    expr: &FilterExpression,
) -> JsonPathResult<FilterValue> {
    match expr {
        FilterExpression::Literal { value } => Ok(value.clone()),
        _ => Ok(FilterValue::Null),
    }
}
