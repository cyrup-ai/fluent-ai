//! Array-specific selector operations
//!
//! Handles Index and Slice selectors for array access with support for
//! negative indexing, from_end indexing, and slice operations with step.

use serde_json::Value;
use crate::json_path::error::JsonPathError;
use super::super::evaluator::CoreJsonPathEvaluator;

type JsonPathResult<T> = Result<T, JsonPathError>;

/// Apply index selector for owned results
pub fn apply_index_selector_owned(
    _evaluator: &CoreJsonPathEvaluator,
    value: &Value,
    index: i64,
    from_end: bool,
    results: &mut Vec<Value>,
) {
    if let Value::Array(arr) = value {
        let actual_index = if from_end && index < 0 {
            // Negative index - count from end (e.g., -1 means last element)
            let abs_index = (-index) as usize;
            if abs_index <= arr.len() && abs_index > 0 {
                arr.len() - abs_index
            } else {
                return; // Index out of bounds
            }
        } else if from_end && index > 0 {
            // Positive from_end index
            if (index as usize) <= arr.len() {
                arr.len() - (index as usize)
            } else {
                return; // Index out of bounds
            }
        } else {
            // Regular positive index
            index as usize
        };

        if actual_index < arr.len() {
            results.push(arr[actual_index].clone());
        }
    }
}

impl CoreJsonPathEvaluator {
    /// Apply slice to array with start, end, step parameters
    pub fn apply_slice_to_array(
        &self,
        arr: &[Value],
        start: Option<i64>,
        end: Option<i64>,
        step: Option<i64>,
    ) -> JsonPathResult<Vec<Value>> {
        let len = arr.len() as i64;
        let step = step.unwrap_or(1);

        if step == 0 {
            return Ok(vec![]); // Invalid step
        }

        let start = start.unwrap_or(if step > 0 { 0 } else { len - 1 });
        let end = end.unwrap_or(if step > 0 { len } else { -1 });

        // Normalize negative indices
        let start = if start < 0 {
            (len + start).max(0)
        } else {
            start.min(len)
        };
        let end = if end < 0 {
            (len + end).max(-1)
        } else {
            end.min(len)
        };

        let mut results = Vec::new();

        if step > 0 {
            let mut i = start;
            while i < end {
                if i >= 0 && (i as usize) < arr.len() {
                    results.push(arr[i as usize].clone());
                }
                i += step;
            }
        } else {
            let mut i = start;
            while i > end {
                if i >= 0 && (i as usize) < arr.len() {
                    results.push(arr[i as usize].clone());
                }
                i += step;
            }
        }

        Ok(results)
    }

    /// Apply index selector for array access
    pub fn apply_index_selector<'a>(
        &self,
        node: &'a Value,
        index: i64,
        from_end: bool,
        results: &mut Vec<&'a Value>,
    ) {
        if let Value::Array(arr) = node {
            let actual_index = if from_end && index < 0 {
                // Negative index - count from end (e.g., -1 means last element)
                let abs_index = (-index) as usize;
                if abs_index <= arr.len() && abs_index > 0 {
                    arr.len() - abs_index
                } else {
                    return; // Index out of bounds
                }
            } else if from_end && index > 0 {
                // Positive from_end index
                if (index as usize) <= arr.len() {
                    arr.len() - (index as usize)
                } else {
                    return; // Index out of bounds
                }
            } else {
                // Regular positive index
                index as usize
            };

            if actual_index < arr.len() {
                results.push(&arr[actual_index]);
            }
        }
    }

    /// Apply slice selector with start, end, step parameters for arrays
    pub fn apply_slice_selector_with_params<'a>(
        &self,
        node: &'a Value,
        start: Option<i64>,
        end: Option<i64>,
        step: Option<i64>,
        results: &mut Vec<&'a Value>,
    ) {
        if let Value::Array(arr) = node {
            let len = arr.len() as i64;
            let step = step.unwrap_or(1);

            if step == 0 {
                return; // Invalid step
            }

            let start = start.unwrap_or(if step > 0 { 0 } else { len - 1 });
            let end = end.unwrap_or(if step > 0 { len } else { -1 });

            // Normalize negative indices
            let start = if start < 0 {
                (len + start).max(0)
            } else {
                start.min(len)
            };
            let end = if end < 0 {
                (len + end).max(-1)
            } else {
                end.min(len)
            };

            if step > 0 {
                let mut i = start;
                while i < end {
                    if i >= 0 && (i as usize) < arr.len() {
                        results.push(&arr[i as usize]);
                    }
                    i += step;
                }
            } else {
                let mut i = start;
                while i > end {
                    if i >= 0 && (i as usize) < arr.len() {
                        results.push(&arr[i as usize]);
                    }
                    i += step;
                }
            }
        }
    }
}