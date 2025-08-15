//! Array operations for JSONPath evaluation
//!
//! This module handles array indexing, slicing, and related operations.

use serde_json::Value;

use crate::json_path::error::JsonPathError;

type JsonPathResult<T> = Result<T, JsonPathError>;

/// Array operations engine for JSONPath evaluation
pub struct ArrayOperations;

impl ArrayOperations {
    /// Apply index selector for array access
    pub fn apply_index(arr: &[Value], index: i64, from_end: bool) -> JsonPathResult<Vec<Value>> {
        let mut results = Vec::new();

        let actual_index = if from_end && index < 0 {
            // Negative index - count from end (e.g., -1 means last element)
            let abs_index = (-index) as usize;
            if abs_index <= arr.len() && abs_index > 0 {
                arr.len() - abs_index
            } else {
                return Ok(results); // Index out of bounds
            }
        } else if from_end && index > 0 {
            // Positive from_end index
            if (index as usize) <= arr.len() {
                arr.len() - (index as usize)
            } else {
                return Ok(results); // Index out of bounds
            }
        } else {
            // Regular positive index
            index as usize
        };

        if actual_index < arr.len() {
            results.push(arr[actual_index].clone());
        }

        Ok(results)
    }

    /// Apply slice selector for array slicing
    pub fn apply_slice(
        arr: &[Value],
        start: Option<i64>,
        end: Option<i64>,
        step: i64,
    ) -> JsonPathResult<Vec<Value>> {
        if step == 0 {
            return Err(JsonPathError::InvalidIndex(
                "Step cannot be zero".to_string(),
            ));
        }

        let len = arr.len() as i64;
        let mut results = Vec::new();

        // Normalize start and end indices
        let start_idx = Self::normalize_index(start, len, step > 0);
        let end_idx = Self::normalize_index(end, len, step > 0);

        if step > 0 {
            // Forward iteration
            let mut i = start_idx;
            while i < end_idx && i < len {
                if i >= 0 {
                    results.push(arr[i as usize].clone());
                }
                i += step;
            }
        } else {
            // Backward iteration
            let mut i = start_idx;
            while i > end_idx && i >= 0 {
                if i < len {
                    results.push(arr[i as usize].clone());
                }
                i += step; // step is negative
            }
        }

        Ok(results)
    }

    /// Normalize slice index according to Python-like semantics
    fn normalize_index(index: Option<i64>, len: i64, forward: bool) -> i64 {
        match index {
            Some(idx) => {
                if idx < 0 {
                    std::cmp::max(0, len + idx)
                } else {
                    std::cmp::min(idx, len)
                }
            }
            None => {
                if forward {
                    0 // Start from beginning for forward iteration
                } else {
                    len - 1 // Start from end for backward iteration
                }
            }
        }
    }

    /// Check if an array index is valid
    pub fn is_valid_index(arr: &[Value], index: i64, from_end: bool) -> bool {
        if from_end && index < 0 {
            let abs_index = (-index) as usize;
            abs_index <= arr.len() && abs_index > 0
        } else if from_end && index > 0 {
            (index as usize) <= arr.len()
        } else {
            index >= 0 && (index as usize) < arr.len()
        }
    }

    /// Get array length safely
    pub fn safe_len(value: &Value) -> Option<usize> {
        match value {
            Value::Array(arr) => Some(arr.len()),
            _ => None,
        }
    }

    /// Check if a value is an array
    pub fn is_array(value: &Value) -> bool {
        matches!(value, Value::Array(_))
    }

    /// Get array element safely
    pub fn get_element(arr: &[Value], index: usize) -> Option<&Value> {
        arr.get(index)
    }

    /// Apply multiple indices to an array
    pub fn apply_multiple_indices(arr: &[Value], indices: &[i64]) -> JsonPathResult<Vec<Value>> {
        let mut results = Vec::new();

        for &index in indices {
            if index >= 0 && (index as usize) < arr.len() {
                results.push(arr[index as usize].clone());
            }
        }

        Ok(results)
    }

    /// Apply range operation (start:end)
    pub fn apply_range(arr: &[Value], start: i64, end: i64) -> JsonPathResult<Vec<Value>> {
        Self::apply_slice(arr, Some(start), Some(end), 1)
    }

    /// Get last N elements from array
    pub fn get_last_n(arr: &[Value], n: usize) -> Vec<Value> {
        if n >= arr.len() {
            arr.to_vec()
        } else {
            arr[arr.len() - n..].to_vec()
        }
    }

    /// Get first N elements from array
    pub fn get_first_n(arr: &[Value], n: usize) -> Vec<Value> {
        if n >= arr.len() {
            arr.to_vec()
        } else {
            arr[..n].to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_positive_index() {
        let arr = vec![json!(1), json!(2), json!(3)];
        let results = ArrayOperations::apply_index(&arr, 1, false).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json!(2));
    }

    #[test]
    fn test_negative_index() {
        let arr = vec![json!(1), json!(2), json!(3)];
        let results = ArrayOperations::apply_index(&arr, -1, true).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json!(3));
    }

    #[test]
    fn test_slice_operation() {
        let arr = vec![json!(1), json!(2), json!(3), json!(4), json!(5)];
        let results = ArrayOperations::apply_slice(&arr, Some(1), Some(4), 1).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results, vec![json!(2), json!(3), json!(4)]);
    }

    #[test]
    fn test_slice_with_step() {
        let arr = vec![json!(1), json!(2), json!(3), json!(4), json!(5)];
        let results = ArrayOperations::apply_slice(&arr, Some(0), Some(5), 2).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results, vec![json!(1), json!(3), json!(5)]);
    }

    #[test]
    fn test_reverse_slice() {
        let arr = vec![json!(1), json!(2), json!(3), json!(4), json!(5)];
        let results = ArrayOperations::apply_slice(&arr, Some(4), Some(0), -1).unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(results, vec![json!(5), json!(4), json!(3), json!(2)]);
    }

    #[test]
    fn test_index_validation() {
        let arr = vec![json!(1), json!(2), json!(3)];
        assert!(ArrayOperations::is_valid_index(&arr, 0, false));
        assert!(ArrayOperations::is_valid_index(&arr, 2, false));
        assert!(!ArrayOperations::is_valid_index(&arr, 3, false));
        assert!(ArrayOperations::is_valid_index(&arr, -1, true));
        assert!(!ArrayOperations::is_valid_index(&arr, -4, true));
    }
}
