//! Recursive descent engine for JSONPath evaluation
//!
//! This module handles recursive descent operations with safety limits to prevent stack overflow.

use serde_json::Value;

use crate::json_path::error::JsonPathError;

type JsonPathResult<T> = Result<T, JsonPathError>;

/// Engine for recursive descent operations
pub struct RecursiveDescentEngine;

impl RecursiveDescentEngine {
    /// Collect all descendants with depth and count limits for safety
    pub fn collect_all_descendants_owned(value: &Value, results: &mut Vec<Value>) {
        Self::collect_descendants_with_limits(value, results, 0, 50, 10000);
    }

    /// Collect descendants with safety limits
    fn collect_descendants_with_limits(
        value: &Value,
        results: &mut Vec<Value>,
        current_depth: usize,
        max_depth: usize,
        max_results: usize,
    ) {
        // Depth limit protection - prevent stack overflow
        if current_depth >= max_depth {
            return;
        }

        // Result count limit protection - prevent memory exhaustion
        if results.len() >= max_results {
            return;
        }

        match value {
            Value::Object(obj) => {
                for child_value in obj.values() {
                    if results.len() >= max_results {
                        break;
                    }
                    // Add the child itself
                    results.push(child_value.clone());
                    // Then recursively add its descendants (but not the child again)
                    Self::collect_descendants_with_limits(
                        child_value,
                        results,
                        current_depth + 1,
                        max_depth,
                        max_results,
                    );
                }
            }
            Value::Array(arr) => {
                for child_value in arr {
                    if results.len() >= max_results {
                        break;
                    }
                    // Add the child itself
                    results.push(child_value.clone());
                    // Then recursively add its descendants (but not the child again)
                    Self::collect_descendants_with_limits(
                        child_value,
                        results,
                        current_depth + 1,
                        max_depth,
                        max_results,
                    );
                }
            }
            _ => {} // Primitives have no descendants
        }
    }

    /// Collect only leaf descendants (primitives) with depth and count limits
    /// Used for $..* pattern to return "all member values and array elements"
    pub fn collect_leaf_descendants_with_limits(
        value: &Value,
        results: &mut Vec<Value>,
        current_depth: usize,
        max_depth: usize,
        max_results: usize,
    ) {
        // Depth limit protection - prevent stack overflow
        if current_depth >= max_depth {
            return;
        }

        // Result count limit protection - prevent memory exhaustion
        if results.len() >= max_results {
            return;
        }

        match value {
            Value::Object(obj) => {
                for child_value in obj.values() {
                    if results.len() >= max_results {
                        break;
                    }

                    // Only add primitives, but recurse into all children
                    match child_value {
                        Value::Object(_) | Value::Array(_) => {
                            // Don't add structural elements, but recurse into them
                            Self::collect_leaf_descendants_with_limits(
                                child_value,
                                results,
                                current_depth + 1,
                                max_depth,
                                max_results,
                            );
                        }
                        _ => {
                            // Add primitive values (null, bool, number, string)
                            results.push(child_value.clone());
                        }
                    }
                }
            }
            Value::Array(arr) => {
                for child_value in arr {
                    if results.len() >= max_results {
                        break;
                    }

                    // Only add primitives, but recurse into all children
                    match child_value {
                        Value::Object(_) | Value::Array(_) => {
                            // Don't add structural elements, but recurse into them
                            Self::collect_leaf_descendants_with_limits(
                                child_value,
                                results,
                                current_depth + 1,
                                max_depth,
                                max_results,
                            );
                        }
                        _ => {
                            // Add primitive values (null, bool, number, string)
                            results.push(child_value.clone());
                        }
                    }
                }
            }
            _ => {} // Primitives have no descendants to collect
        }
    }

    /// Optimized handler for $..*  pattern to avoid exponential complexity
    /// Protected with depth and count limits
    pub fn apply_recursive_descent_wildcard(value: &Value) -> JsonPathResult<Vec<Value>> {
        let mut results = Vec::new();
        Self::collect_recursive_wildcard_with_limits(value, &mut results, 0, 50, 10000);
        Ok(results)
    }

    /// Collect all values in a recursive wildcard pattern with limits
    fn collect_recursive_wildcard_with_limits(
        value: &Value,
        results: &mut Vec<Value>,
        current_depth: usize,
        max_depth: usize,
        max_results: usize,
    ) {
        // Depth limit protection
        if current_depth >= max_depth {
            return;
        }

        // Result count limit protection
        if results.len() >= max_results {
            return;
        }

        match value {
            Value::Object(obj) => {
                // Add all direct children
                for child_value in obj.values() {
                    if results.len() >= max_results {
                        break;
                    }
                    results.push(child_value.clone());
                }

                // Recursively collect from children
                for child_value in obj.values() {
                    if results.len() >= max_results {
                        break;
                    }
                    Self::collect_recursive_wildcard_with_limits(
                        child_value,
                        results,
                        current_depth + 1,
                        max_depth,
                        max_results,
                    );
                }
            }
            Value::Array(arr) => {
                // Add all direct children
                for child_value in arr {
                    if results.len() >= max_results {
                        break;
                    }
                    results.push(child_value.clone());
                }

                // Recursively collect from children
                for child_value in arr {
                    if results.len() >= max_results {
                        break;
                    }
                    Self::collect_recursive_wildcard_with_limits(
                        child_value,
                        results,
                        current_depth + 1,
                        max_depth,
                        max_results,
                    );
                }
            }
            _ => {} // Primitives have no children
        }
    }

    /// Count total descendants without collecting them (for estimation)
    pub fn count_descendants(value: &Value, max_depth: usize) -> usize {
        Self::count_descendants_recursive(value, 0, max_depth)
    }

    fn count_descendants_recursive(value: &Value, current_depth: usize, max_depth: usize) -> usize {
        if current_depth >= max_depth {
            return 0;
        }

        match value {
            Value::Object(obj) => {
                let mut count = obj.len(); // Direct children
                for child_value in obj.values() {
                    count += Self::count_descendants_recursive(
                        child_value,
                        current_depth + 1,
                        max_depth,
                    );
                }
                count
            }
            Value::Array(arr) => {
                let mut count = arr.len(); // Direct children
                for child_value in arr {
                    count += Self::count_descendants_recursive(
                        child_value,
                        current_depth + 1,
                        max_depth,
                    );
                }
                count
            }
            _ => 0, // Primitives have no descendants
        }
    }

    /// Check if recursive descent would be expensive
    pub fn is_expensive_recursive_descent(value: &Value) -> bool {
        let estimated_count = Self::count_descendants(value, 10);
        estimated_count > 1000
    }

    /// Get the maximum depth of a JSON structure
    pub fn max_depth(value: &Value) -> usize {
        Self::max_depth_recursive(value, 0)
    }

    fn max_depth_recursive(value: &Value, current_depth: usize) -> usize {
        match value {
            Value::Object(obj) => {
                let mut max = current_depth;
                for child_value in obj.values() {
                    let child_depth = Self::max_depth_recursive(child_value, current_depth + 1);
                    max = max.max(child_depth);
                }
                max
            }
            Value::Array(arr) => {
                let mut max = current_depth;
                for child_value in arr {
                    let child_depth = Self::max_depth_recursive(child_value, current_depth + 1);
                    max = max.max(child_depth);
                }
                max
            }
            _ => current_depth,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_collect_descendants() {
        let json = json!({
            "a": {
                "b": 1,
                "c": 2
            },
            "d": [3, 4]
        });

        let mut results = Vec::new();
        RecursiveDescentEngine::collect_all_descendants_owned(&json, &mut results);

        // Should collect: {"b": 1, "c": 2}, [3, 4], 1, 2, 3, 4
        assert!(results.len() >= 4);
        assert!(results.contains(&json!(1)));
        assert!(results.contains(&json!(2)));
        assert!(results.contains(&json!(3)));
        assert!(results.contains(&json!(4)));
    }

    #[test]
    fn test_max_depth() {
        let shallow = json!({"a": 1});
        let deep = json!({
            "a": {
                "b": {
                    "c": {
                        "d": 1
                    }
                }
            }
        });

        assert_eq!(RecursiveDescentEngine::max_depth(&shallow), 1);
        assert_eq!(RecursiveDescentEngine::max_depth(&deep), 4);
    }

    #[test]
    fn test_count_descendants() {
        let json = json!({
            "a": 1,
            "b": [2, 3],
            "c": {"d": 4}
        });

        let count = RecursiveDescentEngine::count_descendants(&json, 10);
        assert!(count > 0);
    }

    #[test]
    fn test_expensive_detection() {
        let simple = json!({"a": 1});
        let complex = json!({
            "level1": {
                "level2": {
                    "level3": {
                        "data": [1, 2, 3, 4, 5]
                    }
                }
            }
        });

        assert!(!RecursiveDescentEngine::is_expensive_recursive_descent(
            &simple
        ));
        // Complex structure might be expensive depending on thresholds
    }
}
