//! Advanced descendant operations
//!
//! Depth-specific operations, filtering, and path tracking for descendant processing.

use serde_json::Value;

use super::core::DescendantOperations;

impl DescendantOperations {
    /// Collect descendants at a specific depth level
    pub fn collect_descendants_at_depth(
        node: &Value,
        target_depth: usize,
        current_depth: usize,
        results: &mut Vec<Value>,
    ) {
        if current_depth == target_depth {
            results.push(node.clone());
            return;
        }

        if current_depth < target_depth {
            match node {
                Value::Object(obj) => {
                    for value in obj.values() {
                        Self::collect_descendants_at_depth(value, target_depth, current_depth + 1, results);
                    }
                }
                Value::Array(arr) => {
                    for value in arr {
                        Self::collect_descendants_at_depth(value, target_depth, current_depth + 1, results);
                    }
                }
                _ => {}
            }
        }
    }

    /// Collect descendants with path information
    pub fn collect_descendants_with_paths(
        node: &Value,
        current_path: String,
        results: &mut Vec<(String, Value)>,
    ) {
        match node {
            Value::Object(obj) => {
                for (key, value) in obj {
                    let new_path = if current_path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", current_path, key)
                    };
                    results.push((new_path.clone(), value.clone()));
                    Self::collect_descendants_with_paths(value, new_path, results);
                }
            }
            Value::Array(arr) => {
                for (index, value) in arr.iter().enumerate() {
                    let new_path = if current_path.is_empty() {
                        format!("[{}]", index)
                    } else {
                        format!("{}[{}]", current_path, index)
                    };
                    results.push((new_path.clone(), value.clone()));
                    Self::collect_descendants_with_paths(value, new_path, results);
                }
            }
            _ => {}
        }
    }

    /// Filter descendants by predicate
    pub fn filter_descendants<F>(node: &Value, predicate: F, results: &mut Vec<Value>)
    where
        F: Fn(&Value) -> bool + Copy,
    {
        match node {
            Value::Object(obj) => {
                for value in obj.values() {
                    if predicate(value) {
                        results.push(value.clone());
                    }
                    Self::filter_descendants(value, predicate, results);
                }
            }
            Value::Array(arr) => {
                for value in arr {
                    if predicate(value) {
                        results.push(value.clone());
                    }
                    Self::filter_descendants(value, predicate, results);
                }
            }
            _ => {}
        }
    }

    /// Collect leaf values (values with no descendants)
    pub fn collect_leaf_values(node: &Value, results: &mut Vec<Value>) {
        match node {
            Value::Object(obj) => {
                if obj.is_empty() {
                    results.push(node.clone());
                } else {
                    for value in obj.values() {
                        Self::collect_leaf_values(value, results);
                    }
                }
            }
            Value::Array(arr) => {
                if arr.is_empty() {
                    results.push(node.clone());
                } else {
                    for value in arr {
                        Self::collect_leaf_values(value, results);
                    }
                }
            }
            _ => {
                // Primitive values are always leaves
                results.push(node.clone());
            }
        }
    }
}