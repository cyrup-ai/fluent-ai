//! Recursive property operations
//!
//! Recursive property finding and pattern matching operations.

use serde_json::Value;

use super::core::PropertyOperations;

impl PropertyOperations {
    /// Find property recursively in JSON structure
    pub fn find_property_recursive(json: &Value, property: &str) -> Vec<Value> {
        let mut results = Vec::new();
        Self::find_property_recursive_impl(json, property, &mut results);
        results
    }

    /// Internal implementation for recursive property finding
    fn find_property_recursive_impl(json: &Value, property: &str, results: &mut Vec<Value>) {
        match json {
            Value::Object(obj) => {
                // Check if this object has the property
                if let Some(value) = obj.get(property) {
                    results.push(value.clone());
                }
                // Recurse into all values
                for value in obj.values() {
                    Self::find_property_recursive_impl(value, property, results);
                }
            }
            Value::Array(arr) => {
                // Recurse into all array elements
                for value in arr {
                    Self::find_property_recursive_impl(value, property, results);
                }
            }
            _ => {
                // Leaf values - nothing to do
            }
        }
    }

    /// Find all properties matching a pattern
    pub fn find_properties_matching(json: &Value, pattern: &str) -> Vec<(String, Value)> {
        let mut results = Vec::new();
        Self::find_properties_matching_impl(json, pattern, "", &mut results);
        results
    }

    /// Internal implementation for pattern-based property finding
    fn find_properties_matching_impl(
        json: &Value,
        pattern: &str,
        current_path: &str,
        results: &mut Vec<(String, Value)>,
    ) {
        match json {
            Value::Object(obj) => {
                for (key, value) in obj {
                    let new_path = if current_path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", current_path, key)
                    };

                    // Check if key matches pattern (simple wildcard support)
                    if Self::matches_pattern(key, pattern) {
                        results.push((new_path.clone(), value.clone()));
                    }

                    // Recurse into nested structures
                    Self::find_properties_matching_impl(value, pattern, &new_path, results);
                }
            }
            Value::Array(arr) => {
                for (index, value) in arr.iter().enumerate() {
                    let new_path = if current_path.is_empty() {
                        format!("[{}]", index)
                    } else {
                        format!("{}[{}]", current_path, index)
                    };

                    // Recurse into array elements
                    Self::find_properties_matching_impl(value, pattern, &new_path, results);
                }
            }
            _ => {
                // Leaf values - nothing to do
            }
        }
    }

    /// Simple pattern matching with wildcard support
    fn matches_pattern(text: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        if pattern.contains('*') {
            // Simple wildcard matching
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                return text.starts_with(prefix) && text.ends_with(suffix);
            }
        }

        text == pattern
    }

    /// Check if a property exists at any depth
    pub fn has_property_recursive(json: &Value, property: &str) -> bool {
        match json {
            Value::Object(obj) => {
                if obj.contains_key(property) {
                    return true;
                }
                // Check recursively in all values
                for value in obj.values() {
                    if Self::has_property_recursive(value, property) {
                        return true;
                    }
                }
                false
            }
            Value::Array(arr) => {
                // Check recursively in all array elements
                arr.iter()
                    .any(|value| Self::has_property_recursive(value, property))
            }
            _ => false,
        }
    }

    /// Count occurrences of a property at any depth
    pub fn count_property_occurrences(json: &Value, property: &str) -> usize {
        let results = Self::find_property_recursive(json, property);
        results.len()
    }
}
