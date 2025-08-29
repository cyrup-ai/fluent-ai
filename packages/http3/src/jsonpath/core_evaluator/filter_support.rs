//! Filter support utilities for JSONPath evaluation
//!
//! This module provides helper functions for filter evaluation and context management.

use std::collections::HashSet;

use serde_json::Value;

/// Support utilities for filter evaluation
pub struct FilterSupport;

impl FilterSupport {
    /// Collect all property names that exist across items in an array
    /// This provides context for filter evaluation
    pub fn collect_existing_properties(arr: &[Value]) -> std::collections::HashSet<String> {
        let mut properties = HashSet::new();

        for item in arr {
            if let Value::Object(obj) = item {
                for key in obj.keys() {
                    properties.insert(key.clone());
                }
            }
        }

        properties
    }

    /// Collect property names from a single object
    pub fn collect_object_properties(obj: &Value) -> HashSet<String> {
        let mut properties = HashSet::new();

        if let Value::Object(map) = obj {
            for key in map.keys() {
                properties.insert(key.clone());
            }
        }

        properties
    }

    /// Check if a property exists in a JSON object
    pub fn has_property(obj: &Value, property: &str) -> bool {
        match obj {
            Value::Object(map) => map.contains_key(property),
            _ => false,
        }
    }

    /// Get property value safely
    pub fn get_property<'a>(obj: &'a Value, property: &str) -> Option<&'a Value> {
        match obj {
            Value::Object(map) => map.get(property),
            _ => None,
        }
    }

    /// Check if an array contains objects with specific properties
    pub fn array_has_objects_with_property(arr: &[Value], property: &str) -> bool {
        arr.iter().any(|item| Self::has_property(item, property))
    }

    /// Count objects in array that have a specific property
    pub fn count_objects_with_property(arr: &[Value], property: &str) -> usize {
        arr.iter()
            .filter(|item| Self::has_property(item, property))
            .count()
    }

    /// Get all unique values for a property across array items
    pub fn collect_property_values(arr: &[Value], property: &str) -> Vec<Value> {
        let mut values = Vec::new();
        let mut seen = HashSet::new();

        for item in arr {
            if let Some(value) = Self::get_property(item, property) {
                let value_str = value.to_string();
                if !seen.contains(&value_str) {
                    seen.insert(value_str);
                    values.push(value.clone());
                }
            }
        }

        values
    }

    /// Check if a value matches a type pattern
    pub fn matches_type(value: &Value, type_name: &str) -> bool {
        match type_name.to_lowercase().as_str() {
            "null" => value.is_null(),
            "boolean" | "bool" => value.is_boolean(),
            "number" => value.is_number(),
            "string" => value.is_string(),
            "array" => value.is_array(),
            "object" => value.is_object(),
            _ => false,
        }
    }

    /// Get the JSON type name for a value
    pub fn get_type_name(value: &Value) -> &'static str {
        match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }

    /// Check if a value is considered "truthy" in filter context
    pub fn is_truthy(value: &Value) -> bool {
        match value {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Number(n) => n.as_f64().map_or(false, |f| f != 0.0),
            Value::String(s) => !s.is_empty(),
            Value::Array(arr) => !arr.is_empty(),
            Value::Object(obj) => !obj.is_empty(),
        }
    }

    /// Compare two values for filter operations
    pub fn compare_values(left: &Value, right: &Value) -> Option<std::cmp::Ordering> {
        match (left, right) {
            (Value::Number(a), Value::Number(b)) => {
                let a_f64 = a.as_f64()?;
                let b_f64 = b.as_f64()?;
                a_f64.partial_cmp(&b_f64)
            }
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
            _ => None, // Cannot compare different types
        }
    }

    /// Check if a value contains another value (for arrays and objects)
    pub fn contains_value(container: &Value, target: &Value) -> bool {
        match container {
            Value::Array(arr) => arr.contains(target),
            Value::Object(obj) => obj.values().any(|v| v == target),
            Value::String(s) => {
                if let Value::String(target_str) = target {
                    s.contains(target_str)
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_collect_existing_properties() {
        let arr = vec![
            json!({"name": "Alice", "age": 30}),
            json!({"name": "Bob", "city": "NYC"}),
            json!({"age": 25, "country": "USA"}),
        ];

        let properties = FilterSupport::collect_existing_properties(&arr);
        assert!(properties.contains("name"));
        assert!(properties.contains("age"));
        assert!(properties.contains("city"));
        assert!(properties.contains("country"));
        assert_eq!(properties.len(), 4);
    }

    #[test]
    fn test_has_property() {
        let obj = json!({"name": "Alice", "age": 30});
        assert!(FilterSupport::has_property(&obj, "name"));
        assert!(FilterSupport::has_property(&obj, "age"));
        assert!(!FilterSupport::has_property(&obj, "city"));
    }

    #[test]
    fn test_get_property() {
        let obj = json!({"name": "Alice", "age": 30});
        assert_eq!(
            FilterSupport::get_property(&obj, "name"),
            Some(&json!("Alice"))
        );
        assert_eq!(FilterSupport::get_property(&obj, "age"), Some(&json!(30)));
        assert_eq!(FilterSupport::get_property(&obj, "city"), None);
    }

    #[test]
    fn test_matches_type() {
        assert!(FilterSupport::matches_type(&json!(null), "null"));
        assert!(FilterSupport::matches_type(&json!(true), "boolean"));
        assert!(FilterSupport::matches_type(&json!(42), "number"));
        assert!(FilterSupport::matches_type(&json!("hello"), "string"));
        assert!(FilterSupport::matches_type(&json!([1, 2, 3]), "array"));
        assert!(FilterSupport::matches_type(&json!({"a": 1}), "object"));
    }

    #[test]
    fn test_is_truthy() {
        assert!(!FilterSupport::is_truthy(&json!(null)));
        assert!(!FilterSupport::is_truthy(&json!(false)));
        assert!(!FilterSupport::is_truthy(&json!(0)));
        assert!(!FilterSupport::is_truthy(&json!("")));
        assert!(!FilterSupport::is_truthy(&json!([])));
        assert!(!FilterSupport::is_truthy(&json!({})));

        assert!(FilterSupport::is_truthy(&json!(true)));
        assert!(FilterSupport::is_truthy(&json!(1)));
        assert!(FilterSupport::is_truthy(&json!("hello")));
        assert!(FilterSupport::is_truthy(&json!([1])));
        assert!(FilterSupport::is_truthy(&json!({"a": 1})));
    }

    #[test]
    fn test_compare_values() {
        use std::cmp::Ordering;

        assert_eq!(
            FilterSupport::compare_values(&json!(1), &json!(2)),
            Some(Ordering::Less)
        );
        assert_eq!(
            FilterSupport::compare_values(&json!(2), &json!(1)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            FilterSupport::compare_values(&json!(1), &json!(1)),
            Some(Ordering::Equal)
        );

        assert_eq!(
            FilterSupport::compare_values(&json!("a"), &json!("b")),
            Some(Ordering::Less)
        );
        assert_eq!(
            FilterSupport::compare_values(&json!(true), &json!(false)),
            Some(Ordering::Greater)
        );

        // Different types cannot be compared
        assert_eq!(FilterSupport::compare_values(&json!(1), &json!("1")), None);
    }

    #[test]
    fn test_contains_value() {
        assert!(FilterSupport::contains_value(&json!([1, 2, 3]), &json!(2)));
        assert!(!FilterSupport::contains_value(&json!([1, 2, 3]), &json!(4)));

        assert!(FilterSupport::contains_value(
            &json!({"a": 1, "b": 2}),
            &json!(1)
        ));
        assert!(!FilterSupport::contains_value(
            &json!({"a": 1, "b": 2}),
            &json!(3)
        ));

        assert!(FilterSupport::contains_value(
            &json!("hello world"),
            &json!("world")
        ));
        assert!(!FilterSupport::contains_value(
            &json!("hello world"),
            &json!("foo")
        ));
    }
}
