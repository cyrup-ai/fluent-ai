//! RFC 9535 Null vs Missing Value Semantics (Section 2.6)
//!
//! The JSON `null` value is distinct from missing values. A query may select
//! a node whose value is `null`, and a missing member is different from a
//! member with a `null` value.
//!
//! This module provides utilities for correctly handling this distinction
//! throughout JSONPath evaluation.

use serde_json::Value as JsonValue;
use crate::json_path::error::{JsonPathResult, invalid_expression_error};

/// Represents the result of a property access that distinguishes between
/// null values and missing properties according to RFC 9535 Section 2.6
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyAccessResult {
    /// Property exists and has a null value
    NullValue,
    /// Property exists and has a non-null value
    Value(JsonValue),
    /// Property does not exist (missing)
    Missing,
}

/// Utilities for handling RFC 9535 null vs missing value semantics
pub struct NullSemantics;

impl NullSemantics {
    /// Access a property with proper null vs missing distinction
    ///
    /// Returns PropertyAccessResult to distinguish between:
    /// - A property that exists with null value
    /// - A property that does not exist (missing)
    /// - A property that exists with a non-null value
    #[inline]
    pub fn access_property(object: &JsonValue, property_name: &str) -> PropertyAccessResult {
        match object {
            JsonValue::Object(obj) => {
                match obj.get(property_name) {
                    Some(JsonValue::Null) => PropertyAccessResult::NullValue,
                    Some(value) => PropertyAccessResult::Value(value.clone()),
                    None => PropertyAccessResult::Missing,
                }
            }
            _ => PropertyAccessResult::Missing, // Non-objects don't have properties
        }
    }

    /// Access a nested property path with null vs missing distinction
    ///
    /// Follows a property path through nested objects, maintaining proper
    /// distinction between null values and missing properties at each level.
    #[inline]
    pub fn access_property_path(
        root: &JsonValue,
        path: &[String],
    ) -> PropertyAccessResult {
        let mut current = root;
        
        for (index, property) in path.iter().enumerate() {
            match current {
                JsonValue::Object(obj) => {
                    match obj.get(property) {
                        Some(JsonValue::Null) => {
                            // If this is the final property in the path, return null
                            if index == path.len() - 1 {
                                return PropertyAccessResult::NullValue;
                            } else {
                                // Cannot traverse through null to access deeper properties
                                return PropertyAccessResult::Missing;
                            }
                        }
                        Some(value) => {
                            current = value;
                        }
                        None => {
                            // Property missing at this level
                            return PropertyAccessResult::Missing;
                        }
                    }
                }
                _ => {
                    // Current value is not an object, cannot access properties
                    return PropertyAccessResult::Missing;
                }
            }
        }
        
        // If we've traversed the entire path, return the final value
        match current {
            JsonValue::Null => PropertyAccessResult::NullValue,
            value => PropertyAccessResult::Value(value.clone()),
        }
    }

    /// Check if a value should be considered "present" for filter evaluation
    ///
    /// According to RFC 9535, null values are present but missing values are not.
    /// This affects filter expression evaluation.
    #[inline]
    pub fn is_present(result: &PropertyAccessResult) -> bool {
        match result {
            PropertyAccessResult::NullValue => true,  // null is present
            PropertyAccessResult::Value(_) => true,   // non-null values are present
            PropertyAccessResult::Missing => false,   // missing is not present
        }
    }

    /// Convert PropertyAccessResult to Option<JsonValue> for compatibility
    ///
    /// Used when interacting with existing code that expects Option<JsonValue>.
    /// Note: This loses the null vs missing distinction.
    #[inline]
    pub fn to_option(result: &PropertyAccessResult) -> Option<JsonValue> {
        match result {
            PropertyAccessResult::NullValue => Some(JsonValue::Null),
            PropertyAccessResult::Value(v) => Some(v.clone()),
            PropertyAccessResult::Missing => None,
        }
    }

    /// Convert PropertyAccessResult to JsonValue with explicit missing representation
    ///
    /// For cases where the distinction needs to be preserved, this converts
    /// missing values to a special sentinel value that can be detected later.
    #[inline]
    pub fn to_json_with_missing_marker(result: &PropertyAccessResult) -> JsonValue {
        match result {
            PropertyAccessResult::NullValue => JsonValue::Null,
            PropertyAccessResult::Value(v) => v.clone(),
            PropertyAccessResult::Missing => {
                // Use a special object to represent missing values
                // This should never appear in actual JSON data
                serde_json::json!({"__jsonpath_missing__": true})
            }
        }
    }

    /// Check if a JsonValue is the missing marker
    #[inline]
    pub fn is_missing_marker(value: &JsonValue) -> bool {
        matches!(
            value,
            JsonValue::Object(obj) if obj.len() == 1 && 
                obj.get("__jsonpath_missing__") == Some(&JsonValue::Bool(true))
        )
    }

    /// Array access with proper null vs missing distinction
    ///
    /// Handles array index access while maintaining null vs missing semantics.
    /// Out-of-bounds access is considered missing, not null.
    #[inline]
    pub fn access_array_index(array: &JsonValue, index: i64) -> PropertyAccessResult {
        match array {
            JsonValue::Array(arr) => {
                let actual_index = if index < 0 {
                    // Negative indices count from the end
                    let len = arr.len() as i64;
                    len + index
                } else {
                    index
                };

                if actual_index >= 0 && (actual_index as usize) < arr.len() {
                    let value = &arr[actual_index as usize];
                    match value {
                        JsonValue::Null => PropertyAccessResult::NullValue,
                        v => PropertyAccessResult::Value(v.clone()),
                    }
                } else {
                    // Out of bounds is missing, not null
                    PropertyAccessResult::Missing
                }
            }
            _ => PropertyAccessResult::Missing, // Non-arrays don't have indices
        }
    }

    /// Filter evaluation with proper null vs missing handling
    ///
    /// Evaluates filter expressions while correctly handling the distinction
    /// between null values and missing properties.
    #[inline]
    pub fn evaluate_existence_filter(
        context: &JsonValue,
        property_path: &[String],
    ) -> bool {
        let result = Self::access_property_path(context, property_path);
        Self::is_present(&result)
    }

    /// Comparison with null vs missing distinction
    ///
    /// Handles comparisons involving null values and missing properties
    /// according to RFC 9535 semantics.
    #[inline]
    pub fn compare_with_null_semantics(
        left: &PropertyAccessResult,
        right: &PropertyAccessResult,
    ) -> JsonPathResult<bool> {
        match (left, right) {
            // Both null values
            (PropertyAccessResult::NullValue, PropertyAccessResult::NullValue) => Ok(true),
            
            // Both missing
            (PropertyAccessResult::Missing, PropertyAccessResult::Missing) => Ok(true),
            
            // Null vs missing (different)
            (PropertyAccessResult::NullValue, PropertyAccessResult::Missing) => Ok(false),
            (PropertyAccessResult::Missing, PropertyAccessResult::NullValue) => Ok(false),
            
            // Value comparisons
            (PropertyAccessResult::Value(a), PropertyAccessResult::Value(b)) => {
                Ok(a == b)
            }
            
            // Value vs null (different unless value is explicitly null)
            (PropertyAccessResult::Value(JsonValue::Null), PropertyAccessResult::NullValue) => {
                Ok(true)
            }
            (PropertyAccessResult::NullValue, PropertyAccessResult::Value(JsonValue::Null)) => {
                Ok(true)
            }
            (PropertyAccessResult::Value(_), PropertyAccessResult::NullValue) => Ok(false),
            (PropertyAccessResult::NullValue, PropertyAccessResult::Value(_)) => Ok(false),
            
            // Value vs missing (different)
            (PropertyAccessResult::Value(_), PropertyAccessResult::Missing) => Ok(false),
            (PropertyAccessResult::Missing, PropertyAccessResult::Value(_)) => Ok(false),
        }
    }

    /// Generate test results for different null vs missing scenarios
    ///
    /// Used for testing and validation to ensure proper handling of edge cases.
    #[inline]
    pub fn generate_test_scenarios() -> Vec<(JsonValue, &'static str, PropertyAccessResult)> {
        vec![
            // Null value present
            (
                serde_json::json!({"a": null}),
                "a",
                PropertyAccessResult::NullValue,
            ),
            // Property missing
            (
                serde_json::json!({}),
                "a",
                PropertyAccessResult::Missing,
            ),
            // Non-null value present
            (
                serde_json::json!({"a": "value"}),
                "a",
                PropertyAccessResult::Value(JsonValue::String("value".to_string())),
            ),
            // Root level property with null in nested structure
            (
                serde_json::json!({"nested": null, "obj": {"other": "value"}}),
                "nested", // Access root.nested
                PropertyAccessResult::NullValue,
            ),
            // Root level missing property
            (
                serde_json::json!({"obj": {"nested": null}}),
                "missing", // Access root.missing
                PropertyAccessResult::Missing,
            ),
        ]
    }

    /// Validate that null vs missing semantics are correctly implemented
    ///
    /// Runs validation tests to ensure the implementation correctly distinguishes
    /// between null values and missing properties in various scenarios.
    #[inline]
    pub fn validate_implementation() -> JsonPathResult<()> {
        let test_cases = Self::generate_test_scenarios();
        
        for (json, property, expected) in test_cases {
            let result = Self::access_property(&json, property);
            if result != expected {
                return Err(invalid_expression_error(
                    "",
                    &format!(
                        "null semantics validation failed: expected {:?}, got {:?}",
                        expected, result
                    ),
                    None,
                ));
            }
        }
        
        Ok(())
    }
}

impl PropertyAccessResult {
    /// Check if this result represents a present value (null or non-null)
    #[inline]
    pub fn is_present(&self) -> bool {
        NullSemantics::is_present(self)
    }

    /// Check if this result represents a missing property
    #[inline]
    pub fn is_missing(&self) -> bool {
        matches!(self, PropertyAccessResult::Missing)
    }

    /// Check if this result represents a null value
    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, PropertyAccessResult::NullValue)
    }

    /// Get the JsonValue if present, otherwise return None
    #[inline]
    pub fn value(&self) -> Option<&JsonValue> {
        match self {
            PropertyAccessResult::Value(v) => Some(v),
            PropertyAccessResult::NullValue => None, // Explicitly null
            PropertyAccessResult::Missing => None,   // Missing
        }
    }

    /// Get the JsonValue with null preserved
    #[inline]
    pub fn value_with_null(&self) -> Option<JsonValue> {
        match self {
            PropertyAccessResult::Value(v) => Some(v.clone()),
            PropertyAccessResult::NullValue => Some(JsonValue::Null),
            PropertyAccessResult::Missing => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_vs_missing_distinction() {
        // Test null value present
        let json_with_null = serde_json::json!({"a": null});
        let null_result = NullSemantics::access_property(&json_with_null, "a");
        assert!(matches!(null_result, PropertyAccessResult::NullValue));
        assert!(null_result.is_present());
        assert!(!null_result.is_missing());
        assert!(null_result.is_null());

        // Test property missing
        let json_empty = serde_json::json!({});
        let missing_result = NullSemantics::access_property(&json_empty, "a");
        assert!(matches!(missing_result, PropertyAccessResult::Missing));
        assert!(!missing_result.is_present());
        assert!(missing_result.is_missing());
        assert!(!missing_result.is_null());

        // Test non-null value
        let json_with_value = serde_json::json!({"a": "hello"});
        let value_result = NullSemantics::access_property(&json_with_value, "a");
        assert!(matches!(value_result, PropertyAccessResult::Value(_)));
        assert!(value_result.is_present());
        assert!(!value_result.is_missing());
        assert!(!value_result.is_null());
    }

    #[test]
    fn test_property_path_access() {
        let json = serde_json::json!({
            "store": {
                "book": null,
                "bicycle": {
                    "color": "red"
                }
            }
        });

        // Access null value through path
        let null_path_result = NullSemantics::access_property_path(
            &json, 
            &["store".to_string(), "book".to_string()]
        );
        assert!(matches!(null_path_result, PropertyAccessResult::NullValue));

        // Access missing property through path
        let missing_path_result = NullSemantics::access_property_path(
            &json, 
            &["store".to_string(), "missing".to_string()]
        );
        assert!(matches!(missing_path_result, PropertyAccessResult::Missing));

        // Access existing value through path
        let value_path_result = NullSemantics::access_property_path(
            &json, 
            &["store".to_string(), "bicycle".to_string(), "color".to_string()]
        );
        assert!(matches!(value_path_result, PropertyAccessResult::Value(_)));
    }

    #[test]
    fn test_array_access() {
        let json = serde_json::json!([null, "value", 42]);

        // Access null element
        let null_element = NullSemantics::access_array_index(&json, 0);
        assert!(matches!(null_element, PropertyAccessResult::NullValue));

        // Access regular element  
        let value_element = NullSemantics::access_array_index(&json, 1);
        assert!(matches!(value_element, PropertyAccessResult::Value(_)));

        // Access out of bounds (missing)
        let missing_element = NullSemantics::access_array_index(&json, 10);
        assert!(matches!(missing_element, PropertyAccessResult::Missing));

        // Negative index access
        let last_element = NullSemantics::access_array_index(&json, -1);
        assert!(matches!(last_element, PropertyAccessResult::Value(_)));
    }

    #[test]
    fn test_comparison_semantics() {
        let null_result = PropertyAccessResult::NullValue;
        let missing_result = PropertyAccessResult::Missing;
        let value_result = PropertyAccessResult::Value(serde_json::json!("test"));

        // Null vs null
        assert!(NullSemantics::compare_with_null_semantics(&null_result, &null_result)
            .expect("Failed to compare null vs null"));

        // Missing vs missing
        assert!(NullSemantics::compare_with_null_semantics(&missing_result, &missing_result)
            .expect("Failed to compare missing vs missing"));

        // Null vs missing (different)
        assert!(!NullSemantics::compare_with_null_semantics(&null_result, &missing_result)
            .expect("Failed to compare null vs missing"));

        // Value vs missing (different)
        assert!(!NullSemantics::compare_with_null_semantics(&value_result, &missing_result)
            .expect("Failed to compare value vs missing"));
    }

    #[test]
    fn test_implementation_validation() {
        // This should pass without errors
        NullSemantics::validate_implementation()
            .expect("Failed to validate null semantics implementation");
    }
}