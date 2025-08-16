//! Property access and resolution for filter expressions
//!
//! Handles property path traversal, existence checking, and value conversion
//! with support for RFC 9535 missing vs null semantics.

use std::cell::RefCell;
use std::collections::HashSet;

use super::utils::FilterUtils;
use crate::json_path::error::JsonPathResult;
use crate::json_path::parser::FilterValue;

// Shared thread-local storage for missing property context
thread_local! {
    static MISSING_PROPERTY_CONTEXT: RefCell<Option<(String, bool)>> = RefCell::new(None);
}

/// Property resolution utilities
pub struct PropertyResolver;

impl PropertyResolver {
    /// Check if property path exists and is truthy in filter context
    /// This is the correct semantics for [?@.property] filters  
    #[inline]
    pub fn property_exists_and_truthy(
        context: &serde_json::Value,
        path: &[String],
    ) -> JsonPathResult<bool> {
        println!(
            "DEBUG: property_exists_and_truthy called with context={:?}, path={:?}",
            serde_json::to_string(context).unwrap_or("invalid".to_string()),
            path
        );
        let mut current = context;

        for property in path {
            println!(
                "DEBUG: Checking property '{}' in current={:?}",
                property,
                serde_json::to_string(current).unwrap_or("invalid".to_string())
            );
            if let Some(obj) = current.as_object() {
                if let Some(value) = obj.get(property) {
                    println!(
                        "DEBUG: Found property '{}', value={:?}",
                        property,
                        serde_json::to_string(value).unwrap_or("invalid".to_string())
                    );
                    current = value;
                } else {
                    // Property doesn't exist - return false
                    println!(
                        "DEBUG: Property '{}' does not exist, returning false",
                        property
                    );
                    return Ok(false);
                }
            } else {
                // Current value is not an object - can't access properties
                println!("DEBUG: Current value is not an object, returning false");
                return Ok(false);
            }
        }

        // Property exists - check if it's truthy
        let result = FilterUtils::is_truthy(&Self::json_value_to_filter_value(current));
        println!("DEBUG: Property path exists, is_truthy result={}", result);
        Ok(result)
    }

    /// Resolve property path with context about which properties exist
    #[inline]
    pub fn resolve_property_path_with_context(
        context: &serde_json::Value,
        path: &[String],
        existing_properties: &HashSet<String>,
    ) -> JsonPathResult<FilterValue> {
        let mut current = context;

        for (i, property) in path.iter().enumerate() {
            if let Some(obj) = current.as_object() {
                if let Some(value) = obj.get(property) {
                    current = value;
                } else {
                    // RFC 9535: Missing properties are distinct from null values
                    // For top-level properties, we need to consider context
                    if i == 0 && !path.is_empty() {
                        let exists_in_context = existing_properties.contains(property);
                        println!(
                            "DEBUG: Property '{}' is missing, exists_in_context={}",
                            property, exists_in_context
                        );
                        // Store property name for context-aware comparison
                        MISSING_PROPERTY_CONTEXT.with(|ctx| {
                            *ctx.borrow_mut() = Some((property.clone(), exists_in_context));
                        });
                    }
                    return Ok(FilterValue::Missing);
                }
            } else {
                return Ok(FilterValue::Missing);
            }
        }

        Ok(Self::json_value_to_filter_value(current))
    }

    /// Convert serde_json::Value to FilterValue
    #[inline]
    pub fn json_value_to_filter_value(value: &serde_json::Value) -> FilterValue {
        match value {
            serde_json::Value::Null => FilterValue::Null,
            serde_json::Value::Bool(b) => FilterValue::Boolean(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    FilterValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    FilterValue::Number(f)
                } else {
                    FilterValue::Null
                }
            }
            serde_json::Value::String(s) => FilterValue::String(s.clone()),
            // Arrays and objects should not convert to null - they're distinct values
            // For comparison purposes, we'll handle them specially in compare_values
            serde_json::Value::Array(_) => FilterValue::Boolean(true), // Arrays are truthy
            serde_json::Value::Object(_) => FilterValue::Boolean(true), // Objects are truthy
        }
    }

    /// Access the missing property context for comparison operations
    pub fn with_missing_context<T>(f: impl FnOnce(Option<(String, bool)>) -> T) -> T {
        MISSING_PROPERTY_CONTEXT.with(|ctx| {
            let context_info = ctx.borrow().clone();
            f(context_info)
        })
    }

    /// Clear the missing property context after use
    pub fn clear_missing_context() {
        MISSING_PROPERTY_CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = None;
        });
    }
}
