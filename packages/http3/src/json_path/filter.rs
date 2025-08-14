//! JSONPath Filter Expression Evaluation
//!
//! Handles evaluation of filter expressions including:
//! - Property access (@.property, @.nested.property)
//! - Comparisons (==, !=, <, <=, >, >=)
//! - Logical operations (&&, ||)
//! - Function calls (length(), count(), match(), search(), value())

use std::cell::RefCell;

use crate::json_path::error::{JsonPathResult, deserialization_error};
use crate::json_path::functions::FunctionEvaluator;
use crate::json_path::parser::{ComparisonOp, FilterExpression, FilterValue, LogicalOp};

// Shared thread-local storage for missing property context
thread_local! {
    static MISSING_PROPERTY_CONTEXT: RefCell<Option<(String, bool)>> = RefCell::new(None);
}

/// Filter Expression Evaluator
pub struct FilterEvaluator;

impl FilterEvaluator {
    /// Evaluate filter predicate against JSON context
    #[inline]
    pub fn evaluate_predicate(
        context: &serde_json::Value,
        expr: &FilterExpression,
    ) -> JsonPathResult<bool> {
        // Use empty context for backward compatibility
        let empty_context = std::collections::HashSet::new();
        Self::evaluate_predicate_with_context(context, expr, &empty_context)
    }

    /// Evaluate filter predicate with property existence context
    #[inline]
    pub fn evaluate_predicate_with_context(
        context: &serde_json::Value,
        expr: &FilterExpression,
        existing_properties: &std::collections::HashSet<String>,
    ) -> JsonPathResult<bool> {
        println!(
            "DEBUG: evaluate_predicate called with context={:?}, expr={:?}",
            serde_json::to_string(context).unwrap_or("invalid".to_string()),
            expr
        );
        match expr {
            FilterExpression::Property { path } => {
                // RFC 9535: Property access in filter context checks existence and truthiness
                // For @.author, this should return false if the object doesn't have an 'author' property
                println!("DEBUG: Evaluating property filter with path={:?}", path);
                Self::property_exists_and_truthy(context, path)
            }
            FilterExpression::Comparison {
                left,
                operator,
                right,
            } => {
                let left_val =
                    Self::evaluate_expression_with_context(context, left, existing_properties)?;
                let right_val =
                    Self::evaluate_expression_with_context(context, right, existing_properties)?;
                Self::compare_values_with_context(
                    &left_val,
                    *operator,
                    &right_val,
                    existing_properties,
                )
            }
            FilterExpression::Logical {
                left,
                operator,
                right,
            } => {
                let left_result =
                    Self::evaluate_predicate_with_context(context, left, existing_properties)?;
                let right_result =
                    Self::evaluate_predicate_with_context(context, right, existing_properties)?;
                Ok(match operator {
                    LogicalOp::And => left_result && right_result,
                    LogicalOp::Or => left_result || right_result,
                })
            }
            FilterExpression::Function { name, args } => {
                let value = FunctionEvaluator::evaluate_function_value(
                    context,
                    name,
                    args,
                    &|ctx, expr| {
                        Self::evaluate_expression_with_context(ctx, expr, existing_properties)
                    },
                )?;
                Ok(Self::is_truthy(&value))
            }
            _ => Ok(Self::is_truthy(&Self::evaluate_expression_with_context(
                context,
                expr,
                existing_properties,
            )?)),
        }
    }

    /// Check if property path exists and is truthy in filter context
    /// This is the correct semantics for [?@.property] filters  
    #[inline]
    fn property_exists_and_truthy(
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
        let result = Self::is_truthy(&Self::json_value_to_filter_value(current));
        println!("DEBUG: Property path exists, is_truthy result={}", result);
        Ok(result)
    }

    /// Resolve property path with context about which properties exist
    #[inline]
    fn resolve_property_path_with_context(
        context: &serde_json::Value,
        path: &[String],
        existing_properties: &std::collections::HashSet<String>,
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
    fn json_value_to_filter_value(value: &serde_json::Value) -> FilterValue {
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

    /// Evaluate expression to get its value
    #[inline]
    pub fn evaluate_expression(
        context: &serde_json::Value,
        expr: &FilterExpression,
    ) -> JsonPathResult<FilterValue> {
        let empty_context = std::collections::HashSet::new();
        Self::evaluate_expression_with_context(context, expr, &empty_context)
    }

    /// Evaluate expression with property context
    #[inline]
    pub fn evaluate_expression_with_context(
        context: &serde_json::Value,
        expr: &FilterExpression,
        existing_properties: &std::collections::HashSet<String>,
    ) -> JsonPathResult<FilterValue> {
        match expr {
            FilterExpression::Current => Ok(Self::json_value_to_filter_value(context)),
            FilterExpression::Property { path } => {
                Self::resolve_property_path_with_context(context, path, existing_properties)
            }
            FilterExpression::JsonPath { selectors } => {
                Self::evaluate_jsonpath_selectors(context, selectors)
            }
            FilterExpression::Literal { value } => Ok(value.clone()),
            FilterExpression::Function { name, args } => {
                FunctionEvaluator::evaluate_function_value(context, name, args, &|ctx, expr| {
                    Self::evaluate_expression_with_context(ctx, expr, existing_properties)
                })
            }
            _ => Err(deserialization_error(
                "complex expressions not supported in value context".to_string(),
                format!("{:?}", expr),
                "FilterValue",
            )),
        }
    }

    /// Compare two filter values using the specified operator
    #[inline]
    fn compare_values(
        left: &FilterValue,
        op: ComparisonOp,
        right: &FilterValue,
    ) -> JsonPathResult<bool> {
        let empty_context = std::collections::HashSet::new();
        Self::compare_values_with_context(left, op, right, &empty_context)
    }

    /// Compare two filter values with property existence context
    #[inline]
    fn compare_values_with_context(
        left: &FilterValue,
        op: ComparisonOp,
        right: &FilterValue,
        _existing_properties: &std::collections::HashSet<String>,
    ) -> JsonPathResult<bool> {
        match (left, right) {
            (FilterValue::Integer(a), FilterValue::Integer(b)) => Ok(match op {
                ComparisonOp::Equal => a == b,
                ComparisonOp::NotEqual => a != b,
                ComparisonOp::Less => a < b,
                ComparisonOp::LessEq => a <= b,
                ComparisonOp::Greater => a > b,
                ComparisonOp::GreaterEq => a >= b,
                _ => false,
            }),
            (FilterValue::Number(a), FilterValue::Number(b)) => Ok(match op {
                ComparisonOp::Equal => (a - b).abs() < f64::EPSILON,
                ComparisonOp::NotEqual => (a - b).abs() >= f64::EPSILON,
                ComparisonOp::Less => a < b,
                ComparisonOp::LessEq => a <= b,
                ComparisonOp::Greater => a > b,
                ComparisonOp::GreaterEq => a >= b,
                _ => false,
            }),
            (FilterValue::String(a), FilterValue::String(b)) => Ok(match op {
                ComparisonOp::Equal => a == b,
                ComparisonOp::NotEqual => a != b,
                ComparisonOp::Less => a < b,
                ComparisonOp::LessEq => a <= b,
                ComparisonOp::Greater => a > b,
                ComparisonOp::GreaterEq => a >= b,
                _ => false,
            }),
            (FilterValue::Boolean(a), FilterValue::Boolean(b)) => Ok(match op {
                ComparisonOp::Equal => a == b,
                ComparisonOp::NotEqual => a != b,
                _ => false,
            }),
            // RFC 9535: Missing properties context-aware comparison
            (FilterValue::Missing, FilterValue::Null) => {
                MISSING_PROPERTY_CONTEXT.with(|ctx| {
                    let context_info = ctx.borrow().clone();
                    println!("DEBUG: Missing vs Null comparison, context={:?}, op={:?}", context_info, op);
                    if let Some((property_name, exists_in_context)) = context_info {
                        // Clear the context after use
                        *ctx.borrow_mut() = None;
                        let result = match op {
                            ComparisonOp::Equal => false, // missing is never equal to null
                            ComparisonOp::NotEqual => exists_in_context, // missing != null only if property exists somewhere
                            _ => false,
                        };
                        println!("DEBUG: Context-aware result for property '{}': exists_in_context={}, result={}", property_name, exists_in_context, result);
                        Ok(result)
                    } else {
                        // Fallback: missing properties don't participate in comparisons
                        println!("DEBUG: No context available, returning false");
                        Ok(false)
                    }
                })
            }
            (FilterValue::Null, FilterValue::Missing) => {
                MISSING_PROPERTY_CONTEXT.with(|ctx| {
                    if let Some((_, exists_in_context)) = ctx.borrow().clone() {
                        // Clear the context after use
                        *ctx.borrow_mut() = None;
                        Ok(match op {
                            ComparisonOp::Equal => false, // null is never equal to missing
                            ComparisonOp::NotEqual => exists_in_context, // null != missing only if property exists somewhere
                            _ => false,
                        })
                    } else {
                        // Fallback: missing properties don't participate in comparisons
                        Ok(false)
                    }
                })
            }
            // Other missing property comparisons always false
            (FilterValue::Missing, _) => Ok(false),
            (_, FilterValue::Missing) => Ok(false),
            // RFC 9535: Null value comparisons
            (FilterValue::Null, FilterValue::Null) => Ok(match op {
                ComparisonOp::Equal => true,
                ComparisonOp::NotEqual => false,
                _ => false,
            }),
            (FilterValue::Null, _) => Ok(match op {
                ComparisonOp::Equal => false,
                ComparisonOp::NotEqual => true,
                _ => false,
            }),
            (_, FilterValue::Null) => Ok(match op {
                ComparisonOp::Equal => false,
                ComparisonOp::NotEqual => true,
                _ => false,
            }),
            // Type coercion for number/integer comparisons
            (FilterValue::Integer(a), FilterValue::Number(b)) => Self::compare_values(
                &FilterValue::Number(*a as f64),
                op,
                &FilterValue::Number(*b),
            ),
            (FilterValue::Number(a), FilterValue::Integer(b)) => Self::compare_values(
                &FilterValue::Number(*a),
                op,
                &FilterValue::Number(*b as f64),
            ),
            _ => Ok(false), // Other cross-type comparisons are false
        }
    }

    /// Evaluate complex JSONPath selectors relative to current context
    #[inline]
    fn evaluate_jsonpath_selectors(
        context: &serde_json::Value,
        selectors: &[crate::json_path::parser::JsonSelector],
    ) -> JsonPathResult<FilterValue> {
        use crate::json_path::parser::JsonSelector;

        let mut current = context;

        for selector in selectors {
            match selector {
                JsonSelector::Child { name, .. } => {
                    if let Some(obj) = current.as_object() {
                        if let Some(value) = obj.get(name) {
                            current = value;
                        } else {
                            return Ok(FilterValue::Missing);
                        }
                    } else {
                        return Ok(FilterValue::Missing);
                    }
                }
                JsonSelector::Wildcard => {
                    // For wildcard, return the array itself converted to a suitable representation
                    if current.is_array() {
                        return Ok(Self::json_value_to_filter_value(current));
                    } else {
                        return Ok(FilterValue::Missing);
                    }
                }
                JsonSelector::Index { index, from_end } => {
                    if let Some(arr) = current.as_array() {
                        let actual_index = if *from_end {
                            arr.len().saturating_sub((*index).unsigned_abs() as usize)
                        } else {
                            *index as usize
                        };

                        if let Some(value) = arr.get(actual_index) {
                            current = value;
                        } else {
                            return Ok(FilterValue::Missing);
                        }
                    } else {
                        return Ok(FilterValue::Missing);
                    }
                }
                _ => {
                    // For complex selectors, return the current value
                    return Ok(Self::json_value_to_filter_value(current));
                }
            }
        }

        Ok(Self::json_value_to_filter_value(current))
    }

    /// Check if a filter value is "truthy" for boolean context
    #[inline]
    fn is_truthy(value: &FilterValue) -> bool {
        match value {
            FilterValue::Boolean(b) => *b,
            FilterValue::Integer(i) => *i != 0,
            FilterValue::Number(f) => *f != 0.0 && !f.is_nan(),
            FilterValue::String(s) => !s.is_empty(),
            FilterValue::Null => false,
            FilterValue::Missing => false,
        }
    }
}
