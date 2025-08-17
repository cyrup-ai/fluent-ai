//! Selector application engine for JSONPath evaluation
//!
//! This module handles the core logic for applying individual selectors to JSON values.

use serde_json::Value;

use super::array_operations::ArrayOperations;
use super::filter_support::FilterSupport;
use super::recursive_descent::RecursiveDescentEngine;
use crate::jsonpath::error::JsonPathError;
use crate::jsonpath::filter::FilterEvaluator;
use crate::jsonpath::parser::{FilterExpression, JsonSelector};

type JsonPathResult<T> = Result<T, JsonPathError>;

/// Engine for applying individual selectors to JSON values
pub struct SelectorEngine;

impl SelectorEngine {
    /// Apply a single selector to a JSON value, returning owned values
    pub fn apply_selector(value: &Value, selector: &JsonSelector) -> JsonPathResult<Vec<Value>> {
        let mut results = Vec::new();

        match selector {
            JsonSelector::Root => {
                // Root selector returns the value itself
                results.push(value.clone());
            }
            JsonSelector::Child { name, .. } => {
                if let Value::Object(obj) = value {
                    if let Some(child_value) = obj.get(name) {
                        results.push(child_value.clone());
                    }
                }
            }
            JsonSelector::RecursiveDescent => {
                // Collect all descendants
                RecursiveDescentEngine::collect_all_descendants_owned(value, &mut results);
            }
            JsonSelector::Index { index, from_end } => {
                if let Value::Array(arr) = value {
                    let array_results = ArrayOperations::apply_index(arr, *index, *from_end)?;
                    results.extend(array_results);
                }
            }
            JsonSelector::Slice { start, end, step } => {
                if let Value::Array(arr) = value {
                    let slice_results = ArrayOperations::apply_slice(arr, start, end, *step)?;
                    results.extend(slice_results);
                }
            }
            JsonSelector::Wildcard => {
                Self::apply_wildcard(value, &mut results);
            }
            JsonSelector::Filter { expression } => {
                Self::apply_filter(value, expression, &mut results)?;
            }
            JsonSelector::Union { selectors } => {
                for selector in selectors {
                    let union_results = Self::apply_selector(value, selector)?;
                    results.extend(union_results);
                }
            }
        }

        Ok(results)
    }

    /// Apply wildcard selector to get all children
    fn apply_wildcard(value: &Value, results: &mut Vec<Value>) {
        match value {
            Value::Object(obj) => {
                for child_value in obj.values() {
                    results.push(child_value.clone());
                }
            }
            Value::Array(arr) => {
                for child_value in arr {
                    results.push(child_value.clone());
                }
            }
            _ => {} // Primitives have no children
        }
    }

    /// Apply filter selector using FilterEvaluator
    fn apply_filter(
        value: &Value,
        expression: &FilterExpression,
        results: &mut Vec<Value>,
    ) -> JsonPathResult<()> {
        // RFC 9535 Section 2.3.5.2: Filter selector tests children of input value
        match value {
            Value::Array(arr) => {
                // For arrays: collect existing properties first for context-aware evaluation
                let existing_properties = FilterSupport::collect_existing_properties(arr);

                log::debug!(
                    "Array filter - collected {} existing properties: {:?}",
                    existing_properties.len(),
                    existing_properties
                );

                // For arrays: test each element (child) against filter with context
                for item in arr {
                    if FilterEvaluator::evaluate_predicate_with_context(
                        item,
                        expression,
                        &existing_properties,
                    )? {
                        results.push(item.clone());
                    }
                }
            }
            Value::Object(_obj) => {
                // For objects, apply filter to the object itself
                // Create context with properties from this object
                let existing_properties: std::collections::HashSet<String> =
                    std::collections::HashSet::new();

                if FilterEvaluator::evaluate_predicate_with_context(
                    value,
                    expression,
                    &existing_properties,
                )? {
                    results.push(value.clone());
                }
            }
            _ => {
                // For primitives, the filter doesn't apply (no children to test)
                // This is correct per RFC 9535 - filters only apply to structured values
            }
        }
        Ok(())
    }

    /// Apply multiple selectors in sequence
    pub fn apply_selectors(
        initial_value: &Value,
        selectors: &[JsonSelector],
    ) -> JsonPathResult<Vec<Value>> {
        let mut current_results = vec![initial_value.clone()];

        for selector in selectors {
            let mut next_results = Vec::new();

            for value in &current_results {
                let selector_results = Self::apply_selector(value, selector)?;
                next_results.extend(selector_results);

                // Safety check: prevent memory exhaustion
                if next_results.len() > 10000 {
                    log::warn!(
                        "Selector application stopped - result set too large ({})",
                        next_results.len()
                    );
                    return Ok(vec![]);
                }
            }

            current_results = next_results;

            // Early termination if no results
            if current_results.is_empty() {
                return Ok(vec![]);
            }
        }

        Ok(current_results)
    }

    /// Check if a selector is potentially expensive
    pub fn is_expensive_selector(selector: &JsonSelector) -> bool {
        match selector {
            JsonSelector::RecursiveDescent => true,
            JsonSelector::Wildcard => true,
            JsonSelector::Filter { .. } => true,
            JsonSelector::Slice { .. } => true,
            JsonSelector::Union { .. } => true,
            _ => false,
        }
    }

    /// Estimate the complexity of a selector
    pub fn selector_complexity(selector: &JsonSelector) -> u32 {
        match selector {
            JsonSelector::Root => 1,
            JsonSelector::Child { .. } => 1,
            JsonSelector::Index { .. } => 1,
            JsonSelector::Wildcard => 10,
            JsonSelector::Slice { .. } => 5,
            JsonSelector::Filter { .. } => 20,
            JsonSelector::RecursiveDescent => 50,
            JsonSelector::Union { selectors } => {
                selectors
                    .iter()
                    .map(|s| Self::selector_complexity(s))
                    .sum::<u32>()
                    + 5
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_root_selector() {
        let json = json!({"test": "value"});
        let selector = JsonSelector::Root;
        let results = SelectorEngine::apply_selector(&json, &selector)
            .expect("Failed to apply root selector");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json);
    }

    #[test]
    fn test_child_selector() {
        let json = json!({"store": {"name": "test"}});
        let selector = JsonSelector::Child {
            name: "store".to_string(),
            quoted: false,
        };
        let results = SelectorEngine::apply_selector(&json, &selector)
            .expect("Failed to apply child selector");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json!({"name": "test"}));
    }

    #[test]
    fn test_wildcard_selector() {
        let json = json!({"a": 1, "b": 2, "c": 3});
        let selector = JsonSelector::Wildcard;
        let results = SelectorEngine::apply_selector(&json, &selector)
            .expect("Failed to apply wildcard selector");
        assert_eq!(results.len(), 3);
        assert!(results.contains(&json!(1)));
        assert!(results.contains(&json!(2)));
        assert!(results.contains(&json!(3)));
    }

    #[test]
    fn test_selector_complexity() {
        assert_eq!(SelectorEngine::selector_complexity(&JsonSelector::Root), 1);
        assert_eq!(
            SelectorEngine::selector_complexity(&JsonSelector::Wildcard),
            10
        );
        assert_eq!(
            SelectorEngine::selector_complexity(&JsonSelector::RecursiveDescent),
            50
        );
    }
}
