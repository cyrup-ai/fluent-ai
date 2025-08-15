//! JSONPath selector application with zero-allocation patterns
//! 
//! Handles individual selector types: child, index, wildcard, filter, slice, union.

use serde_json::Value;
use crate::json_path::error::JsonPathError;
use crate::json_path::filter::FilterEvaluator;
use crate::json_path::parser::{FilterExpression, JsonSelector};
use super::evaluator::CoreJsonPathEvaluator;

type JsonPathResult<T> = Result<T, JsonPathError>;

impl CoreJsonPathEvaluator {
    /// Apply a single selector to a JSON value, returning owned values
    pub fn apply_selector_to_value(
        &self,
        value: &Value,
        selector: &JsonSelector,
    ) -> JsonPathResult<Vec<Value>> {
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
                self.collect_all_descendants_owned(value, &mut results);
            }
            JsonSelector::Index { index, from_end } => {
                if let Value::Array(arr) = value {
                    let actual_index = if *from_end && *index < 0 {
                        // Negative index - count from end (e.g., -1 means last element)
                        let abs_index = (-*index) as usize;
                        if abs_index <= arr.len() && abs_index > 0 {
                            arr.len() - abs_index
                        } else {
                            return Ok(results); // Index out of bounds
                        }
                    } else if *from_end && *index > 0 {
                        // Positive from_end index
                        if (*index as usize) <= arr.len() {
                            arr.len() - (*index as usize)
                        } else {
                            return Ok(results); // Index out of bounds
                        }
                    } else {
                        // Regular positive index
                        *index as usize
                    };

                    if actual_index < arr.len() {
                        results.push(arr[actual_index].clone());
                    }
                }
            }
            JsonSelector::Wildcard => {
                self.apply_wildcard_owned(value, &mut results, 1000); // Reasonable limit
            }
            JsonSelector::Filter { expression } => {
                self.apply_filter_selector_owned(value, expression, &mut results)?;
            }
            JsonSelector::Slice { start, end, step } => {
                if let Value::Array(arr) = value {
                    let slice_results = self.apply_slice_to_array(arr, *start, *end, *step)?;
                    results.extend(slice_results);
                }
            }
            JsonSelector::Union { selectors } => {
                // Apply each selector in the union and collect all results
                // RFC 9535: Union preserves order and duplicates
                for union_selector in selectors {
                    let union_results = self.apply_selector_to_value(value, union_selector)?;
                    results.extend(union_results);
                }
            }
        }

        Ok(results)
    }

    /// Apply wildcard selector with result limit for performance
    fn apply_wildcard_owned(&self, value: &Value, results: &mut Vec<Value>, max_results: usize) {
        match value {
            Value::Object(obj) => {
                // Wildcard on object returns all object values
                for child_value in obj.values() {
                    if results.len() >= max_results {
                        break;
                    }
                    results.push(child_value.clone());
                }
            }
            Value::Array(arr) => {
                // Wildcard on array returns all array elements
                for child_value in arr {
                    if results.len() >= max_results {
                        break;
                    }
                    results.push(child_value.clone());
                }
            }
            _ => {
                // Primitives have no children - wildcard returns nothing
            }
        }
    }

    /// Apply filter selector using FilterEvaluator with owned results
    fn apply_filter_selector_owned(
        &self,
        node: &Value,
        expression: &FilterExpression,
        results: &mut Vec<Value>,
    ) -> JsonPathResult<()> {
        match node {
            Value::Array(arr) => {
                log::debug!(
                    "apply_filter_selector_owned called on array with {} items",
                    arr.len()
                );
                // Collect all property names that exist across items in this array
                let existing_properties = self.collect_existing_properties(arr);

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
                    node,
                    expression,
                    &existing_properties,
                )? {
                    results.push(node.clone());
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Collect all property names that exist across any item in the array
    pub fn collect_existing_properties(&self, arr: &[Value]) -> std::collections::HashSet<String> {
        let mut properties = std::collections::HashSet::new();

        for item in arr {
            if let Some(obj) = item.as_object() {
                for key in obj.keys() {
                    properties.insert(key.clone());
                }
            }
        }

        log::debug!("Collected existing properties: {:?}", properties);
        properties
    }

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

    /// Apply child selector to a node - handles object property access
    pub fn apply_child_selector<'a>(&self, node: &'a Value, name: &str, results: &mut Vec<&'a Value>) {
        if let Value::Object(obj) = node {
            if let Some(value) = obj.get(name) {
                results.push(value);
            }
        }
    }

    /// Collect all descendants using recursive descent (..)
    pub fn collect_all_descendants<'a>(&self, node: &'a Value, results: &mut Vec<&'a Value>) {
        match node {
            Value::Object(obj) => {
                for value in obj.values() {
                    results.push(value);
                    self.collect_all_descendants(value, results);
                }
            }
            Value::Array(arr) => {
                for value in arr {
                    results.push(value);
                    self.collect_all_descendants(value, results);
                }
            }
            _ => {}
        }
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

    /// Apply wildcard selector to get all children
    pub fn apply_wildcard_selector<'a>(&self, node: &'a Value, results: &mut Vec<&'a Value>) {
        match node {
            Value::Object(obj) => {
                for value in obj.values() {
                    results.push(value);
                }
            }
            Value::Array(arr) => {
                for value in arr {
                    results.push(value);
                }
            }
            _ => {}
        }
    }

    /// Apply filter selector using FilterEvaluator
    pub fn apply_filter_selector<'a>(
        &self,
        node: &'a Value,
        expression: &FilterExpression,
        results: &mut Vec<&'a Value>,
    ) -> JsonPathResult<()> {
        match node {
            Value::Array(arr) => {
                println!(
                    "DEBUG: apply_filter_selector called on array with {} items",
                    arr.len()
                );
                // Collect all property names that exist across items in this array
                let existing_properties = self.collect_existing_properties(arr);

                for item in arr {
                    if FilterEvaluator::evaluate_predicate_with_context(
                        item,
                        expression,
                        &existing_properties,
                    )? {
                        results.push(item);
                    }
                }
            }
            Value::Object(_obj) => {
                // For objects, apply filter to the object itself
                // Create context with properties from this object
                let existing_properties: std::collections::HashSet<String> =
                    std::collections::HashSet::new();

                if FilterEvaluator::evaluate_predicate_with_context(
                    node,
                    expression,
                    &existing_properties,
                )? {
                    results.push(node);
                }
            }
            _ => {}
        }
        Ok(())
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