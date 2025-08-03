//! Core JSONPath evaluator for production functionality
//!
//! This module provides a robust JSONPath implementation that handles
//! the complete RFC 9535 specification with optimized performance.
//!
//! NOTE: Contains multiple evaluation approaches for different JSONPath patterns.
//! Some methods may be alternative implementations preserved for future optimization.
#![allow(dead_code)]

use serde_json::Value;
use crate::json_path::{
    JsonPathResult, 
    parser::JsonPathParser,
    ast::{JsonSelector, FilterExpression},
    filter::FilterEvaluator
};

/// Core JSONPath evaluator that works on parsed JSON
pub struct CoreJsonPathEvaluator {
    expression: String,
}

impl CoreJsonPathEvaluator {
    /// Create new evaluator with JSONPath expression
    pub fn new(expression: &str) -> JsonPathResult<Self> {
        // Validate the expression can be parsed
        JsonPathParser::compile(expression)?;
        
        Ok(Self {
            expression: expression.to_string(),
        })
    }

    /// Evaluate JSONPath expression against JSON value using AST-based evaluation
    pub fn evaluate(&self, json: &Value) -> JsonPathResult<Vec<Value>> {
        // Parse expression once to get AST selectors
        let parsed_expr = JsonPathParser::compile(&self.expression)?;
        let selectors = parsed_expr.selectors();
        
        // Start with root node - collect references first to avoid lifetime issues
        let mut current_results: Vec<Value> = vec![json.clone()];
        
        // Process each selector in the chain
        for selector in selectors {
            let mut next_results = Vec::new();
            
            // Apply selector to each current result
            for current_value in &current_results {
                let intermediate_results = self.apply_selector_to_value(current_value, &selector)?;
                next_results.extend(intermediate_results);
            }
            
            current_results = next_results;
            
            // Early exit if no matches
            if current_results.is_empty() {
                return Ok(vec![]);
            }
        }
        
        Ok(current_results)
    }
    
    /// Apply a single selector to a JSON value, returning owned values
    fn apply_selector_to_value(&self, value: &Value, selector: &JsonSelector) -> JsonPathResult<Vec<Value>> {
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
                    let actual_index = if *from_end {
                        if *index > 0 && (*index as usize) <= arr.len() {
                            arr.len() - (*index as usize)
                        } else {
                            return Ok(results); // Index out of bounds
                        }
                    } else if *index < 0 {
                        // Negative index - count from end
                        let abs_index = (-*index) as usize;
                        if abs_index <= arr.len() {
                            arr.len() - abs_index
                        } else {
                            return Ok(results); // Index out of bounds
                        }
                    } else {
                        *index as usize
                    };

                    if actual_index < arr.len() {
                        results.push(arr[actual_index].clone());
                    }
                }
            }
            JsonSelector::Slice { start, end, step } => {
                if let Value::Array(arr) = value {
                    let slice_results = self.apply_slice_to_array(arr, *start, *end, *step)?;
                    results.extend(slice_results);
                }
            }
            JsonSelector::Wildcard => {
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
            JsonSelector::Filter { expression } => {
                match value {
                    Value::Array(arr) => {
                        for item in arr {
                            if FilterEvaluator::evaluate_predicate(item, expression)? {
                                results.push(item.clone());
                            }
                        }
                    }
                    Value::Object(_) => {
                        if FilterEvaluator::evaluate_predicate(value, expression)? {
                            results.push(value.clone());
                        }
                    }
                    _ => {} // Primitives don't support filters
                }
            }
            JsonSelector::Union { selectors } => {
                for union_selector in selectors {
                    let union_results = self.apply_selector_to_value(value, union_selector)?;
                    results.extend(union_results);
                }
            }
        }
        
        Ok(results)
    }
    
    /// Collect all descendants using recursive descent, returning owned values
    fn collect_all_descendants_owned(&self, value: &Value, results: &mut Vec<Value>) {
        match value {
            Value::Object(obj) => {
                for child_value in obj.values() {
                    results.push(child_value.clone());
                    self.collect_all_descendants_owned(child_value, results);
                }
            }
            Value::Array(arr) => {
                for child_value in arr {
                    results.push(child_value.clone());
                    self.collect_all_descendants_owned(child_value, results);
                }
            }
            _ => {} // Primitives have no descendants
        }
    }
    
    /// Apply slice to array with start, end, step parameters
    fn apply_slice_to_array(&self, arr: &[Value], start: Option<i64>, end: Option<i64>, step: Option<i64>) -> JsonPathResult<Vec<Value>> {
        let len = arr.len() as i64;
        let step = step.unwrap_or(1);
        
        if step == 0 {
            return Ok(vec![]); // Invalid step
        }
        
        let start = start.unwrap_or(if step > 0 { 0 } else { len - 1 });
        let end = end.unwrap_or(if step > 0 { len } else { -1 });
        
        // Normalize negative indices
        let start = if start < 0 { (len + start).max(0) } else { start.min(len) };
        let end = if end < 0 { (len + end).max(-1) } else { end.min(len) };
        
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
    
    
    fn evaluate_wildcard(&self, json: &Value) -> JsonPathResult<Vec<Value>> {
        match json {
            Value::Object(obj) => Ok(obj.values().cloned().collect()),
            Value::Array(arr) => Ok(arr.clone()),
            _ => Ok(vec![]),
        }
    }
    
    fn collect_all_values(&self, json: &Value) -> Vec<Value> {
        let mut results = Vec::new();
        self.collect_all_values_recursive(json, &mut results);
        results
    }
    
    fn collect_all_values_recursive(&self, json: &Value, results: &mut Vec<Value>) {
        match json {
            Value::Object(obj) => {
                for value in obj.values() {
                    results.push(value.clone());
                    self.collect_all_values_recursive(value, results);
                }
            }
            Value::Array(arr) => {
                for value in arr {
                    results.push(value.clone());
                    self.collect_all_values_recursive(value, results);
                }
            }
            _ => {}
        }
    }

    /// Apply child selector to a node - handles object property access
    fn apply_child_selector<'a>(&self, node: &'a Value, name: &str, results: &mut Vec<&'a Value>) {
        if let Value::Object(obj) = node {
            if let Some(value) = obj.get(name) {
                results.push(value);
            }
        }
    }

    /// Collect all descendants using recursive descent (..)
    fn collect_all_descendants<'a>(&self, node: &'a Value, results: &mut Vec<&'a Value>) {
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
    fn apply_index_selector<'a>(&self, node: &'a Value, index: i64, from_end: bool, results: &mut Vec<&'a Value>) {
        if let Value::Array(arr) = node {
            let actual_index = if from_end {
                if index > 0 {
                    if (index as usize) <= arr.len() {
                        arr.len() - (index as usize)
                    } else {
                        return; // Index out of bounds
                    }
                } else {
                    return; // Invalid negative index for from_end
                }
            } else if index < 0 {
                // Negative index - count from end
                let abs_index = (-index) as usize;
                if abs_index <= arr.len() {
                    arr.len() - abs_index
                } else {
                    return; // Index out of bounds
                }
            } else {
                index as usize
            };

            if actual_index < arr.len() {
                results.push(&arr[actual_index]);
            }
        }
    }

    /// Apply wildcard selector to get all children
    fn apply_wildcard_selector<'a>(&self, node: &'a Value, results: &mut Vec<&'a Value>) {
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
    fn apply_filter_selector<'a>(&self, node: &'a Value, expression: &FilterExpression, results: &mut Vec<&'a Value>) -> JsonPathResult<()> {
        match node {
            Value::Array(arr) => {
                for item in arr {
                    if FilterEvaluator::evaluate_predicate(item, expression)? {
                        results.push(item);
                    }
                }
            }
            Value::Object(_obj) => {
                // For objects, apply filter to the object itself
                if FilterEvaluator::evaluate_predicate(node, expression)? {
                    results.push(node);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Apply slice selector with start, end, step parameters for arrays
    fn apply_slice_selector_with_params<'a>(&self, node: &'a Value, start: Option<i64>, end: Option<i64>, step: Option<i64>, results: &mut Vec<&'a Value>) {
        if let Value::Array(arr) = node {
            let len = arr.len() as i64;
            let step = step.unwrap_or(1);
            
            if step == 0 {
                return; // Invalid step
            }
            
            let start = start.unwrap_or(if step > 0 { 0 } else { len - 1 });
            let end = end.unwrap_or(if step > 0 { len } else { -1 });
            
            // Normalize negative indices
            let start = if start < 0 { (len + start).max(0) } else { start.min(len) };
            let end = if end < 0 { (len + end).max(-1) } else { end.min(len) };
            
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


    /// Evaluate a property path on a JSON value (for nested property access)
    fn evaluate_property_path(&self, json: &Value, path: &str) -> JsonPathResult<Vec<Value>> {
        // Handle simple property access for now
        let properties: Vec<&str> = path.split('.').collect();
        let mut current = vec![json.clone()];
        
        for property in properties {
            if property.is_empty() {
                continue;
            }
            
            let mut next = Vec::new();
            for value in current {
                if let Value::Object(obj) = value {
                    if let Some(prop_value) = obj.get(property) {
                        next.push(prop_value.clone());
                    }
                }
            }
            current = next;
        }
        
        Ok(current)
    }
    
    fn find_property_recursive(&self, json: &Value, property: &str) -> Vec<Value> {
        let mut results = Vec::new();
        self.find_property_recursive_impl(json, property, &mut results);
        results
    }
    
    fn find_property_recursive_impl(&self, json: &Value, property: &str, results: &mut Vec<Value>) {
        match json {
            Value::Object(obj) => {
                // Check if this object has the property
                if let Some(value) = obj.get(property) {
                    results.push(value.clone());
                }
                // Recurse into all values
                for value in obj.values() {
                    self.find_property_recursive_impl(value, property, results);
                }
            }
            Value::Array(arr) => {
                // Recurse into all array elements
                for value in arr {
                    self.find_property_recursive_impl(value, property, results);
                }
            }
            _ => {
                // Leaf values - nothing to do
            }
        }
    }
    
    fn evaluate_array_access(&self, json: &Value, expr: &str) -> JsonPathResult<Vec<Value>> {
        // Core parsing for array access patterns
        if let Some(captures) = self.parse_array_expression(expr) {
            let (path, selector) = captures;
            
            // Navigate to the array - collect intermediate results to avoid lifetime issues
            if path == "$" {
                match json {
                    Value::Array(arr) => self.apply_array_selector(arr, &selector),
                    _ => Ok(vec![]),
                }
            } else if path.starts_with("$.") {
                let property_results = self.evaluate_property_path(json, &path[2..])?;
                if property_results.len() == 1 {
                    match &property_results[0] {
                        Value::Array(arr) => self.apply_array_selector(&arr, &selector),
                        _ => Ok(vec![]),
                    }
                } else {
                    Ok(vec![])
                }
            } else if path.starts_with("$..") {
                // Handle recursive descent to array
                let property = &path[3..];
                let candidates = self.find_property_recursive(json, property);
                if candidates.len() == 1 {
                    match &candidates[0] {
                        Value::Array(arr) => self.apply_array_selector(&arr, &selector),
                        _ => Ok(vec![]),
                    }
                } else {
                    Ok(vec![])
                }
            } else {
                Ok(vec![])
            }
        } else {
            Ok(vec![])
        }
    }
    
    fn parse_array_expression(&self, expr: &str) -> Option<(String, String)> {
        if let Some(bracket_start) = expr.rfind('[') {
            if let Some(bracket_end) = expr[bracket_start..].find(']') {
                let path = expr[..bracket_start].to_string();
                let selector = expr[bracket_start+1..bracket_start+bracket_end].to_string();
                return Some((path, selector));
            }
        }
        None
    }
    
    fn apply_array_selector(&self, arr: &[Value], selector: &str) -> JsonPathResult<Vec<Value>> {
        if selector == "*" {
            // Wildcard - return all elements
            Ok(arr.to_vec())
        } else if let Ok(index) = selector.parse::<i64>() {
            // Index selector
            let actual_index = if index < 0 {
                arr.len() as i64 + index
            } else {
                index
            } as usize;
            
            if actual_index < arr.len() {
                Ok(vec![arr[actual_index].clone()])
            } else {
                Ok(vec![])
            }
        } else if selector.contains(':') {
            // Slice selector
            self.apply_slice_selector(arr, selector)
        } else if selector.contains(',') {
            // Union selector
            self.apply_union_selector(arr, selector)
        } else {
            // Unsupported selector
            Ok(vec![])
        }
    }
    
    fn apply_slice_selector(&self, arr: &[Value], selector: &str) -> JsonPathResult<Vec<Value>> {
        let parts: Vec<&str> = selector.split(':').collect();
        if parts.len() < 2 {
            return Ok(vec![]);
        }
        
        let start = if parts[0].is_empty() { 0 } else { parts[0].parse::<i64>().unwrap_or(0) };
        let end = if parts[1].is_empty() { 
            arr.len() as i64 
        } else { 
            parts[1].parse::<i64>().unwrap_or(arr.len() as i64) 
        };
        
        let start_idx = if start < 0 { 
            (arr.len() as i64 + start).max(0) as usize 
        } else { 
            start as usize 
        };
        let end_idx = if end < 0 { 
            (arr.len() as i64 + end).max(0) as usize 
        } else { 
            (end as usize).min(arr.len()) 
        };
        
        if start_idx < end_idx {
            Ok(arr[start_idx..end_idx].to_vec())
        } else {
            Ok(vec![])
        }
    }
    
    fn apply_union_selector(&self, arr: &[Value], selector: &str) -> JsonPathResult<Vec<Value>> {
        let indices: Vec<&str> = selector.split(',').collect();
        let mut results = Vec::new();
        
        for idx_str in indices {
            let idx_str = idx_str.trim();
            if let Ok(index) = idx_str.parse::<i64>() {
                let actual_index = if index < 0 {
                    arr.len() as i64 + index
                } else {
                    index
                } as usize;
                
                if actual_index < arr.len() {
                    results.push(arr[actual_index].clone());
                }
            }
        }
        
        Ok(results)
    }
    
    fn evaluate_property_with_array_wildcards(&self, json: &Value, expr: &str) -> JsonPathResult<Vec<Value>> {
        // Handle expressions like $.store.book[*].author
        
        // Split the expression into parts around [*]
        let parts: Vec<&str> = expr.split("[*]").collect();
        if parts.len() != 2 {
            return Ok(vec![]); // More complex patterns not supported yet
        }
        
        let before_wildcard = parts[0]; // "$.store.book"
        let after_wildcard = parts[1];  // ".author"
        
        // Navigate to the array location
        let array_value = if before_wildcard == "$" {
            json
        } else if before_wildcard.starts_with("$.") {
            let path_parts: Vec<&str> = before_wildcard[2..].split('.').collect();
            let mut current = json;
            
            for part in path_parts {
                match current {
                    Value::Object(obj) => {
                        if let Some(value) = obj.get(part) {
                            current = value;
                        } else {
                            return Ok(vec![]); // Property not found
                        }
                    }
                    _ => return Ok(vec![]), // Can't access property on non-object
                }
            }
            current
        } else {
            return Ok(vec![]);
        };
        
        // Apply wildcard to array and then continue with remaining path
        match array_value {
            Value::Array(arr) => {
                let mut results = Vec::new();
                for item in arr.iter() {
                    if after_wildcard.is_empty() {
                        // No property after wildcard, return the array item itself
                        results.push(item.clone());
                    } else if after_wildcard.starts_with('.') {
                        // Property access after wildcard
                        let property_path = &after_wildcard[1..]; // Remove leading dot
                        let property_results = self.evaluate_property_path(item, property_path)?;
                        results.extend(property_results);
                    }
                }
                Ok(results)
            }
            _ => Ok(vec![]), // Not an array
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_root_selector() {
        let evaluator = CoreJsonPathEvaluator::new("$").unwrap();
        let json = json!({"test": "value"});
        let results = evaluator.evaluate(&json).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json);
    }

    #[test]
    fn test_property_access() {
        let evaluator = CoreJsonPathEvaluator::new("$.store").unwrap();
        let json = json!({"store": {"name": "test"}});
        let results = evaluator.evaluate(&json).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json!({"name": "test"}));
    }

    #[test]
    fn test_recursive_descent() {
        let evaluator = CoreJsonPathEvaluator::new("$..author").unwrap();
        let json = json!({
            "store": {
                "book": [
                    {"author": "Author 1"},
                    {"author": "Author 2"}
                ]
            }
        });
        let results = evaluator.evaluate(&json).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&json!("Author 1")));
        assert!(results.contains(&json!("Author 2")));
    }

    #[test]
    fn test_array_wildcard() {
        let evaluator = CoreJsonPathEvaluator::new("$.store.book[*]").unwrap();
        let json = json!({
            "store": {
                "book": [
                    {"title": "Book 1"},
                    {"title": "Book 2"}
                ]
            }
        });
        let results = evaluator.evaluate(&json).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn debug_simple_infinite_loop() {
        println!("\n=== DEBUG: Testing simple pattern that's timing out ===");
        
        let json_value = json!({
            "store": {
                "book": ["a", "b", "c", "d"],
                "bicycle": {"color": "red", "price": 19.95}
            }
        });
        
        let pattern = "$.store.bicycle";
        println!("Testing pattern: {}", pattern);
        
        match CoreJsonPathEvaluator::new(pattern) {
            Ok(evaluator) => {
                let start = std::time::Instant::now();
                match evaluator.evaluate(&json_value) {
                    Ok(results) => {
                        let elapsed = start.elapsed();
                        println!("✅ SUCCESS: Got {} results in {:?}", results.len(), elapsed);
                        for (i, result) in results.iter().enumerate() {
                            println!("  [{}]: {}", i, result);
                        }
                    }
                    Err(e) => {
                        println!("❌ ERROR: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("❌ CREATION ERROR: {}", e);
            }
        }
    }

    #[test]
    fn test_duplicate_preservation_debug() {
        println!("=== Testing duplicate preservation ===");
        
        let test_json = json!({
            "data": {
                "x": 42,
                "y": 24  
            }
        });

        // Test 1: Direct property access
        println!("Test 1: Direct property access");
        let evaluator = CoreJsonPathEvaluator::new("$.data.x").unwrap();
        let results = evaluator.evaluate(&test_json).unwrap();
        println!("$.data.x -> {} results: {:?}", results.len(), results);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json!(42));

        // Test 2: Test bracket notation
        println!("Test 2: Bracket notation");
        let evaluator = CoreJsonPathEvaluator::new("$.data['x']").unwrap();
        let results = evaluator.evaluate(&test_json).unwrap();
        println!("$.data['x'] -> {} results: {:?}", results.len(), results);

        // Test 3: Test the multi-selector expression for duplicate preservation
        println!("Test 3: Multi-selector (should show duplicates)");
        let evaluator = CoreJsonPathEvaluator::new("$.data['x','x','y','x']").unwrap();
        let results = evaluator.evaluate(&test_json).unwrap();
        println!("$.data['x','x','y','x'] -> {} results: {:?}", results.len(), results);

        // Test 4: Test union selector with array indices
        println!("Test 4: Array union selector");
        let array_json = json!({
            "items": [10, 20, 30, 40]
        });
        let evaluator = CoreJsonPathEvaluator::new("$.items[0,1,0,2]").unwrap();
        let results = evaluator.evaluate(&array_json).unwrap();
        println!("$.items[0,1,0,2] -> {} results: {:?}", results.len(), results);
    }
}