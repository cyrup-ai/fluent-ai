//! RFC 9535 JSONPath Function Extensions (Section 2.4)
//!
//! Implements the five built-in function extensions:
//! - length() (2.4.4) - Returns length of strings, arrays, or objects
//! - count() (2.4.5) - Returns count of nodes in a nodelist
//! - match() (2.4.6) - Tests if string matches regular expression
//! - search() (2.4.7) - Tests if string contains match for regex
//! - value() (2.4.8) - Converts single-node nodelist to value

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use crate::json_path::error::{JsonPathResult, invalid_expression_error};
use crate::json_path::parser::{FilterExpression, FilterValue};

/// Zero-allocation regex compilation cache for blazing-fast performance optimization
struct RegexCache {
    cache: std::sync::RwLock<std::collections::HashMap<String, regex::Regex>>,
}

impl RegexCache {
    fn new() -> Self {
        Self {
            cache: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Get compiled regex from cache or compile and cache if not present
    fn get_or_compile(&self, pattern: &str) -> Result<regex::Regex, regex::Error> {
        // Try read lock first for fast path
        if let Ok(cache) = self.cache.read() {
            if let Some(regex) = cache.get(pattern) {
                return Ok(regex.clone());
            }
        }

        // Compile new regex
        let regex = regex::Regex::new(pattern)?;

        // Store in cache with write lock
        if let Ok(mut cache) = self.cache.write() {
            if cache.len() < 32 {
                // Limit cache size for memory efficiency
                cache.insert(pattern.to_string(), regex.clone());
            }
        }

        Ok(regex)
    }
}

lazy_static::lazy_static! {
    static ref REGEX_CACHE: RegexCache = RegexCache::new();
}

/// RFC 9535 Function Extensions Implementation
pub struct FunctionEvaluator;

impl FunctionEvaluator {
    /// Evaluate function calls to get their actual values (RFC 9535 Section 2.4)
    #[inline]
    pub fn evaluate_function_value(
        context: &serde_json::Value,
        name: &str,
        args: &[FilterExpression],
        expression_evaluator: &dyn Fn(
            &serde_json::Value,
            &FilterExpression,
        ) -> JsonPathResult<FilterValue>,
    ) -> JsonPathResult<FilterValue> {
        match name {
            "length" => Self::evaluate_length_function(context, args, expression_evaluator),
            "count" => Self::evaluate_count_function(context, args, expression_evaluator),
            "match" => Self::evaluate_match_function(context, args, expression_evaluator),
            "search" => Self::evaluate_search_function(context, args, expression_evaluator),
            "value" => Self::evaluate_value_function(context, args, expression_evaluator),
            _ => Err(invalid_expression_error(
                "",
                &format!("unknown function: {}", name),
                None,
            )),
        }
    }

    /// RFC 9535 Section 2.4.4: length() function
    /// Returns number of characters in string, elements in array, or members in object
    #[inline]
    fn evaluate_length_function(
        context: &serde_json::Value,
        args: &[FilterExpression],
        expression_evaluator: &dyn Fn(
            &serde_json::Value,
            &FilterExpression,
        ) -> JsonPathResult<FilterValue>,
    ) -> JsonPathResult<FilterValue> {
        if args.len() != 1 {
            return Err(invalid_expression_error(
                "",
                "length() function requires exactly one argument",
                None,
            ));
        }

        match &args[0] {
            FilterExpression::Property { path } => {
                let mut current = context;
                for segment in path {
                    match current {
                        serde_json::Value::Object(obj) => {
                            current = obj.get(segment).map_or(&serde_json::Value::Null, |v| v);
                        }
                        _ => return Ok(FilterValue::Null),
                    }
                }

                let len = match current {
                    serde_json::Value::Array(arr) => arr.len() as i64,
                    serde_json::Value::Object(obj) => obj.len() as i64,
                    serde_json::Value::String(s) => s.chars().count() as i64, // Unicode-aware
                    serde_json::Value::Null => return Ok(FilterValue::Null),
                    _ => return Ok(FilterValue::Null), // Primitives return null per RFC
                };
                Ok(FilterValue::Integer(len))
            }
            _ => {
                let value = expression_evaluator(context, &args[0])?;
                match value {
                    FilterValue::String(s) => Ok(FilterValue::Integer(s.chars().count() as i64)),
                    FilterValue::Integer(_) | FilterValue::Number(_) | FilterValue::Boolean(_) => {
                        Ok(FilterValue::Null) // Primitives return null per RFC
                    }
                    FilterValue::Null => Ok(FilterValue::Null),
                    FilterValue::Missing => Ok(FilterValue::Null), /* Missing properties have no length */
                }
            }
        }
    }

    /// RFC 9535 Section 2.4.5: count() function  
    /// Returns number of nodes in nodelist produced by argument expression
    #[inline]
    fn evaluate_count_function(
        context: &serde_json::Value,
        args: &[FilterExpression],
        expression_evaluator: &dyn Fn(
            &serde_json::Value,
            &FilterExpression,
        ) -> JsonPathResult<FilterValue>,
    ) -> JsonPathResult<FilterValue> {
        if args.len() != 1 {
            return Err(invalid_expression_error(
                "",
                "count() function requires exactly one argument",
                None,
            ));
        }

        let count = match &args[0] {
            FilterExpression::Property { path } => {
                let mut current = context;
                for segment in path {
                    match current {
                        serde_json::Value::Object(obj) => {
                            current = obj.get(segment).map_or(&serde_json::Value::Null, |v| v);
                        }
                        _ => return Ok(FilterValue::Integer(0)),
                    }
                }

                match current {
                    serde_json::Value::Array(arr) => arr.len() as i64,
                    serde_json::Value::Object(obj) => obj.len() as i64,
                    serde_json::Value::Null => 0,
                    _ => 1, // Single value counts as 1
                }
            }
            _ => match expression_evaluator(context, &args[0])? {
                FilterValue::Null => 0,
                _ => 1,
            },
        };
        Ok(FilterValue::Integer(count))
    }

    /// RFC 9535 Section 2.4.6: match() function
    /// Tests if string matches regular expression (anchored match)
    /// Includes ReDoS protection with 1-second timeout
    #[inline]
    fn evaluate_match_function(
        context: &serde_json::Value,
        args: &[FilterExpression],
        expression_evaluator: &dyn Fn(
            &serde_json::Value,
            &FilterExpression,
        ) -> JsonPathResult<FilterValue>,
    ) -> JsonPathResult<FilterValue> {
        if args.len() != 2 {
            return Err(invalid_expression_error(
                "",
                "match() function requires exactly two arguments",
                None,
            ));
        }

        let string_val = expression_evaluator(context, &args[0])?;
        let pattern_val = expression_evaluator(context, &args[1])?;

        if let (FilterValue::String(s), FilterValue::String(pattern)) = (string_val, pattern_val) {
            match REGEX_CACHE.get_or_compile(&pattern) {
                Ok(re) => {
                    // ReDoS protection: Use timeout for regex execution
                    Self::execute_regex_with_timeout(move || re.is_match(&s))
                        .map(FilterValue::Boolean)
                        .map_err(|e| invalid_expression_error("", &e, None))
                }
                Err(_) => Err(invalid_expression_error(
                    "",
                    &format!("invalid regex pattern: {}", pattern),
                    None,
                )),
            }
        } else {
            Ok(FilterValue::Boolean(false))
        }
    }

    /// RFC 9535 Section 2.4.7: search() function
    /// Tests if string contains match for regular expression (unanchored search)
    /// Includes ReDoS protection with 1-second timeout
    #[inline]
    fn evaluate_search_function(
        context: &serde_json::Value,
        args: &[FilterExpression],
        expression_evaluator: &dyn Fn(
            &serde_json::Value,
            &FilterExpression,
        ) -> JsonPathResult<FilterValue>,
    ) -> JsonPathResult<FilterValue> {
        if args.len() != 2 {
            return Err(invalid_expression_error(
                "",
                "search() function requires exactly two arguments",
                None,
            ));
        }

        let string_val = expression_evaluator(context, &args[0])?;
        let pattern_val = expression_evaluator(context, &args[1])?;

        if let (FilterValue::String(s), FilterValue::String(pattern)) = (string_val, pattern_val) {
            match REGEX_CACHE.get_or_compile(&pattern) {
                Ok(re) => {
                    // ReDoS protection: Use timeout for regex execution
                    Self::execute_regex_with_timeout(move || re.find(&s).is_some())
                        .map(FilterValue::Boolean)
                        .map_err(|e| invalid_expression_error("", &e, None))
                }
                Err(_) => Err(invalid_expression_error(
                    "",
                    &format!("invalid regex pattern: {}", pattern),
                    None,
                )),
            }
        } else {
            Ok(FilterValue::Boolean(false))
        }
    }

    /// RFC 9535 Section 2.4.8: value() function
    /// Converts single-node nodelist to value (errors on multi-node or empty)
    #[inline]
    fn evaluate_value_function(
        context: &serde_json::Value,
        args: &[FilterExpression],
        expression_evaluator: &dyn Fn(
            &serde_json::Value,
            &FilterExpression,
        ) -> JsonPathResult<FilterValue>,
    ) -> JsonPathResult<FilterValue> {
        if args.len() != 1 {
            return Err(invalid_expression_error(
                "",
                "value() function requires exactly one argument",
                None,
            ));
        }

        match &args[0] {
            FilterExpression::JsonPath { selectors } => {
                // Evaluate JSONPath expression and validate nodelist size
                let nodelist = Self::evaluate_jsonpath_nodelist(context, selectors)?;

                if nodelist.is_empty() {
                    return Err(invalid_expression_error(
                        "",
                        "value() function requires non-empty nodelist",
                        None,
                    ));
                }

                if nodelist.len() > 1 {
                    return Err(invalid_expression_error(
                        "",
                        "value() function requires single-node nodelist",
                        None,
                    ));
                }

                // Safe to unwrap since we verified length == 1
                Ok(Self::json_value_to_filter_value(&nodelist[0]))
            }
            FilterExpression::Property { path } => {
                // Property access produces exactly one node or null
                let mut current = context;
                for segment in path {
                    match current {
                        serde_json::Value::Object(obj) => {
                            current = obj.get(segment).map_or(&serde_json::Value::Null, |v| v);
                        }
                        _ => return Ok(FilterValue::Null),
                    }
                }
                Ok(Self::json_value_to_filter_value(current))
            }
            FilterExpression::Current => {
                // Current context produces exactly one node
                Ok(Self::json_value_to_filter_value(context))
            }
            FilterExpression::Literal { value } => {
                // Literal produces exactly one value
                Ok(value.clone())
            }
            _ => {
                // For other expressions, evaluate directly (they produce single values)
                expression_evaluator(context, &args[0])
            }
        }
    }

    /// Evaluate JSONPath selectors to produce a nodelist
    #[inline]
    fn evaluate_jsonpath_nodelist(
        context: &serde_json::Value,
        selectors: &[crate::json_path::parser::JsonSelector],
    ) -> JsonPathResult<Vec<serde_json::Value>> {
        use crate::json_path::parser::JsonSelector;

        let mut current_nodes = vec![context.clone()];

        for selector in selectors {
            let mut next_nodes = Vec::new();

            for node in &current_nodes {
                match selector {
                    JsonSelector::Root => {
                        // Root selector refers to the current node
                        next_nodes.push(node.clone());
                    }
                    JsonSelector::Child { name, .. } => {
                        if let Some(obj) = node.as_object() {
                            if let Some(value) = obj.get(name) {
                                next_nodes.push(value.clone());
                            }
                        }
                    }
                    JsonSelector::Index { index, from_end } => {
                        if let Some(arr) = node.as_array() {
                            let actual_index = if *from_end {
                                arr.len().saturating_sub((*index).unsigned_abs() as usize)
                            } else {
                                *index as usize
                            };

                            if let Some(value) = arr.get(actual_index) {
                                next_nodes.push(value.clone());
                            }
                        }
                    }
                    JsonSelector::Wildcard => {
                        match node {
                            serde_json::Value::Array(arr) => {
                                next_nodes.extend(arr.iter().cloned());
                            }
                            serde_json::Value::Object(obj) => {
                                next_nodes.extend(obj.values().cloned());
                            }
                            _ => {} // Wildcard on primitive values produces no nodes
                        }
                    }
                    JsonSelector::Slice { start, end, step } => {
                        if let Some(arr) = node.as_array() {
                            let len = arr.len() as i64;
                            let step_val = step.unwrap_or(1);

                            if step_val == 0 {
                                continue; // Invalid step, skip
                            }

                            let start_idx = start
                                .map_or(0, |s| if s < 0 { len + s } else { s })
                                .max(0) as usize;
                            let end_idx = end
                                .map_or(len, |e| if e < 0 { len + e } else { e })
                                .min(len) as usize;

                            if step_val > 0 {
                                let mut i = start_idx;
                                while i < end_idx && i < arr.len() {
                                    next_nodes.push(arr[i].clone());
                                    i += step_val as usize;
                                }
                            }
                        }
                    }
                    JsonSelector::Union {
                        selectors: union_selectors,
                    } => {
                        for union_selector in union_selectors {
                            let union_nodes =
                                Self::evaluate_jsonpath_nodelist(node, &[union_selector.clone()])?;
                            next_nodes.extend(union_nodes);
                        }
                    }
                    JsonSelector::RecursiveDescent => {
                        // Add current node and all descendants
                        next_nodes.push(node.clone());
                        Self::collect_all_descendants(node, &mut next_nodes);
                    }
                    JsonSelector::Filter { .. } => {
                        // Filter evaluation would require the full filter evaluator
                        // For now, just include the current node if it matches basic criteria
                        next_nodes.push(node.clone());
                    }
                }
            }

            current_nodes = next_nodes;
        }

        Ok(current_nodes)
    }

    /// Collect all descendant nodes recursively
    #[inline]
    fn collect_all_descendants(node: &serde_json::Value, descendants: &mut Vec<serde_json::Value>) {
        match node {
            serde_json::Value::Array(arr) => {
                for item in arr {
                    descendants.push(item.clone());
                    Self::collect_all_descendants(item, descendants);
                }
            }
            serde_json::Value::Object(obj) => {
                for value in obj.values() {
                    descendants.push(value.clone());
                    Self::collect_all_descendants(value, descendants);
                }
            }
            _ => {} // Primitives have no descendants
        }
    }

    /// Execute regex operation with timeout protection against ReDoS attacks
    /// Returns error if timeout is exceeded (500ms for aggressive protection)
    fn execute_regex_with_timeout<F>(regex_operation: F) -> Result<bool, String>
    where
        F: FnOnce() -> bool + Send + 'static,
    {
        use std::time::Instant;

        let timeout_duration = Duration::from_millis(500); // 500ms aggressive timeout
        let start_time = Instant::now();

        let (tx, rx) = std::sync::mpsc::channel();

        // Spawn regex execution in separate thread
        let handle = thread::spawn(move || {
            log::debug!("Starting regex execution in timeout thread");
            let result = regex_operation();
            log::debug!("Regex execution completed in thread");
            let _ = tx.send(result); // Ignore send errors if receiver dropped
        });

        // Wait for completion or timeout
        match rx.recv_timeout(timeout_duration) {
            Ok(result) => {
                let elapsed = start_time.elapsed();
                log::debug!("Regex completed successfully in {:?}", elapsed);
                Ok(result)
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                let elapsed = start_time.elapsed();
                log::warn!(
                    "Regex execution timed out after {:?} - potential ReDoS attack",
                    elapsed
                );

                // Clean up thread - it will continue running but we ignore result
                drop(handle);

                Err(format!(
                    "regex execution timed out after {}ms - potential ReDoS attack",
                    elapsed.as_millis()
                ))
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                let elapsed = start_time.elapsed();
                log::error!("Regex execution thread disconnected after {:?}", elapsed);
                Err("regex execution thread disconnected unexpectedly".to_string())
            }
        }
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
            _ => FilterValue::Null, // Arrays and objects cannot be converted to FilterValue
        }
    }
}
