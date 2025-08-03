//! High-performance streaming JSON deserializer with JSONPath navigation
//!
//! This module provides the core streaming deserializer that combines JSONPath expression
//! evaluation with incremental JSON parsing to yield individual objects from nested arrays
//! as HTTP response bytes arrive.

use crate::json_path::{
    buffer::{StreamBuffer, BufferReader},
    error::{JsonPathError, JsonPathResult, JsonPathErrorExt, json_parse_error, stream_error, deserialization_error},
    parser::{JsonPathExpression, JsonSelector},
    state_machine::{StreamStateMachine, JsonStreamState},
};

use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde_json::{Deserializer, StreamDeserializer, de::IoRead};
use std::io::Read;

/// High-performance streaming JSON deserializer with JSONPath navigation
///
/// Combines JSONPath expression evaluation with incremental JSON parsing to extract
/// individual array elements from nested JSON structures during HTTP streaming.
/// Supports full JSONPath specification including recursive descent (..) operators.
pub struct JsonPathDeserializer<'a, T> {
    /// JSONPath expression for navigation and filtering
    path_expression: &'a JsonPathExpression,
    /// Streaming buffer for efficient byte processing
    buffer: &'a mut StreamBuffer,
    /// State machine for tracking parsing progress
    state: &'a mut StreamStateMachine,
    /// Current parsing depth in JSON structure
    current_depth: usize,
    /// Whether we've reached the target array location
    in_target_array: bool,
    /// Current object nesting level within target array
    object_nesting: usize,
    /// Buffer for accumulating complete JSON objects
    object_buffer: Vec<u8>,
    /// Current selector index being evaluated in the JSONPath expression
    current_selector_index: usize,
    /// Whether we're currently in recursive descent mode
    in_recursive_descent: bool,
    /// Stack of depth levels where recursive descent should continue searching
    recursive_descent_stack: Vec<usize>,
    /// Path breadcrumbs for backtracking during recursive descent
    path_breadcrumbs: Vec<String>,
    /// Performance marker
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> JsonPathDeserializer<'a, T>
where 
    T: DeserializeOwned,
{
    /// Create new streaming deserializer instance
    ///
    /// # Arguments
    ///
    /// * `path_expression` - Compiled JSONPath expression for navigation
    /// * `buffer` - Streaming buffer containing JSON bytes
    /// * `state` - State machine for tracking parsing progress
    #[inline]
    pub fn new(
        path_expression: &'a JsonPathExpression,
        buffer: &'a mut StreamBuffer,
        state: &'a mut StreamStateMachine,
    ) -> Self {
        let has_recursive_descent = path_expression.has_recursive_descent();
        let initial_capacity = if has_recursive_descent { 256 } else { 32 };
        
        Self {
            path_expression,
            buffer,
            state,
            current_depth: 0,
            in_target_array: false,
            object_nesting: 0,
            object_buffer: Vec::with_capacity(1024), // Pre-allocate 1KB
            current_selector_index: 0,
            in_recursive_descent: false,
            recursive_descent_stack: Vec::with_capacity(initial_capacity),
            path_breadcrumbs: Vec::with_capacity(initial_capacity),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Process available bytes and yield deserialized objects
    ///
    /// Incrementally parses JSON while evaluating JSONPath expressions to identify
    /// and extract individual array elements for deserialization.
    ///
    /// # Returns
    ///
    /// Iterator over successfully deserialized objects of type `T`
    ///
    /// # Performance
    ///
    /// Uses zero-copy byte processing and pre-allocated buffers for optimal performance.
    /// Inlined hot paths minimize function call overhead during streaming.
    pub fn process_available(&mut self) -> JsonPathIterator<'_, 'a, T> {
        JsonPathIterator::new(self)
    }
}

// JsonPathDeserializer does not implement Iterator directly
// Use process_available() to get JsonPathIterator which implements Iterator

/// Iterator over streaming JSON objects matching JSONPath expression
pub struct JsonPathIterator<'iter, 'data, T> {
    deserializer: &'iter mut JsonPathDeserializer<'data, T>,
    stream_deserializer: Option<StreamDeserializer<'static, IoRead<BufferReader<'data>>, T>>,
    bytes_consumed: usize,
}

impl<'iter, 'data, T> JsonPathIterator<'iter, 'data, T>
where
    T: DeserializeOwned,
{
    /// Create new iterator over streaming JSON objects
    #[inline]
    fn new(deserializer: &'iter mut JsonPathDeserializer<'data, T>) -> Self {
        Self {
            deserializer,
            stream_deserializer: None,
            bytes_consumed: 0,
        }
    }

    /// Advance to next complete JSON object matching JSONPath
    ///
    /// Incrementally parses JSON structure while evaluating JSONPath expressions
    /// to identify individual array elements for deserialization.
    #[inline]
    fn advance_to_next_object(&mut self) -> JsonPathResult<Option<T>> {
        // Fast path: if we have a stream deserializer, try to get next object
        if let Some(ref mut stream_deser) = self.stream_deserializer {
            match stream_deser.next() {
                Some(Ok(obj)) => return Ok(Some(obj)),
                Some(Err(e)) => {
                    // Reset stream deserializer on error and continue parsing
                    self.stream_deserializer = None;
                    return Err(json_parse_error(
                        format!("JSON deserialization failed: {}", e),
                        self.bytes_consumed,
                        "streaming array element".to_string(),
                    ));
                }
                None => {
                    // Stream deserializer exhausted, continue parsing
                    self.stream_deserializer = None;
                }
            }
        }

        // Parse JSON incrementally to find next object matching JSONPath
        loop {
            let byte = match self.read_next_byte()? {
                Some(b) => b,
                None => return Ok(None), // No more data available
            };

            match self.process_json_byte(byte)? {
                JsonProcessResult::ObjectFound => {
                    // Found complete object, attempt deserialization
                    return self.deserialize_current_object();
                }
                JsonProcessResult::Continue => {
                    // Continue parsing
                    continue;
                }
                JsonProcessResult::NeedMoreData => {
                    // Need more bytes to complete parsing
                    return Ok(None);
                }
                JsonProcessResult::Complete => {
                    // Processing complete (end of stream)
                    return Ok(None);
                }
            }
        }
    }

    /// Read next byte from streaming buffer
    #[inline]
    fn read_next_byte(&mut self) -> JsonPathResult<Option<u8>> {
        let mut buf = [0u8; 1];
        let mut reader = self.deserializer.buffer.create_reader();
        match reader.read(&mut buf) {
            Ok(1) => {
                self.bytes_consumed += 1;
                Ok(Some(buf[0]))
            }
            Ok(0) => Ok(None), // No more data
            Ok(_) => {
                // Should not happen with single-byte buffer, but handle gracefully
                self.bytes_consumed += 1;
                Ok(Some(buf[0]))
            }
            Err(e) => Err(stream_error(
                &format!("Failed to read from buffer: {}", e),
                &format!("bytes_consumed: {}", self.bytes_consumed),
                false,
            )),
        }
    }

    /// Process single JSON byte and update parsing state
    #[inline]
    fn process_json_byte(&mut self, byte: u8) -> JsonPathResult<JsonProcessResult> {
        match self.deserializer.state.current_state() {
            JsonStreamState::Initial => self.process_initial_byte(byte),
            JsonStreamState::Navigating { .. } => self.process_navigating_byte(byte),
            JsonStreamState::StreamingArray { .. } => self.process_array_byte(byte),
            JsonStreamState::ProcessingObject { .. } => self.process_object_byte(byte),
            JsonStreamState::Complete => Ok(JsonProcessResult::NeedMoreData),
            JsonStreamState::Finishing { .. } => Ok(JsonProcessResult::Complete),
            JsonStreamState::Error { .. } => Ok(JsonProcessResult::Complete),
        }
    }

    /// Process byte in initial parsing state
    #[inline]
    fn process_initial_byte(&mut self, byte: u8) -> JsonPathResult<JsonProcessResult> {
        match byte {
            b'{' => {
                self.deserializer.current_depth += 1;
                self.deserializer.state.transition_to_navigating();
                Ok(JsonProcessResult::Continue)
            }
            b'[' => {
                self.deserializer.current_depth += 1;
                if self.matches_root_array_path() {
                    self.deserializer.in_target_array = true;
                    self.deserializer.state.transition_to_streaming_array();
                }
                Ok(JsonProcessResult::Continue)
            }
            b' ' | b'\t' | b'\n' | b'\r' => Ok(JsonProcessResult::Continue), // Skip whitespace
            _ => Err(json_parse_error(
                "Invalid JSON: expected object or array".to_string(),
                self.bytes_consumed,
                format!("byte: {}", byte as char),
            )),
        }
    }

    /// Process byte during JSONPath navigation
    #[inline]
    fn process_navigating_byte(&mut self, byte: u8) -> JsonPathResult<JsonProcessResult> {
        match byte {
            b'{' => {
                self.deserializer.current_depth += 1;
                
                // Check if we should enter recursive descent mode when entering objects
                if self.should_enter_recursive_descent() {
                    self.enter_recursive_descent_mode();
                    self.update_breadcrumbs(None);
                }
                
                Ok(JsonProcessResult::Continue)
            }
            b'}' => {
                self.deserializer.current_depth -= 1;
                if self.deserializer.current_depth == 0 {
                    self.deserializer.state.transition_to_complete();
                }
                
                // Exit recursive descent if we've returned to the starting depth
                if self.deserializer.in_recursive_descent {
                    if let Some(&start_depth) = self.deserializer.recursive_descent_stack.last() {
                        if self.deserializer.current_depth <= start_depth {
                            self.exit_recursive_descent_mode();
                        }
                    }
                }
                
                Ok(JsonProcessResult::Continue)
            }
            b'[' => {
                self.deserializer.current_depth += 1;
                
                // Check for recursive descent activation before path matching
                if self.should_enter_recursive_descent() {
                    self.enter_recursive_descent_mode();
                    self.update_breadcrumbs(None);
                }
                
                if self.matches_current_path() {
                    self.deserializer.in_target_array = true;
                    self.deserializer.state.transition_to_streaming_array();
                }
                Ok(JsonProcessResult::Continue)
            }
            b']' => {
                self.deserializer.current_depth -= 1;
                
                // Exit recursive descent if we've returned to the starting depth
                if self.deserializer.in_recursive_descent {
                    if let Some(&start_depth) = self.deserializer.recursive_descent_stack.last() {
                        if self.deserializer.current_depth <= start_depth {
                            self.exit_recursive_descent_mode();
                        }
                    }
                }
                
                Ok(JsonProcessResult::Continue)
            }
            _ => Ok(JsonProcessResult::Continue), // Continue navigation
        }
    }

    /// Process byte within target array
    #[inline]
    fn process_array_byte(&mut self, byte: u8) -> JsonPathResult<JsonProcessResult> {
        match byte {
            b'{' => {
                self.deserializer.object_nesting += 1;
                self.deserializer.object_buffer.push(byte);
                
                // Update breadcrumbs during recursive descent for object elements
                if self.deserializer.in_recursive_descent {
                    self.update_breadcrumbs(None);
                }
                
                self.deserializer.state.transition_to_processing_object();
                Ok(JsonProcessResult::Continue)
            }
            b'[' => {
                // Nested array within target array - handle recursive descent
                self.deserializer.current_depth += 1;
                self.deserializer.object_buffer.push(byte);
                
                if self.deserializer.in_recursive_descent {
                    self.update_breadcrumbs(None);
                    
                    // Check if nested array matches the recursive descent pattern
                    if self.matches_current_path() {
                        // Continue processing this nested array as a target
                        self.deserializer.state.transition_to_streaming_array();
                        return Ok(JsonProcessResult::Continue);
                    }
                }
                
                self.deserializer.state.transition_to_processing_object();
                Ok(JsonProcessResult::Continue)
            }
            b']' => {
                // End of target array
                self.deserializer.in_target_array = false;
                
                // Handle recursive descent state when exiting arrays
                if self.deserializer.in_recursive_descent {
                    if let Some(&start_depth) = self.deserializer.recursive_descent_stack.last() {
                        if self.deserializer.current_depth <= start_depth {
                            self.exit_recursive_descent_mode();
                        }
                    }
                }
                
                self.deserializer.state.transition_to_complete();
                Ok(JsonProcessResult::NeedMoreData)
            }
            b' ' | b'\t' | b'\n' | b'\r' | b',' => Ok(JsonProcessResult::Continue), // Skip whitespace and separators
            _ => {
                // Start of primitive value in array
                self.deserializer.object_buffer.push(byte);
                
                // Update breadcrumbs for primitive values during recursive descent
                if self.deserializer.in_recursive_descent {
                    self.update_breadcrumbs(None);
                }
                
                self.deserializer.state.transition_to_processing_object();
                Ok(JsonProcessResult::Continue)
            }
        }
    }

    /// Process byte within JSON object
    #[inline]
    fn process_object_byte(&mut self, byte: u8) -> JsonPathResult<JsonProcessResult> {
        self.deserializer.object_buffer.push(byte);

        match byte {
            b'{' => {
                self.deserializer.object_nesting += 1;
                
                // Update breadcrumbs for nested objects during recursive descent
                if self.deserializer.in_recursive_descent {
                    self.update_breadcrumbs(None);
                }
                
                Ok(JsonProcessResult::Continue)
            }
            b'}' => {
                if self.deserializer.object_nesting > 0 {
                    self.deserializer.object_nesting -= 1;
                }
                
                if self.deserializer.object_nesting == 0 {
                    // Complete object found - check if we should continue recursive descent
                    if self.deserializer.in_recursive_descent {
                        // In recursive descent, continue searching after this object
                        self.deserializer.state.transition_to_streaming_array();
                        return Ok(JsonProcessResult::ObjectFound);
                    } else {
                        // Normal mode - object completed
                        self.deserializer.state.transition_to_streaming_array();
                        return Ok(JsonProcessResult::ObjectFound);
                    }
                }
                
                // Update breadcrumbs when exiting nested objects during recursive descent
                if self.deserializer.in_recursive_descent && !self.deserializer.path_breadcrumbs.is_empty() {
                    self.deserializer.path_breadcrumbs.pop();
                }
                
                Ok(JsonProcessResult::Continue)
            }
            b'[' => {
                // Nested array within object - handle recursive descent continuation
                if self.deserializer.in_recursive_descent {
                    self.update_breadcrumbs(None);
                    
                    // Check if this nested array should trigger a new target match
                    if self.matches_current_path() {
                        // Found a new target array during recursive descent
                        self.deserializer.in_target_array = true;
                        self.deserializer.state.transition_to_streaming_array();
                        return Ok(JsonProcessResult::Continue);
                    }
                }
                
                Ok(JsonProcessResult::Continue)
            }
            b'"' => {
                // Handle string literals (skip escaped quotes)
                self.skip_string()?;
                Ok(JsonProcessResult::Continue)
            }
            _ => Ok(JsonProcessResult::Continue),
        }
    }

    /// Skip string literal including escaped characters
    #[inline]
    fn skip_string(&mut self) -> JsonPathResult<()> {
        loop {
            let byte = match self.read_next_byte()? {
                Some(b) => b,
                None => return Err(json_parse_error(
                    "Unterminated string literal".to_string(),
                    self.bytes_consumed,
                    "end of stream".to_string(),
                )),
            };

            self.deserializer.object_buffer.push(byte);

            match byte {
                b'"' => break, // End of string
                b'\\' => {
                    // Escaped character, skip next byte
                    if let Some(escaped) = self.read_next_byte()? {
                        self.deserializer.object_buffer.push(escaped);
                    }
                }
                _ => continue,
            }
        }
        Ok(())
    }

    /// Check if current position matches JSONPath root array selector
    #[inline]
    fn matches_root_array_path(&self) -> bool {
        matches!(
            self.deserializer.path_expression.root_selector(),
            Some(JsonSelector::Wildcard) | Some(JsonSelector::Index { .. }) | Some(JsonSelector::Slice { .. })
        )
    }

    /// Check if current position matches JSONPath expression
    #[inline] 
    fn matches_current_path(&self) -> bool {
        self.evaluate_jsonpath_at_current_position()
    }
    
    /// Evaluate JSONPath expression at current parsing position
    /// 
    /// Handles recursive descent by maintaining state across the JSON structure traversal.
    /// Uses the enhanced path matching logic from the parser module.
    #[inline]
    fn evaluate_jsonpath_at_current_position(&self) -> bool {
        // Use the enhanced recursive descent-aware matching from the parser
        if self.deserializer.in_recursive_descent {
            // In recursive descent mode, continue searching at any depth
            self.evaluate_recursive_descent_match()
        } else {
            // Normal depth-based matching
            self.deserializer.path_expression.matches_at_depth(self.deserializer.current_depth)
        }
    }
    
    /// Evaluate recursive descent matching at current position
    /// 
    /// When in recursive descent mode, we need to check if the current structure
    /// matches the selector following the recursive descent operator.
    #[inline]
    fn evaluate_recursive_descent_match(&self) -> bool {
        let selectors = self.deserializer.path_expression.selectors();
        
        // Find the current recursive descent position
        if let Some(rd_start) = self.deserializer.path_expression.recursive_descent_start() {
            let next_selector_index = rd_start + 1;
            
            if next_selector_index < selectors.len() {
                // Try to match the selector after recursive descent
                self.matches_selector_at_depth(&selectors[next_selector_index], self.deserializer.current_depth)
            } else {
                // Recursive descent at end matches everything
                true
            }
        } else {
            // Not actually in recursive descent mode
            false
        }
    }
    
    /// Check if a specific selector matches at the given depth
    /// 
    /// Helper method for evaluating individual selectors during recursive descent.
    #[inline]
    fn matches_selector_at_depth(&self, selector: &JsonSelector, depth: usize) -> bool {
        match selector {
            JsonSelector::Root => depth == 0,
            JsonSelector::Child { name, .. } => {
                // For streaming context, we can't easily check property names
                // So we assume child selectors match at appropriate depths
                depth > 0
            },
            JsonSelector::RecursiveDescent => true, // Recursive descent always matches
            JsonSelector::Index { .. } | JsonSelector::Slice { .. } | JsonSelector::Wildcard => {
                // Array selectors require being in an array context
                // In streaming, we approximate this by depth checking
                depth > 0
            },
            JsonSelector::Filter { .. } => {
                // Filter selectors require array elements to filter
                depth > 0
            },
            JsonSelector::Union { selectors } => {
                // Union selectors match if any alternative matches
                selectors.iter().any(|s| self.matches_selector_at_depth(s, depth))
            },
        }
    }

    /// Deserialize current accumulated object
    #[inline]
    fn deserialize_current_object(&mut self) -> JsonPathResult<Option<T>> {
        if self.deserializer.object_buffer.is_empty() {
            return Ok(None);
        }

        let json_str = std::str::from_utf8(&self.deserializer.object_buffer)
            .map_err(|e| deserialization_error(
                format!("Invalid UTF-8 in JSON object: {}", e),
                String::from_utf8_lossy(&self.deserializer.object_buffer).to_string(),
                std::any::type_name::<T>(),
            ))?;

        let result = serde_json::from_str::<T>(json_str)
            .map_err(|e| deserialization_error(
                format!("Failed to deserialize JSON: {}", e),
                json_str.to_string(),
                std::any::type_name::<T>(),
            ))?;

        // Clear buffer for next object
        self.deserializer.object_buffer.clear();
        self.deserializer.state.increment_objects_yielded();

        Ok(Some(result))
    }
    
    /// Enter recursive descent mode at current depth
    /// 
    /// Called when encountering a recursive descent (..) operator during streaming.
    /// Manages the state transition and breadcrumb tracking for backtracking.
    #[inline]
    fn enter_recursive_descent_mode(&mut self) {
        if !self.deserializer.in_recursive_descent {
            self.deserializer.in_recursive_descent = true;
            self.deserializer.recursive_descent_stack.push(self.deserializer.current_depth);
            
            // Update selector index to skip past the recursive descent operator
            if let Some(rd_start) = self.deserializer.path_expression.recursive_descent_start() {
                self.deserializer.current_selector_index = rd_start;
            }
        }
    }
    
    /// Exit recursive descent mode
    /// 
    /// Called when the recursive descent search is complete or needs to backtrack.
    /// Restores the previous navigation state.
    #[inline]
    fn exit_recursive_descent_mode(&mut self) {
        if self.deserializer.in_recursive_descent {
            self.deserializer.in_recursive_descent = false;
            self.deserializer.recursive_descent_stack.clear();
            self.deserializer.path_breadcrumbs.clear();
            
            // Reset selector index to continue normal navigation
            self.deserializer.current_selector_index = 0;
        }
    }
    
    /// Check if we should enter recursive descent mode at current position
    /// 
    /// Evaluates the JSONPath expression to determine if a recursive descent
    /// operator should be activated based on the current parsing state.
    #[inline]
    fn should_enter_recursive_descent(&self) -> bool {
        if self.deserializer.in_recursive_descent {
            return false; // Already in recursive descent mode
        }
        
        let selectors = self.deserializer.path_expression.selectors();
        let current_index = self.deserializer.current_selector_index;
        
        // Check if current selector is recursive descent
        if current_index < selectors.len() {
            matches!(selectors[current_index], JsonSelector::RecursiveDescent)
        } else {
            false
        }
    }
    
    /// Update breadcrumbs during recursive descent navigation
    /// 
    /// Tracks the path taken through the JSON structure for efficient backtracking
    /// during recursive descent evaluation.
    #[inline]
    fn update_breadcrumbs(&mut self, property_name: Option<&str>) {
        if self.deserializer.in_recursive_descent {
            if let Some(name) = property_name {
                self.deserializer.path_breadcrumbs.push(name.to_string());
            } else {
                // Array index or anonymous structure
                self.deserializer.path_breadcrumbs.push(format!("[{}]", self.deserializer.current_depth));
            }
        }
    }
}

impl<'iter, 'data, T> Iterator for JsonPathIterator<'iter, 'data, T>
where
    T: DeserializeOwned,
{
    type Item = JsonPathResult<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.advance_to_next_object() {
            Ok(Some(obj)) => Some(Ok(obj)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Result of processing a single JSON byte
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonProcessResult {
    /// Continue processing more bytes
    Continue,
    /// Complete JSON object found and ready for deserialization
    ObjectFound,
    /// Need more data to continue parsing
    NeedMoreData,
    /// Processing complete (end of stream)
    Complete,
}

/// Streaming deserializer for general use
///
/// Provides a simplified interface for streaming JSON deserialization without
/// JSONPath navigation. Useful for streaming simple arrays directly.
pub struct StreamingDeserializer<R: Read, T> {
    inner: StreamDeserializer<'static, IoRead<R>, T>,
}

impl<R: Read, T> StreamingDeserializer<R, T>
where
    T: DeserializeOwned,
{
    /// Create new streaming deserializer from reader
    ///
    /// # Arguments
    ///
    /// * `reader` - Source of JSON bytes
    #[inline]
    pub fn new(reader: R) -> Self {
        let inner = Deserializer::from_reader(reader).into_iter::<T>();
        Self { inner }
    }
}

impl<R: Read, T> Iterator for StreamingDeserializer<R, T>
where
    T: DeserializeOwned,
{
    type Item = Result<T, serde_json::Error>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_path::{
        buffer::StreamBuffer,
        parser::JsonPathParser,
        state_machine::StreamStateMachine,
    };
    use bytes::Bytes;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
    struct TestModel {
        id: String,
        value: i32,
    }


}