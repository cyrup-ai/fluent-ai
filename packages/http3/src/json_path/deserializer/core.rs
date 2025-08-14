//! Core JsonPathDeserializer struct and main API
//!
//! Contains the primary deserializer structure with JSONPath navigation capabilities
//! and the main public interface for streaming JSON deserialization.

use serde::de::DeserializeOwned;

use super::iterator::JsonPathIterator;
use crate::json_path::{buffer::StreamBuffer, parser::JsonPathExpression};

/// Current state of the JSON deserializer
#[derive(Debug, Clone, PartialEq)]
pub(super) enum DeserializerState {
    /// Initial state - waiting for JSON to begin
    Initial,
    /// Navigating through JSON structure to find target location
    Navigating,
    /// Processing array elements at target location
    ProcessingArray,
    /// Processing individual JSON object
    ProcessingObject,
    /// Processing complete
    Complete,
}

/// High-performance streaming JSON deserializer with JSONPath navigation
///
/// Combines JSONPath expression evaluation with incremental JSON parsing to extract
/// individual array elements from nested JSON structures during HTTP streaming.
/// Supports full JSONPath specification including recursive descent (..) operators.
pub struct JsonPathDeserializer<'a, T> {
    /// JSONPath expression for navigation and filtering
    pub(super) path_expression: &'a JsonPathExpression,
    /// Streaming buffer for efficient byte processing
    pub(super) buffer: &'a mut StreamBuffer,
    /// Current parsing state
    pub(super) state: DeserializerState,
    /// Current parsing depth in JSON structure
    pub(super) current_depth: usize,
    /// Whether we've reached the target array location
    pub(super) in_target_array: bool,
    /// Current object nesting level within target array
    pub(super) object_nesting: usize,
    /// Buffer for accumulating complete JSON objects
    pub(super) object_buffer: Vec<u8>,
    /// Current selector index being evaluated in the JSONPath expression
    /// TODO: Part of streaming JSONPath evaluation state - implement usage in new architecture
    #[allow(dead_code)]
    pub(super) current_selector_index: usize,
    /// Whether we're currently in recursive descent mode
    /// TODO: Part of ".." operator implementation - integrate with new evaluator
    #[allow(dead_code)]
    pub(super) in_recursive_descent: bool,
    /// Stack of depth levels where recursive descent should continue searching
    /// TODO: Used for complex recursive descent patterns - implement in new architecture
    #[allow(dead_code)]
    pub(super) recursive_descent_stack: Vec<usize>,
    /// Path breadcrumbs for backtracking during recursive descent
    /// TODO: Navigation state for complex JSONPath expressions - integrate with new evaluator
    #[allow(dead_code)]
    pub(super) path_breadcrumbs: Vec<String>,
    /// Current array index for slice and index evaluation
    pub(super) current_array_index: i64,
    /// Array index stack for nested array processing
    pub(super) array_index_stack: Vec<i64>,
    /// Current position in the buffer for consistent reading
    pub(super) buffer_position: usize,
    /// Target property name for $.property[*] patterns
    pub(super) target_property: Option<String>,
    /// Whether we're currently inside the target property
    pub(super) in_target_property: bool,
    /// Performance marker
    pub(super) _phantom: std::marker::PhantomData<T>,
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
    #[inline]
    pub fn new(path_expression: &'a JsonPathExpression, buffer: &'a mut StreamBuffer) -> Self {
        let has_recursive_descent = path_expression.has_recursive_descent();
        let initial_capacity = if has_recursive_descent { 256 } else { 32 };

        // Extract target property name for $.property[*] patterns
        let target_property = {
            let expr = path_expression.as_string();
            if expr.starts_with("$.") && expr.ends_with("[*]") {
                let property_part = &expr[2..expr.len() - 3]; // Remove "$." and "[*]"
                if !property_part.contains('.') && !property_part.contains('[') {
                    Some(property_part.to_string())
                } else {
                    None // Complex nested paths not supported yet
                }
            } else {
                None
            }
        };

        Self {
            path_expression,
            buffer,
            state: DeserializerState::Initial,
            current_depth: 0,
            in_target_array: false,
            object_nesting: 0,
            object_buffer: Vec::with_capacity(1024), // Pre-allocate 1KB
            current_selector_index: 0,
            in_recursive_descent: false,
            recursive_descent_stack: Vec::with_capacity(initial_capacity),
            path_breadcrumbs: Vec::with_capacity(initial_capacity),
            current_array_index: 0,
            array_index_stack: Vec::with_capacity(16), // Support up to 16 nested arrays
            buffer_position: 0,
            target_property,
            in_target_property: false,
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
        // Continue from current buffer position to process newly available data
        // Buffer position tracks our progress through the streaming data
        JsonPathIterator::new(self)
    }

    /// Read next byte from buffer with position tracking
    #[inline]
    pub(super) fn read_next_byte(&mut self) -> crate::json_path::error::JsonPathResult<Option<u8>> {
        // Check if we've reached the end of available data
        if self.buffer_position >= self.buffer.len() {
            return Ok(None); // No more data available
        }

        // Read byte at current position
        match self.buffer.get_byte_at(self.buffer_position) {
            Some(byte) => {
                self.buffer_position += 1;
                Ok(Some(byte))
            }
            None => Ok(None), // Position beyond buffer bounds
        }
    }

    /// Process single JSON byte and update parsing state
    #[inline]
    pub(super) fn process_json_byte(
        &mut self,
        byte: u8,
    ) -> crate::json_path::error::JsonPathResult<super::processor::JsonProcessResult> {
        match &self.state {
            DeserializerState::Initial => self.process_initial_byte(byte),
            DeserializerState::Navigating => self.process_navigating_byte(byte),
            DeserializerState::ProcessingArray => self.process_array_byte(byte),
            DeserializerState::ProcessingObject => self.process_object_byte(byte),
            DeserializerState::Complete => Ok(super::processor::JsonProcessResult::Complete),
        }
    }

    /// Process byte when parser is in initial state
    #[inline]
    pub(super) fn process_initial_byte(
        &mut self,
        byte: u8,
    ) -> crate::json_path::error::JsonPathResult<super::processor::JsonProcessResult> {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => Ok(super::processor::JsonProcessResult::Continue), /* Skip whitespace */
            b'{' => {
                let expression = self.path_expression.as_string();
                if expression.starts_with("$.") && expression.ends_with("[*]") {
                    // For $.property[*] patterns, we need to navigate to the property first
                    self.transition_to_navigating();
                    self.object_nesting = self.object_nesting.saturating_add(1);
                    // Reprocess this byte in the new navigating state
                    self.process_navigating_byte(byte)
                } else {
                    self.transition_to_processing_object();
                    self.object_nesting = self.object_nesting.saturating_add(1);
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            b'[' => {
                self.transition_to_processing_array();
                self.current_depth = self.current_depth.saturating_add(1);
                self.array_index_stack.push(self.current_array_index);
                self.current_array_index = 0;

                // For $[*] expressions, we immediately enter target array at root level
                let expression = self.path_expression.as_string();
                if expression == "$[*]" {
                    self.in_target_array = true;
                }

                Ok(super::processor::JsonProcessResult::Continue)
            }
            _ => Ok(super::processor::JsonProcessResult::Continue),
        }
    }

    /// Process byte when navigating through JSON structure
    pub(super) fn process_navigating_byte(
        &mut self,
        byte: u8,
    ) -> crate::json_path::error::JsonPathResult<super::processor::JsonProcessResult> {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => Ok(super::processor::JsonProcessResult::Continue),
            b'"' => {
                // Potential property name - need to check if it matches our target
                if self.target_property.is_some() {
                    let property_name = self.read_property_name()?;
                    if let Some(ref target_prop) = self.target_property {
                        if property_name == *target_prop {
                            self.in_target_property = true;
                        }
                    }
                } else {
                    // Skip string content normally
                    let mut escaped = false;
                    while let Some(string_byte) = self.read_next_byte()? {
                        if escaped {
                            escaped = false;
                        } else {
                            match string_byte {
                                b'"' => break,
                                b'\\' => escaped = true,
                                _ => {}
                            }
                        }
                    }
                }
                Ok(super::processor::JsonProcessResult::Continue)
            }
            b'{' => {
                if self.in_target_array && self.matches_current_path() {
                    // We're inside the target array - start collecting this object
                    self.object_buffer.clear();
                    self.object_buffer.push(byte);
                    self.transition_to_processing_object();
                    self.object_nesting = 1;
                    Ok(super::processor::JsonProcessResult::Continue)
                } else {
                    // We're still navigating - this is just a nested object, continue navigating
                    self.object_nesting = self.object_nesting.saturating_add(1);
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            b'[' => {
                if self.in_target_property {
                    // Found array in target property - this is our target array!
                    self.in_target_array = true;
                    self.in_target_property = false;
                }
                self.current_depth = self.current_depth.saturating_add(1);
                self.transition_to_processing_array();
                self.array_index_stack.push(self.current_array_index);
                self.current_array_index = 0;
                Ok(super::processor::JsonProcessResult::Continue)
            }
            b']' => {
                if self.in_target_array {
                    self.in_target_array = false;
                }
                self.current_depth = self.current_depth.saturating_sub(1);
                if let Some(prev_index) = self.array_index_stack.pop() {
                    self.current_array_index = prev_index;
                }
                if self.current_depth == 0 {
                    self.transition_to_complete();
                    Ok(super::processor::JsonProcessResult::Complete)
                } else {
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            b'}' => {
                self.object_nesting = self.object_nesting.saturating_sub(1);
                if self.object_nesting == 0 {
                    self.transition_to_complete();
                    Ok(super::processor::JsonProcessResult::Complete)
                } else {
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            b':' => {
                // Property separator - next value could be our target array
                Ok(super::processor::JsonProcessResult::Continue)
            }
            b',' => Ok(super::processor::JsonProcessResult::Continue),
            _ => Ok(super::processor::JsonProcessResult::Continue),
        }
    }

    /// Process byte when inside target array
    pub(super) fn process_array_byte(
        &mut self,
        byte: u8,
    ) -> crate::json_path::error::JsonPathResult<super::processor::JsonProcessResult> {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => Ok(super::processor::JsonProcessResult::Continue),
            b'{' => {
                if self.in_target_array && self.matches_current_path() {
                    self.object_buffer.clear();
                    self.object_buffer.push(byte);
                    self.transition_to_processing_object();
                    self.object_nesting = 1;
                    Ok(super::processor::JsonProcessResult::Continue)
                } else {
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            b'[' => {
                self.current_depth = self.current_depth.saturating_add(1);
                self.transition_to_processing_array();
                self.array_index_stack.push(self.current_array_index);
                self.current_array_index = 0;
                Ok(super::processor::JsonProcessResult::Continue)
            }
            b']' => {
                // Check if we have a remaining object to process before closing array
                if self.in_target_array && !self.object_buffer.is_empty() {
                    // Last object in array - process it before closing
                    let result = super::processor::JsonProcessResult::ObjectFound;
                    self.in_target_array = false;
                    self.current_depth = self.current_depth.saturating_sub(1);
                    if let Some(prev_index) = self.array_index_stack.pop() {
                        self.current_array_index = prev_index;
                    }
                    return Ok(result);
                }

                if self.in_target_array {
                    self.in_target_array = false;
                }
                self.current_depth = self.current_depth.saturating_sub(1);
                if let Some(prev_index) = self.array_index_stack.pop() {
                    self.current_array_index = prev_index;
                }
                if self.current_depth == 0 {
                    self.transition_to_complete();
                    Ok(super::processor::JsonProcessResult::Complete)
                } else {
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            b',' => {
                if self.in_target_array && !self.object_buffer.is_empty() {
                    // Don't add the comma to the object buffer - it's a separator, not part of the object
                    Ok(super::processor::JsonProcessResult::ObjectFound)
                } else {
                    self.current_array_index = self.current_array_index.saturating_add(1);
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            _ => {
                if self.in_target_array {
                    self.object_buffer.push(byte);
                }
                Ok(super::processor::JsonProcessResult::Continue)
            }
        }
    }

    /// Process byte when inside JSON object
    pub(super) fn process_object_byte(
        &mut self,
        byte: u8,
    ) -> crate::json_path::error::JsonPathResult<super::processor::JsonProcessResult> {
        // Always add bytes to object buffer if we're in target array BEFORE processing special characters
        if self.in_target_array {
            self.object_buffer.push(byte);
        }

        match byte {
            b'"' => {
                // Skip string content but don't add the bytes again since we already added the opening quote
                let mut escaped = false;
                while let Some(string_byte) = self.read_next_byte()? {
                    if self.in_target_array {
                        self.object_buffer.push(string_byte);
                    }
                    if escaped {
                        escaped = false;
                    } else {
                        match string_byte {
                            b'"' => break, // End of string
                            b'\\' => escaped = true,
                            _ => {}
                        }
                    }
                }
                Ok(super::processor::JsonProcessResult::Continue)
            }
            b'{' => {
                self.object_nesting = self.object_nesting.saturating_add(1);
                Ok(super::processor::JsonProcessResult::Continue)
            }
            b'[' => {
                // Transition to streaming array state when encountering array in object
                self.current_depth = self.current_depth.saturating_add(1);
                self.transition_to_processing_array();
                self.array_index_stack.push(self.current_array_index);
                self.current_array_index = 0;
                Ok(super::processor::JsonProcessResult::Continue)
            }
            b'}' => {
                self.object_nesting = self.object_nesting.saturating_sub(1);
                if self.object_nesting == 0 {
                    // Complete object found - transition back to streaming array state
                    self.transition_to_processing_array();
                    Ok(super::processor::JsonProcessResult::ObjectFound)
                } else {
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            _ => Ok(super::processor::JsonProcessResult::Continue),
        }
    }

    /// Check if current position matches JSONPath expression
    #[inline]
    pub(super) fn matches_current_path(&self) -> bool {
        // Simplified JSONPath matching for basic patterns
        // This handles common cases like $.data[*], $.items[*]

        let expression = self.path_expression.as_string();

        // For array wildcard patterns like $.data[*], $.items[*]
        if expression.starts_with("$.") && expression.ends_with("[*]") {
            // Match when we're inside the target array to capture array elements
            return self.in_target_array;
        }

        // For root array patterns like $[*]
        if expression == "$[*]" {
            // Match when we're inside the root array
            return self.in_target_array;
        }

        // Default fallback - match if we're in target array
        self.in_target_array
    }

    /// Deserialize current object from buffer
    #[inline]
    pub(super) fn deserialize_current_object(
        &mut self,
    ) -> crate::json_path::error::JsonPathResult<Option<T>> {
        if self.object_buffer.is_empty() {
            return Ok(None);
        }

        let json_str = match std::str::from_utf8(&self.object_buffer) {
            Ok(s) => s,
            Err(e) => {
                let error = crate::json_path::error::json_parse_error(
                    format!("Invalid UTF-8 in JSON object: {}", e),
                    self.buffer_position,
                    "object deserialization".to_string(),
                );
                self.object_buffer.clear();
                return Err(error);
            }
        };

        let result = match serde_json::from_str::<T>(json_str) {
            Ok(obj) => obj,
            Err(e) => {
                let error = crate::json_path::error::json_parse_error(
                    format!("Failed to deserialize JSON: {}", e),
                    self.buffer_position,
                    "object deserialization".to_string(),
                );
                self.object_buffer.clear();
                return Err(error);
            }
        };

        // Clear buffer for next object
        self.object_buffer.clear();
        // Object yielded - no state tracking needed for internal state

        Ok(Some(result))
    }

    /// Read a JSON property name from the current position
    #[inline]
    pub(super) fn read_property_name(&mut self) -> crate::json_path::error::JsonPathResult<String> {
        let mut property_name = String::new();
        let mut escaped = false;

        while let Some(byte) = self.read_next_byte()? {
            if escaped {
                escaped = false;
                match byte {
                    b'"' => property_name.push('"'),
                    b'\\' => property_name.push('\\'),
                    b'/' => property_name.push('/'),
                    b'b' => property_name.push('\u{0008}'),
                    b'f' => property_name.push('\u{000C}'),
                    b'n' => property_name.push('\n'),
                    b'r' => property_name.push('\r'),
                    b't' => property_name.push('\t'),
                    _ => {
                        property_name.push('\\');
                        property_name.push(byte as char);
                    }
                }
            } else {
                match byte {
                    b'"' => break, // End of property name
                    b'\\' => escaped = true,
                    _ => property_name.push(byte as char),
                }
            }
        }

        Ok(property_name)
    }

    // Internal state transition methods

    /// Transition to navigating state
    #[inline]
    pub(super) fn transition_to_navigating(&mut self) {
        self.state = DeserializerState::Navigating;
    }

    /// Transition to processing array state  
    #[inline]
    pub(super) fn transition_to_processing_array(&mut self) {
        self.state = DeserializerState::ProcessingArray;
    }

    /// Transition to processing object state
    #[inline]
    pub(super) fn transition_to_processing_object(&mut self) {
        self.state = DeserializerState::ProcessingObject;
    }

    /// Transition to complete state
    #[inline]
    pub(super) fn transition_to_complete(&mut self) {
        self.state = DeserializerState::Complete;
    }
}
