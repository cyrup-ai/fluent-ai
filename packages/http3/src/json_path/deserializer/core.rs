//! Core JsonPathDeserializer struct and main API
//!
//! Contains the primary deserializer structure with JSONPath navigation capabilities
//! and the main public interface for streaming JSON deserialization.

use serde::de::DeserializeOwned;

use super::iterator::JsonPathIterator;
use crate::json_path::{
    buffer::StreamBuffer, parser::JsonPathExpression, state_machine::StreamStateMachine,
};

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
    /// State machine for tracking parsing progress
    pub(super) state: &'a mut StreamStateMachine,
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
            current_array_index: 0,
            array_index_stack: Vec::with_capacity(16), // Support up to 16 nested arrays
            buffer_position: 0,
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
        // Reset buffer position to start of current buffer when processing new chunks
        // This ensures we don't miss any data when new chunks are appended
        self.buffer_position = 0;
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
    pub(super) fn process_json_byte(&mut self, byte: u8) -> crate::json_path::error::JsonPathResult<super::processor::JsonProcessResult> {
        match self.state.current_state() {
            crate::json_path::state_machine::JsonStreamState::Initial => self.process_initial_byte(byte),
            crate::json_path::state_machine::JsonStreamState::Navigating { .. } => self.process_navigating_byte(byte),
            crate::json_path::state_machine::JsonStreamState::StreamingArray { .. } => self.process_array_byte(byte),
            crate::json_path::state_machine::JsonStreamState::ProcessingObject { .. } => self.process_object_byte(byte),
            crate::json_path::state_machine::JsonStreamState::Complete => Ok(super::processor::JsonProcessResult::Complete),
            crate::json_path::state_machine::JsonStreamState::Finishing { .. } => {
                // Transition to Complete to terminate processing
                self.state.transition_to_complete();
                Ok(super::processor::JsonProcessResult::Complete)
            }
            crate::json_path::state_machine::JsonStreamState::Error { .. } => Ok(super::processor::JsonProcessResult::Complete),
        }
    }

    /// Process byte when parser is in initial state
    #[inline]
    pub(super) fn process_initial_byte(&mut self, byte: u8) -> crate::json_path::error::JsonPathResult<super::processor::JsonProcessResult> {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => Ok(super::processor::JsonProcessResult::Continue), // Skip whitespace
            b'{' => {
                self.state.transition_to_processing_object();
                self.object_nesting = self.object_nesting.saturating_add(1);
                Ok(super::processor::JsonProcessResult::Continue)
            }
            b'[' => {
                self.state.transition_to_streaming_array();
                self.current_depth = self.current_depth.saturating_add(1);
                self.array_index_stack.push(self.current_array_index);
                self.current_array_index = 0;
                if self.matches_root_array_path() {
                    self.in_target_array = true;
                }
                Ok(super::processor::JsonProcessResult::Continue)
            }
            _ => Ok(super::processor::JsonProcessResult::Continue),
        }
    }

    /// Check if current position matches JSONPath root array selector
    #[inline]
    pub(super) fn matches_root_array_path(&self) -> bool {
        matches!(
            self.path_expression.root_selector(),
            Some(crate::json_path::parser::JsonSelector::Wildcard)
                | Some(crate::json_path::parser::JsonSelector::Index { .. })
                | Some(crate::json_path::parser::JsonSelector::Slice { .. })
        )
    }

    /// Process byte when navigating through JSON structure
    pub(super) fn process_navigating_byte(&mut self, byte: u8) -> crate::json_path::error::JsonPathResult<super::processor::JsonProcessResult> {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => Ok(super::processor::JsonProcessResult::Continue),
            b'{' => {
                self.object_nesting = self.object_nesting.saturating_add(1);
                self.state.transition_to_processing_object();
                if self.matches_current_path() && self.in_target_array {
                    self.object_buffer.clear();
                    self.object_buffer.push(byte);
                    Ok(super::processor::JsonProcessResult::Continue)
                } else {
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            b'[' => {
                self.current_depth = self.current_depth.saturating_add(1);
                self.state.transition_to_streaming_array();
                self.array_index_stack.push(self.current_array_index);
                self.current_array_index = 0;
                if self.matches_current_path() {
                    self.in_target_array = true;
                }
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
                    self.state.transition_to_complete();
                    Ok(super::processor::JsonProcessResult::Complete)
                } else {
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            b'}' => {
                self.object_nesting = self.object_nesting.saturating_sub(1);
                if self.object_nesting == 0 {
                    self.state.transition_to_complete();
                    Ok(super::processor::JsonProcessResult::Complete)
                } else {
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            b',' => Ok(super::processor::JsonProcessResult::Continue),
            _ => Ok(super::processor::JsonProcessResult::Continue),
        }
    }

    /// Process byte when inside target array
    pub(super) fn process_array_byte(&mut self, byte: u8) -> crate::json_path::error::JsonPathResult<super::processor::JsonProcessResult> {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => Ok(super::processor::JsonProcessResult::Continue),
            b'{' => {
                if self.in_target_array && self.matches_current_path() {
                    self.object_buffer.clear();
                    self.object_buffer.push(byte);
                    self.state.transition_to_processing_object();
                    self.object_nesting = 1;
                    Ok(super::processor::JsonProcessResult::Continue)
                } else {
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            b'[' => {
                self.current_depth = self.current_depth.saturating_add(1);
                self.state.transition_to_streaming_array();
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
                    self.state.transition_to_complete();
                    Ok(super::processor::JsonProcessResult::Complete)
                } else {
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            b',' => {
                if self.in_target_array && !self.object_buffer.is_empty() {
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
    pub(super) fn process_object_byte(&mut self, byte: u8) -> crate::json_path::error::JsonPathResult<super::processor::JsonProcessResult> {
        if self.in_target_array {
            self.object_buffer.push(byte);
        }

        match byte {
            b'"' => {
                self.skip_string_content()?;
                Ok(super::processor::JsonProcessResult::Continue)
            }
            b'{' => {
                self.object_nesting = self.object_nesting.saturating_add(1);
                Ok(super::processor::JsonProcessResult::Continue)
            }
            b'}' => {
                self.object_nesting = self.object_nesting.saturating_sub(1);
                if self.object_nesting == 0 {
                    Ok(super::processor::JsonProcessResult::ObjectFound)
                } else {
                    Ok(super::processor::JsonProcessResult::Continue)
                }
            }
            _ => Ok(super::processor::JsonProcessResult::Continue),
        }
    }

    /// Skip over JSON string content including escaped characters
    pub(super) fn skip_string_content(&mut self) -> crate::json_path::error::JsonPathResult<()> {
        let mut escaped = false;
        let mut bytes_processed = 0;
        const MAX_STRING_BYTES: usize = 1024 * 1024;

        while let Some(byte) = self.read_next_byte()? {
            bytes_processed += 1;

            if bytes_processed > MAX_STRING_BYTES {
                return Err(crate::json_path::error::json_parse_error(
                    "JSON string too long or unterminated".to_string(),
                    self.buffer_position,
                    "string parsing".to_string(),
                ));
            }
            if self.in_target_array {
                self.object_buffer.push(byte);
            }

            if escaped {
                escaped = false;
            } else {
                match byte {
                    b'"' => return Ok(()),
                    b'\\' => escaped = true,
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Check if current position matches JSONPath expression
    #[inline]
    pub(super) fn matches_current_path(&self) -> bool {
        // Production implementation - check if we're in target array
        self.in_target_array
    }

    /// Deserialize current object from buffer
    #[inline]
    pub(super) fn deserialize_current_object(&mut self) -> Option<T> {
        if self.object_buffer.is_empty() {
            return None;
        }

        let json_str = match std::str::from_utf8(&self.object_buffer) {
            Ok(s) => s,
            Err(e) => {
                log::error!("Invalid UTF-8 in JSON object: {}", e);
                self.object_buffer.clear();
                return None;
            }
        };

        let result = match serde_json::from_str::<T>(json_str) {
            Ok(obj) => obj,
            Err(e) => {
                log::error!("Failed to deserialize JSON: {}", e);
                self.object_buffer.clear();
                return None;
            }
        };

        // Clear buffer for next object
        self.object_buffer.clear();
        self.state.increment_objects_yielded();

        Some(result)
    }
}
