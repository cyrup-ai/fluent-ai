//! JSON byte-by-byte processing logic
//!
//! Contains the core logic for processing individual JSON bytes during streaming,
//! including state transitions and JSONPath evaluation integration.

use serde::de::DeserializeOwned;

use super::iterator::JsonPathIterator;
use crate::json_path::{error::JsonPathResult, state_machine::JsonStreamState};

/// Result of processing a single JSON byte
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonProcessResult {
    /// Continue processing more bytes
    Continue,
    /// Complete JSON object found and ready for deserialization
    ObjectFound,
    /// Need more data to continue parsing
    NeedMoreData,
    /// Processing complete (end of stream)
    Complete,
}

impl<'iter, 'data, T> JsonPathIterator<'iter, 'data, T>
where
    T: DeserializeOwned,
{
    /// Read next byte from streaming buffer using persistent position tracking
    #[inline]
    pub(super) fn read_next_byte(&mut self) -> JsonPathResult<Option<u8>> {
        // Check if we've reached the end of available data
        if self.deserializer.buffer_position >= self.deserializer.buffer.len() {
            return Ok(None); // No more data available
        }

        // Read byte at current position using buffer accessor
        match self
            .deserializer
            .buffer
            .get_byte_at(self.deserializer.buffer_position)
        {
            Some(byte) => {
                self.deserializer.buffer_position += 1;
                self.bytes_consumed += 1;
                Ok(Some(byte))
            }
            None => Ok(None), // Position beyond buffer bounds
        }
    }

    /// Process single JSON byte and update parsing state
    #[inline]
    pub(super) fn process_json_byte(&mut self, byte: u8) -> JsonPathResult<JsonProcessResult> {
        match self.deserializer.state.current_state() {
            JsonStreamState::Initial => self.process_initial_byte(byte),
            JsonStreamState::Navigating { .. } => self.process_navigating_byte(byte),
            JsonStreamState::StreamingArray { .. } => self.process_array_byte(byte),
            JsonStreamState::ProcessingObject { .. } => self.process_object_byte(byte),
            JsonStreamState::Complete => Ok(JsonProcessResult::Complete),
            JsonStreamState::Finishing { .. } => {
                // Transition to Complete to terminate processing
                self.deserializer.state.transition_to_complete();
                Ok(JsonProcessResult::Complete)
            }
            JsonStreamState::Error { .. } => Ok(JsonProcessResult::Complete),
        }
    }

    /// Process byte when parser is in initial state
    #[inline]
    pub(super) fn process_initial_byte(&mut self, byte: u8) -> JsonPathResult<JsonProcessResult> {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => Ok(JsonProcessResult::Continue), // Skip whitespace
            b'{' => {
                self.deserializer.state.transition_to_processing_object();
                self.deserializer.object_nesting =
                    self.deserializer.object_nesting.saturating_add(1);
                if self.matches_root_object_path() {
                    Ok(JsonProcessResult::Continue)
                } else {
                    Ok(JsonProcessResult::Continue)
                }
            }
            b'[' => {
                self.deserializer.state.transition_to_streaming_array();
                self.deserializer.current_depth = self.deserializer.current_depth.saturating_add(1);
                // Push current array index to stack for nested arrays
                self.deserializer
                    .array_index_stack
                    .push(self.deserializer.current_array_index);
                self.deserializer.current_array_index = 0; // Reset for new array
                if self.matches_root_array_path() {
                    self.deserializer.in_target_array = true;
                }
                Ok(JsonProcessResult::Continue)
            }
            _ => Ok(JsonProcessResult::Continue),
        }
    }

    /// Check if current position matches JSONPath root object selector
    #[inline]
    pub(super) fn matches_root_object_path(&self) -> bool {
        // Check if the root selector matches object access
        match self.deserializer.path_expression.root_selector() {
            Some(crate::json_path::parser::JsonSelector::Child { .. }) => true,
            Some(crate::json_path::parser::JsonSelector::Filter { .. }) => true,
            _ => false,
        }
    }

    /// Check if current position matches JSONPath root array selector
    #[inline]
    pub(super) fn matches_root_array_path(&self) -> bool {
        matches!(
            self.deserializer.path_expression.root_selector(),
            Some(crate::json_path::parser::JsonSelector::Wildcard)
                | Some(crate::json_path::parser::JsonSelector::Index { .. })
                | Some(crate::json_path::parser::JsonSelector::Slice { .. })
        )
    }

    /// Process byte when navigating through JSON structure
    pub(super) fn process_navigating_byte(
        &mut self,
        byte: u8,
    ) -> JsonPathResult<JsonProcessResult> {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => Ok(JsonProcessResult::Continue), // Skip whitespace
            b'{' => {
                self.deserializer.object_nesting =
                    self.deserializer.object_nesting.saturating_add(1);
                self.deserializer.state.transition_to_processing_object();
                if self.matches_current_path() && self.deserializer.in_target_array {
                    self.deserializer.object_buffer.clear();
                    self.deserializer.object_buffer.push(byte);
                    Ok(JsonProcessResult::Continue)
                } else {
                    Ok(JsonProcessResult::Continue)
                }
            }
            b'[' => {
                self.deserializer.current_depth = self.deserializer.current_depth.saturating_add(1);
                self.deserializer.state.transition_to_streaming_array();
                // Push current array index to stack for nested arrays
                self.deserializer
                    .array_index_stack
                    .push(self.deserializer.current_array_index);
                self.deserializer.current_array_index = 0; // Reset for new array
                if self.matches_current_path() {
                    self.deserializer.in_target_array = true;
                }
                Ok(JsonProcessResult::Continue)
            }
            b']' => {
                if self.deserializer.in_target_array {
                    self.deserializer.in_target_array = false;
                }
                self.deserializer.current_depth = self.deserializer.current_depth.saturating_sub(1);
                // Restore previous array index from stack
                if let Some(prev_index) = self.deserializer.array_index_stack.pop() {
                    self.deserializer.current_array_index = prev_index;
                }
                if self.deserializer.current_depth == 0 {
                    self.deserializer.state.transition_to_complete();
                    Ok(JsonProcessResult::Complete)
                } else {
                    Ok(JsonProcessResult::Continue)
                }
            }
            b'}' => {
                self.deserializer.object_nesting =
                    self.deserializer.object_nesting.saturating_sub(1);
                if self.deserializer.object_nesting == 0 {
                    self.deserializer.state.transition_to_complete();
                    Ok(JsonProcessResult::Complete)
                } else {
                    Ok(JsonProcessResult::Continue)
                }
            }
            b',' => Ok(JsonProcessResult::Continue), // Array/object separator
            _ => Ok(JsonProcessResult::Continue),
        }
    }

    /// Process byte when inside target array
    pub(super) fn process_array_byte(&mut self, byte: u8) -> JsonPathResult<JsonProcessResult> {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => Ok(JsonProcessResult::Continue), // Skip whitespace
            b'{' => {
                if self.deserializer.in_target_array && self.matches_current_path() {
                    self.deserializer.object_buffer.clear();
                    self.deserializer.object_buffer.push(byte);
                    self.deserializer.state.transition_to_processing_object();
                    self.deserializer.object_nesting = 1;
                    Ok(JsonProcessResult::Continue)
                } else {
                    Ok(JsonProcessResult::Continue)
                }
            }
            b'[' => {
                self.deserializer.current_depth = self.deserializer.current_depth.saturating_add(1);
                self.deserializer.state.transition_to_streaming_array();
                // Push current array index to stack for nested arrays
                self.deserializer
                    .array_index_stack
                    .push(self.deserializer.current_array_index);
                self.deserializer.current_array_index = 0; // Reset for new array
                Ok(JsonProcessResult::Continue)
            }
            b']' => {
                if self.deserializer.in_target_array {
                    self.deserializer.in_target_array = false;
                }
                self.deserializer.current_depth = self.deserializer.current_depth.saturating_sub(1);
                // Restore previous array index from stack
                if let Some(prev_index) = self.deserializer.array_index_stack.pop() {
                    self.deserializer.current_array_index = prev_index;
                }
                if self.deserializer.current_depth == 0 {
                    self.deserializer.state.transition_to_complete();
                    Ok(JsonProcessResult::Complete)
                } else {
                    Ok(JsonProcessResult::Continue)
                }
            }
            b',' => {
                if self.deserializer.in_target_array && !self.deserializer.object_buffer.is_empty()
                {
                    // Complete object found - object_buffer contains a full JSON object
                    Ok(JsonProcessResult::ObjectFound)
                } else {
                    // Increment array index for next element
                    self.deserializer.current_array_index =
                        self.deserializer.current_array_index.saturating_add(1);
                    Ok(JsonProcessResult::Continue)
                }
            }
            _ => {
                if self.deserializer.in_target_array {
                    self.deserializer.object_buffer.push(byte);
                }
                Ok(JsonProcessResult::Continue)
            }
        }
    }

    /// Process byte when inside JSON object
    pub(super) fn process_object_byte(&mut self, byte: u8) -> JsonPathResult<JsonProcessResult> {
        if self.deserializer.in_target_array {
            self.deserializer.object_buffer.push(byte);
        }

        match byte {
            b'"' => {
                self.skip_string_content()?;
                Ok(JsonProcessResult::Continue)
            }
            b'{' => {
                self.deserializer.object_nesting =
                    self.deserializer.object_nesting.saturating_add(1);
                Ok(JsonProcessResult::Continue)
            }
            b'}' => {
                self.deserializer.object_nesting =
                    self.deserializer.object_nesting.saturating_sub(1);
                if self.deserializer.object_nesting == 0 {
                    // Complete object found
                    Ok(JsonProcessResult::ObjectFound)
                } else {
                    Ok(JsonProcessResult::Continue)
                }
            }
            _ => Ok(JsonProcessResult::Continue),
        }
    }

    /// Skip over JSON string content including escaped characters
    pub(super) fn skip_string_content(&mut self) -> JsonPathResult<()> {
        let mut escaped = false;
        let mut bytes_processed = 0;
        const MAX_STRING_BYTES: usize = 1024 * 1024; // 1MB limit for JSON strings

        while let Some(byte) = self.read_next_byte()? {
            bytes_processed += 1;

            // Prevent infinite loops on malformed JSON
            if bytes_processed > MAX_STRING_BYTES {
                return Err(crate::json_path::error::json_parse_error(
                    "JSON string too long or unterminated".to_string(),
                    self.bytes_consumed,
                    "string parsing".to_string(),
                ));
            }
            if self.deserializer.in_target_array {
                self.deserializer.object_buffer.push(byte);
            }

            if escaped {
                escaped = false;
            } else {
                match byte {
                    b'"' => return Ok(()), // End of string
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
        self.evaluate_jsonpath_at_current_position()
    }

    /// Evaluate JSONPath expression at current parsing position
    #[inline]
    pub(super) fn evaluate_jsonpath_at_current_position(&self) -> bool {
        if self.deserializer.in_recursive_descent {
            self.evaluate_recursive_descent_match()
        } else {
            self.evaluate_selector_match()
        }
    }

    /// Evaluate current selector considering array indices and slice notation
    #[inline]
    pub(super) fn evaluate_selector_match(&self) -> bool {
        // Get the current selector from the path expression
        let selectors = self.deserializer.path_expression.selectors();
        let selector_index = self
            .deserializer
            .current_selector_index
            .min(selectors.len().saturating_sub(1));

        if selector_index >= selectors.len() {
            return false;
        }

        let current_selector = &selectors[selector_index];
        self.evaluate_single_selector(current_selector)
    }

    /// Evaluate a single selector against current streaming context
    #[inline]
    pub(super) fn evaluate_single_selector(
        &self,
        selector: &crate::json_path::parser::JsonSelector,
    ) -> bool {
        use crate::json_path::parser::JsonSelector;

        match selector {
            JsonSelector::Root => self.deserializer.current_depth == 0,
            JsonSelector::Child { .. } => self.deserializer.current_depth > 0,
            JsonSelector::RecursiveDescent => true, // Always matches
            JsonSelector::Wildcard => self.deserializer.current_depth > 0,
            JsonSelector::Index { index, from_end } => {
                self.evaluate_index_selector(*index, *from_end)
            }
            JsonSelector::Slice { start, end, step } => {
                self.evaluate_slice_selector(*start, *end, *step)
            }
            JsonSelector::Filter { expression } => {
                // Evaluate the filter expression against the current JSON context
                if !self.deserializer.object_buffer.is_empty() {
                    if let Ok(json_str) = std::str::from_utf8(&self.deserializer.object_buffer) {
                        if let Ok(context) = serde_json::from_str::<serde_json::Value>(json_str) {
                            use crate::json_path::JsonPathResultExt;
                            use crate::json_path::filter::FilterEvaluator;
                            return FilterEvaluator::evaluate_predicate(&context, expression)
                                .handle_or_default(false);
                        }
                    }
                }
                false
            }
            JsonSelector::Union { selectors } => {
                // Union matches if any selector matches
                selectors.iter().any(|s| self.evaluate_single_selector(s))
            }
        }
    }

    /// Evaluate array index selector against current array position
    #[inline]
    pub(super) fn evaluate_index_selector(&self, index: i64, from_end: bool) -> bool {
        if !self.deserializer.in_target_array {
            return false;
        }

        let current_idx = self.deserializer.current_array_index;

        if from_end || index < 0 {
            // Negative indices require knowing array length, which we don't have in streaming
            // For streaming context, we skip negative index matching
            false
        } else {
            current_idx == index
        }
    }

    /// Evaluate array slice selector against current array position
    #[inline]
    pub(super) fn evaluate_slice_selector(
        &self,
        start: Option<i64>,
        end: Option<i64>,
        step: Option<i64>,
    ) -> bool {
        if !self.deserializer.in_target_array {
            return false;
        }

        let current_idx = self.deserializer.current_array_index;
        let step = step.unwrap_or(1);

        // Handle step size
        if step <= 0 {
            return false; // Invalid step
        }

        // Check start boundary
        let start_idx = match start {
            Some(s) if s >= 0 => s,
            Some(_) => return false, // Negative start not supported in streaming
            None => 0,               // Default start
        };

        // Check end boundary (None means no upper limit in streaming context)
        let within_end = match end {
            Some(e) if e >= 0 => current_idx < e,
            Some(_) => false, // Negative end not supported in streaming
            None => true,     // No upper limit
        };

        // Check if current index is within slice bounds and matches step
        let within_start = current_idx >= start_idx;
        let matches_step = (current_idx - start_idx) % step == 0;

        within_start && within_end && matches_step
    }
}
