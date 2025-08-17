//! Core JSON byte processing logic
//!
//! Contains the main entry points and core logic for processing individual JSON bytes
//! during streaming, including state transitions and basic byte reading.

use serde::de::DeserializeOwned;

use super::super::iterator::JsonPathIterator;
use crate::jsonpath::error::JsonPathResult;

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
    /// TODO: This appears to duplicate functionality in core.rs - investigate architectural intent
    #[allow(dead_code)]
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
    /// TODO: This appears to duplicate functionality in core.rs - investigate architectural intent
    #[allow(dead_code)]
    #[inline]
    pub(super) fn process_json_byte(&mut self, byte: u8) -> JsonPathResult<JsonProcessResult> {
        match &self.deserializer.state {
            super::super::core::DeserializerState::Initial => self.process_initial_byte(byte),
            super::super::core::DeserializerState::Navigating => self.process_navigating_byte(byte),
            super::super::core::DeserializerState::ProcessingArray => self.process_array_byte(byte),
            super::super::core::DeserializerState::ProcessingObject => {
                self.process_object_byte(byte)
            }
            super::super::core::DeserializerState::Complete => Ok(JsonProcessResult::Complete),
        }
    }

    /// Check if current position matches JSONPath root object selector
    #[inline]
    pub(super) fn matches_root_object_path(&self) -> bool {
        // Check if the root selector matches object access
        match self.deserializer.path_expression.root_selector() {
            Some(crate::jsonpath::parser::JsonSelector::Child { .. }) => true,
            Some(crate::jsonpath::parser::JsonSelector::Filter { .. }) => true,
            _ => false,
        }
    }

    /// Check if current position matches JSONPath root array selector
    #[inline]
    pub(super) fn matches_root_array_path(&self) -> bool {
        matches!(
            self.deserializer.path_expression.root_selector(),
            Some(crate::jsonpath::parser::JsonSelector::Wildcard)
                | Some(crate::jsonpath::parser::JsonSelector::Index { .. })
                | Some(crate::jsonpath::parser::JsonSelector::Slice { .. })
        )
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
                return Err(crate::jsonpath::error::json_parse_error(
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
}
