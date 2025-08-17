use serde::de::DeserializeOwned;

use super::types::{DeserializerState, JsonPathDeserializer};

impl<'a, T> JsonPathDeserializer<'a, T>
where
    T: DeserializeOwned,
{
    /// Read next byte from buffer with position tracking
    #[inline]
    pub(super) fn read_next_byte(&mut self) -> crate::jsonpath::error::JsonPathResult<Option<u8>> {
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
    ) -> crate::jsonpath::error::JsonPathResult<super::super::processor::JsonProcessResult> {
        match &self.state {
            DeserializerState::Initial => self.process_initial_byte(byte),
            DeserializerState::Navigating => self.process_navigating_byte(byte),
            DeserializerState::ProcessingArray => self.process_array_byte(byte),
            DeserializerState::ProcessingObject => self.process_object_byte(byte),
            DeserializerState::Complete => Ok(super::super::processor::JsonProcessResult::Complete),
        }
    }
}
