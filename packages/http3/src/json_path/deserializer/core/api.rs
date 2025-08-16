//! Public API methods for JsonPathDeserializer
//!
//! Contains the main public interface methods for streaming JSON deserialization
//! and buffer management operations.

use serde::de::DeserializeOwned;

use super::iterator::JsonPathIterator;
use super::types::JsonPathDeserializer;

impl<'a, T> JsonPathDeserializer<'a, T>
where
    T: DeserializeOwned,
{
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
}