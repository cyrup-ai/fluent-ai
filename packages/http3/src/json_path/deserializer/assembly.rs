//! Object assembly and deserialization logic
//!
//! Handles the accumulation of complete JSON objects from byte streams
//! and their deserialization into target types.

use serde::de::DeserializeOwned;

use super::iterator::JsonPathIterator;
use crate::json_path::error::deserialization_error;

impl<'iter, 'data, T> JsonPathIterator<'iter, 'data, T>
where
    T: DeserializeOwned,
{
    /// Deserialize current accumulated object
    #[inline]
    pub(super) fn deserialize_current_object(&mut self) -> Option<T> {
        if self.deserializer.object_buffer.is_empty() {
            return None;
        }

        let json_str = match std::str::from_utf8(&self.deserializer.object_buffer) {
            Ok(s) => s,
            Err(e) => {
                let error = deserialization_error(
                    format!("Invalid UTF-8 in JSON object: {}", e),
                    String::from_utf8_lossy(&self.deserializer.object_buffer).to_string(),
                    std::any::type_name::<T>(),
                );
                log::error!("JSON deserialization UTF-8 error: {}", error);
                // Clear buffer and continue processing
                self.deserializer.object_buffer.clear();
                return None;
            }
        };

        let result = match serde_json::from_str::<T>(json_str) {
            Ok(obj) => obj,
            Err(e) => {
                let error = deserialization_error(
                    format!("Failed to deserialize JSON: {}", e),
                    json_str.to_string(),
                    std::any::type_name::<T>(),
                );
                log::error!("JSON deserialization parse error: {}", error);
                // Clear buffer and continue processing
                self.deserializer.object_buffer.clear();
                return None;
            }
        };

        // Clear buffer for next object
        self.deserializer.object_buffer.clear();
        self.deserializer.state.increment_objects_yielded();

        Some(result)
    }
}
