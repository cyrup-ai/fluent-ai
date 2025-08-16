use serde::de::DeserializeOwned;

use super::types::JsonPathDeserializer;

impl<'a, T> JsonPathDeserializer<'a, T>
where
    T: DeserializeOwned,
{
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
}
