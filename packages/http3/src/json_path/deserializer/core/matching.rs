//! JSONPath matching and deserialization logic
//!
//! Contains methods for JSONPath expression matching and object deserialization
//! during streaming JSON processing.

use serde::de::DeserializeOwned;

use super::types::JsonPathDeserializer;

impl<'a, T> JsonPathDeserializer<'a, T>
where
    T: DeserializeOwned,
{
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
}
