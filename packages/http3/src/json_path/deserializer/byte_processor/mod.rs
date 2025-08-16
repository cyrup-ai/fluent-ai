//! Byte processing utilities for JSON deserialization
//!
//! This module provides byte-level processing functionality for the JSON deserializer.

/// Result type for JSON processing operations
#[derive(Debug, Clone)]
pub enum JsonProcessResult {
    /// Processing completed successfully
    Success,
    /// Processing needs more data
    NeedsMoreData,
    /// Processing encountered an error
    Error(String),
}

impl JsonProcessResult {
    /// Check if the result indicates success
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, JsonProcessResult::Success)
    }

    /// Check if the result indicates more data is needed
    #[must_use]
    pub fn needs_more_data(&self) -> bool {
        matches!(self, JsonProcessResult::NeedsMoreData)
    }

    /// Check if the result indicates an error
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, JsonProcessResult::Error(_))
    }

    /// Get the error message if this is an error result
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        match self {
            JsonProcessResult::Error(msg) => Some(msg),
            _ => None,
        }
    }
}

impl Default for JsonProcessResult {
    fn default() -> Self {
        JsonProcessResult::Success
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_process_result_success() {
        let result = JsonProcessResult::Success;
        assert!(result.is_success());
        assert!(!result.needs_more_data());
        assert!(!result.is_error());
        assert_eq!(result.error_message(), None);
    }

    #[test]
    fn test_json_process_result_needs_more_data() {
        let result = JsonProcessResult::NeedsMoreData;
        assert!(!result.is_success());
        assert!(result.needs_more_data());
        assert!(!result.is_error());
        assert_eq!(result.error_message(), None);
    }

    #[test]
    fn test_json_process_result_error() {
        let error_msg = "Parse error";
        let result = JsonProcessResult::Error(error_msg.to_string());
        assert!(!result.is_success());
        assert!(!result.needs_more_data());
        assert!(result.is_error());
        assert_eq!(result.error_message(), Some(error_msg));
    }

    #[test]
    fn test_json_process_result_default() {
        let result = JsonProcessResult::default();
        assert!(result.is_success());
        assert!(!result.needs_more_data());
        assert!(!result.is_error());
    }
}
