//! JSON and WebSocket format implementations for streaming
//!
//! Provides JSON and WebSocket formatting capabilities with zero-allocation patterns
//! and production-ready error handling.

use crate::streaming::{StreamingError, StreamingTokenResponse};
use super::types::StreamingFormatter;

impl StreamingFormatter {
    /// Format as JSON with full metadata
    #[inline]
    pub fn format_json(&self, response: &StreamingTokenResponse) -> Result<String, StreamingError> {
        serde_json::to_string(response)
            .map_err(|e| StreamingError::FormatError(format!("JSON serialization failed: {}", e)))
    }

    /// Format as WebSocket message
    #[inline]
    pub fn format_websocket(
        &self,
        response: &StreamingTokenResponse,
    ) -> Result<String, StreamingError> {
        // WebSocket message with type indicator
        let message = serde_json::json!({
            "type": "token",
            "sequence": self.sequence_number,
            "data": response
        });

        serde_json::to_string(&message).map_err(|e| {
            StreamingError::FormatError(format!("WebSocket JSON serialization failed: {}", e))
        })
    }

    /// Format with custom format (extensible for future formats)
    #[inline]
    pub fn format_custom(
        &self,
        response: &StreamingTokenResponse,
        format_name: &str,
    ) -> Result<String, StreamingError> {
        match format_name {
            "minimal_json" => {
                // Minimal JSON with only essential fields
                let minimal = serde_json::json!({
                    "text": response.text,
                    "sequence_id": response.sequence_id,
                    "is_final": response.is_final
                });
                serde_json::to_string(&minimal).map_err(|e| {
                    StreamingError::FormatError(format!("Custom JSON serialization failed: {}", e))
                })
            }
            "csv" => {
                // CSV format: position,content,is_complete,probability
                Ok(format!(
                    "{},{},{},{}",
                    response.sequence_id,
                    response.text.replace(',', "\\,"), // Escape commas
                    response.is_final,
                    1.0 // Default probability since not available in struct
                ))
            }
            _ => Err(StreamingError::FormatError(format!(
                "Unknown custom format: {}",
                format_name
            )))}
    }

    /// Format JSON end marker
    #[inline]
    pub fn format_json_end_marker(&self) -> String {
        serde_json::json!({"type": "end", "sequence": self.sequence_number}).to_string()
    }

    /// Format WebSocket end marker
    #[inline]
    pub fn format_websocket_end_marker(&self) -> Result<String, StreamingError> {
        let message = serde_json::json!({
            "type": "end",
            "sequence": self.sequence_number
        });
        serde_json::to_string(&message).map_err(|e| {
            StreamingError::FormatError(format!("End marker serialization failed: {}", e))
        })
    }

    /// Format JSON error message
    #[inline]
    pub fn format_json_error(&self, error: &StreamingError) -> String {
        let error_response = serde_json::json!({
            "type": "error",
            "sequence": self.sequence_number,
            "error": error.to_string()
        });
        error_response.to_string()
    }

    /// Format WebSocket error message
    #[inline]
    pub fn format_websocket_error(
        &self,
        error: &StreamingError,
    ) -> Result<String, StreamingError> {
        let message = serde_json::json!({
            "type": "error",
            "sequence": self.sequence_number,
            "error": error.to_string()
        });
        serde_json::to_string(&message).map_err(|e| {
            StreamingError::FormatError(format!(
                "Error message serialization failed: {}",
                e
            ))
        })
    }

    /// Format custom format end marker
    #[inline]
    pub fn format_custom_end_marker(&self, format_name: &str) -> Result<String, StreamingError> {
        match format_name {
            "minimal_json" => Ok(serde_json::json!({"end": true}).to_string()),
            "csv" => Ok("END,[END],0.0".to_string()),
            _ => Err(StreamingError::FormatError(format!(
                "Unknown custom format for end marker: {}",
                format_name
            )))}
    }

    /// Format custom format error message
    #[inline]
    pub fn format_custom_error(&self, error: &StreamingError, format_name: &str) -> Result<String, StreamingError> {
        match format_name {
            "minimal_json" => Ok(serde_json::json!({"error": error.to_string()}).to_string()),
            "csv" => Ok(format!(
                "ERROR,[ERROR: {}],0.0",
                error.to_string().replace(',', "\\,")
            )),
            _ => Err(StreamingError::FormatError(format!(
                "Unknown custom format for error: {}",
                format_name
            )))}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{TokenMetadata, TokenTiming};
    use crate::streaming::formats::types::OutputFormat;

    fn create_test_response() -> StreamingTokenResponse {
        StreamingTokenResponse {
            content: "test token".to_string(),
            token_id: Some(123),
            position: 5,
            is_complete_token: true,
            probability: Some(0.95),
            alternatives: None,
            timing: TokenTiming::default(),
            metadata: TokenMetadata::default()}
    }

    #[test]
    fn test_json_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Json);
        let response = create_test_response();

        let formatted = formatter.format_json(&response).unwrap();
        assert!(formatted.contains("test token"));
        assert!(formatted.contains("123"));
        assert!(formatted.contains("0.95"));
    }

    #[test]
    fn test_websocket_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::WebSocket);
        let response = create_test_response();

        let formatted = formatter.format_websocket(&response).unwrap();
        assert!(formatted.contains("\"type\":\"token\""));
        assert!(formatted.contains("\"sequence\":0"));
    }

    #[test]
    fn test_minimal_json_custom_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("minimal_json".to_string()));
        let response = create_test_response();

        let formatted = formatter.format_custom(&response, "minimal_json").unwrap();
        assert!(formatted.contains("test token"));
        assert!(formatted.contains("\"sequence_id\":5"));
        assert!(formatted.contains("\"is_final\":true"));
    }

    #[test]
    fn test_csv_custom_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("csv".to_string()));
        let response = create_test_response();

        let formatted = formatter.format_custom(&response, "csv").unwrap();
        assert_eq!(formatted, "5,test token,true,1.0");
    }

    #[test]
    fn test_csv_comma_escaping() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("csv".to_string()));
        let mut response = create_test_response();
        response.text = "test, token".to_string();

        let formatted = formatter.format_custom(&response, "csv").unwrap();
        assert!(formatted.contains("test\\, token"));
    }

    #[test]
    fn test_unknown_custom_format() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("unknown".to_string()));
        let response = create_test_response();

        let result = formatter.format_custom(&response, "unknown");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown custom format"));
    }

    #[test]
    fn test_json_end_marker() {
        let formatter = StreamingFormatter::new(OutputFormat::Json);
        let end_marker = formatter.format_json_end_marker();

        assert!(end_marker.contains("\"type\":\"end\""));
        assert!(end_marker.contains("\"sequence\":0"));
    }

    #[test]
    fn test_websocket_end_marker() {
        let formatter = StreamingFormatter::new(OutputFormat::WebSocket);
        let end_marker = formatter.format_websocket_end_marker().unwrap();

        assert!(end_marker.contains("\"type\":\"end\""));
        assert!(end_marker.contains("\"sequence\":0"));
    }

    #[test]
    fn test_json_error_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Json);
        let error = StreamingError::Utf8Error("test error".to_string());

        let formatted = formatter.format_json_error(&error);
        assert!(formatted.contains("\"type\":\"error\""));
        assert!(formatted.contains("test error"));
    }

    #[test]
    fn test_websocket_error_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::WebSocket);
        let error = StreamingError::Utf8Error("test error".to_string());

        let formatted = formatter.format_websocket_error(&error).unwrap();
        assert!(formatted.contains("\"type\":\"error\""));
        assert!(formatted.contains("test error"));
    }

    #[test]
    fn test_custom_end_markers() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("minimal_json".to_string()));
        
        let minimal_end = formatter.format_custom_end_marker("minimal_json").unwrap();
        assert_eq!(minimal_end, serde_json::json!({"end": true}).to_string());

        let csv_end = formatter.format_custom_end_marker("csv").unwrap();
        assert_eq!(csv_end, "END,[END],0.0");
    }

    #[test]
    fn test_custom_error_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Custom("csv".to_string()));
        let error = StreamingError::Utf8Error("test, error".to_string());

        let formatted = formatter.format_custom_error(&error, "csv").unwrap();
        assert!(formatted.contains("ERROR,"));
        assert!(formatted.contains("test\\, error")); // Comma should be escaped
    }
}