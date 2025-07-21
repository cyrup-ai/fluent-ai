//! Output format handling for streaming responses
//!
//! Provides multiple output format support for streaming token responses:
//! - JSON format for structured API responses
//! - Server-Sent Events (SSE) for web browser streaming
//! - WebSocket messages for real-time applications
//! - Plain text for simple streaming applications
//! - Raw format for high-performance scenarios

use std::fmt;
use serde::{Serialize, Deserialize};
use crate::streaming::{StreamingTokenResponse, StreamingError};

/// Supported output formats for streaming
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// JSON format with full metadata
    Json,
    /// Server-Sent Events format for web streaming
    ServerSentEvents,
    /// WebSocket message format
    WebSocket,
    /// Plain text content only
    PlainText,
    /// Raw format with minimal overhead
    Raw,
    /// Custom format with user-defined structure
    Custom(String),
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::ServerSentEvents => write!(f, "sse"),
            OutputFormat::WebSocket => write!(f, "websocket"),
            OutputFormat::PlainText => write!(f, "text"),
            OutputFormat::Raw => write!(f, "raw"),
            OutputFormat::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Json
    }
}

/// Streaming formatter that converts token responses to different output formats
pub struct StreamingFormatter {
    format: OutputFormat,
    sequence_number: u64,
    event_id_prefix: String,
}

impl StreamingFormatter {
    /// Create a new formatter with specified output format
    #[inline]
    pub fn new(format: OutputFormat) -> Self {
        Self {
            format,
            sequence_number: 0,
            event_id_prefix: "token".to_string(),
        }
    }

    /// Set custom event ID prefix for SSE format
    #[inline]
    pub fn set_event_id_prefix(&mut self, prefix: String) {
        self.event_id_prefix = prefix;
    }

    /// Format a streaming token response according to the configured format
    #[inline]
    pub fn format_response(&mut self, response: &StreamingTokenResponse) -> Result<String, StreamingError> {
        self.sequence_number += 1;
        
        match &self.format {
            OutputFormat::Json => self.format_json(response),
            OutputFormat::ServerSentEvents => self.format_sse(response),
            OutputFormat::WebSocket => self.format_websocket(response),
            OutputFormat::PlainText => self.format_plain_text(response),
            OutputFormat::Raw => self.format_raw(response),
            OutputFormat::Custom(format_name) => self.format_custom(response, format_name),
        }
    }

    /// Format as JSON with full metadata
    #[inline]
    fn format_json(&self, response: &StreamingTokenResponse) -> Result<String, StreamingError> {
        serde_json::to_string(response)
            .map_err(|e| StreamingError::FormatError(format!("JSON serialization failed: {}", e)))
    }

    /// Format as Server-Sent Events
    #[inline]
    fn format_sse(&self, response: &StreamingTokenResponse) -> Result<String, StreamingError> {
        let data = serde_json::to_string(response)
            .map_err(|e| StreamingError::FormatError(format!("SSE JSON serialization failed: {}", e)))?;
        
        let mut sse_message = String::with_capacity(data.len() + 100);
        sse_message.push_str(&format!("id: {}-{}\n", self.event_id_prefix, self.sequence_number));
        sse_message.push_str("event: token\n");
        
        // Handle multi-line data by prefixing each line with "data: "
        for line in data.lines() {
            sse_message.push_str("data: ");
            sse_message.push_str(line);
            sse_message.push('\n');
        }
        sse_message.push('\n'); // Extra newline to mark end of SSE event
        
        Ok(sse_message)
    }

    /// Format as WebSocket message
    #[inline]
    fn format_websocket(&self, response: &StreamingTokenResponse) -> Result<String, StreamingError> {
        // WebSocket message with type indicator
        let message = serde_json::json!({
            "type": "token",
            "sequence": self.sequence_number,
            "data": response
        });
        
        serde_json::to_string(&message)
            .map_err(|e| StreamingError::FormatError(format!("WebSocket JSON serialization failed: {}", e)))
    }

    /// Format as plain text (content only)
    #[inline]
    fn format_plain_text(&self, response: &StreamingTokenResponse) -> Result<String, StreamingError> {
        Ok(response.content.clone())
    }

    /// Format as raw data with minimal overhead
    #[inline]
    fn format_raw(&self, response: &StreamingTokenResponse) -> Result<String, StreamingError> {
        // Raw format: token_id|position|content
        Ok(format!("{}|{}|{}", 
            response.token_id.unwrap_or(0),
            response.position,
            response.content
        ))
    }

    /// Format with custom format (extensible for future formats)
    #[inline]
    fn format_custom(&self, response: &StreamingTokenResponse, format_name: &str) -> Result<String, StreamingError> {
        match format_name {
            "minimal_json" => {
                // Minimal JSON with only essential fields
                let minimal = serde_json::json!({
                    "content": response.content,
                    "position": response.position,
                    "is_complete": response.is_complete_token
                });
                serde_json::to_string(&minimal)
                    .map_err(|e| StreamingError::FormatError(format!("Custom JSON serialization failed: {}", e)))
            },
            "csv" => {
                // CSV format: position,content,is_complete,probability
                Ok(format!("{},{},{},{}", 
                    response.position,
                    response.content.replace(',', "\\,"), // Escape commas
                    response.is_complete_token,
                    response.probability.unwrap_or(0.0)
                ))
            },
            _ => Err(StreamingError::FormatError(
                format!("Unknown custom format: {}", format_name)
            ))
        }
    }

    /// Format end-of-stream marker
    #[inline]
    pub fn format_end_marker(&mut self) -> Result<String, StreamingError> {
        self.sequence_number += 1;
        
        match &self.format {
            OutputFormat::Json => Ok(serde_json::json!({"type": "end", "sequence": self.sequence_number}).to_string()),
            OutputFormat::ServerSentEvents => {
                Ok(format!("id: {}-{}\nevent: end\ndata: {{}}\n\n", self.event_id_prefix, self.sequence_number))
            },
            OutputFormat::WebSocket => {
                let message = serde_json::json!({
                    "type": "end",
                    "sequence": self.sequence_number
                });
                serde_json::to_string(&message)
                    .map_err(|e| StreamingError::FormatError(format!("End marker serialization failed: {}", e)))
            },
            OutputFormat::PlainText => Ok("\n[END]\n".to_string()),
            OutputFormat::Raw => Ok("END|0|".to_string()),
            OutputFormat::Custom(format_name) => {
                match format_name.as_str() {
                    "minimal_json" => Ok(serde_json::json!({"end": true}).to_string()),
                    "csv" => Ok("END,,[END],0.0".to_string()),
                    _ => Err(StreamingError::FormatError(
                        format!("Unknown custom format for end marker: {}", format_name)
                    ))
                }
            }
        }
    }

    /// Format error message for streaming
    #[inline]
    pub fn format_error(&mut self, error: &StreamingError) -> Result<String, StreamingError> {
        self.sequence_number += 1;
        
        match &self.format {
            OutputFormat::Json => {
                let error_response = serde_json::json!({
                    "type": "error",
                    "sequence": self.sequence_number,
                    "error": error.to_string()
                });
                Ok(error_response.to_string())
            },
            OutputFormat::ServerSentEvents => {
                Ok(format!("id: {}-{}\nevent: error\ndata: {{}}\n\n", self.event_id_prefix, self.sequence_number))
            },
            OutputFormat::WebSocket => {
                let message = serde_json::json!({
                    "type": "error",
                    "sequence": self.sequence_number,
                    "error": error.to_string()
                });
                serde_json::to_string(&message)
                    .map_err(|e| StreamingError::FormatError(format!("Error message serialization failed: {}", e)))
            },
            OutputFormat::PlainText => Ok(format!("\n[ERROR: {}]\n", error)),
            OutputFormat::Raw => Ok(format!("ERROR|0|{}", error)),
            OutputFormat::Custom(format_name) => {
                match format_name.as_str() {
                    "minimal_json" => Ok(serde_json::json!({"error": error.to_string()}).to_string()),
                    "csv" => Ok(format!("ERROR,,[ERROR: {}],0.0", error.to_string().replace(',', "\\,"))),
                    _ => Err(StreamingError::FormatError(
                        format!("Unknown custom format for error: {}", format_name)
                    ))
                }
            }
        }
    }

    /// Get current format
    #[inline]
    pub fn format(&self) -> &OutputFormat {
        &self.format
    }

    /// Get current sequence number
    #[inline]
    pub fn sequence_number(&self) -> u64 {
        self.sequence_number
    }

    /// Reset sequence number (useful for new streams)
    #[inline]
    pub fn reset_sequence(&mut self) {
        self.sequence_number = 0;
    }

    /// Check if format supports metadata
    #[inline]
    pub fn supports_metadata(&self) -> bool {
        matches!(self.format, 
            OutputFormat::Json | 
            OutputFormat::ServerSentEvents | 
            OutputFormat::WebSocket |
            OutputFormat::Custom(_)
        )
    }

    /// Check if format is suitable for web streaming
    #[inline]
    pub fn is_web_compatible(&self) -> bool {
        matches!(self.format, 
            OutputFormat::Json | 
            OutputFormat::ServerSentEvents | 
            OutputFormat::WebSocket
        )
    }

    /// Get content-type header for HTTP responses
    #[inline]
    pub fn content_type(&self) -> &'static str {
        match &self.format {
            OutputFormat::Json => "application/json",
            OutputFormat::ServerSentEvents => "text/event-stream",
            OutputFormat::WebSocket => "application/json", // WebSocket upgrade changes this
            OutputFormat::PlainText => "text/plain",
            OutputFormat::Raw => "application/octet-stream",
            OutputFormat::Custom(format_name) => {
                match format_name.as_str() {
                    "minimal_json" => "application/json",
                    "csv" => "text/csv",
                    _ => "application/octet-stream"
                }
            }
        }
    }
}

/// Utility functions for format handling
pub mod format_utils {
    use super::*;

    /// Parse output format from string
    #[inline]
    pub fn parse_format(format_str: &str) -> Result<OutputFormat, StreamingError> {
        match format_str.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "sse" | "server-sent-events" => Ok(OutputFormat::ServerSentEvents),
            "websocket" | "ws" => Ok(OutputFormat::WebSocket),
            "text" | "plain" | "plaintext" => Ok(OutputFormat::PlainText),
            "raw" => Ok(OutputFormat::Raw),
            custom if custom.starts_with("custom:") => {
                let name = custom.strip_prefix("custom:").unwrap_or(custom);
                Ok(OutputFormat::Custom(name.to_string()))
            },
            _ => Err(StreamingError::FormatError(
                format!("Unknown output format: {}", format_str)
            ))
        }
    }

    /// Get all supported format names
    #[inline]
    pub fn supported_formats() -> Vec<&'static str> {
        vec!["json", "sse", "websocket", "text", "raw", "custom:*"]
    }

    /// Check if format requires buffering
    #[inline]
    pub fn requires_buffering(format: &OutputFormat) -> bool {
        matches!(format, OutputFormat::ServerSentEvents | OutputFormat::WebSocket)
    }

    /// Get optimal buffer size for format
    #[inline]
    pub fn optimal_buffer_size(format: &OutputFormat) -> usize {
        match format {
            OutputFormat::Json => 1024,
            OutputFormat::ServerSentEvents => 2048, // SSE has overhead
            OutputFormat::WebSocket => 1024,
            OutputFormat::PlainText => 512,
            OutputFormat::Raw => 256, // Minimal overhead
            OutputFormat::Custom(_) => 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{TokenTiming, TokenMetadata};

    fn create_test_response() -> StreamingTokenResponse {
        StreamingTokenResponse {
            content: "test token".to_string(),
            token_id: Some(123),
            position: 5,
            is_complete_token: true,
            probability: Some(0.95),
            alternatives: None,
            timing: TokenTiming::default(),
            metadata: TokenMetadata::default(),
        }
    }

    #[test]
    fn test_json_formatting() {
        let mut formatter = StreamingFormatter::new(OutputFormat::Json);
        let response = create_test_response();
        
        let formatted = formatter.format_response(&response).unwrap();
        assert!(formatted.contains("test token"));
        assert!(formatted.contains("123"));
        assert!(formatted.contains("0.95"));
    }

    #[test]
    fn test_sse_formatting() {
        let mut formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        let response = create_test_response();
        
        let formatted = formatter.format_response(&response).unwrap();
        assert!(formatted.starts_with("id: token-1"));
        assert!(formatted.contains("event: token"));
        assert!(formatted.contains("data: "));
        assert!(formatted.ends_with("\n\n"));
    }

    #[test]
    fn test_websocket_formatting() {
        let mut formatter = StreamingFormatter::new(OutputFormat::WebSocket);
        let response = create_test_response();
        
        let formatted = formatter.format_response(&response).unwrap();
        assert!(formatted.contains("\"type\":\"token\""));
        assert!(formatted.contains("\"sequence\":1"));
    }

    #[test]
    fn test_plain_text_formatting() {
        let mut formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let response = create_test_response();
        
        let formatted = formatter.format_response(&response).unwrap();
        assert_eq!(formatted, "test token");
    }

    #[test]
    fn test_raw_formatting() {
        let mut formatter = StreamingFormatter::new(OutputFormat::Raw);
        let response = create_test_response();
        
        let formatted = formatter.format_response(&response).unwrap();
        assert_eq!(formatted, "123|5|test token");
    }

    #[test]
    fn test_custom_formatting() {
        let mut formatter = StreamingFormatter::new(OutputFormat::Custom("minimal_json".to_string()));
        let response = create_test_response();
        
        let formatted = formatter.format_response(&response).unwrap();
        assert!(formatted.contains("test token"));
        assert!(formatted.contains("\"position\":5"));
        assert!(formatted.contains("\"is_complete\":true"));
    }

    #[test]
    fn test_csv_custom_formatting() {
        let mut formatter = StreamingFormatter::new(OutputFormat::Custom("csv".to_string()));
        let response = create_test_response();
        
        let formatted = formatter.format_response(&response).unwrap();
        assert_eq!(formatted, "5,test token,true,0.95");
    }

    #[test]
    fn test_end_marker_formatting() {
        let mut formatter = StreamingFormatter::new(OutputFormat::Json);
        let end_marker = formatter.format_end_marker().unwrap();
        
        assert!(end_marker.contains("\"type\":\"end\""));
        assert!(end_marker.contains("\"sequence\":1"));
    }

    #[test]
    fn test_error_formatting() {
        let mut formatter = StreamingFormatter::new(OutputFormat::Json);
        let error = StreamingError::Utf8Error("test error".to_string());
        
        let formatted = formatter.format_error(&error).unwrap();
        assert!(formatted.contains("\"type\":\"error\""));
        assert!(formatted.contains("test error"));
    }

    #[test]
    fn test_sequence_numbering() {
        let mut formatter = StreamingFormatter::new(OutputFormat::Json);
        let response = create_test_response();
        
        formatter.format_response(&response).unwrap();
        assert_eq!(formatter.sequence_number(), 1);
        
        formatter.format_response(&response).unwrap();
        assert_eq!(formatter.sequence_number(), 2);
        
        formatter.reset_sequence();
        assert_eq!(formatter.sequence_number(), 0);
    }

    #[test]
    fn test_metadata_support() {
        let json_formatter = StreamingFormatter::new(OutputFormat::Json);
        assert!(json_formatter.supports_metadata());
        
        let text_formatter = StreamingFormatter::new(OutputFormat::PlainText);
        assert!(!text_formatter.supports_metadata());
    }

    #[test]
    fn test_web_compatibility() {
        let sse_formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        assert!(sse_formatter.is_web_compatible());
        
        let raw_formatter = StreamingFormatter::new(OutputFormat::Raw);
        assert!(!raw_formatter.is_web_compatible());
    }

    #[test]
    fn test_content_type() {
        let json_formatter = StreamingFormatter::new(OutputFormat::Json);
        assert_eq!(json_formatter.content_type(), "application/json");
        
        let sse_formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        assert_eq!(sse_formatter.content_type(), "text/event-stream");
    }

    #[test]
    fn test_format_parsing() {
        assert_eq!(format_utils::parse_format("json").unwrap(), OutputFormat::Json);
        assert_eq!(format_utils::parse_format("sse").unwrap(), OutputFormat::ServerSentEvents);
        assert_eq!(format_utils::parse_format("websocket").unwrap(), OutputFormat::WebSocket);
        assert_eq!(format_utils::parse_format("text").unwrap(), OutputFormat::PlainText);
        assert_eq!(format_utils::parse_format("raw").unwrap(), OutputFormat::Raw);
        
        if let OutputFormat::Custom(name) = format_utils::parse_format("custom:test").unwrap() {
            assert_eq!(name, "test");
        } else {
            panic!("Expected custom format");
        }
        
        assert!(format_utils::parse_format("invalid").is_err());
    }

    #[test]
    fn test_buffer_size_optimization() {
        assert_eq!(format_utils::optimal_buffer_size(&OutputFormat::Raw), 256);
        assert_eq!(format_utils::optimal_buffer_size(&OutputFormat::PlainText), 512);
        assert_eq!(format_utils::optimal_buffer_size(&OutputFormat::Json), 1024);
        assert_eq!(format_utils::optimal_buffer_size(&OutputFormat::ServerSentEvents), 2048);
    }

    #[test]
    fn test_buffering_requirements() {
        assert!(format_utils::requires_buffering(&OutputFormat::ServerSentEvents));
        assert!(format_utils::requires_buffering(&OutputFormat::WebSocket));
        assert!(!format_utils::requires_buffering(&OutputFormat::PlainText));
        assert!(!format_utils::requires_buffering(&OutputFormat::Raw));
    }
}