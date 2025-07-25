//! Core types and enums for streaming format handling
//!
//! Defines the fundamental types used across all streaming format implementations.
//! All types are zero-allocation where possible and use Arc<str> for string sharing.

use std::fmt;

use serde::{Deserialize, Serialize};

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

/// Core streaming formatter that converts token responses to different output formats
#[derive(Debug)]
pub struct StreamingFormatter {
    pub(crate) format: OutputFormat,
    pub(crate) sequence_number: u64,
    pub(crate) event_id_prefix: String,
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
        matches!(
            self.format,
            OutputFormat::Json
                | OutputFormat::ServerSentEvents
                | OutputFormat::WebSocket
                | OutputFormat::Custom(_)
        )
    }

    /// Check if format is suitable for web streaming
    #[inline]
    pub fn is_web_compatible(&self) -> bool {
        matches!(
            self.format,
            OutputFormat::Json | OutputFormat::ServerSentEvents | OutputFormat::WebSocket
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
            OutputFormat::Custom(format_name) => match format_name.as_str() {
                "minimal_json" => "application/json",
                "csv" => "text/csv",
                _ => "application/octet-stream",
            },
        }
    }

    /// Increment sequence number (used by format implementations)
    #[inline]
    pub(crate) fn increment_sequence(&mut self) {
        self.sequence_number += 1;
    }
}

/// Format utility functions and constants
pub mod format_constants {
    use super::OutputFormat;

    /// Get all supported format names
    #[inline]
    pub fn supported_formats() -> Vec<&'static str> {
        vec!["json", "sse", "websocket", "text", "raw", "custom:*"]
    }

    /// Check if format requires buffering
    #[inline]
    pub fn requires_buffering(format: &OutputFormat) -> bool {
        matches!(
            format,
            OutputFormat::ServerSentEvents | OutputFormat::WebSocket
        )
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

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::ServerSentEvents.to_string(), "sse");
        assert_eq!(OutputFormat::WebSocket.to_string(), "websocket");
        assert_eq!(OutputFormat::PlainText.to_string(), "text");
        assert_eq!(OutputFormat::Raw.to_string(), "raw");
        assert_eq!(
            OutputFormat::Custom("test".to_string()).to_string(),
            "custom:test"
        );
    }

    #[test]
    fn test_output_format_default() {
        assert_eq!(OutputFormat::default(), OutputFormat::Json);
    }

    #[test]
    fn test_streaming_formatter_creation() {
        let formatter = StreamingFormatter::new(OutputFormat::Json);
        assert_eq!(formatter.format(), &OutputFormat::Json);
        assert_eq!(formatter.sequence_number(), 0);
    }

    #[test]
    fn test_event_id_prefix() {
        let mut formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        formatter.set_event_id_prefix("custom".to_string());
        assert_eq!(formatter.event_id_prefix, "custom");
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
    fn test_sequence_reset() {
        let mut formatter = StreamingFormatter::new(OutputFormat::Json);
        formatter.increment_sequence();
        formatter.increment_sequence();
        assert_eq!(formatter.sequence_number(), 2);

        formatter.reset_sequence();
        assert_eq!(formatter.sequence_number(), 0);
    }

    #[test]
    fn test_buffer_size_optimization() {
        assert_eq!(format_constants::optimal_buffer_size(&OutputFormat::Raw), 256);
        assert_eq!(
            format_constants::optimal_buffer_size(&OutputFormat::PlainText),
            512
        );
        assert_eq!(format_constants::optimal_buffer_size(&OutputFormat::Json), 1024);
        assert_eq!(
            format_constants::optimal_buffer_size(&OutputFormat::ServerSentEvents),
            2048
        );
    }

    #[test]
    fn test_buffering_requirements() {
        assert!(format_constants::requires_buffering(
            &OutputFormat::ServerSentEvents
        ));
        assert!(format_constants::requires_buffering(&OutputFormat::WebSocket));
        assert!(!format_constants::requires_buffering(&OutputFormat::PlainText));
        assert!(!format_constants::requires_buffering(&OutputFormat::Raw));
    }
}