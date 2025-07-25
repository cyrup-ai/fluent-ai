//! Output format handling for streaming responses
//!
//! Provides multiple output format support for streaming token responses:
//! - JSON format for structured API responses
//! - Server-Sent Events (SSE) for web browser streaming
//! - WebSocket messages for real-time applications
//! - Plain text for simple streaming applications
//! - Raw format for high-performance scenarios
//!
//! ## Architecture
//! The module is decomposed into focused submodules:
//! - `types`: Core format types and StreamingFormatter struct
//! - `json`: JSON and WebSocket format implementations
//! - `text`: Plain text, SSE, and raw format implementations
//! - `utils`: Utility functions, buffer management, and performance tools
//!
//! ## Usage
//! ```rust
//! use fluent_ai_candle::streaming::formats::{OutputFormat, StreamingFormatter};
//!
//! let mut formatter = StreamingFormatter::new(OutputFormat::Json);
//! let formatted = formatter.format_response(&response)?;
//! ```

// Core modules
pub mod types;
pub mod json;
pub mod text;
pub mod utils;

// Re-export commonly used types for ergonomic API
pub use types::{
    OutputFormat,
    StreamingFormatter,
    format_constants,
};

// Re-export utility modules for advanced usage
pub use utils::{
    format_utils,
    buffer_utils,
    perf_utils,
};

pub use text::text_utils;

// Legacy compatibility re-exports to maintain API compatibility
pub use format_utils::*;

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::streaming::{StreamingTokenResponse, TokenMetadata, TokenTiming, StreamingError};

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
    fn test_format_integration() {
        let response = create_test_response();

        // Test JSON formatting
        let mut json_formatter = StreamingFormatter::new(OutputFormat::Json);
        let json_result = json_formatter.format_response(&response).unwrap();
        assert!(json_result.contains("test token"));
        assert_eq!(json_formatter.sequence_number(), 1);

        // Test SSE formatting
        let mut sse_formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        let sse_result = sse_formatter.format_response(&response).unwrap();
        assert!(sse_result.starts_with("id: token-1"));
        assert!(sse_result.contains("event: token"));

        // Test WebSocket formatting
        let mut ws_formatter = StreamingFormatter::new(OutputFormat::WebSocket);
        let ws_result = ws_formatter.format_response(&response).unwrap();
        assert!(ws_result.contains("\"type\":\"token\""));

        // Test plain text formatting
        let mut text_formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let text_result = text_formatter.format_response(&response).unwrap();
        assert_eq!(text_result, "test token");

        // Test raw formatting
        let mut raw_formatter = StreamingFormatter::new(OutputFormat::Raw);
        let raw_result = raw_formatter.format_response(&response).unwrap();
        assert_eq!(raw_result, "5|0|test token");
    }

    #[test]
    fn test_end_marker_integration() {
        // Test JSON end marker
        let mut json_formatter = StreamingFormatter::new(OutputFormat::Json);
        let json_end = json_formatter.format_end_marker().unwrap();
        assert!(json_end.contains("\"type\":\"end\""));

        // Test SSE end marker
        let mut sse_formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        let sse_end = sse_formatter.format_end_marker().unwrap();
        assert!(sse_end.contains("event: end"));

        // Test WebSocket end marker
        let mut ws_formatter = StreamingFormatter::new(OutputFormat::WebSocket);
        let ws_end = ws_formatter.format_end_marker().unwrap();
        assert!(ws_end.contains("\"type\":\"end\""));

        // Test plain text end marker
        let mut text_formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let text_end = text_formatter.format_end_marker().unwrap();
        assert_eq!(text_end, "\n[END]\n");

        // Test raw end marker
        let mut raw_formatter = StreamingFormatter::new(OutputFormat::Raw);
        let raw_end = raw_formatter.format_end_marker().unwrap();
        assert_eq!(raw_end, "END|0|");
    }

    #[test]
    fn test_error_handling_integration() {
        let error = StreamingError::Utf8Error("test error".to_string());

        // Test JSON error formatting
        let mut json_formatter = StreamingFormatter::new(OutputFormat::Json);
        let json_error = json_formatter.format_error(&error).unwrap();
        assert!(json_error.contains("\"type\":\"error\""));
        assert!(json_error.contains("test error"));

        // Test SSE error formatting
        let mut sse_formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        let sse_error = sse_formatter.format_error(&error).unwrap();
        assert!(sse_error.contains("event: error"));

        // Test WebSocket error formatting
        let mut ws_formatter = StreamingFormatter::new(OutputFormat::WebSocket);
        let ws_error = ws_formatter.format_error(&error).unwrap();
        assert!(ws_error.contains("\"type\":\"error\""));

        // Test plain text error formatting
        let mut text_formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let text_error = text_formatter.format_error(&error).unwrap();
        assert!(text_error.contains("[ERROR: test error]"));

        // Test raw error formatting
        let mut raw_formatter = StreamingFormatter::new(OutputFormat::Raw);
        let raw_error = raw_formatter.format_error(&error).unwrap();
        assert_eq!(raw_error, "ERROR|0|test error");
    }

    #[test]
    fn test_custom_format_integration() {
        let response = create_test_response();

        // Test minimal JSON custom format
        let mut minimal_formatter = StreamingFormatter::new(OutputFormat::Custom("minimal_json".to_string()));
        let minimal_result = minimal_formatter.format_response(&response).unwrap();
        assert!(minimal_result.contains("test token"));
        assert!(minimal_result.contains("\"sequence_id\":5"));

        // Test CSV custom format
        let mut csv_formatter = StreamingFormatter::new(OutputFormat::Custom("csv".to_string()));
        let csv_result = csv_formatter.format_response(&response).unwrap();
        assert_eq!(csv_result, "5,test token,true,1.0");
    }

    #[test]
    fn test_format_metadata_integration() {
        // Test metadata support
        let json_formatter = StreamingFormatter::new(OutputFormat::Json);
        assert!(json_formatter.supports_metadata());

        let sse_formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        assert!(sse_formatter.supports_metadata());

        let text_formatter = StreamingFormatter::new(OutputFormat::PlainText);
        assert!(!text_formatter.supports_metadata());

        // Test web compatibility
        let ws_formatter = StreamingFormatter::new(OutputFormat::WebSocket);
        assert!(ws_formatter.is_web_compatible());

        let raw_formatter = StreamingFormatter::new(OutputFormat::Raw);
        assert!(!raw_formatter.is_web_compatible());
    }

    #[test]
    fn test_content_type_integration() {
        let json_formatter = StreamingFormatter::new(OutputFormat::Json);
        assert_eq!(json_formatter.content_type(), "application/json");

        let sse_formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        assert_eq!(sse_formatter.content_type(), "text/event-stream");

        let text_formatter = StreamingFormatter::new(OutputFormat::PlainText);
        assert_eq!(text_formatter.content_type(), "text/plain");

        let raw_formatter = StreamingFormatter::new(OutputFormat::Raw);
        assert_eq!(raw_formatter.content_type(), "application/octet-stream");
    }

    #[test]
    fn test_sequence_number_consistency() {
        let response = create_test_response();
        let mut formatter = StreamingFormatter::new(OutputFormat::Json);

        // First response
        formatter.format_response(&response).unwrap();
        assert_eq!(formatter.sequence_number(), 1);

        // Second response
        formatter.format_response(&response).unwrap();
        assert_eq!(formatter.sequence_number(), 2);

        // End marker
        formatter.format_end_marker().unwrap();
        assert_eq!(formatter.sequence_number(), 3);

        // Reset sequence
        formatter.reset_sequence();
        assert_eq!(formatter.sequence_number(), 0);
    }

    #[test]
    fn test_buffer_optimization_integration() {
        // Test optimal buffer sizes
        assert_eq!(format_utils::optimal_buffer_size(&OutputFormat::Raw), 256);
        assert_eq!(format_utils::optimal_buffer_size(&OutputFormat::PlainText), 512);
        assert_eq!(format_utils::optimal_buffer_size(&OutputFormat::Json), 1024);
        assert_eq!(format_utils::optimal_buffer_size(&OutputFormat::ServerSentEvents), 2048);

        // Test buffering requirements
        assert!(format_utils::requires_buffering(&OutputFormat::ServerSentEvents));
        assert!(format_utils::requires_buffering(&OutputFormat::WebSocket));
        assert!(!format_utils::requires_buffering(&OutputFormat::PlainText));
        assert!(!format_utils::requires_buffering(&OutputFormat::Raw));
    }

    #[test]
    fn test_format_parsing_integration() {
        // Test case-insensitive parsing
        assert_eq!(format_utils::parse_format("JSON").unwrap(), OutputFormat::Json);
        assert_eq!(format_utils::parse_format("sse").unwrap(), OutputFormat::ServerSentEvents);
        assert_eq!(format_utils::parse_format("WebSocket").unwrap(), OutputFormat::WebSocket);
        assert_eq!(format_utils::parse_format("PLAIN").unwrap(), OutputFormat::PlainText);
        assert_eq!(format_utils::parse_format("raw").unwrap(), OutputFormat::Raw);

        // Test custom format parsing
        if let OutputFormat::Custom(name) = format_utils::parse_format("custom:test").unwrap() {
            assert_eq!(name, "test");
        } else {
            panic!("Expected custom format");
        }

        // Test invalid format
        assert!(format_utils::parse_format("invalid").is_err());
    }
}