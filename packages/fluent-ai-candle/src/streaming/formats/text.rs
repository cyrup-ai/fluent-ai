//! Text-based format implementations for streaming
//!
//! Provides plain text, Server-Sent Events (SSE), and raw format capabilities
//! with zero-allocation patterns and optimized string handling.

use crate::streaming::{StreamingError, StreamingTokenResponse};
use super::types::StreamingFormatter;

impl StreamingFormatter {
    /// Format as Server-Sent Events
    #[inline]
    pub fn format_sse(&self, response: &StreamingTokenResponse) -> Result<String, StreamingError> {
        let data = serde_json::to_string(response).map_err(|e| {
            StreamingError::FormatError(format!("SSE JSON serialization failed: {}", e))
        })?;

        let mut sse_message = String::with_capacity(data.len() + 100);
        sse_message.push_str(&format!(
            "id: {}-{}\n",
            self.event_id_prefix, self.sequence_number
        ));
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

    /// Format as plain text (content only)
    #[inline]
    pub fn format_plain_text(
        &self,
        response: &StreamingTokenResponse,
    ) -> Result<String, StreamingError> {
        Ok(response.text.clone())
    }

    /// Format as raw data with minimal overhead
    #[inline]
    pub fn format_raw(&self, response: &StreamingTokenResponse) -> Result<String, StreamingError> {
        // Raw format: sequence_id|timestamp|text
        Ok(format!(
            "{}|{}|{}",
            response.sequence_id, response.timestamp, response.text
        ))
    }

    /// Format SSE end marker
    #[inline]
    pub fn format_sse_end_marker(&self) -> String {
        format!(
            "id: {}-{}\nevent: end\ndata: {{}}\n\n",
            self.event_id_prefix, self.sequence_number
        )
    }

    /// Format plain text end marker
    #[inline]
    pub fn format_plain_text_end_marker(&self) -> String {
        "\n[END]\n".to_string()
    }

    /// Format raw end marker
    #[inline]
    pub fn format_raw_end_marker(&self) -> String {
        "END|0|".to_string()
    }

    /// Format SSE error message
    #[inline]
    pub fn format_sse_error(&self, error: &StreamingError) -> String {
        format!(
            "id: {}-{}\nevent: error\ndata: {{}}\n\n",
            self.event_id_prefix, self.sequence_number
        )
    }

    /// Format plain text error message
    #[inline]
    pub fn format_plain_text_error(&self, error: &StreamingError) -> String {
        format!("\n[ERROR: {}]\n", error)
    }

    /// Format raw error message
    #[inline]
    pub fn format_raw_error(&self, error: &StreamingError) -> String {
        format!("ERROR|0|{}", error)
    }

    /// Format response using the configured format with unified interface
    #[inline]
    pub fn format_response(
        &mut self,
        response: &StreamingTokenResponse,
    ) -> Result<String, StreamingError> {
        self.increment_sequence();

        match &self.format {
            crate::streaming::formats::types::OutputFormat::Json => self.format_json(response),
            crate::streaming::formats::types::OutputFormat::ServerSentEvents => self.format_sse(response),
            crate::streaming::formats::types::OutputFormat::WebSocket => self.format_websocket(response),
            crate::streaming::formats::types::OutputFormat::PlainText => self.format_plain_text(response),
            crate::streaming::formats::types::OutputFormat::Raw => self.format_raw(response),
            crate::streaming::formats::types::OutputFormat::Custom(format_name) => {
                self.format_custom(response, format_name)
            },
        }
    }

    /// Format end-of-stream marker using the configured format
    #[inline]
    pub fn format_end_marker(&mut self) -> Result<String, StreamingError> {
        self.increment_sequence();

        match &self.format {
            crate::streaming::formats::types::OutputFormat::Json => Ok(self.format_json_end_marker()),
            crate::streaming::formats::types::OutputFormat::ServerSentEvents => Ok(self.format_sse_end_marker()),
            crate::streaming::formats::types::OutputFormat::WebSocket => self.format_websocket_end_marker(),
            crate::streaming::formats::types::OutputFormat::PlainText => Ok(self.format_plain_text_end_marker()),
            crate::streaming::formats::types::OutputFormat::Raw => Ok(self.format_raw_end_marker()),
            crate::streaming::formats::types::OutputFormat::Custom(format_name) => {
                self.format_custom_end_marker(format_name)
            },
        }
    }

    /// Format error message using the configured format
    #[inline]
    pub fn format_error(&mut self, error: &StreamingError) -> Result<String, StreamingError> {
        self.increment_sequence();

        match &self.format {
            crate::streaming::formats::types::OutputFormat::Json => Ok(self.format_json_error(error)),
            crate::streaming::formats::types::OutputFormat::ServerSentEvents => Ok(self.format_sse_error(error)),
            crate::streaming::formats::types::OutputFormat::WebSocket => self.format_websocket_error(error),
            crate::streaming::formats::types::OutputFormat::PlainText => Ok(self.format_plain_text_error(error)),
            crate::streaming::formats::types::OutputFormat::Raw => Ok(self.format_raw_error(error)),
            crate::streaming::formats::types::OutputFormat::Custom(format_name) => {
                self.format_custom_error(error, format_name)
            },
        }
    }
}

/// Utility functions for text format parsing and handling
pub mod text_utils {
    use super::*;
    use crate::streaming::formats::types::OutputFormat;

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
            }
            _ => Err(StreamingError::FormatError(format!(
                "Unknown output format: {}",
                format_str
            ))),
        }
    }

    /// Escape special characters for SSE format
    #[inline]
    pub fn escape_sse_data(data: &str) -> String {
        data.replace('\n', "\\n").replace('\r', "\\r")
    }

    /// Escape special characters for raw format
    #[inline]
    pub fn escape_raw_data(data: &str) -> String {
        data.replace('|', "\\|").replace('\n', "\\n")
    }

    /// Split text into optimal chunks for streaming
    #[inline]
    pub fn chunk_text_for_streaming(text: &str, max_chunk_size: usize) -> Vec<&str> {
        if text.len() <= max_chunk_size {
            return vec![text];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = std::cmp::min(start + max_chunk_size, text.len());
            
            // Try to break at word boundaries
            let chunk_end = if end < text.len() {
                text[start..end]
                    .rfind(char::is_whitespace)
                    .map(|pos| start + pos + 1)
                    .unwrap_or(end)
            } else {
                end
            };

            chunks.push(&text[start..chunk_end]);
            start = chunk_end;
        }

        chunks
    }

    /// Validate SSE event format
    #[inline]
    pub fn validate_sse_format(data: &str) -> Result<(), StreamingError> {
        if data.contains("\n\n") && !data.ends_with("\n\n") {
            return Err(StreamingError::FormatError(
                "SSE data contains embedded event separators".to_string(),
            ));
        }
        Ok(())
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
            metadata: TokenMetadata::default(),
        }
    }

    #[test]
    fn test_sse_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        let response = create_test_response();

        let formatted = formatter.format_sse(&response).unwrap();
        assert!(formatted.starts_with("id: token-0"));
        assert!(formatted.contains("event: token"));
        assert!(formatted.contains("data: "));
        assert!(formatted.ends_with("\n\n"));
    }

    #[test]
    fn test_plain_text_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let response = create_test_response();

        let formatted = formatter.format_plain_text(&response).unwrap();
        assert_eq!(formatted, "test token");
    }

    #[test]
    fn test_raw_formatting() {
        let formatter = StreamingFormatter::new(OutputFormat::Raw);
        let response = create_test_response();

        let formatted = formatter.format_raw(&response).unwrap();
        assert_eq!(formatted, "5|0|test token");
    }

    #[test]
    fn test_unified_format_response() {
        let mut formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let response = create_test_response();

        let formatted = formatter.format_response(&response).unwrap();
        assert_eq!(formatted, "test token");
        assert_eq!(formatter.sequence_number(), 1);
    }

    #[test]
    fn test_end_markers() {
        let mut json_formatter = StreamingFormatter::new(OutputFormat::Json);
        let json_end = json_formatter.format_end_marker().unwrap();
        assert!(json_end.contains("\"type\":\"end\""));

        let mut sse_formatter = StreamingFormatter::new(OutputFormat::ServerSentEvents);
        let sse_end = sse_formatter.format_end_marker().unwrap();
        assert!(sse_end.contains("event: end"));

        let mut text_formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let text_end = text_formatter.format_end_marker().unwrap();
        assert_eq!(text_end, "\n[END]\n");

        let mut raw_formatter = StreamingFormatter::new(OutputFormat::Raw);
        let raw_end = raw_formatter.format_end_marker().unwrap();
        assert_eq!(raw_end, "END|0|");
    }

    #[test]
    fn test_error_formatting() {
        let mut formatter = StreamingFormatter::new(OutputFormat::PlainText);
        let error = StreamingError::Utf8Error("test error".to_string());

        let formatted = formatter.format_error(&error).unwrap();
        assert!(formatted.contains("ERROR: test error"));
    }

    #[test]
    fn test_format_parsing() {
        assert_eq!(
            text_utils::parse_format("json").unwrap(),
            OutputFormat::Json
        );
        assert_eq!(
            text_utils::parse_format("sse").unwrap(),
            OutputFormat::ServerSentEvents
        );
        assert_eq!(
            text_utils::parse_format("websocket").unwrap(),
            OutputFormat::WebSocket
        );
        assert_eq!(
            text_utils::parse_format("text").unwrap(),
            OutputFormat::PlainText
        );
        assert_eq!(
            text_utils::parse_format("raw").unwrap(),
            OutputFormat::Raw
        );

        if let OutputFormat::Custom(name) = text_utils::parse_format("custom:test").unwrap() {
            assert_eq!(name, "test");
        } else {
            panic!("Expected custom format");
        }

        assert!(text_utils::parse_format("invalid").is_err());
    }

    #[test]
    fn test_sse_data_escaping() {
        let data = "line1\nline2\rline3";
        let escaped = text_utils::escape_sse_data(data);
        assert_eq!(escaped, "line1\\nline2\\rline3");
    }

    #[test]
    fn test_raw_data_escaping() {
        let data = "field1|field2\nfield3";
        let escaped = text_utils::escape_raw_data(data);
        assert_eq!(escaped, "field1\\|field2\\nfield3");
    }

    #[test]
    fn test_text_chunking() {
        let text = "This is a long text that needs to be chunked for optimal streaming performance";
        let chunks = text_utils::chunk_text_for_streaming(text, 20);
        
        assert!(chunks.len() > 1);
        for chunk in chunks {
            assert!(chunk.len() <= 25); // Allow for word boundary adjustments
        }
    }

    #[test]
    fn test_sse_format_validation() {
        let valid_data = "data: some content\n\n";
        assert!(text_utils::validate_sse_format(valid_data).is_ok());

        let invalid_data = "data: some\n\ncontent in middle";
        assert!(text_utils::validate_sse_format(invalid_data).is_err());
    }
}