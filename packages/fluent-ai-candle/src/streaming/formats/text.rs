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
    pub fn format_sse_error(&self, _error: &StreamingError) -> String {
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
            }}
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
            }}
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
            }}
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
            )))}
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
