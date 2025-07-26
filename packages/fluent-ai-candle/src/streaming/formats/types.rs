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
    Custom(String)}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::ServerSentEvents => write!(f, "sse"),
            OutputFormat::WebSocket => write!(f, "websocket"),
            OutputFormat::PlainText => write!(f, "text"),
            OutputFormat::Raw => write!(f, "raw"),
            OutputFormat::Custom(name) => write!(f, "custom:{}", name)}
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
    pub(crate) event_id_prefix: String}

impl StreamingFormatter {
    /// Create a new formatter with specified output format
    #[inline]
    pub fn new(format: OutputFormat) -> Self {
        Self {
            format,
            sequence_number: 0,
            event_id_prefix: "token".to_string()}
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
                _ => "application/octet-stream"}}
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
            OutputFormat::Custom(_) => 1024}
    }
}
