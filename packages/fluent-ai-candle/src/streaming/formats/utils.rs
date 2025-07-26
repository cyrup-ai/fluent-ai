//! Utility functions and performance optimizations for format handling
//!
//! Provides format parsing, buffer management, and performance optimizations
//! with zero-allocation patterns where possible.

use crate::streaming::StreamingError;
use super::types::OutputFormat;

/// Format utility functions and constants
pub mod format_utils {
    use super::*;

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

    /// Parse output format from string with case-insensitive matching
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

    /// Get MIME type for HTTP Content-Type header
    #[inline]
    pub fn get_mime_type(format: &OutputFormat) -> &'static str {
        match format {
            OutputFormat::Json => "application/json",
            OutputFormat::ServerSentEvents => "text/event-stream",
            OutputFormat::WebSocket => "application/json", // Before upgrade
            OutputFormat::PlainText => "text/plain; charset=utf-8",
            OutputFormat::Raw => "application/octet-stream",
            OutputFormat::Custom(format_name) => match format_name.as_str() {
                "minimal_json" => "application/json",
                "csv" => "text/csv; charset=utf-8",
                "xml" => "application/xml",
                "yaml" | "yml" => "application/yaml",
                _ => "application/octet-stream"}}
    }

    /// Check if format supports real-time streaming
    #[inline]
    pub fn supports_streaming(format: &OutputFormat) -> bool {
        matches!(
            format,
            OutputFormat::ServerSentEvents
                | OutputFormat::WebSocket
                | OutputFormat::Raw
                | OutputFormat::PlainText
        )
    }

    /// Get recommended cache headers for format
    #[inline]
    pub fn get_cache_headers(format: &OutputFormat) -> Vec<(&'static str, &'static str)> {
        match format {
            OutputFormat::ServerSentEvents => vec![
                ("Cache-Control", "no-cache"),
                ("Connection", "keep-alive"),
            ],
            OutputFormat::WebSocket => vec![
                ("Cache-Control", "no-cache"),
                ("Connection", "Upgrade"),
                ("Upgrade", "websocket"),
            ],
            OutputFormat::Json | OutputFormat::PlainText => vec![
                ("Cache-Control", "no-cache, no-store, must-revalidate"),
                ("Pragma", "no-cache"),
                ("Expires", "0"),
            ],
            OutputFormat::Raw => vec![("Cache-Control", "no-cache")],
            OutputFormat::Custom(_) => vec![("Cache-Control", "no-cache")]}
    }
}

/// Buffer management utilities for optimal performance
pub mod buffer_utils {
    use super::*;
    use std::io::Write;

    /// Pre-allocated buffer for string formatting
    pub struct FormatBuffer {
        buffer: Vec<u8>,
        capacity: usize}

    impl FormatBuffer {
        /// Create new buffer with format-specific capacity
        #[inline]
        pub fn new(format: &OutputFormat) -> Self {
            let capacity = format_utils::optimal_buffer_size(format);
            Self {
                buffer: Vec::with_capacity(capacity),
                capacity}
        }

        /// Write formatted data to buffer
        #[inline]
        pub fn write_formatted(&mut self, data: &str) -> Result<&str, StreamingError> {
            self.buffer.clear();
            self.buffer
                .write_all(data.as_bytes())
                .map_err(|e| StreamingError::FormatError(format!("Buffer write failed: {}", e)))?;

            std::str::from_utf8(&self.buffer)
                .map_err(|e| StreamingError::Utf8Error(format!("Buffer UTF-8 conversion failed: {}", e)))
        }

        /// Get current buffer contents as string
        #[inline]
        pub fn as_str(&self) -> Result<&str, StreamingError> {
            std::str::from_utf8(&self.buffer)
                .map_err(|e| StreamingError::Utf8Error(format!("Buffer UTF-8 conversion failed: {}", e)))
        }

        /// Clear buffer and prepare for reuse
        #[inline]
        pub fn clear(&mut self) {
            self.buffer.clear();
        }

        /// Get buffer capacity
        #[inline]
        pub fn capacity(&self) -> usize {
            self.capacity
        }

        /// Ensure buffer has minimum capacity
        #[inline]
        pub fn ensure_capacity(&mut self, min_capacity: usize) {
            if self.buffer.capacity() < min_capacity {
                self.buffer.reserve(min_capacity - self.buffer.capacity());
                self.capacity = self.buffer.capacity();
            }
        }
    }

    /// Calculate optimal chunk size for streaming format
    #[inline]
    pub fn calculate_chunk_size(format: &OutputFormat, content_size: usize) -> usize {
        let base_size = format_utils::optimal_buffer_size(format);
        let overhead = match format {
            OutputFormat::ServerSentEvents => 100, // SSE headers
            OutputFormat::WebSocket => 50,         // WebSocket frame
            OutputFormat::Json => 20,              // JSON structure
            OutputFormat::PlainText => 0,          // No overhead
            OutputFormat::Raw => 10,               // Minimal delimiters
            OutputFormat::Custom(_) => 30,         // Variable overhead
        };

        std::cmp::min(base_size - overhead, content_size)
    }

    /// Optimize string allocation for format
    #[inline]
    pub fn optimize_string_capacity(format: &OutputFormat, estimated_size: usize) -> usize {
        let multiplier = match format {
            OutputFormat::ServerSentEvents => 1.5, // SSE has significant overhead
            OutputFormat::WebSocket => 1.2,        // WebSocket has frame overhead
            OutputFormat::Json => 1.1,             // JSON has minimal overhead
            OutputFormat::PlainText => 1.0,        // No overhead
            OutputFormat::Raw => 1.05,             // Minimal overhead
            OutputFormat::Custom(_) => 1.3,        // Conservative estimate
        };

        (estimated_size as f64 * multiplier) as usize
    }
}

/// Performance monitoring and metrics
pub mod perf_utils {
    use std::time::{Duration, Instant};

    /// Format performance metrics
    #[derive(Debug, Clone)]
    pub struct FormatMetrics {
        pub format_time: Duration,
        pub buffer_size: usize,
        pub output_size: usize,
        pub compression_ratio: f64}

    impl FormatMetrics {
        /// Create new metrics instance
        #[inline]
        pub fn new() -> Self {
            Self {
                format_time: Duration::default(),
                buffer_size: 0,
                output_size: 0,
                compression_ratio: 1.0}
        }

        /// Start timing a format operation
        #[inline]
        pub fn start_timing(&mut self) -> Instant {
            Instant::now()
        }

        /// End timing and record duration
        #[inline]
        pub fn end_timing(&mut self, start: Instant) {
            self.format_time = start.elapsed();
        }

        /// Record buffer and output sizes
        #[inline]
        pub fn record_sizes(&mut self, buffer_size: usize, output_size: usize) {
            self.buffer_size = buffer_size;
            self.output_size = output_size;
            self.compression_ratio = if buffer_size > 0 {
                output_size as f64 / buffer_size as f64
            } else {
                1.0
            };
        }

        /// Get formatting throughput in bytes per second
        #[inline]
        pub fn throughput_bps(&self) -> f64 {
            if self.format_time.as_secs_f64() > 0.0 {
                self.output_size as f64 / self.format_time.as_secs_f64()
            } else {
                0.0
            }
        }
    }

    impl Default for FormatMetrics {
        fn default() -> Self {
            Self::new()
        }
    }
}
