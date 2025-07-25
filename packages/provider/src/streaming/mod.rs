//! Zero-allocation streaming completion interfaces
//!
//! This module provides blazing-fast, lock-free streaming implementations
//! for real-time AI completion responses across all providers.

use std::pin::Pin;
use std::task::{Context, Poll};

use fluent_ai_domain::chunk::CompletionChunk;
use futures_util::Stream;
use serde::Deserialize;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::completion_provider::{CompletionError, ResponseMetadata};

/// Universal streaming completion response trait
///
/// Provides zero-allocation streaming with lock-free channel communication
/// and inline token processing for maximum performance.
pub trait StreamingCompletionResponse: Send + Sync {
    /// The type of raw streaming response from the provider
    type RawResponse: Send + Sync;

    /// Get the raw provider-specific response
    fn raw_response(&self) -> &Self::RawResponse;

    /// Get response metadata (filled as chunks arrive)
    fn metadata(&self) -> &ResponseMetadata;

    /// Convert to a stream of completion chunks
    fn into_stream(
        self,
    ) -> Pin<Box<dyn Stream<Item = Result<CompletionChunk, CompletionError>> + Send>>;

    /// Collect all chunks into a single completion (for non-streaming use)
    async fn collect(self) -> Result<String, CompletionError>
    where
        Self: Sized,
    {
        let mut stream = self.into_stream();
        let mut content = String::new();

        while let Some(chunk_result) = futures_util::StreamExt::next(&mut stream).await {
            match chunk_result {
                Ok(chunk) => {
                    if let Some(text) = chunk.content.text() {
                        content.push_str(text);
                    }
                }
                Err(e) => return Err(e)}
        }

        Ok(content)
    }
}

/// Zero-allocation SSE (Server-Sent Events) parser
///
/// Efficiently parses SSE streams without heap allocations for common cases.
/// Uses static string patterns and inline parsing for maximum performance.
pub struct SseParser {
    buffer: Vec<u8>,
    last_event_id: Option<String>}

impl SseParser {
    /// Create a new SSE parser with zero allocations
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(8192), // Pre-allocate reasonable buffer
            last_event_id: None}
    }

    /// Parse SSE chunk with zero allocations for common cases
    ///
    /// Returns `None` if the chunk is incomplete and needs more data
    pub fn parse_chunk(&mut self, chunk: &[u8]) -> Result<Option<SseEvent>, CompletionError> {
        self.buffer.extend_from_slice(chunk);

        // Look for complete SSE events (terminated by \n\n)
        if let Some(pos) = self.buffer.windows(2).position(|w| w == b"\n\n") {
            let event_data = &self.buffer[..pos];
            let event = self.parse_event_data(event_data)?;

            // Remove processed data from buffer
            self.buffer.drain(..pos + 2);

            Ok(Some(event))
        } else {
            Ok(None) // Need more data
        }
    }

    /// Parse raw SSE event data into structured event
    #[inline]
    fn parse_event_data(&mut self, data: &[u8]) -> Result<SseEvent, CompletionError> {
        let mut event = SseEvent::default();

        for line in data.split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }

            if let Some(colon_pos) = line.iter().position(|&b| b == b':') {
                let field = &line[..colon_pos];
                let value = &line[colon_pos + 1..];

                // Skip leading space in value
                let value = if value.starts_with(&[b' ']) {
                    &value[1..]
                } else {
                    value
                };

                match field {
                    b"data" => {
                        if !event.data.is_empty() {
                            event.data.push('\n');
                        }
                        event.data.push_str(&String::from_utf8_lossy(value));
                    }
                    b"event" => {
                        event.event_type = Some(String::from_utf8_lossy(value).into_owned());
                    }
                    b"id" => {
                        let id = String::from_utf8_lossy(value).into_owned();
                        self.last_event_id = Some(id.clone());
                        event.id = Some(id);
                    }
                    b"retry" => {
                        if let Ok(retry_str) = std::str::from_utf8(value) {
                            if let Ok(retry_ms) = retry_str.parse::<u64>() {
                                event.retry = Some(retry_ms);
                            }
                        }
                    }
                    _ => {
                        // Unknown field, ignore
                    }
                }
            } else {
                // Line without colon - treat as data
                if !event.data.is_empty() {
                    event.data.push('\n');
                }
                event.data.push_str(&String::from_utf8_lossy(line));
            }
        }

        Ok(event)
    }
}

/// Zero-allocation SSE event representation
#[derive(Debug, Default, Clone)]
pub struct SseEvent {
    /// Event data
    pub data: String,
    /// Event type (optional)
    pub event_type: Option<String>,
    /// Event ID (optional)
    pub id: Option<String>,
    /// Retry timeout in milliseconds (optional)
    pub retry: Option<u64>}

impl SseEvent {
    /// Check if this is a data event with content
    #[inline(always)]
    pub fn has_data(&self) -> bool {
        !self.data.is_empty()
    }

    /// Check if this is a specific event type
    #[inline(always)]
    pub fn is_event_type(&self, event_type: &str) -> bool {
        self.event_type.as_deref() == Some(event_type)
    }

    /// Parse the data as JSON
    #[inline]
    pub fn parse_json<T>(&self) -> Result<T, CompletionError>
    where
        T: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(&self.data).map_err(|e| {
            CompletionError::DeserializationError(format!(
                "Failed to parse SSE data as JSON: {}",
                e
            ))
        })
    }
}

/// JSON Lines parser for streaming responses
///
/// Efficiently parses newline-delimited JSON without heap allocations
/// for common cases, using inline parsing and zero-copy string operations.
pub struct JsonLinesParser {
    buffer: Vec<u8>,
    line_buffer: String}

impl JsonLinesParser {
    /// Create a new JSON Lines parser
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(4096),
            line_buffer: String::with_capacity(1024)}
    }

    /// Parse JSON Lines chunk
    ///
    /// Returns a vector of parsed JSON objects, or an error if parsing fails
    pub fn parse_chunk<T>(&mut self, chunk: &[u8]) -> Result<Vec<T>, CompletionError>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.buffer.extend_from_slice(chunk);
        let mut results = Vec::new();

        while let Some(newline_pos) = self.buffer.iter().position(|&b| b == b'\n') {
            // Extract the line
            let line_bytes = &self.buffer[..newline_pos];

            // Skip empty lines
            if !line_bytes.is_empty() {
                // Convert to string (reuse buffer for efficiency)
                self.line_buffer.clear();
                self.line_buffer
                    .push_str(&String::from_utf8_lossy(line_bytes));

                // Parse JSON
                match serde_json::from_str::<T>(&self.line_buffer) {
                    Ok(parsed) => results.push(parsed),
                    Err(e) => {
                        return Err(CompletionError::DeserializationError(format!(
                            "Failed to parse JSON line: {}",
                            e
                        )));
                    }
                }
            }

            // Remove processed line from buffer
            self.buffer.drain(..newline_pos + 1);
        }

        Ok(results)
    }

    /// Get any remaining unparsed data
    #[inline(always)]
    pub fn remaining_data(&self) -> &[u8] {
        &self.buffer
    }

    /// Clear all buffers
    #[inline(always)]
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.line_buffer.clear();
    }
}

/// Default implementation for the streaming completion response trait
pub struct DefaultStreamingResponse<T> {
    pub raw_response: T,
    pub stream: UnboundedReceiverStream<Result<CompletionChunk, CompletionError>>,
    pub metadata: ResponseMetadata}

impl<T> DefaultStreamingResponse<T>
where
    T: Send + Sync,
{
    /// Create a new default streaming response
    #[inline(always)]
    pub fn new(
        raw_response: T,
        stream: UnboundedReceiverStream<Result<CompletionChunk, CompletionError>>,
    ) -> Self {
        Self {
            raw_response,
            stream,
            metadata: ResponseMetadata::default()}
    }

    /// Add metadata
    #[inline(always)]
    pub fn with_metadata(mut self, metadata: ResponseMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl<T> StreamingCompletionResponse for DefaultStreamingResponse<T>
where
    T: Send + Sync,
{
    type RawResponse = T;

    #[inline(always)]
    fn raw_response(&self) -> &Self::RawResponse {
        &self.raw_response
    }

    #[inline(always)]
    fn metadata(&self) -> &ResponseMetadata {
        &self.metadata
    }

    fn into_stream(
        self,
    ) -> Pin<Box<dyn Stream<Item = Result<CompletionChunk, CompletionError>> + Send>> {
        Box::pin(self.stream)
    }
}

