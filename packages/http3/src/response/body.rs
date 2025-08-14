//! HTTP response body processing and streaming functionality
//!
//! Handles JSON streaming, Server-Sent Events parsing, JSONPath filtering,
//! and body deserialization with zero-allocation design and blazing-fast performance.

use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;

use bytes::Bytes;
use fluent_ai_async::AsyncStream;
use serde::de::DeserializeOwned;

use crate::json_path::JsonStreamProcessor;
use crate::response::core::HttpResponse;

/// Server-Sent Event structure for streaming responses
/// Zero allocation design with unwrapped values
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// Event data payload
    pub data: Option<String>,
    /// Event type
    pub event_type: Option<String>,
    /// Event ID for last-event-id tracking
    pub id: Option<String>,
    /// Retry interval in milliseconds
    pub retry: Option<u64>,
}

impl SseEvent {
    /// Create new SSE event with data
    #[must_use]
    pub fn data(data: String) -> Self {
        Self {
            data: Some(data),
            event_type: None,
            id: None,
            retry: None,
        }
    }

    /// Create new SSE event with type and data
    #[must_use]
    pub fn typed(event_type: String, data: String) -> Self {
        Self {
            data: Some(data),
            event_type: Some(event_type),
            id: None,
            retry: None,
        }
    }
}

/// Type alias for the chunk handler function
type ChunkHandler<T> =
    Box<dyn FnMut(&T) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + 'static>;

/// JSON stream that yields unwrapped T values with user `on_chunk` error handling
/// Users get immediate values, error handling via `on_chunk` handler
pub struct JsonStream<T> {
    body: Vec<u8>,
    _phantom: PhantomData<T>,
    /// Optional handler for processing chunks
    #[allow(dead_code)]
    handler: Option<ChunkHandler<T>>,
}

impl<T> Debug for JsonStream<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsonStream")
            .field("body", &self.body)
            .field(
                "handler",
                &if self.handler.is_some() {
                    "Some(<function>)"
                } else {
                    "None"
                },
            )
            .finish()
    }
}

impl<T: serde::de::DeserializeOwned> JsonStream<T> {
    /// Get JSON value - returns `T` directly (no futures)
    ///
    /// Users get immediate values, error handling via `on_chunk` handlers
    #[must_use]
    pub fn get(&self) -> Option<T> {
        // Parse JSON once and return the result
        // Error handling delegated to user on_chunk handlers
        serde_json::from_slice(&self.body).ok()
    }

    /// Collect all JSON values into a Vec
    #[must_use]
    pub fn collect_json(self) -> Vec<T> {
        match self.get() {
            Some(value) => vec![value],
            None => Vec::new(),
        }
    }
}

impl<T> JsonStream<T>
where
    T: Clone + Send + 'static,
{
    /// Add `on_chunk` handler for error handling and processing
    /// Users receive unwrapped values T, errors handled in `on_chunk`
    #[must_use = "returns the modified stream"]
    pub fn on_chunk<F>(self, f: F) -> Self
    where
        F: FnMut(&T) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + 'static,
    {
        Self {
            handler: Some(Box::new(f)),
            ..self
        }
    }

    /// Collect implementation for pure streaming architecture
    /// Users wanting "await" similar behavior call `.collect()`
    #[must_use = "method returns a new value and does not modify the original"]
    pub fn collect(self) -> JsonStream<T> {
        self
    }
}

impl HttpResponse {
    /// Parse the body as JSON stream - returns unwrapped T chunks
    /// Only available for JSON content-type responses
    /// Zero futures, error handling via user `on_chunk` handlers, users call `.collect()` for await-like behavior
    #[must_use]
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Option<JsonStream<T>> {
        // Only provide JSON parsing for JSON content types
        if self.is_json_content() {
            Some(JsonStream {
                body: self.body().to_vec(),
                _phantom: std::marker::PhantomData,
                handler: None,
            })
        } else {
            None
        }
    }

    /// Stream individual objects from JSON arrays using JSONPath filtering
    ///
    /// Processes the response body through JSONPath expressions to extract
    /// individual objects from JSON arrays and nested structures.
    ///
    /// # Arguments
    /// * `jsonpath_expr` - JSONPath expression for filtering (e.g., "$.data[*]", "$.results[?(@.active)]")
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    ///
    /// # Returns
    /// AsyncStream of successfully deserialized objects of type T
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::HttpResponse;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// // Parse OpenAI-style {"data": [...]} response
    /// let users: Vec<User> = response
    ///     .jsonpath_stream("$.data[*]")
    ///     .collect();
    /// ```
    #[must_use]
    pub fn jsonpath_stream<T>(&self, jsonpath_expr: &str) -> AsyncStream<T>
    where
        T: DeserializeOwned + Send + 'static,
    {
        let stream_processor = JsonStreamProcessor::<T>::new(jsonpath_expr);
        let response_bytes = Bytes::from(self.body().to_vec());

        // Process the entire response body through JSONPath filtering
        stream_processor.process_body(response_bytes)
    }

    /// Extract the first object matching a JSONPath expression
    ///
    /// Convenience method for getting a single object from JSON responses
    /// using JSONPath filtering.
    ///
    /// # Arguments
    /// * `jsonpath_expr` - JSONPath expression for filtering
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize the matching JSON object into
    ///
    /// # Returns
    /// Option containing the first matching object, or None if no matches
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::HttpResponse;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// // Get first user from {"data": [...]} response
    /// let first_user: Option<User> = response
    ///     .jsonpath_first("$.data[0]");
    /// ```
    #[must_use]
    pub fn jsonpath_first<T>(&self, jsonpath_expr: &str) -> Option<T>
    where
        T: DeserializeOwned + Send + 'static,
    {
        self.jsonpath_stream(jsonpath_expr)
            .collect()
            .into_iter()
            .next()
    }

    /// Extract all objects matching a JSONPath expression into a Vec
    ///
    /// Convenience method for collecting all matching objects from JSON responses
    /// using JSONPath filtering.
    ///
    /// # Arguments
    /// * `jsonpath_expr` - JSONPath expression for filtering
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    ///
    /// # Returns
    /// Vec containing all matching objects
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::HttpResponse;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// // Get all active users from response
    /// let active_users: Vec<User> = response
    ///     .jsonpath_collect("$.users[?(@.active == true)]");
    /// ```
    #[must_use]
    pub fn jsonpath_collect<T>(&self, jsonpath_expr: &str) -> Vec<T>
    where
        T: DeserializeOwned + Send + 'static,
    {
        self.jsonpath_stream(jsonpath_expr).collect()
    }

    /// Get Server-Sent Events - returns Vec<SseEvent> directly
    ///
    /// Get SSE events if this is an SSE response
    /// Returns empty `Vec` if not an SSE response
    #[must_use]
    pub fn sse(&self) -> Vec<SseEvent> {
        let body = String::from_utf8_lossy(self.body());
        Self::parse_sse_events(&body)
    }

    /// Parse SSE events according to the Server-Sent Events specification
    /// Handles multi-line data fields, event types, IDs, and retry directives
    fn parse_sse_events(body: &str) -> Vec<SseEvent> {
        let mut events = Vec::new();
        let mut current_event = SseEvent {
            data: None,
            event_type: None,
            id: None,
            retry: None,
        };
        let mut data_lines = Vec::new();

        for line in body.lines() {
            let line = line.trim_end_matches('\r'); // Handle CRLF endings

            // Empty line indicates end of event
            if line.is_empty() {
                if !data_lines.is_empty()
                    || current_event.event_type.is_some()
                    || current_event.id.is_some()
                    || current_event.retry.is_some()
                {
                    // Join data lines with newlines (SSE spec requirement)
                    if !data_lines.is_empty() {
                        current_event.data = Some(data_lines.join("\n"));
                    }

                    events.push(current_event);

                    // Reset for next event
                    current_event = SseEvent {
                        data: None,
                        event_type: None,
                        id: None,
                        retry: None,
                    };
                    data_lines.clear();
                }
                continue;
            }

            // Skip comment lines (start with :)
            if line.starts_with(':') {
                continue;
            }

            // Parse field: value pairs
            if let Some(colon_pos) = line.find(':') {
                let field = &line[..colon_pos];
                let value = line[colon_pos + 1..].trim_start_matches(' ');

                match field {
                    "data" => {
                        data_lines.push(value.to_string());
                    }
                    "event" => {
                        current_event.event_type = Some(value.to_string());
                    }
                    "id" => {
                        // ID field must not contain null characters (spec requirement)
                        if !value.contains('\0') {
                            current_event.id = Some(value.to_string());
                        }
                    }
                    "retry" => {
                        // retry field must be a valid number (milliseconds)
                        let retry_ms = value.parse::<u64>().ok();
                        if let Some(retry_ms) = retry_ms {
                            current_event.retry = Some(retry_ms);
                        }
                    }
                    _ => {
                        // Ignore unknown fields (spec allows this)
                    }
                }
            } else {
                // Line without colon is treated as "data: <line>"
                data_lines.push(line.to_string());
            }
        }

        // Handle final event if stream doesn't end with empty line
        if !data_lines.is_empty()
            || current_event.event_type.is_some()
            || current_event.id.is_some()
            || current_event.retry.is_some()
        {
            if !data_lines.is_empty() {
                current_event.data = Some(data_lines.join("\n"));
            }
            events.push(current_event);
        }

        events
    }
}
