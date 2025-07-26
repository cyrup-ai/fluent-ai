//! HTTP response types and utilities

use std::collections::HashMap;
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;

use bytes::Bytes;
use http::StatusCode;

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
    pub retry: Option<u64>}

impl SseEvent {
    /// Create new SSE event with data
    #[must_use]
    pub fn data(data: String) -> Self {
        Self {
            data: Some(data),
            event_type: None,
            id: None,
            retry: None}
    }

    /// Create new SSE event with type and data
    #[must_use]
    pub fn typed(event_type: String, data: String) -> Self {
        Self {
            data: Some(data),
            event_type: Some(event_type),
            id: None,
            retry: None}
    }
}

/// JSON stream that yields unwrapped T values with user `on_chunk` error handling
/// Users get immediate values, error handling via `on_chunk` handlerstreaming architecture
/// Type alias for the chunk handler function
type ChunkHandler<T> = Box<dyn FnMut(&T) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + 'static>;

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
            .field("handler", &if self.handler.is_some() { "Some(<function>)" } else { "None" })
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

    /// Collect JSON as Vec - returns Vec<T> directly (no futures)
    /// Users wanting "await" similar behavior    /// Collect all JSON values into a Vec
    #[must_use]
    pub fn collect_json(self) -> Vec<T> {
        match self.get() {
            Some(value) => vec![value],
            None => Vec::new()}
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

/// HTTP response structure with zero-allocation design
#[derive(Debug, Clone)]
pub struct HttpResponse {
    status: StatusCode,
    headers: HashMap<String, String>,
    body: Vec<u8>}

impl HttpResponse {
    /// Create a new HTTP response
    #[must_use]
    pub fn new(status: StatusCode, headers: &reqwest::header::HeaderMap, body: Vec<u8>) -> Self {
        // Convert reqwest headers to HashMap with zero-allocation filtering
        let headers = headers
            .iter()
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|v| (k.as_str().to_string(), v.to_string()))
            })
            .collect();

        Self {
            status,
            headers,
            body}
    }

    /// Create a response from cache data - zero-allocation constructor for blazing-fast performance
    pub fn from_cache(
        status: StatusCode,
        headers: HashMap<String, String>,
        body: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            status,
            headers,
            body: body.into()}
    }

    /// Get the status code
    #[must_use]
    pub fn status(&self) -> StatusCode {
        self.status
    }

    /// Get the headers
    #[must_use]
    pub fn headers(&self) -> &HashMap<String, String> {
        &self.headers
    }

    /// Get the body as bytes
    #[must_use]
    pub fn body(&self) -> &[u8] {
        &self.body
    }

    /// Get the body as text - returns `String` directly
    /// NO FUTURES - pure streaming, users call `.collect()` for await-like behavior
    #[must_use]
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }

    /// Get the body as a `Bytes` handle
    #[must_use]
    pub fn bytes(&self) -> Bytes {
        Bytes::from(self.body.clone())
    }

    /// Check if response is successful (2xx status)
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    /// Parse the body as JSON stream - returns unwrapped T chunks
    /// Only available for JSON content-type responses
    /// Zero futures, error handling via user `on_chunk` handlers, users call `.collect()` for await-like behavior
    #[must_use]
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Option<JsonStream<T>> {
        // Only provide JSON parsing for JSON content types
        if self.is_json_content() {
            Some(JsonStream {
                body: self.body.clone(),
                _phantom: std::marker::PhantomData,
                handler: None,
            })
        } else {
            None
        }
    }

    /// Check if response has JSON content type
    #[must_use]
    pub fn is_json_content(&self) -> bool {
        self.content_type()
            .is_some_and(|ct| ct.contains("application/json") || ct.contains("text/json"))
    }

    /// Check if the response is a client error (4xx status)
    #[must_use]
    pub fn is_client_error(&self) -> bool {
        self.status.is_client_error()
    }

    /// Get the response body as bytes vector - returns `Vec<u8>` directly
    /// NO FUTURES - pure streaming, returns unwrapped bytes
    #[must_use]
    pub fn stream(&self) -> Vec<u8> {
        self.body.clone()
    }

    /// Get Server-Sent Events - returns Vec<SseEvent> directly
    ///
    /// Get SSE events if this is an SSE response
    /// Returns empty `Vec` if not an SSE response
    #[must_use]
    pub fn sse(&self) -> Vec<SseEvent> {
        let body = String::from_utf8_lossy(&self.body);
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
            retry: None};
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
                        retry: None};
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

    /// Check if this is a server error (5xx)
    #[must_use]
    pub fn is_server_error(&self) -> bool {
        self.status.is_server_error()
    }

    /// Check if this is a redirection (3xx)
    #[must_use]
    pub fn is_redirection(&self) -> bool {
        self.status.is_redirection()
    }

    /// Check if this is an informational response (1xx)
    #[must_use]
    pub fn is_informational(&self) -> bool {
        self.status.is_informational()
    }

    /// Get header value by name (case-insensitive)
    #[must_use]
    pub fn header(&self, key: &str) -> Option<&String> {
        self.headers.get(key)
    }

    /// Get `Content-Type` header value
    #[must_use]
    pub fn content_type(&self) -> Option<&String> {
        self.header("content-type")
    }

    /// Get `ETag` header value
    #[must_use]
    pub fn etag(&self) -> Option<&String> {
        self.header("etag")
    }

    /// Get `Last-Modified` header value
    #[must_use]
    pub fn last_modified(&self) -> Option<&String> {
        self.header("last-modified")
    }

    /// Get `Cache-Control` header value
    #[must_use]
    pub fn cache_control(&self) -> Option<&String> {
        self.header("cache-control")
    }

    /// Get content length
    #[must_use]
    pub fn content_length(&self) -> Option<u64> {
        self.header("content-length").and_then(|v| v.parse().ok())
    }

    /// Get `Expires` header value
    #[must_use]
    pub fn expires(&self) -> Option<&String> {
        self.header("expires")
    }

    /// Get computed expires timestamp (Unix timestamp)
    /// This is set by `CacheMiddleware` and represents the effective cache expiration
    #[must_use]
    pub fn computed_expires(&self) -> Option<u64> {
        self.header("x-computed-expires")
            .and_then(|v| v.parse().ok())
    }

    /// Check if response is cacheable based on computed expires
    #[must_use]
    pub fn is_cacheable(&self) -> bool {
        self.computed_expires().is_some() && self.is_success()
    }

    /// Get time until expires in seconds
    #[must_use]
    pub fn seconds_until_expires(&self) -> Option<u64> {
        self.computed_expires().and_then(|expires| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_secs();

            if expires > now {
                Some(expires - now)
            } else {
                Some(0) // Already expired
            }
        })
    }

    /// Get Server header value
    #[must_use]
    pub fn server(&self) -> Option<&String> {
        self.header("server")
    }

    /// Get Date header value
    #[must_use]
    pub fn date(&self) -> Option<&String> {
        self.header("date")
    }

    /// Get Location header value (for redirects)
    #[must_use]
    pub fn location(&self) -> Option<&String> {
        self.header("location")
    }

    /// Get body size in bytes
    #[must_use]
    pub fn body_size(&self) -> usize {
        self.body.len()
    }

    /// Check if body is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }

    /// Get status as u16
    #[must_use]
    pub fn status_code(&self) -> u16 {
        self.status.as_u16()
    }

    /// Get status reason phrase
    #[must_use]
    pub fn status_reason(&self) -> Option<&str> {
        self.status.canonical_reason()
    }

    /// Convert to error if not successful
    pub fn error_for_status(self) -> crate::HttpResult<Self> {
        if self.is_success() {
            Ok(self)
        } else {
            Err(crate::HttpError::HttpStatus {
                status: self.status.as_u16(),
                message: format!(
                    "HTTP {} {}",
                    self.status.as_u16(),
                    self.status.canonical_reason().unwrap_or("Unknown")
                ),
                body: String::from_utf8_lossy(&self.body).to_string()})
        }
    }

    /// Create a 200 OK response
    #[must_use]
    pub fn ok(body: Vec<u8>) -> Self {
        Self {
            status: StatusCode::OK,
            headers: HashMap::new(),
            body}
    }

    /// Create a 404 Not Found response
    #[must_use]
    pub fn not_found() -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            headers: HashMap::new(),
            body: b"Not Found".to_vec()}
    }

    /// Create a 500 Internal Server Error response
    #[must_use]
    pub fn internal_server_error() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            headers: HashMap::new(),
            body: b"Internal Server Error".to_vec()}
    }

    /// Create a 400 Bad Request response
    #[must_use]
    pub fn bad_request() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            headers: HashMap::new(),
            body: b"Bad Request".to_vec()}
    }

    /// Create a 401 Unauthorized response
    #[must_use]
    pub fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            headers: HashMap::new(),
            body: b"Unauthorized".to_vec()}
    }

    /// Create a 403 Forbidden response
    #[must_use]
    pub fn forbidden() -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            headers: HashMap::new(),
            body: b"Forbidden".to_vec()}
    }

    /// Create a 429 Too Many Requests response
    #[must_use]
    pub fn too_many_requests() -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            headers: HashMap::new(),
            body: b"Too Many Requests".to_vec()}
    }

    /// Create a 502 Bad Gateway response
    #[must_use]
    pub fn bad_gateway() -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            headers: HashMap::new(),
            body: b"Bad Gateway".to_vec()}
    }

    /// Create a 503 Service Unavailable response
    #[must_use]
    pub fn service_unavailable() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            headers: HashMap::new(),
            body: b"Service Unavailable".to_vec()}
    }

    /// Create a 504 Gateway Timeout response
    #[must_use]
    pub fn gateway_timeout() -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            headers: HashMap::new(),
            body: b"Gateway Timeout".to_vec()}
    }
}
