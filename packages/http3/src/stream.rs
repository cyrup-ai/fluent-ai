//! HTTP streaming utilities - Pure streams-first architecture
//! Zero futures, zero Result wrapping - all AsyncStream based

use bytes::Bytes;
use fluent_ai_async::prelude::{ChunkHandler, MessageChunk};
use fluent_ai_async::{AsyncStream, AsyncStreamSender, emit};
use http::{HeaderMap, StatusCode};

// MessageChunk implementations moved to wrappers.rs to avoid orphan rule violations
// Use StringWrapper and BytesWrapper instead of implementing directly on external types

/// Represents a chunk of an HTTP response stream.
#[derive(Debug, Clone)]
pub enum HttpChunk {
    /// The head of the response, containing status and headers.
    Head(StatusCode, HeaderMap),
    /// A chunk of the response body.
    Body(Bytes),
    /// Deserialized data from response body for pattern matching
    Deserialized(serde_json::Value),
    /// Error that occurred during processing - stores error message directly for fast access
    Error(String),
}

impl Default for HttpChunk {
    fn default() -> Self {
        HttpChunk::Head(StatusCode::OK, HeaderMap::new())
    }
}

impl cyrup_sugars::prelude::MessageChunk for DownloadChunk {
    fn bad_chunk(error: String) -> Self {
        DownloadChunk {
            data: bytes::Bytes::new(),
            chunk_number: 0,
            total_size: None,
            bytes_downloaded: 0,
            error_message: Some(error),
        }
    }

    fn is_error(&self) -> bool {
        self.error_message.is_some()
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl fluent_ai_async::prelude::MessageChunk for DownloadChunk {
    fn bad_chunk(error: String) -> Self {
        DownloadChunk {
            data: bytes::Bytes::new(),
            chunk_number: 0,
            total_size: None,
            bytes_downloaded: 0,
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl cyrup_sugars::prelude::MessageChunk for HttpChunk {
    fn bad_chunk(error: String) -> Self {
        HttpChunk::Error(error)
    }

    fn is_error(&self) -> bool {
        matches!(self, HttpChunk::Error(_))
    }

    fn error(&self) -> Option<&str> {
        match self {
            HttpChunk::Error(msg) => Some(msg),
            _ => None,
        }
    }
}

impl fluent_ai_async::prelude::MessageChunk for HttpChunk {
    fn bad_chunk(error: String) -> Self {
        HttpChunk::Error(error)
    }

    fn error(&self) -> Option<&str> {
        match self {
            HttpChunk::Error(msg) => Some(msg),
            _ => None,
        }
    }
}

impl serde::Serialize for HttpChunk {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        match self {
            HttpChunk::Head(status, headers) => {
                let mut state = serializer.serialize_struct("HttpChunk", 3)?;
                state.serialize_field("type", "head")?;
                state.serialize_field("status", &status.as_u16())?;

                // Convert HeaderMap to serializable format
                let headers_map: std::collections::HashMap<String, String> = headers
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();
                state.serialize_field("headers", &headers_map)?;
                state.end()
            }
            HttpChunk::Body(bytes) => {
                let mut state = serializer.serialize_struct("HttpChunk", 2)?;
                state.serialize_field("type", "body")?;
                state.serialize_field("length", &bytes.len())?;
                state.end()
            }
            HttpChunk::Deserialized(value) => {
                let mut state = serializer.serialize_struct("HttpChunk", 2)?;
                state.serialize_field("type", "deserialized")?;
                state.serialize_field("data", value)?;
                state.end()
            }
            HttpChunk::Error(message) => {
                let mut state = serializer.serialize_struct("HttpChunk", 2)?;
                state.serialize_field("type", "error")?;
                state.serialize_field("message", message)?;
                state.end()
            }
        }
    }
}

/// Represents a bad/error chunk in HTTP response processing.
#[derive(Debug, Clone)]
pub enum BadChunk {
    /// An error chunk containing the HTTP error that occurred
    Error(crate::HttpError),
    /// A chunk that failed processing with context
    ProcessingFailed {
        /// The original error
        error: crate::HttpError,
        /// Additional context about the processing failure
        context: String,
    },
}

impl BadChunk {
    /// Create a BadChunk from an HttpError
    pub fn from_err(error: crate::HttpError) -> Self {
        BadChunk::Error(error)
    }

    /// Create a BadChunk from an HttpError with processing context
    pub fn from_processing_error(error: crate::HttpError, context: String) -> Self {
        BadChunk::ProcessingFailed { error, context }
    }

    /// Get the underlying HttpError
    pub fn error(&self) -> &crate::HttpError {
        match self {
            BadChunk::Error(err) => err,
            BadChunk::ProcessingFailed { error, .. } => error,
        }
    }
}

/// Represents a chunk of a file download stream.
#[derive(Debug, Clone)]
pub struct DownloadChunk {
    /// Raw bytes of this chunk
    pub data: Bytes,
    /// Sequential number of this chunk
    pub chunk_number: u64,
    /// Total file size if known from headers
    pub total_size: Option<u64>,
    /// Total bytes downloaded so far
    pub bytes_downloaded: u64,
    /// Error message if this is an error chunk
    pub error_message: Option<String>,
}

impl Default for DownloadChunk {
    fn default() -> Self {
        DownloadChunk {
            data: Bytes::new(),
            chunk_number: 0,
            total_size: None,
            bytes_downloaded: 0,
            error_message: None,
        }
    }
}

impl DownloadChunk {
    /// Calculate progress percentage if total size is known
    pub fn progress_percentage(&self) -> Option<f64> {
        self.total_size.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.bytes_downloaded as f64 / total as f64) * 100.0
            }
        })
    }

    /// Calculate download speed in bytes per second (requires timing context)
    pub fn download_speed(&self, elapsed_seconds: f64) -> Option<f64> {
        if elapsed_seconds > 0.0 {
            Some(self.bytes_downloaded as f64 / elapsed_seconds)
        } else {
            None
        }
    }
}

/// Server-Sent Event data
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// Event type
    pub event_type: Option<String>,
    /// Event data
    pub data: String,
    /// Event ID
    pub id: Option<String>,
    /// Retry timeout
    pub retry: Option<u64>,
}

/// Pure AsyncStream of HTTP response chunks - NO Result wrapping
pub struct HttpStream {
    inner: AsyncStream<HttpChunk>,
    chunk_handler:
        Option<Box<dyn Fn(Result<HttpChunk, crate::HttpError>) -> HttpChunk + Send + Sync>>,
}

impl HttpStream {
    /// Create new HttpStream from AsyncStream
    pub fn new(stream: AsyncStream<HttpChunk>) -> Self {
        Self {
            inner: stream,
            chunk_handler: None,
        }
    }

    /// Create from generator function
    pub fn with_channel<F>(producer: F) -> Self
    where
        F: FnOnce(AsyncStreamSender<HttpChunk>) + Send + 'static,
    {
        Self {
            inner: AsyncStream::with_channel(producer),
            chunk_handler: None,
        }
    }

    /// Get next chunk
    pub fn poll_next(&mut self) -> Option<HttpChunk> {
        self.inner.try_next()
    }

    /// Collect all chunks as bytes
    pub fn collect_bytes(mut self) -> Vec<u8> {
        let mut bytes = Vec::new();
        while let Some(chunk) = self.poll_next() {
            if let HttpChunk::Body(chunk_bytes) = chunk {
                bytes.extend_from_slice(&chunk_bytes);
            }
        }
        bytes
    }

    /// Collect and deserialize JSON
    pub fn collect_json<T>(self) -> Option<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let bytes = self.collect_bytes();
        if bytes.is_empty() {
            return None;
        }
        serde_json::from_slice(&bytes).ok()
    }

    /// Collect single response and deserialize - PUBLIC API COMPATIBILITY
    pub fn collect_one<T>(self) -> T
    where
        T: serde::de::DeserializeOwned + From<BadChunk> + Default,
    {
        match self.collect_json::<T>() {
            Some(result) => result,
            None => T::default(), // Fallback to default if parsing fails
        }
    }

    /// Collect single response with error handler - fluent_ai_async pattern
    pub fn collect_or_else<T, F>(self, mut error_handler: F) -> T
    where
        T: serde::de::DeserializeOwned + MessageChunk + Default + Send + 'static,
        F: FnMut(&T) -> T,
    {
        let bytes = self.collect_bytes();
        if bytes.is_empty() {
            return error_handler(&T::bad_chunk("Empty response".to_string()));
        }
        match serde_json::from_slice::<T>(&bytes) {
            Ok(result) => result,
            Err(_) => error_handler(&T::bad_chunk("Failed to parse response".to_string())),
        }
    }

    /// Collect all responses as vector - PUBLIC API COMPATIBILITY
    pub fn collect_all<T>(self) -> Vec<T>
    where
        T: serde::de::DeserializeOwned + From<BadChunk>,
    {
        // For single response HTTP, this typically returns a single-item vector
        match self.collect_json::<T>() {
            Some(result) => vec![result],
            None => vec![],
        }
    }

    /// Create HTTP stream from raw bytes - pure streams only
    #[must_use]
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self::with_channel(move |sender| {
            std::thread::spawn(move || {
                // Emit the data as a single body chunk
                let chunk = HttpChunk::Body(bytes::Bytes::from(data));
                let _ = sender.send(chunk);
            });
        })
    }
}

/// Iterator implementation for backwards compatibility with existing code
impl Iterator for HttpStream {
    type Item = HttpChunk;

    fn next(&mut self) -> Option<Self::Item> {
        self.poll_next()
    }
}

/// Pure AsyncStream of download chunks - NO Result wrapping
pub struct DownloadStream {
    inner: AsyncStream<DownloadChunk>,
    chunk_handler:
        Option<Box<dyn Fn(Result<DownloadChunk, crate::HttpError>) -> DownloadChunk + Send + Sync>>,
}

impl DownloadStream {
    /// Create new download stream
    #[must_use]
    pub fn new(stream: AsyncStream<DownloadChunk>) -> Self {
        Self {
            inner: stream,
            chunk_handler: None,
        }
    }

    /// Create download stream with progress tracking
    #[must_use]
    pub fn with_progress<F>(producer: F) -> Self
    where
        F: FnOnce(AsyncStreamSender<DownloadChunk>) + Send + 'static,
    {
        Self {
            inner: AsyncStream::with_channel(producer),
            chunk_handler: None,
        }
    }

    /// Get next download chunk
    pub fn poll_next(&mut self) -> Option<DownloadChunk> {
        self.inner.try_next()
    }

    /// Stream extension method equivalent - NO async, pure streams only
    pub fn try_next(&mut self) -> Option<DownloadChunk> {
        self.poll_next()
    }
}

impl ChunkHandler<DownloadChunk, crate::HttpError> for DownloadStream {
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<DownloadChunk, crate::HttpError>) -> DownloadChunk + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }
}

/// Pure AsyncStream of text lines - NO Result wrapping
pub type LinesStream = AsyncStream<String>;

/// Pure AsyncStream of Server-Sent Events - NO Result wrapping  
pub type SseStream = AsyncStream<SseEvent>;

/// Pure AsyncStream of JSON objects - NO Result wrapping
pub type JsonStream = AsyncStream<serde_json::Value>;

// Conversions between BadChunk and HttpChunk for elegant error handling
impl From<BadChunk> for HttpChunk {
    fn from(bad_chunk: BadChunk) -> Self {
        match bad_chunk {
            BadChunk::Error(error) => HttpChunk::Error(error.to_string()),
            BadChunk::ProcessingFailed { error, .. } => HttpChunk::Error(error.to_string()),
        }
    }
}

impl From<crate::HttpError> for BadChunk {
    fn from(error: crate::HttpError) -> Self {
        BadChunk::Error(error)
    }
}

// Enable BadChunk -> serde_json::Value conversion for JSON processing
impl From<BadChunk> for serde_json::Value {
    fn from(bad_chunk: BadChunk) -> Self {
        match bad_chunk {
            BadChunk::Error(error) => {
                serde_json::json!({
                    "error": true,
                    "message": error.to_string(),
                    "type": "http_error"
                })
            }
            BadChunk::ProcessingFailed { error, context } => {
                serde_json::json!({
                    "error": true,
                    "message": error.to_string(),
                    "context": context,
                    "type": "processing_error"
                })
            }
        }
    }
}

// Enable BadChunk -> Vec<u8> conversion for raw byte collection
impl From<BadChunk> for Vec<u8> {
    fn from(bad_chunk: BadChunk) -> Self {
        // Convert error to JSON bytes
        let error_json = serde_json::Value::from(bad_chunk);
        serde_json::to_vec(&error_json).unwrap_or_else(|_| b"[]".to_vec())
    }
}

// Remove duplicate implementations - already defined above

/// Implement ChunkHandler trait for HttpStream - pure fluent_ai_async pattern
impl ChunkHandler<HttpChunk, crate::HttpError> for HttpStream {
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<HttpChunk, crate::HttpError>) -> HttpChunk + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }
}

// Remove all orphan trait implementations - these violate Rust's orphan rule
// Only implement traits for types we own in this crate
