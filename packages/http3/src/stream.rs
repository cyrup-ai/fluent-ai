//! HTTP streaming utilities - Zero futures, pure unwrapped value streams

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_util::Stream;
use http::{HeaderMap, StatusCode};
use pin_project_lite::pin_project;

use crate::HttpResult;

// Implement NotResult for HttpChunk to work with cyrup_sugars AsyncStream
impl cyrup_sugars::NotResult for HttpChunk {}

/// Represents a chunk of an HTTP response stream.
#[derive(Debug, Clone)]
pub enum HttpChunk {
    /// The head of the response, containing status and headers.
    Head(StatusCode, HeaderMap),
    /// A chunk of the response body.
    Body(Bytes),
    /// Deserialized data from response body for pattern matching
    Deserialized(serde_json::Value),
    /// Error that occurred during processing
    Error(crate::HttpError),
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
            HttpChunk::Error(error) => {
                let mut state = serializer.serialize_struct("HttpChunk", 2)?;
                state.serialize_field("type", "error")?;
                state.serialize_field("message", &error.to_string())?;
                state.end()
            }
        }
    }
}

/// Represents the result of processing a chunk through an on_chunk handler
#[derive(Debug, Clone)]
pub enum ProcessedChunk<T> {
    /// Successfully processed chunk with transformed data
    Ok(T),
    /// Error occurred during chunk processing
    Err(crate::HttpError),
    /// Chunk should be skipped/ignored
    Skip,
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
}

// Stream newtype wrappers to allow for extension methods

pin_project! {
    /// A stream of HTTP response chunks.
    pub struct HttpStream {
        #[pin]
        inner: Pin<Box<dyn Stream<Item = HttpResult<HttpChunk>> + Send>>,
        // Not pinned because processors are invoked during polling, not streamed
        chunk_processors: Vec<Box<dyn FnMut(HttpChunk) -> HttpChunk + Send>>
    }
}

impl HttpStream {
    /// Create a new HTTP response stream
    #[must_use]
    pub fn new(inner: Pin<Box<dyn Stream<Item = HttpResult<HttpChunk>> + Send>>) -> Self {
        Self {
            inner,
            chunk_processors: Vec::new(),
        }
    }

    /// Create a new HTTP stream with chunk processors
    #[must_use]
    pub fn with_processors(
        inner: Pin<Box<dyn Stream<Item = HttpResult<HttpChunk>> + Send>>,
        processors: Vec<Box<dyn FnMut(HttpChunk) -> HttpChunk + Send>>,
    ) -> Self {
        Self {
            inner,
            chunk_processors: processors,
        }
    }

    /// Add a chunk processor to this stream
    pub fn add_processor<F>(&mut self, processor: F)
    where
        F: FnMut(HttpChunk) -> HttpChunk + Send + 'static,
    {
        self.chunk_processors.push(Box::new(processor));
    }
}

impl Stream for HttpStream {
    type Item = HttpResult<HttpChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        match this.inner.poll_next(cx) {
            Poll::Ready(Some(Ok(mut chunk))) => {
                // Apply all chunk processors to the chunk
                for processor in this.chunk_processors.iter_mut() {
                    chunk = processor(chunk);
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

pin_project! {
    /// A stream of downloaded file chunks.
    pub struct DownloadStream {
        #[pin]
        inner: Pin<Box<dyn Stream<Item = HttpResult<DownloadChunk>> + Send>>
    }
}

impl DownloadStream {
    /// Create a new file download stream
    #[must_use]
    pub fn new(inner: Pin<Box<dyn Stream<Item = HttpResult<DownloadChunk>> + Send>>) -> Self {
        Self { inner }
    }
}

impl Stream for DownloadStream {
    type Item = HttpResult<DownloadChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().inner.poll_next(cx)
    }
}

pin_project! {
    /// A stream of lines from a response body.
    pub struct LinesStream {
        #[pin]
        inner: Pin<Box<dyn Stream<Item = HttpResult<String>> + Send>>
    }
}

impl LinesStream {
    /// Create a new line-by-line text stream
    #[must_use]
    pub fn new(inner: Pin<Box<dyn Stream<Item = HttpResult<String>> + Send>>) -> Self {
        Self { inner }
    }
}

impl Stream for LinesStream {
    type Item = HttpResult<String>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().inner.poll_next(cx)
    }
}

pin_project! {
    /// A stream of Server-Sent Events (SSE).
    pub struct SseStream {
        #[pin]
        inner: Pin<Box<dyn Stream<Item = HttpResult<crate::SseEvent>> + Send>>
    }
}

impl SseStream {
    /// Create a new Server-Sent Events stream
    #[must_use]
    pub fn new(inner: Pin<Box<dyn Stream<Item = HttpResult<crate::SseEvent>> + Send>>) -> Self {
        Self { inner }
    }
}

impl Stream for SseStream {
    type Item = HttpResult<crate::SseEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().inner.poll_next(cx)
    }
}

// Conversions between BadChunk and HttpChunk for elegant error handling
impl From<BadChunk> for HttpChunk {
    fn from(bad_chunk: BadChunk) -> Self {
        match bad_chunk {
            BadChunk::Error(error) => HttpChunk::Error(error),
            BadChunk::ProcessingFailed { error, .. } => HttpChunk::Error(error),
        }
    }
}

impl From<crate::HttpError> for BadChunk {
    fn from(error: crate::HttpError) -> Self {
        BadChunk::Error(error)
    }
}

// Implement cyrup_sugars StreamExt trait for HttpStream to enable on_chunk method
impl cyrup_sugars::StreamExt<HttpChunk> for HttpStream {
    fn on_result<F>(self, mut f: F) -> cyrup_sugars::AsyncStream<HttpChunk>
    where
        F: FnMut(Result<HttpChunk, Box<dyn std::error::Error + Send + Sync>>) -> Result<HttpChunk, Box<dyn std::error::Error + Send + Sync>> + Send + 'static,
        HttpChunk: cyrup_sugars::NotResult,
    {
        use futures_util::StreamExt;
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let mut stream = self;
            while let Some(chunk_result) = stream.next().await {
                let converted_result = match chunk_result {
                    Ok(chunk) => Ok(chunk),
                    Err(http_error) => Err(Box::new(http_error) as Box<dyn std::error::Error + Send + Sync>),
                };
                
                match f(converted_result) {
                    Ok(chunk) => {
                        if tx.send(chunk).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        cyrup_sugars::AsyncStream::new(rx)
    }

    fn on_chunk<F, U>(self, mut f: F) -> cyrup_sugars::AsyncStream<U>
    where
        F: FnMut(Result<HttpChunk, Box<dyn std::error::Error + Send + Sync>>) -> U + Send + 'static,
        U: Send + 'static + cyrup_sugars::NotResult,
    {
        use futures_util::StreamExt;
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let mut stream = self;
            while let Some(chunk_result) = stream.next().await {
                // Convert HttpResult<HttpChunk> to Result<HttpChunk, Box<dyn Error>>
                let converted_result = match chunk_result {
                    Ok(chunk) => Ok(chunk),
                    Err(http_error) => Err(Box::new(http_error) as Box<dyn std::error::Error + Send + Sync>),
                };
                
                let processed = f(converted_result);
                if tx.send(processed).is_err() {
                    break;
                }
            }
        });

        cyrup_sugars::AsyncStream::new(rx)
    }

    fn on_error<F>(self, _f: F) -> cyrup_sugars::AsyncStream<HttpChunk>
    where
        F: FnMut(Box<dyn std::error::Error + Send + Sync>) + Send + 'static,
        HttpChunk: cyrup_sugars::NotResult,
    {
        use futures_util::StreamExt;
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let mut stream = self;
            while let Some(chunk_result) = stream.next().await {
                if let Ok(chunk) = chunk_result {
                    if tx.send(chunk).is_err() {
                        break;
                    }
                }
            }
        });

        cyrup_sugars::AsyncStream::new(rx)
    }

    fn tap_each(self, mut f: impl FnMut(&HttpChunk) + Send + 'static) -> cyrup_sugars::AsyncStream<HttpChunk>
    where
        HttpChunk: cyrup_sugars::NotResult,
    {
        use futures_util::StreamExt;
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let mut stream = self;
            while let Some(chunk_result) = stream.next().await {
                if let Ok(chunk) = &chunk_result {
                    f(chunk);
                }
                if let Ok(chunk) = chunk_result {
                    if tx.send(chunk).is_err() {
                        break;
                    }
                }
            }
        });

        cyrup_sugars::AsyncStream::new(rx)
    }

    fn tee_each(self, mut f: impl FnMut(HttpChunk) + Send + 'static) -> cyrup_sugars::AsyncStream<HttpChunk>
    where
        HttpChunk: cyrup_sugars::NotResult,
    {
        use futures_util::StreamExt;
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let mut stream = self;
            while let Some(chunk_result) = stream.next().await {
                if let Ok(chunk) = chunk_result {
                    f(chunk.clone());
                    if tx.send(chunk).is_err() {
                        break;
                    }
                }
            }
        });

        cyrup_sugars::AsyncStream::new(rx)
    }

    fn map_stream<U: Send + 'static + cyrup_sugars::NotResult>(
        self,
        mut f: impl FnMut(HttpChunk) -> U + Send + 'static,
    ) -> cyrup_sugars::AsyncStream<U> {
        use futures_util::StreamExt;
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let mut stream = self;
            while let Some(chunk_result) = stream.next().await {
                if let Ok(chunk) = chunk_result {
                    if tx.send(f(chunk)).is_err() {
                        break;
                    }
                }
            }
        });

        cyrup_sugars::AsyncStream::new(rx)
    }

    fn filter_stream(self, mut f: impl FnMut(&HttpChunk) -> bool + Send + 'static) -> cyrup_sugars::AsyncStream<HttpChunk>
    where
        HttpChunk: cyrup_sugars::NotResult,
    {
        use futures_util::StreamExt;
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let mut stream = self;
            while let Some(chunk_result) = stream.next().await {
                if let Ok(chunk) = chunk_result {
                    if f(&chunk) && tx.send(chunk).is_err() {
                        break;
                    }
                }
            }
        });

        cyrup_sugars::AsyncStream::new(rx)
    }

    fn partition_chunks(self, chunk_size: usize) -> cyrup_sugars::AsyncStream<Vec<HttpChunk>>
    where
        Vec<HttpChunk>: cyrup_sugars::NotResult,
    {
        use futures_util::StreamExt;
        
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let mut stream = self;
            let mut buffer = Vec::with_capacity(chunk_size);

            while let Some(chunk_result) = stream.next().await {
                if let Ok(chunk) = chunk_result {
                    buffer.push(chunk);
                    if buffer.len() >= chunk_size {
                        let chunk_vec = std::mem::replace(&mut buffer, Vec::with_capacity(chunk_size));
                        if tx.send(chunk_vec).is_err() {
                            break;
                        }
                    }
                }
            }

            // Send remaining items
            if !buffer.is_empty() {
                let _ = tx.send(buffer);
            }
        });

        cyrup_sugars::AsyncStream::new(rx)
    }

    fn collect(self) -> cyrup_sugars::AsyncTask<Vec<HttpChunk>>
    where
        Vec<HttpChunk>: cyrup_sugars::NotResult,
    {
        use futures_util::StreamExt;
        
        let (tx, rx) = tokio::sync::oneshot::channel();

        tokio::spawn(async move {
            let mut stream = self;
            let mut result = Vec::new();

            while let Some(chunk_result) = stream.next().await {
                if let Ok(chunk) = chunk_result {
                    result.push(chunk);
                }
            }

            let _ = tx.send(result);
        });

        cyrup_sugars::AsyncTask::new(cyrup_sugars::ZeroOneOrMany::one(rx))
    }

    fn await_result<F, Fut>(self, mut _f: F) -> cyrup_sugars::AsyncTask<()>
    where
        F: FnMut(HttpChunk) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = tx.send(());
        cyrup_sugars::AsyncTask::new(cyrup_sugars::ZeroOneOrMany::one(rx))
    }

    fn await_ok<F, Fut>(self, mut _f: F) -> cyrup_sugars::AsyncTask<()>
    where
        F: FnMut(HttpChunk) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = tx.send(());
        cyrup_sugars::AsyncTask::new(cyrup_sugars::ZeroOneOrMany::one(rx))
    }
}

// Enable BadChunk -> serde_json::Value conversion for collect_internal
impl From<BadChunk> for serde_json::Value {
    fn from(bad_chunk: BadChunk) -> Self {
        match bad_chunk {
            BadChunk::Error(error) => {
                serde_json::json!({
                    "error": true,
                    "message": error.to_string(),
                    "type": "http_error"
                })
            },
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
