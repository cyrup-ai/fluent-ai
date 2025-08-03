//! HTTP streaming utilities - Zero futures, pure unwrapped value streams

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_util::Stream;
use http::{HeaderMap, StatusCode};
use pin_project_lite::pin_project;

use crate::HttpResult;

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
