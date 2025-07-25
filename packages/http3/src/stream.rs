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
    Body(Bytes)}

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
    pub bytes_downloaded: u64}

// Stream newtype wrappers to allow for extension methods

pin_project! {
    /// A stream of HTTP response chunks.
    pub struct HttpStream {
        #[pin]
        inner: Pin<Box<dyn Stream<Item = HttpResult<HttpChunk>> + Send>>
    }
}

impl HttpStream {
    /// Create a new HTTP response stream
    pub fn new(inner: Pin<Box<dyn Stream<Item = HttpResult<HttpChunk>> + Send>>) -> Self {
        Self { inner }
    }
}

impl Stream for HttpStream {
    type Item = HttpResult<HttpChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().inner.poll_next(cx)
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
