//! HTTP streaming utilities - Zero futures, pure unwrapped value streams

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use fluent_ai_async::AsyncStream;
use futures_util::Stream;
use http::{HeaderMap, StatusCode};
use pin_project_lite::pin_project;

use crate::{HttpError, HttpResponse, HttpResult};

/// Represents a chunk of an HTTP response stream.
#[derive(Debug, Clone)]
pub enum HttpChunk {
    /// The head of the response, containing status and headers.
    Head(StatusCode, HeaderMap),
    /// A chunk of the response body.
    Body(Bytes),
}

/// Represents a chunk of a file download stream.
#[derive(Debug, Clone)]
pub struct DownloadChunk {
    pub data: Bytes,
    pub chunk_number: u64,
    pub total_size: Option<u64>,
    pub bytes_downloaded: u64,
}

// Stream newtype wrappers to allow for extension methods

pin_project! {
    /// A stream of HTTP response chunks.
    pub struct HttpStream {
        #[pin]
        inner: Pin<Box<dyn Stream<Item = HttpResult<HttpChunk>> + Send>>
    }
}

impl HttpStream {
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
        inner: AsyncStream<HttpResult<String>>
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
        inner: AsyncStream<HttpResult<crate::SseEvent>>
    }
}

impl Stream for SseStream {
    type Item = HttpResult<crate::SseEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().inner.poll_next(cx)
    }
}
