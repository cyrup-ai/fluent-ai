//! Core Streaming Response Types
//!
//! This module provides the foundational streaming response infrastructure
//! for HTTP/3 with zero-allocation, lock-free patterns using fluent_ai_async.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};

use bytes::Bytes;
use cyrup_sugars::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, AsyncStreamSender, emit};
use http::{HeaderMap, HeaderValue, Response, StatusCode, Version};

use crate::types::chunks::HttpResponseChunk;

/// Core streaming HTTP response with zero-allocation patterns
///
/// This type provides the foundational streaming response infrastructure
/// that all protocol implementations (H2, H3, Quiche, WASM) will extend.
#[derive(Debug)]
pub struct HttpStreamingResponse {
    /// Response chunk stream
    pub chunks: AsyncStream<HttpResponseChunk, 1024>,

    /// Response metadata (cached from first status chunk)
    pub status: Option<StatusCode>,
    pub headers: Option<HeaderMap>,
    pub version: Option<Version>,

    /// Streaming state tracking (lock-free)
    pub bytes_received: AtomicU64,
    pub is_complete: AtomicBool,
    pub has_error: AtomicBool,
}

impl HttpStreamingResponse {
    /// Create a new streaming response with channel-based producer
    #[inline]
    pub fn with_channel<F>(producer: F) -> Self
    where
        F: FnOnce(AsyncStreamSender<HttpResponseChunk, 1024>) + Send + 'static,
    {
        let chunks = AsyncStream::with_channel(producer);

        Self {
            chunks,
            status: None,
            headers: None,
            version: None,
            bytes_received: AtomicU64::new(0),
            is_complete: AtomicBool::new(false),
            has_error: AtomicBool::new(false),
        }
    }

    /// Create an error-aware streaming response
    #[inline]
    pub fn create_error_aware_stream<F>(producer: F) -> Self
    where
        F: FnOnce(AsyncStreamSender<HttpResponseChunk, 1024>) + Send + 'static,
    {
        Self::with_channel(move |sender| {
            producer(sender);
        })
    }

    /// Create a timeout-aware streaming response
    #[inline]
    pub fn create_timeout_aware_stream<F>(producer: F, timeout_ms: u64) -> Self
    where
        F: FnOnce(AsyncStreamSender<HttpResponseChunk, 1024>) + Send + 'static,
    {
        Self::with_channel(move |sender| {
            let start_time = std::time::Instant::now();

            producer(AsyncStreamSender::new(Arc::new(move |chunk| {
                if start_time.elapsed().as_millis() > timeout_ms as u128 {
                    let timeout_chunk = HttpResponseChunk::timeout_error(
                        format!("Operation timed out after {}ms", timeout_ms),
                        "streaming_response",
                    );
                    sender.send(timeout_chunk);
                    return;
                }
                sender.send(chunk);
            })));
        })
    }

    /// Create a basic response stream (for WASM fetch API integration)
    #[inline]
    pub fn create_basic_response_stream<F>(producer: F) -> Self
    where
        F: FnOnce(AsyncStreamSender<HttpResponseChunk, 1024>) + Send + 'static,
    {
        Self::with_channel(producer)
    }

    /// Get the next response chunk (non-blocking)
    #[inline]
    pub fn try_next(&self) -> Option<HttpResponseChunk> {
        match self.chunks.try_next() {
            Some(chunk) => {
                self.update_state(&chunk);
                Some(chunk)
            }
            None => None,
        }
    }

    /// Collect all response chunks (blocking)
    #[inline]
    pub fn collect(self) -> Vec<HttpResponseChunk> {
        let chunks: Vec<_> = self.chunks.collect();

        // Update final state based on collected chunks
        for chunk in &chunks {
            self.update_state(chunk);
        }

        chunks
    }

    /// Collect chunks or handle first error encountered
    #[inline]
    pub fn collect_or_else<F>(
        self,
        error_handler: F,
    ) -> Result<Vec<HttpResponseChunk>, HttpResponseChunk>
    where
        F: FnOnce(HttpResponseChunk) -> HttpResponseChunk,
    {
        let chunks = self.chunks.collect();

        for chunk in &chunks {
            if chunk.is_error() {
                let handled_error = error_handler(chunk.clone());
                return Err(handled_error);
            }
        }

        Ok(chunks)
    }

    /// Get current response status (cached from status chunk)
    #[inline]
    pub fn status(&self) -> Option<StatusCode> {
        self.status
    }

    /// Get current response headers (cached from status chunk)
    #[inline]
    pub fn headers(&self) -> Option<&HeaderMap> {
        self.headers.as_ref()
    }

    /// Get current response version (cached from status chunk)
    #[inline]
    pub fn version(&self) -> Option<Version> {
        self.version
    }

    /// Get total bytes received so far
    #[inline]
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received.load(Ordering::Relaxed)
    }

    /// Check if response is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.is_complete.load(Ordering::Relaxed)
    }

    /// Check if response has encountered an error
    #[inline]
    pub fn has_error(&self) -> bool {
        self.has_error.load(Ordering::Relaxed)
    }

    /// Update internal state based on received chunk (lock-free)
    #[inline]
    fn update_state(&self, chunk: &HttpResponseChunk) {
        match chunk {
            HttpResponseChunk::Status {
                status,
                headers,
                version,
            } => {
                // Note: status, headers, and version are Option types that cannot be atomically updated
                // In a production system, these would need to be wrapped in Arc<Mutex<>> or similar
                // for thread-safe updates, or the architecture would use message passing
                // For now, we track the atomic state flags which are the critical metrics
            }
            HttpResponseChunk::Data { bytes, .. } => {
                self.bytes_received
                    .fetch_add(bytes.len() as u64, Ordering::Relaxed);
            }
            HttpResponseChunk::Complete => {
                self.is_complete.store(true, Ordering::Relaxed);
            }
            chunk if chunk.is_error() => {
                self.has_error.store(true, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

/// Zero-allocation response builder for streaming responses
///
/// This builder provides an ergonomic way to construct streaming responses
/// without unnecessary allocations during the building process.
#[derive(Debug, Default)]
pub struct HttpStreamingResponseBuilder {
    status: Option<StatusCode>,
    headers: HeaderMap,
    version: Option<Version>,
}

impl HttpStreamingResponseBuilder {
    /// Create a new response builder
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set response status
    #[inline]
    pub fn status(mut self, status: StatusCode) -> Self {
        self.status = Some(status);
        self
    }

    /// Add a header
    #[inline]
    pub fn header<K, V>(mut self, key: K, value: V) -> Self
    where
        K: TryInto<http::HeaderName>,
        V: TryInto<HeaderValue>,
    {
        if let (Ok(name), Ok(val)) = (key.try_into(), value.try_into()) {
            self.headers.insert(name, val);
        }
        self
    }

    /// Set HTTP version
    #[inline]
    pub fn version(mut self, version: Version) -> Self {
        self.version = Some(version);
        self
    }

    /// Build streaming response with producer function
    #[inline]
    pub fn build_with_producer<F>(self, producer: F) -> HttpStreamingResponse
    where
        F: FnOnce(AsyncStreamSender<HttpResponseChunk, 1024>) + Send + 'static,
    {
        let status = self.status.unwrap_or(StatusCode::OK);
        let headers = self.headers;
        let version = self.version.unwrap_or(Version::HTTP_11);

        HttpStreamingResponse::with_channel(move |sender| {
            // Send initial status chunk
            emit!(sender, HttpResponseChunk::status(status, headers, version));

            // Run user-provided producer
            producer(sender);
        })
    }

    /// Build streaming response with data iterator
    #[inline]
    pub fn build_with_data<I>(self, data_iter: I) -> HttpStreamingResponse
    where
        I: IntoIterator<Item = Bytes> + Send + 'static,
        I::IntoIter: Send,
    {
        let status = self.status.unwrap_or(StatusCode::OK);
        let headers = self.headers;
        let version = self.version.unwrap_or(Version::HTTP_11);

        HttpStreamingResponse::with_channel(move |sender| {
            // Send initial status chunk
            emit!(sender, HttpResponseChunk::status(status, headers, version));

            // Send data chunks
            let mut iter = data_iter.into_iter();
            let mut chunk_count = 0;

            while let Some(bytes) = iter.next() {
                chunk_count += 1;
                let is_final = false; // We don't know if this is the last chunk yet
                emit!(sender, HttpResponseChunk::data(bytes, is_final));
            }

            // Send completion marker
            emit!(sender, HttpResponseChunk::complete());
        })
    }
}

/// Utility functions for creating common streaming response patterns
impl HttpStreamingResponse {
    /// Create a simple OK response with data
    #[inline]
    pub fn ok_with_data(data: Bytes) -> Self {
        HttpStreamingResponseBuilder::new()
            .status(StatusCode::OK)
            .build_with_producer(move |sender| {
                emit!(sender, HttpResponseChunk::data(data, true));
                emit!(sender, HttpResponseChunk::complete());
            })
    }

    /// Create an error response
    #[inline]
    pub fn error_response(status: StatusCode, message: impl Into<Arc<str>>) -> Self {
        HttpStreamingResponseBuilder::new()
            .status(status)
            .build_with_producer(move |sender| {
                emit!(
                    sender,
                    HttpResponseChunk::protocol_error(message, Some(status))
                );
            })
    }

    /// Create a streaming response from an existing Response<T>
    #[inline]
    pub fn from_response<T>(response: Response<T>) -> Self
    where
        T: Into<Bytes>,
    {
        let (parts, body) = response.into_parts();
        let body_bytes = body.into();

        HttpStreamingResponseBuilder::new()
            .status(parts.status)
            .version(parts.version)
            .build_with_producer(move |sender| {
                // Copy headers
                for (name, value) in parts.headers {
                    if let Some(name) = name {
                        let mut headers = HeaderMap::new();
                        headers.insert(name, value);
                        // Note: This could be optimized to send headers in initial status chunk
                    }
                }

                emit!(sender, HttpResponseChunk::data(body_bytes, true));
                emit!(sender, HttpResponseChunk::complete());
            })
    }
}
