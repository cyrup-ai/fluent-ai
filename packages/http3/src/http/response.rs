//! HTTP response types with component-level streaming
//!
//! This module provides the CANONICAL HttpResponse implementation where each
//! HTTP component (status, headers, body) is exposed as an individual AsyncStream,
//! enabling real-time processing as data arrives from the wire.

use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Instant;

use bytes::Bytes;
use fluent_ai_async::AsyncStream;
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode, Version};

/// HTTP response with component-level streaming
///
/// This is the CANONICAL HttpResponse implementation that exposes each HTTP
/// component as a separate AsyncStream, allowing processing of headers before
/// the body arrives, and enabling constant-memory processing of large responses.
pub struct HttpResponse {
    /// HTTP status code - set once, read many times (atomic, 0 = not yet received)
    status: AtomicU16,

    /// Headers stream - delivers headers one by one as they're decoded
    pub headers_stream: AsyncStream<HttpHeader, 256>,

    /// Body stream - delivers body chunks as they arrive from the wire
    pub body_stream: AsyncStream<HttpBodyChunk, 1024>,

    /// Trailers stream - delivers trailing headers after body completes
    pub trailers_stream: AsyncStream<HttpHeader, 64>,

    /// HTTP version used for this response
    pub version: Version,

    /// Stream ID for HTTP/2 and HTTP/3 multiplexing
    pub stream_id: u64,
}

/// HTTP status information
#[derive(Debug, Clone)]
pub struct HttpStatus {
    /// HTTP status code
    pub code: StatusCode,

    /// Status reason phrase (may be empty in HTTP/2 and HTTP/3)
    pub reason: String,

    /// HTTP version
    pub version: Version,

    /// Timestamp when status was received
    pub timestamp: Instant,
}

/// Individual HTTP header
#[derive(Debug, Clone)]
pub struct HttpHeader {
    /// Header name
    pub name: HeaderName,

    /// Header value
    pub value: HeaderValue,

    /// Timestamp when header was received
    pub timestamp: Instant,
}

/// HTTP chunk types for streaming data
#[derive(Debug, Clone, Default)]
pub enum HttpChunk {
    /// Response body data chunk
    Body(Bytes),
    
    /// Raw data chunk
    Data(Bytes),
    
    /// Generic chunk data
    Chunk(Bytes),
    
    /// HTTP headers chunk
    Headers(StatusCode, HeaderMap),
    
    /// HTTP trailers chunk - headers that come after the body
    Trailers(HeaderMap),
    
    /// Error occurred during streaming
    Error(String),
    
    /// End of stream marker
    #[default]
    End,
}

/// Protocol-agnostic download chunk for file downloads with progress tracking
///
/// Used by DownloadBuilder to provide consistent download functionality
/// across all protocols (HTTP/2, HTTP/3, QUIC) via the strategy pattern.
#[derive(Debug, Clone)]
pub enum HttpDownloadChunk {
    /// Data chunk with download progress information
    Data { 
        /// Raw chunk data
        chunk: Vec<u8>, 
        /// Total bytes downloaded so far
        downloaded: u64, 
        /// Total file size if known
        total_size: Option<u64> 
    },
    
    /// Progress update without data (for progress-only notifications)
    Progress { 
        /// Total bytes downloaded so far
        downloaded: u64, 
        /// Total file size if known
        total_size: Option<u64> 
    },
    
    /// Download completed successfully
    Complete,
    
    /// Error occurred during download
    Error { 
        /// Error message
        message: String 
    },
}

/// Type alias for download streams - protocol-agnostic download streaming
pub type HttpDownloadStream = AsyncStream<HttpDownloadChunk, 1024>;

/// HTTP body chunk with metadata
#[derive(Debug, Clone)]
pub struct HttpBodyChunk {
    /// Chunk data
    pub data: Bytes,

    /// Offset in the overall body stream
    pub offset: u64,

    /// Whether this is the final chunk
    pub is_final: bool,

    /// Timestamp when chunk was received
    pub timestamp: Instant,
}

impl HttpBodyChunk {
    /// Get the data bytes from this chunk
    pub fn data(&self) -> Option<&[u8]> {
        Some(&self.data)
    }
    
    /// Check if this chunk has an error
    pub fn error(&self) -> Option<&str> {
        None // HttpBodyChunk doesn't carry errors directly
    }
}

impl From<HttpBodyChunk> for HttpChunk {
    fn from(chunk: HttpBodyChunk) -> Self {
        HttpChunk::Body(chunk.data)
    }
}



impl HttpChunk {
    /// Get the data bytes from any chunk variant that contains data
    pub fn data(&self) -> Option<&Bytes> {
        match self {
            HttpChunk::Body(data) | HttpChunk::Data(data) | HttpChunk::Chunk(data) => Some(data),
            HttpChunk::Headers(_, _) | HttpChunk::Trailers(_) | HttpChunk::Error(_) | HttpChunk::End => None,
        }
    }
    
    /// Check if this is an error chunk
    pub fn is_error(&self) -> bool {
        matches!(self, HttpChunk::Error(_))
    }
    
    /// Check if this is the end marker
    pub fn is_end(&self) -> bool {
        matches!(self, HttpChunk::End)
    }
}

impl fluent_ai_async::prelude::MessageChunk for HttpChunk {
    #[inline]
    fn bad_chunk(error_message: String) -> Self {
        HttpChunk::Error(error_message)
    }
    
    #[inline]
    fn error(&self) -> Option<&str> {
        match self {
            HttpChunk::Error(msg) => Some(msg.as_str()),
            _ => None,
        }
    }
}

impl fluent_ai_async::prelude::MessageChunk for HttpDownloadChunk {
    #[inline]
    fn bad_chunk(error_message: String) -> Self {
        HttpDownloadChunk::Error { message: error_message }
    }
    
    #[inline]
    fn error(&self) -> Option<&str> {
        match self {
            HttpDownloadChunk::Error { message } => Some(message.as_str()),
            _ => None,
        }
    }
}

impl HttpDownloadChunk {
    /// Get the data bytes from this download chunk
    pub fn data(&self) -> Option<&[u8]> {
        match self {
            HttpDownloadChunk::Data { chunk, .. } => Some(chunk.as_slice()),
            _ => None,
        }
    }
    
    /// Get download progress information
    pub fn progress(&self) -> Option<(u64, Option<u64>)> {
        match self {
            HttpDownloadChunk::Data { downloaded, total_size, .. } => Some((*downloaded, *total_size)),
            HttpDownloadChunk::Progress { downloaded, total_size } => Some((*downloaded, *total_size)),
            _ => None,
        }
    }
    
    /// Check if this is the completion marker
    pub fn is_complete(&self) -> bool {
        matches!(self, HttpDownloadChunk::Complete)
    }
}

impl HttpResponse {
    /// Create a new HttpResponse with the given component streams
    pub fn new(
        headers_stream: AsyncStream<HttpHeader, 256>,
        body_stream: AsyncStream<HttpBodyChunk, 1024>,
        trailers_stream: AsyncStream<HttpHeader, 64>,
        version: Version,
        stream_id: u64,
    ) -> Self {
        Self {
            status: AtomicU16::new(0),
            headers_stream,
            body_stream,
            trailers_stream,
            version,
            stream_id,
        }
    }

    /// Get status code (0 if not yet received) - lock-free, zero-cost
    #[inline(always)]
    pub fn status(&self) -> u16 {
        self.status.load(Ordering::Acquire)
    }
    
    /// Get StatusCode if available
    #[inline(always)]
    pub fn status_code(&self) -> Option<StatusCode> {
        match self.status() {
            0 => None,
            code => StatusCode::from_u16(code).ok()
        }
    }
    
    /// Set status (called by protocol layers only)
    #[inline(always)]
    pub(crate) fn set_status(&self, status: StatusCode) {
        self.status.store(status.as_u16(), Ordering::Release);
    }
    
    /// Check methods - zero-cost, lock-free
    #[inline(always)]
    pub fn is_success(&self) -> bool {
        self.status_code().map_or(false, |s| s.is_success())
    }
    
    #[inline(always)]
    pub fn is_error(&self) -> bool {
        self.status_code().map_or(false, |s| s.is_client_error() || s.is_server_error())
    }
    
    #[inline(always)]
    pub fn is_redirect(&self) -> bool {
        self.status_code().map_or(false, |s| s.is_redirection())
    }

    /// Create an empty response (used for errors)
    pub fn empty() -> Self {
        // Create proper AsyncStreams using channel factory method
        let (_, headers_stream) = AsyncStream::channel();
        let (_, body_stream) = AsyncStream::channel();
        let (_, trailers_stream) = AsyncStream::channel();

        Self {
            status: AtomicU16::new(0),
            headers_stream,
            body_stream,
            trailers_stream,
            version: Version::HTTP_11,
            stream_id: 0,
        }
    }

    /// Create HttpResponse from HTTP/1.1 response
    pub fn from_http1_response(
        status: StatusCode,
        headers: HeaderMap,
        body_stream: AsyncStream<HttpBodyChunk, 1024>,
        stream_id: u64,
    ) -> Self {
        // Create channels for headers only
        let (headers_sender, headers_stream) = AsyncStream::channel();
        let (_, trailers_stream) = AsyncStream::channel(); // HTTP/1.1 has no trailers

        // Emit headers immediately
        for (name, value) in headers.iter() {
            let http_header = HttpHeader {
                name: name.clone(),
                value: value.clone(),
                timestamp: Instant::now(),
            };
            let _ = headers_sender.send(http_header);
        }

        Self {
            status: AtomicU16::new(status.as_u16()),
            headers_stream,
            body_stream,
            trailers_stream,
            version: Version::HTTP_11,
            stream_id,
        }
    }

    /// Create HttpResponse from HTTP/2 response
    pub fn from_http2_response(
        status: StatusCode,
        headers: HeaderMap,
        body_stream: AsyncStream<HttpBodyChunk, 1024>,
        trailers_stream: AsyncStream<HttpHeader, 64>,
        stream_id: u64,
    ) -> Self {
        // Create channels for headers only
        let (headers_sender, headers_stream) = AsyncStream::channel();

        // Emit headers immediately
        for (name, value) in headers.iter() {
            let http_header = HttpHeader {
                name: name.clone(),
                value: value.clone(),
                timestamp: Instant::now(),
            };
            let _ = headers_sender.send(http_header);
        }

        Self {
            status: AtomicU16::new(status.as_u16()),
            headers_stream,
            body_stream,
            trailers_stream,
            version: Version::HTTP_2,
            stream_id,
        }
    }

    /// Create HttpResponse from HTTP/3 response
    pub fn from_http3_response(
        status: StatusCode,
        headers: HeaderMap,
        body_stream: AsyncStream<HttpBodyChunk, 1024>,
        trailers_stream: AsyncStream<HttpHeader, 64>,
        stream_id: u64,
    ) -> Self {
        // Create channels for headers only
        let (headers_sender, headers_stream) = AsyncStream::channel();

        // Emit headers immediately
        for (name, value) in headers.iter() {
            let http_header = HttpHeader {
                name: name.clone(),
                value: value.clone(),
                timestamp: Instant::now(),
            };
            let _ = headers_sender.send(http_header);
        }

        Self {
            status: AtomicU16::new(status.as_u16()),
            headers_stream,
            body_stream,
            trailers_stream,
            version: Version::HTTP_3,
            stream_id,
        }
    }

    /// Create error HttpResponse
    pub fn error(status_code: StatusCode, message: String) -> Self {
        use bytes::Bytes;

        // Create channels
        let (headers_sender, headers_stream) = AsyncStream::channel();
        let (body_sender, body_stream) = AsyncStream::channel();
        let (_, trailers_stream) = AsyncStream::channel();

        // Emit content-type header
        let content_type_header = HttpHeader {
            name: http::header::CONTENT_TYPE,
            value: HeaderValue::from_static("text/plain"),
            timestamp: Instant::now(),
        };
        let _ = headers_sender.send(content_type_header);

        // Emit error message as body
        let error_chunk = HttpBodyChunk {
            data: Bytes::from(message),
            offset: 0,
            is_final: true,
            timestamp: Instant::now(),
        };
        let _ = body_sender.send(error_chunk);

        Self {
            status: AtomicU16::new(status_code.as_u16()),
            headers_stream,
            body_stream,
            trailers_stream,
            version: Version::HTTP_11,
            stream_id: 0,
        }
    }

    /// Collect all headers into a HeaderMap
    /// Note: This consumes the headers stream
    pub async fn collect_headers(&mut self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        while let Some(header) = self.headers_stream.next().await {
            headers.insert(header.name, header.value);
        }
        headers
    }

    /// Collect the entire body into a single Bytes buffer
    /// Note: This consumes the body stream and may use significant memory
    pub async fn collect_body(&mut self) -> Bytes {
        let mut chunks = Vec::new();
        let mut total_size = 0;

        while let Some(chunk) = self.body_stream.next().await {
            total_size += chunk.data.len();
            chunks.push(chunk.data);
        }

        // Combine all chunks into a single Bytes
        if chunks.is_empty() {
            Bytes::new()
        } else if chunks.len() == 1 {
            chunks.into_iter().next().unwrap()
        } else {
            let mut combined = Vec::with_capacity(total_size);
            for chunk in chunks {
                combined.extend_from_slice(&chunk);
            }
            Bytes::from(combined)
        }
    }




}

impl HttpStatus {
    /// Create a new HttpStatus
    pub fn new(code: StatusCode, reason: String, version: Version) -> Self {
        Self {
            code,
            reason,
            version,
            timestamp: Instant::now(),
        }
    }

    /// Create from just a status code (for HTTP/2 and HTTP/3)
    pub fn from_code(code: StatusCode, version: Version) -> Self {
        Self {
            code,
            reason: String::new(),
            version,
            timestamp: Instant::now(),
        }
    }
}

impl HttpHeader {
    /// Create a new HttpHeader
    pub fn new(name: HeaderName, value: HeaderValue) -> Self {
        Self {
            name,
            value,
            timestamp: Instant::now(),
        }
    }
}

impl HttpBodyChunk {
    /// Create a new HttpBodyChunk
    pub fn new(data: Bytes, offset: u64, is_final: bool) -> Self {
        Self {
            data,
            offset,
            is_final,
            timestamp: Instant::now(),
        }
    }
}

impl fluent_ai_async::prelude::MessageChunk for HttpResponse {
    fn bad_chunk(error: String) -> Self {
        use fluent_ai_async::AsyncStream;

        let (_, headers_stream) = AsyncStream::channel();
        let (_, body_stream) = AsyncStream::channel();
        let (_, trailers_stream) = AsyncStream::channel();

        Self {
            status: AtomicU16::new(StatusCode::INTERNAL_SERVER_ERROR.as_u16()),
            headers_stream,
            body_stream,
            trailers_stream,
            version: Version::HTTP_11,
            stream_id: 0,
        }
    }

    fn is_error(&self) -> bool {
        false
    }

    fn error(&self) -> Option<&str> {
        None
    }
}

impl Default for HttpResponse {
    fn default() -> Self {
        Self::empty()
    }
}

impl fluent_ai_async::prelude::MessageChunk for HttpStatus {
    fn bad_chunk(error: String) -> Self {
        Self {
            code: StatusCode::INTERNAL_SERVER_ERROR,
            reason: error,
            version: Version::HTTP_11,
            timestamp: Instant::now(),
        }
    }

    fn is_error(&self) -> bool {
        self.code.is_server_error() || self.code.is_client_error()
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some(&self.reason)
        } else {
            None
        }
    }
}

impl Default for HttpStatus {
    fn default() -> Self {
        Self {
            code: StatusCode::OK,
            reason: "OK".to_string(),
            version: Version::HTTP_11,
            timestamp: Instant::now(),
        }
    }
}

impl fluent_ai_async::prelude::MessageChunk for HttpHeader {
    fn bad_chunk(error: String) -> Self {
        Self {
            name: HeaderName::from_static("x-error"),
            value: HeaderValue::from_str(&error)
                .unwrap_or_else(|_| HeaderValue::from_static("invalid")),
            timestamp: Instant::now(),
        }
    }

    fn is_error(&self) -> bool {
        self.name == "x-error"
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            self.value.to_str().ok()
        } else {
            None
        }
    }
}

impl Default for HttpHeader {
    fn default() -> Self {
        Self {
            name: HeaderName::from_static("content-length"),
            value: HeaderValue::from_static("0"),
            timestamp: Instant::now(),
        }
    }
}

impl fluent_ai_async::prelude::MessageChunk for HttpBodyChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            data: Bytes::from(error),
            offset: 0,
            is_final: true,
            timestamp: Instant::now(),
        }
    }

    fn is_error(&self) -> bool {
        false // Body chunks don't represent errors by themselves
    }

    fn error(&self) -> Option<&str> {
        None
    }
}

impl Default for HttpBodyChunk {
    fn default() -> Self {
        Self {
            data: Bytes::new(),
            offset: 0,
            is_final: false,
            timestamp: Instant::now(),
        }
    }
}
