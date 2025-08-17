//! Streaming request/response pipeline for HTTP/2 and HTTP/3
//!
//! Zero-allocation request/response handling using ONLY fluent_ai_async patterns.
//! This module provides the CANONICAL StreamingResponse implementation.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use fluent_ai_async::prelude::*;
use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri, Version};

use crate::http::request::HttpRequest;
use crate::streaming::chunks::HttpChunk;

/// HTTP request for streaming pipeline
#[derive(Debug, Clone)]
pub struct StreamingRequest {
    pub method: Method,
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
    pub stream_id: Option<u64>,
    pub version: Version,
}

impl MessageChunk for StreamingRequest {
    fn bad_chunk(error: String) -> Self {
        StreamingRequest {
            method: Method::GET,
            uri: Uri::from_static("/"),
            headers: HeaderMap::new(),
            body: error.into_bytes(),
            stream_id: None,
            version: Version::HTTP_3,
        }
    }

    fn is_error(&self) -> bool {
        self.body.len() > 0 && self.method == Method::GET && self.uri.path() == "/"
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            std::str::from_utf8(&self.body).ok()
        } else {
            None
        }
    }
}

/// HTTP response for streaming pipeline
///
/// This is the CANONICAL StreamingResponse implementation that consolidates all
/// previous response variants into a single, comprehensive streaming response type.
#[derive(Debug, Clone)]
pub struct StreamingResponse {
    /// Response status and headers
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub version: Version,

    /// Response body as streaming chunks
    pub chunks: AsyncStream<HttpChunk, 1024>,

    /// Response metadata
    pub stream_id: Option<u64>,
    pub is_complete: AtomicBool,
    pub content_length: Option<u64>,
    pub content_type: Option<String>,

    /// Streaming statistics
    pub bytes_received: AtomicU64,
    pub chunks_received: AtomicU64,
    pub start_time: Instant,

    /// Response configuration
    pub follow_redirects: bool,
    pub max_redirects: u32,
    pub timeout: Option<Duration>,

    /// Caching information
    pub cache_control: Option<String>,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
    pub expires: Option<String>,
}

impl Default for StreamingResponse {
    fn default() -> Self {
        Self {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            version: Version::HTTP_3,
            chunks: AsyncStream::with_channel(|_| {}),
            stream_id: None,
            is_complete: AtomicBool::new(false),
            content_length: None,
            content_type: None,
            bytes_received: AtomicU64::new(0),
            chunks_received: AtomicU64::new(0),
            start_time: Instant::now(),
            follow_redirects: true,
            max_redirects: 10,
            timeout: Some(Duration::from_secs(30)),
            cache_control: None,
            etag: None,
            last_modified: None,
            expires: None,
        }
    }
}

impl MessageChunk for StreamingResponse {
    #[inline]
    fn bad_chunk(error: String) -> Self {
        let chunks = AsyncStream::with_channel(move |sender| {
            emit!(
                sender,
                HttpChunk::Error(error.clone())
            );
        });

        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            headers: HeaderMap::new(),
            version: Version::HTTP_3,
            chunks,
            stream_id: None,
            is_complete: AtomicBool::new(true),
            content_length: None,
            content_type: None,
            bytes_received: AtomicU64::new(0),
            chunks_received: AtomicU64::new(0),
            timeout: None,
            cache_control: None,
            etag: None,
            last_modified: None,
            expires: None,
            start_time: Instant::now(),
            follow_redirects: false,
            max_redirects: 0,
            redirect_count: AtomicU32::new(0),
        }
    }

    #[inline]
    fn is_error(&self) -> bool {
        self.status.is_server_error() || self.status.is_client_error()
    }

    #[inline]
    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some(self.status.canonical_reason().unwrap_or("Unknown error"))
        } else {
            None
        }
    }
}

impl StreamingResponse {
    /// Create new streaming response
    #[inline]
    pub fn new(status: StatusCode, headers: HeaderMap, version: Version) -> Self {
        let mut response = Self::default();
        response.status = status;
        response.headers = headers;
        response.version = version;
        response.extract_metadata();
        response
    }

    /// Create response with chunk stream
    #[inline]
    pub fn with_chunks(
        status: StatusCode,
        headers: HeaderMap,
        version: Version,
        chunks: AsyncStream<HttpChunk, 1024>,
    ) -> Self {
        let mut response = Self {
            status,
            headers,
            version,
            chunks,
            ..Default::default()
        };
        response.extract_metadata();
        response
    }

    /// Create response from HttpRequest execution
    pub fn from_request_execution<F>(request: HttpRequest, executor: F) -> Self
    where
        F: FnOnce(HttpRequest) -> AsyncStream<HttpChunk, 1024> + Send + 'static,
    {
        let chunks = executor(request);

        Self {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            version: Version::HTTP_3,
            chunks,
            ..Default::default()
        }
    }

    /// Extract metadata from headers
    fn extract_metadata(&mut self) {
        // Extract content length
        if let Some(cl) = self.headers.get(http::header::CONTENT_LENGTH) {
            if let Ok(cl_str) = cl.to_str() {
                self.content_length = cl_str.parse().ok();
            }
        }

        // Extract content type
        if let Some(ct) = self.headers.get(http::header::CONTENT_TYPE) {
            if let Ok(ct_str) = ct.to_str() {
                self.content_type = Some(ct_str.to_string());
            }
        }

        // Extract cache control
        if let Some(cc) = self.headers.get(http::header::CACHE_CONTROL) {
            if let Ok(cc_str) = cc.to_str() {
                self.cache_control = Some(cc_str.to_string());
            }
        }

        // Extract ETag
        if let Some(etag) = self.headers.get(http::header::ETAG) {
            if let Ok(etag_str) = etag.to_str() {
                self.etag = Some(etag_str.to_string());
            }
        }

        // Extract Last-Modified
        if let Some(lm) = self.headers.get(http::header::LAST_MODIFIED) {
            if let Ok(lm_str) = lm.to_str() {
                self.last_modified = Some(lm_str.to_string());
            }
        }

        // Extract Expires
        if let Some(exp) = self.headers.get(http::header::EXPIRES) {
            if let Ok(exp_str) = exp.to_str() {
                self.expires = Some(exp_str.to_string());
            }
        }
    }

    /// Get next response chunk
    #[inline]
    pub fn try_next(&mut self) -> Option<HttpChunk> {
        if let Some(chunk) = self.chunks.try_next() {
            self.update_statistics(&chunk);
            Some(chunk)
        } else {
            None
        }
    }

    /// Update streaming statistics
    fn update_statistics(&self, chunk: &HttpChunk) {
        self.chunks_received.fetch_add(1, Ordering::Relaxed);

        match chunk {
            HttpChunk::Body(data) => {
                self.bytes_received
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
            }
            HttpChunk::Error(_) => {
                self.is_complete.store(true, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Collect all response chunks
    pub fn collect_chunks(self) -> Vec<HttpChunk> {
        self.chunks.collect()
    }

    /// Collect response body as bytes
    pub fn collect_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();

        for chunk in self.chunks {
            match chunk {
                HttpChunk::Body(data) => {
                    bytes.extend_from_slice(&data);
                }
                HttpChunk::Error(_) => break,
                _ => {}
            }
        }

        bytes
    }

    /// Collect response body as string
    pub fn collect_string(self) -> Result<String, std::string::FromUtf8Error> {
        let bytes = self.collect_bytes();
        String::from_utf8(bytes)
    }

    /// Collect and deserialize JSON
    pub fn collect_json<T>(self) -> Result<T, serde_json::Error>
    where
        T: serde::de::DeserializeOwned,
    {
        let bytes = self.collect_bytes();
        serde_json::from_slice(&bytes)
    }

    /// Check if response is successful (2xx status)
    #[inline]
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    /// Check if response is redirect (3xx status)
    #[inline]
    pub fn is_redirect(&self) -> bool {
        self.status.is_redirection()
    }

    /// Check if response is client error (4xx status)
    #[inline]
    pub fn is_client_error(&self) -> bool {
        self.status.is_client_error()
    }

    /// Check if response is server error (5xx status)
    #[inline]
    pub fn is_server_error(&self) -> bool {
        self.status.is_server_error()
    }

    /// Check if response is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.is_complete.load(Ordering::Relaxed)
    }

    /// Get response duration
    #[inline]
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get response statistics
    #[inline]
    pub fn stats(&self) -> ResponseStats {
        ResponseStats {
            status: self.status,
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            chunks_received: self.chunks_received.load(Ordering::Relaxed),
            duration: self.duration(),
            is_complete: self.is_complete(),
            content_length: self.content_length,
            content_type: self.content_type.clone(),
        }
    }

    /// Transform response chunks with a mapping function
    pub fn map_chunks<F, T>(self, mapper: F) -> AsyncStream<T, 1024>
    where
        F: Fn(HttpChunk) -> T + Send + 'static,
        T: Send + 'static,
    {
        AsyncStream::with_channel(move |sender| {
            for chunk in self.chunks {
                let mapped = mapper(chunk);
                emit!(sender, mapped);
            }
        })
    }

    /// Filter response chunks based on predicate
    pub fn filter_chunks<F>(self, predicate: F) -> AsyncStream<HttpChunk, 1024>
    where
        F: Fn(&HttpChunk) -> bool + Send + 'static,
    {
        AsyncStream::with_channel(move |sender| {
            for chunk in self.chunks {
                if predicate(&chunk) {
                    emit!(sender, chunk);
                }
            }
        })
    }

    /// Take only the first N chunks
    pub fn take_chunks(self, n: usize) -> AsyncStream<HttpChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            let mut count = 0;
            for chunk in self.chunks {
                if count >= n {
                    break;
                }
                emit!(sender, chunk);
                count += 1;
            }
        })
    }

    /// Add header to response
    #[inline]
    pub fn header<K, V>(mut self, key: K, value: V) -> Self
    where
        K: TryInto<HeaderName>,
        V: TryInto<HeaderValue>,
    {
        if let (Ok(name), Ok(val)) = (key.try_into(), value.try_into()) {
            self.headers.insert(name, val);
        }
        self
    }

    /// Set status code
    #[inline]
    pub fn status(mut self, status: StatusCode) -> Self {
        self.status = status;
        self
    }

    /// Set HTTP version
    #[inline]
    pub fn version(mut self, version: Version) -> Self {
        self.version = version;
        self
    }

    /// Set stream ID
    #[inline]
    pub fn stream_id(mut self, stream_id: u64) -> Self {
        self.stream_id = Some(stream_id);
        self
    }
}

/// Response statistics
#[derive(Debug, Clone)]
pub struct ResponseStats {
    pub status: StatusCode,
    pub bytes_received: u64,
    pub chunks_received: u64,
    pub duration: Duration,
    pub is_complete: bool,
    pub content_length: Option<u64>,
    pub content_type: Option<String>,
}

/// Request/response pipeline using ONLY AsyncStream patterns
#[derive(Debug)]
pub struct StreamingPipeline {
    pub active_requests: HashMap<String, StreamingRequest>,
    pub next_request_id: u64,
    pub default_timeout: Duration,
    pub max_concurrent_requests: usize,
}

impl Default for StreamingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingPipeline {
    /// Create new streaming pipeline
    #[inline]
    pub fn new() -> Self {
        StreamingPipeline {
            active_requests: HashMap::new(),
            next_request_id: 1,
            default_timeout: Duration::from_secs(30),
            max_concurrent_requests: 100,
        }
    }

    /// Execute HTTP request using AsyncStream patterns
    pub fn execute_request(
        &mut self,
        request: StreamingRequest,
    ) -> AsyncStream<StreamingResponse, 1024> {
        let request_id = format!("req-{}", self.next_request_id);
        self.next_request_id += 1;

        self.active_requests
            .insert(request_id.clone(), request.clone());

        AsyncStream::with_channel(move |sender| {
            // Create response chunks based on request
            let response_chunks = AsyncStream::with_channel(move |chunk_sender| {
                // Emit status
                emit!(
                    chunk_sender,
                    HttpChunk::Head(StatusCode::OK, HeaderMap::new())
                );

                // Emit body (echo request body for now)
                if !request.body.is_empty() {
                    emit!(
                        chunk_sender,
                        HttpChunk::Body(Bytes::from(request.body))
                    );
                }
            });

            let response = StreamingResponse::with_chunks(
                StatusCode::OK,
                HeaderMap::new(),
                Version::HTTP_3,
                response_chunks,
            );

            emit!(sender, response);
        })
    }

    /// Get active request count
    #[inline]
    pub fn active_request_count(&self) -> usize {
        self.active_requests.len()
    }

    /// Remove completed request
    #[inline]
    pub fn remove_request(&mut self, request_id: &str) -> Option<StreamingRequest> {
        self.active_requests.remove(request_id)
    }

    /// Set default timeout
    #[inline]
    pub fn set_default_timeout(&mut self, timeout: Duration) {
        self.default_timeout = timeout;
    }

    /// Set max concurrent requests
    #[inline]
    pub fn set_max_concurrent_requests(&mut self, max: usize) {
        self.max_concurrent_requests = max;
    }
}

impl StreamingRequest {
    /// Create new streaming request
    #[inline]
    pub fn new(method: Method, uri: Uri) -> Self {
        Self {
            method,
            uri,
            headers: HeaderMap::new(),
            body: Vec::new(),
            stream_id: None,
            version: Version::HTTP_3,
        }
    }

    /// Create from HttpRequest
    pub fn from_http_request(request: HttpRequest) -> Self {
        let mut headers = request.headers().clone();
        let body = match request.body() {
            Some(body) => match body {
                crate::http::request::RequestBody::Bytes(bytes) => bytes.to_vec(),
                crate::http::request::RequestBody::Text(text) => text.as_bytes().to_vec(),
                crate::http::request::RequestBody::Json(json) => {
                    serde_json::to_vec(json).unwrap_or_default()
                }
                _ => Vec::new(),
            },
            None => Vec::new(),
        };

        // Convert URL to URI
        let uri = request
            .url()
            .as_str()
            .parse()
            .unwrap_or_else(|_| Uri::from_static("/"));

        Self {
            method: request.method().clone(),
            uri,
            headers,
            body,
            stream_id: None,
            version: request.version(),
        }
    }

    /// Add header
    #[inline]
    pub fn header<K, V>(mut self, key: K, value: V) -> Self
    where
        K: TryInto<HeaderName>,
        V: TryInto<HeaderValue>,
    {
        if let (Ok(name), Ok(val)) = (key.try_into(), value.try_into()) {
            self.headers.insert(name, val);
        }
        self
    }

    /// Set body
    #[inline]
    pub fn body<B: Into<Vec<u8>>>(mut self, body: B) -> Self {
        self.body = body.into();
        self
    }

    /// Set stream ID
    #[inline]
    pub fn stream_id(mut self, stream_id: u64) -> Self {
        self.stream_id = Some(stream_id);
        self
    }
}
