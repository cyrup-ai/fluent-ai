//! Canonical HTTP client with telemetry integration and zero-allocation streaming
//!
//! Provides the unified HttpClient struct using fluent_ai_async directly
//! with integrated telemetry, connection pooling, and pure streaming protocols

use std::sync::Arc;
use std::time::{Duration, Instant};

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit};
use http::{HeaderMap, Method, Uri};

use crate::telemetry::{ClientStats, ClientStatsSnapshot};
use crate::{HttpChunk, HttpConfig, HttpRequest};

/// Canonical HTTP client with integrated telemetry and zero-allocation streaming
///
/// Combines HTTP/1.1, HTTP/2, and HTTP/3 protocol support with comprehensive
/// telemetry tracking, connection pooling, and fluent_ai_async streaming architecture.
#[derive(Debug, Clone)]
pub struct HttpClient {
    /// Client configuration with protocol preferences and timeouts
    pub(crate) config: HttpConfig,
    /// Zero-allocation atomic telemetry for performance tracking
    pub(crate) stats: ClientStats,
    /// Default headers applied to all requests
    pub(crate) default_headers: HeaderMap,
    /// Connection timeout for establishing new connections
    pub(crate) connect_timeout: Option<Duration>,
    /// Request timeout for individual requests
    pub(crate) request_timeout: Option<Duration>,
}

impl HttpClient {
    /// Create new HttpClient with integrated telemetry and streaming architecture
    #[inline]
    pub fn new(config: HttpConfig) -> Self {
        Self {
            config,
            stats: ClientStats::default(),
            default_headers: HeaderMap::new(),
            connect_timeout: Some(Duration::from_secs(30)),
            request_timeout: Some(Duration::from_secs(60)),
        }
    }

    /// Create HttpClient with custom configuration and telemetry
    #[inline]
    pub fn with_config_and_stats(
        config: HttpConfig,
        stats: ClientStats,
        default_headers: HeaderMap,
        connect_timeout: Option<Duration>,
        request_timeout: Option<Duration>,
    ) -> Self {
        Self {
            config,
            stats,
            default_headers,
            connect_timeout,
            request_timeout,
        }
    }

    /// Get current telemetry snapshot
    #[inline]
    pub fn stats(&self) -> ClientStatsSnapshot {
        self.stats.snapshot()
    }

    /// Convenience method to make a GET request
    #[inline]
    pub fn get(&self, uri: Uri) -> HttpRequest {
        HttpRequest::builder()
            .method(Method::GET)
            .uri(uri)
            .headers(self.default_headers.clone())
            .build()
    }

    /// Convenience method to make a POST request
    #[inline]
    pub fn post(&self, uri: Uri) -> HttpRequest {
        HttpRequest::builder()
            .method(Method::POST)
            .uri(uri)
            .headers(self.default_headers.clone())
            .build()
    }

    /// Convenience method to make a PUT request
    #[inline]
    pub fn put(&self, uri: Uri) -> HttpRequest {
        HttpRequest::builder()
            .method(Method::PUT)
            .uri(uri)
            .headers(self.default_headers.clone())
            .build()
    }

    /// Convenience method to make a DELETE request
    #[inline]
    pub fn delete(&self, uri: Uri) -> HttpRequest {
        HttpRequest::builder()
            .method(Method::DELETE)
            .uri(uri)
            .headers(self.default_headers.clone())
            .build()
    }

    /// Execute HTTP request with telemetry tracking and protocol selection
    ///
    /// Automatically selects optimal protocol (H3 -> H2 -> H1.1) based on server
    /// capabilities and tracks comprehensive telemetry metrics.
    #[inline]
    pub fn execute(
        &self,
        mut request: HttpRequest,
    ) -> AsyncStream<crate::streaming::pipeline::StreamingResponse, 1024> {
        let stats = self.stats.clone();
        let start_time = Instant::now();

        // Merge default headers
        for (key, value) in &self.default_headers {
            if !request.headers().contains_key(key) {
                request.headers_mut().insert(key.clone(), value.clone());
            }
        }

        AsyncStream::with_channel(move |sender| {
            // Update request count
            stats
                .request_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // Determine protocol based on URL scheme and server capabilities
            let uri = request.uri();
            let scheme = uri.scheme_str().unwrap_or("https");

            match scheme {
                "https" => {
                    // Try H3 first, fallback to H2
                    let response_stream =
                        crate::streaming::pipeline::StreamingResponse::from_request(
                            request.clone(),
                        );

                    // Process response stream with telemetry
                    for response_result in response_stream {
                        match response_result {
                            Ok(response) => {
                                // Track successful response
                                stats
                                    .successful_requests
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                emit!(sender, response);
                            }
                            Err(e) => {
                                // Track failed request
                                stats
                                    .failed_requests
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                emit!(
                                    sender,
                                    crate::streaming::pipeline::StreamingResponse::bad_chunk(
                                        e.to_string()
                                    )
                                );
                                return;
                            }
                        }
                    }
                }
                "http" => {
                    // Use H2 for HTTP
                    let response_stream =
                        crate::streaming::pipeline::StreamingResponse::from_request(
                            request.clone(),
                        );

                    // Process response stream with telemetry
                    for response_result in response_stream {
                        match response_result {
                            Ok(response) => {
                                stats
                                    .successful_requests
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                emit!(sender, response);
                            }
                            Err(e) => {
                                stats
                                    .failed_requests
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                emit!(
                                    sender,
                                    crate::streaming::pipeline::StreamingResponse::bad_chunk(
                                        e.to_string()
                                    )
                                );
                                return;
                            }
                        }
                    }
                }
                _ => {
                    // Unsupported scheme
                    stats
                        .failed_requests
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    emit!(
                        sender,
                        crate::streaming::pipeline::StreamingResponse::bad_chunk(format!(
                            "Unsupported URI scheme: {}",
                            scheme
                        ))
                    );
                    return;
                }
            }

            // Track successful request completion
            stats
                .successful_requests
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let elapsed = start_time.elapsed();
            stats.total_response_time_nanos.fetch_add(
                elapsed.as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        })
    }

    /// Execute request with streaming response processing
    ///
    /// Returns an AsyncStream of HttpChunk for low-level streaming control
    #[inline]
    pub fn execute_streaming(&self, request: HttpRequest) -> AsyncStream<HttpChunk, 1024> {
        let response_stream = self.execute(request);

        AsyncStream::with_channel(move |sender| {
            for response in response_stream {
                // Convert StreamingResponse to HttpChunk stream
                let chunk_stream = response.into_chunk_stream();
                for chunk in chunk_stream {
                    emit!(sender, chunk);
                }
            }
        })
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new(HttpConfig::default())
    }
}
