//! Thin HTTP client orchestration layer - delegates to specialized operation modules

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[allow(unused_imports)]
use std::time::SystemTime;

use async_stream::stream;
use futures_util::StreamExt;

use crate::{
    DownloadChunk, DownloadStream, HttpChunk, HttpConfig, HttpError, HttpRequest, HttpStream,
};

/// Lightweight HTTP client that orchestrates specialized operation modules
#[derive(Debug, Clone)]
pub struct HttpClient {
    /// Underlying HTTP client (delegated to operation modules)
    inner: reqwest::Client,
    /// Client configuration
    config: HttpConfig,
    /// Atomic metrics for lock-free statistics
    stats: ClientStats,
}

impl HttpClient {
    /// Returns a reference to the inner `reqwest::Client`.
    pub fn inner(&self) -> &reqwest::Client {
        &self.inner
    }

    /// Returns a reference to the `HttpConfig`.
    pub fn config(&self) -> &HttpConfig {
        &self.config
    }

    /// Get client statistics snapshot
    pub fn stats_snapshot(&self) -> ClientStatsSnapshot {
        self.stats.snapshot()
    }

    /// Executes a request and returns a stream of `HttpChunk`s.
    pub fn execute_streaming(&self, request: HttpRequest) -> HttpStream {
        self.execute_streaming_with_debug(request, false)
    }

    /// Executes a request with optional debug logging
    pub fn execute_streaming_with_debug(&self, request: HttpRequest, debug: bool) -> HttpStream {
        if debug {
            println!("🚀 HTTP Request: {} {}", request.method(), request.url());
            if let Some(body) = request.body() {
                println!("📤 Request Body: {} bytes", body.len());
            }
            println!("📋 Request Headers:");
            for (name, value) in request.headers().iter() {
                if let Ok(value_str) = value.to_str() {
                    println!("  {}: {}", name, value_str);
                }
            }
        }

        let client = self.inner.clone();
        let reqwest_request = match client
            .request(request.method().clone(), request.url())
            .headers(request.headers().clone())
            .body(request.body().cloned().unwrap_or_else(Vec::new))
            .build()
        {
            Ok(req) => req,
            Err(e) => {
                let error_stream = stream! { yield Err(HttpError::from(e)); };
                return HttpStream::new(Box::pin(error_stream));
            }
        };

        let response_stream = stream! {
            match client.execute(reqwest_request).await {
                Ok(response) => {
                    let status = response.status();
                    let headers = response.headers().clone();
                    log::debug!("HTTP Response: {} {}", status.as_u16(), status.canonical_reason().unwrap_or(""));
                    yield Ok(HttpChunk::Head(status, headers));

                    let mut byte_stream = response.bytes_stream();
                    while let Some(item) = byte_stream.next().await {
                        match item {
                            Ok(bytes) => yield Ok(HttpChunk::Body(bytes)),
                            Err(e) => {
                                yield Err(HttpError::from(e));
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    yield Err(HttpError::from(e));
                }
            }
        };

        HttpStream::new(Box::pin(response_stream))
    }

    /// Executes a download request and returns a stream of `DownloadChunk`s.
    pub fn download_file(&self, request: HttpRequest) -> DownloadStream {
        let client = self.inner.clone();
        let reqwest_request = match client
            .request(request.method().clone(), request.url())
            .headers(request.headers().clone())
            .body(request.body().cloned().unwrap_or_else(Vec::new))
            .build()
        {
            Ok(req) => req,
            Err(e) => {
                let error_stream = stream! { yield Err(HttpError::from(e)); };
                return DownloadStream::new(Box::pin(error_stream));
            }
        };

        let download_stream = stream! {
            match client.execute(reqwest_request).await {
                Ok(response) => {
                    let total_size = response.content_length();
                    let mut bytes_downloaded = 0;
                    let mut chunk_number = 0;

                    let mut byte_stream = response.bytes_stream();
                    while let Some(item) = byte_stream.next().await {
                        match item {
                            Ok(bytes) => {
                                bytes_downloaded += bytes.len() as u64;
                                chunk_number += 1;
                                let chunk = DownloadChunk {
                                    data: bytes,
                                    chunk_number,
                                    total_size,
                                    bytes_downloaded,
                                };
                                yield Ok(chunk);
                            }
                            Err(e) => {
                                yield Err(HttpError::from(e));
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    yield Err(HttpError::from(e));
                }
            }
        };

        DownloadStream::new(Box::pin(download_stream))
    }
}

/// Thread-safe client statistics with atomic counters
#[derive(Debug, Default, Clone)]
pub struct ClientStats {
    /// Total number of HTTP requests made
    pub request_count: Arc<AtomicUsize>,
    /// Total number of connections established
    pub connection_count: Arc<AtomicUsize>,
    /// Total bytes sent in request bodies
    pub total_bytes_sent: Arc<AtomicU64>,
    /// Total bytes received in response bodies
    pub total_bytes_received: Arc<AtomicU64>,
    /// Total response time across all requests in nanoseconds
    pub total_response_time_nanos: Arc<AtomicU64>,
    /// Number of successful requests (2xx status codes)
    pub successful_requests: Arc<AtomicUsize>,
    /// Number of failed requests (4xx/5xx status codes or network errors)
    pub failed_requests: Arc<AtomicUsize>,
    /// Number of cache hits
    pub cache_hits: Arc<AtomicUsize>,
    /// Number of cache misses
    pub cache_misses: Arc<AtomicUsize>,
}

/// Immutable snapshot of client statistics at a point in time
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientStatsSnapshot {
    /// Total number of HTTP requests made
    pub request_count: usize,
    /// Total number of connections established
    pub connection_count: usize,
    /// Total bytes sent in request bodies
    pub total_bytes_sent: u64,
    /// Total bytes received in response bodies
    pub total_bytes_received: u64,
    /// Total response time across all requests in nanoseconds
    pub total_response_time_nanos: u64,
    /// Number of successful requests (2xx status codes)
    pub successful_requests: usize,
    /// Number of failed requests (4xx/5xx status codes or network errors)
    pub failed_requests: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
}

impl ClientStats {
    /// Create a snapshot of the current statistics.
    pub fn snapshot(&self) -> ClientStatsSnapshot {
        ClientStatsSnapshot {
            request_count: self.request_count.load(Ordering::Relaxed),
            connection_count: self.connection_count.load(Ordering::Relaxed),
            total_bytes_sent: self.total_bytes_sent.load(Ordering::Relaxed),
            total_bytes_received: self.total_bytes_received.load(Ordering::Relaxed),
            total_response_time_nanos: self.total_response_time_nanos.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
        }
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        let config = HttpConfig::default();
        let inner = reqwest::Client::builder()
            .pool_max_idle_per_host(config.pool_max_idle_per_host)
            .pool_idle_timeout(config.pool_idle_timeout)
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .tcp_nodelay(config.tcp_nodelay)
            .http2_prior_knowledge()
            .http2_adaptive_window(config.http2_adaptive_window)
            .use_rustls_tls()
            .tls_built_in_root_certs(true)
            .https_only(config.https_only)
            .user_agent(&config.user_agent)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            inner,
            config,
            stats: ClientStats::default(),
        }
    }
}
