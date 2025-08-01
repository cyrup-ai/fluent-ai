//! Thin HTTP client orchestration layer - delegates to specialized operation modules

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[allow(unused_imports)]
use std::time::SystemTime;

use async_stream::stream;
use futures_util::StreamExt;

use crate::{
    DownloadChunk, DownloadStream, HttpChunk, HttpConfig, HttpError, HttpRequest, HttpStream};

/// Lightweight HTTP client that orchestrates specialized operation modules
#[derive(Debug, Clone)]
pub struct HttpClient {
    /// Underlying HTTP client (delegated to operation modules)
    inner: reqwest::Client,
    /// Client configuration
    config: HttpConfig,
    /// Atomic metrics for lock-free statistics
    stats: ClientStats}

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
                                    bytes_downloaded};
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
    pub cache_misses: Arc<AtomicUsize>}

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
    pub cache_misses: usize}

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
            cache_misses: self.cache_misses.load(Ordering::Relaxed)}
    }
}

impl HttpClient {
    /// Create a new HttpClient with the specified configuration
    /// Enables HTTP/3 (QUIC) optimization when config.http3_enabled is true
    pub fn with_config(config: HttpConfig) -> Result<Self, crate::HttpError> {
        let mut builder = reqwest::Client::builder()
            .pool_max_idle_per_host(config.pool_max_idle_per_host)
            .pool_idle_timeout(config.pool_idle_timeout)
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .tcp_nodelay(config.tcp_nodelay)
            .use_rustls_tls()
            .tls_built_in_root_certs(config.use_native_certs)
            .https_only(config.https_only)
            .user_agent(&config.user_agent);

        // HTTP/3 (QUIC) configuration when enabled
        if config.http3_enabled {
            // Enable HTTP/3 protocol version preference
            builder = builder.http3_prior_knowledge();
            
            // Note: Advanced QUIC configuration methods require the 'reqwest_unstable' feature
            // which is not enabled by default. For full HTTP/3 optimization, enable this feature.
            #[cfg(feature = "reqwest_unstable")]
            {
                // Configure QUIC connection parameters when unstable features are enabled
                if let Some(idle_timeout) = config.quic_max_idle_timeout {
                    builder = builder.http3_max_idle_timeout(idle_timeout);
                }
                
                if let Some(stream_window) = config.quic_stream_receive_window {
                    builder = builder.http3_stream_receive_window(stream_window as u64);
                }
                
                if let Some(conn_window) = config.quic_receive_window {
                    builder = builder.http3_conn_receive_window(conn_window as u64);
                }
                
                if let Some(send_window) = config.quic_send_window {
                    builder = builder.http3_send_window(send_window);
                }
                
                // Enable BBR congestion control if requested
                if config.quic_congestion_bbr {
                    builder = builder.http3_congestion_bbr();
                }
                
                // Configure HTTP/3 protocol parameters
                if let Some(header_size) = config.h3_max_field_section_size {
                    builder = builder.http3_max_field_section_size(header_size);
                }
                
                if config.h3_enable_grease {
                    builder = builder.http3_send_grease(true);
                }
            }
            
            #[cfg(not(feature = "reqwest_unstable"))]
            {
                // Basic HTTP/3 is enabled but advanced QUIC optimizations are not available
                // without the reqwest_unstable feature. HTTP/3 will still provide significant
                // performance improvements over HTTP/2 for most use cases.
                log::debug!("HTTP/3 enabled but advanced QUIC optimizations require 'reqwest_unstable' feature");
            }
        } else {
            // HTTP/2 configuration when HTTP/3 is disabled
            builder = builder.http2_prior_knowledge()
                .http2_adaptive_window(config.http2_adaptive_window);
                
            if let Some(frame_size) = config.http2_max_frame_size {
                builder = builder.http2_max_frame_size(frame_size);
            }
        }

        // Compression configuration
        builder = builder
            .gzip(config.gzip)
            .brotli(config.brotli)
            .deflate(config.deflate);

        // Build the client with error handling
        let inner = builder.build()
            .map_err(|e| crate::HttpError::Configuration(format!("Failed to build HTTP client: {}", e)))?;

        Ok(Self {
            inner,
            config,
            stats: ClientStats::default(),
        })
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        // Use the new with_config constructor with default configuration
        // If configuration fails, fall back to a basic reqwest client
        let config = HttpConfig::default();
        Self::with_config(config)
            .unwrap_or_else(|_| Self {
                inner: reqwest::Client::new(),
                config: HttpConfig::default(),
                stats: ClientStats::default(),
            })
    }
}
