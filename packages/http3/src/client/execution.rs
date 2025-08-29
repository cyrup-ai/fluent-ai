//! HTTP request execution with streaming support
//!
//! Provides streaming execution methods for HTTP requests and downloads
//! with zero-allocation streaming patterns and efficient error handling.

use fluent_ai_async::{AsyncStream, emit, spawn_task};

use crate::prelude::*;
/// Download stream type alias for download operations  
pub type DownloadStream = crate::http::response::HttpDownloadStream;


impl HttpClient {

    /// Execute request with strategy directly (avoiding circular dependency)
    fn execute_with_strategy_direct(&self, request: HttpRequest) -> Result<crate::http::response::HttpResponse, String> {
        let uri_str = request.uri();
        let is_https = uri_str.starts_with("https://");

        match self.strategy() {
            crate::protocols::HttpProtocolStrategy::Auto { prefer, fallback_chain, configs } => {
                // Try H3 -> H2 -> H1.1 fallback for HTTPS, H2 -> H1.1 for HTTP
                if is_https {
                    self.try_http3_fallback_http2_direct(request)
                } else {
                    self.try_http2_fallback_http1_direct(request)
                }
            }
            crate::protocols::HttpProtocolStrategy::Http3Only => {
                if !is_https {
                    return Err("HTTP/3 requires HTTPS".to_string());
                }
                self.execute_http3_direct(request)
            }
            crate::protocols::HttpProtocolStrategy::Http2Only => self.execute_http2_direct(request),
            crate::protocols::HttpProtocolStrategy::Http1Only => self.execute_http1_direct(request),
            crate::protocols::HttpProtocolStrategy::AiOptimized => {
                // Enhanced retry logic for AI providers
                if is_https {
                    self.try_http3_fallback_http2_direct(request)
                } else {
                    self.try_http2_fallback_http1_direct(request)
                }
            }
            crate::protocols::HttpProtocolStrategy::StreamingOptimized => {
                // Prefer HTTP/2 for streaming due to multiplexing
                if is_https {
                    self.try_http2_fallback_http3_direct(request)
                } else {
                    self.try_http2_fallback_http1_direct(request)
                }
            }
            crate::protocols::HttpProtocolStrategy::LowLatency => {
                // Prefer HTTP/3 for low latency due to QUIC
                if is_https {
                    self.try_http3_fallback_http2_direct(request)
                } else {
                    self.execute_http2_direct(request)
                }
            }
        }
    }

    /// Direct HTTP/3 execution  
    fn execute_http3_direct(&self, request: HttpRequest) -> Result<crate::http::response::HttpResponse, String> {
        // Use HTTP/3 only strategy for direct execution
        let h3_strategy = crate::protocols::HttpProtocolStrategy::Http3(crate::protocols::H3Config::default());
        h3_strategy.execute(request)
    }

    /// Direct HTTP/2 execution
    fn execute_http2_direct(&self, request: HttpRequest) -> Result<crate::http::response::HttpResponse, String> {
        // Use HTTP/2 only strategy for direct execution
        let h2_strategy = crate::protocols::HttpProtocolStrategy::Http2(crate::protocols::H2Config::default());
        h2_strategy.execute(request)
    }

    /// Direct HTTP/1.1 execution
    fn execute_http1_direct(&self, request: HttpRequest) -> Result<crate::http::response::HttpResponse, String> {
        // Use HTTP/1.1 only strategy for direct execution
        let h1_strategy = crate::protocols::HttpProtocolStrategy::Http1Only;
        h1_strategy.execute(request)
    }

    /// Try HTTP/3 with fallback to HTTP/2 (direct)
    fn try_http3_fallback_http2_direct(&self, request: HttpRequest) -> Result<crate::http::response::HttpResponse, String> {
        match self.execute_http3_direct(request.clone()) {
            Ok(response) => Ok(response),
            Err(_) => self.execute_http2_direct(request),
        }
    }

    /// Try HTTP/2 with fallback to HTTP/1.1 (direct)
    fn try_http2_fallback_http1_direct(&self, request: HttpRequest) -> Result<crate::http::response::HttpResponse, String> {
        match self.execute_http2_direct(request.clone()) {
            Ok(response) => Ok(response),
            Err(_) => self.execute_http1_direct(request),
        }
    }

    /// Try HTTP/2 with fallback to HTTP/3 (direct, for streaming optimization)
    fn try_http2_fallback_http3_direct(&self, request: HttpRequest) -> Result<crate::http::response::HttpResponse, String> {
        match self.execute_http2_direct(request.clone()) {
            Ok(response) => Ok(response),
            Err(_) => self.execute_http3_direct(request),
        }
    }


    /// Executes a download request and returns a stream of `DownloadChunk`s.
    ///
    /// Optimized for file downloads with progress tracking. Each chunk includes
    /// download progress information including bytes downloaded and total size.
    pub fn download_file(&self, request: HttpRequest) -> DownloadStream {
        use std::sync::atomic::Ordering;
        use std::time::Instant;

        // Record download request start
        let _start_time = Instant::now();
        self.stats().total_requests.fetch_add(1, Ordering::Relaxed);

        // Record request body size if present
        if let Some(body) = request.body() {
            self.stats()
                .bytes_sent
                .fetch_add(body.len() as u64, Ordering::Relaxed);
        }

        // Use strategy pattern for protocol-agnostic downloads
        match self.strategy().download_stream(request) {
            Ok(download_stream) => download_stream,
            Err(error_msg) => {
                // Return error in download stream
                AsyncStream::with_channel(move |sender| {
                    spawn_task(move || {
                        let error_chunk = crate::http::response::HttpDownloadChunk::Error { 
                            message: error_msg
                        };
                        emit!(sender, error_chunk);
                    });
                })
            }
        }
    }
}
