//! HTTP request execution with streaming support
//!
//! Provides streaming execution methods for HTTP requests and downloads
//! with zero-allocation streaming patterns and efficient error handling.

use async_stream::stream;
use futures_util::StreamExt;

use super::core::HttpClient;
use crate::{DownloadChunk, DownloadStream, HttpChunk, HttpError, HttpRequest, HttpStream};

impl HttpClient {
    /// Executes a request and returns a stream of `HttpChunk`s.
    ///
    /// Provides streaming response processing with minimal memory allocation.
    /// Response headers are emitted first, followed by body chunks as they arrive.
    #[inline]
    pub fn execute_streaming(&self, request: HttpRequest) -> HttpStream {
        self.execute_streaming_with_debug(request, false)
    }

    /// Executes a request with optional debug logging
    ///
    /// When debug is enabled, logs request details including method, URL, headers,
    /// and body size for debugging and monitoring purposes.
    pub fn execute_streaming_with_debug(&self, request: HttpRequest, debug: bool) -> HttpStream {
        use std::sync::atomic::Ordering;
        use std::time::Instant;
        
        // Record request start and increment request counter
        let start_time = Instant::now();
        self.stats.request_count.fetch_add(1, Ordering::Relaxed);
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
        // Record request body size if present
        if let Some(body) = request.body() {
            self.stats.total_bytes_sent.fetch_add(body.len() as u64, Ordering::Relaxed);
        }
        
        let reqwest_request = match client
            .request(request.method().clone(), request.url())
            .headers(request.headers().clone())
            .body(request.body().cloned().unwrap_or_else(Vec::new))
            .build()
        {
            Ok(req) => req,
            Err(e) => {
                // Record failed request
                self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                let error_stream = stream! { yield Err(HttpError::from(e)); };
                return HttpStream::new(Box::pin(error_stream));
            }
        };

        let stats = self.stats.clone();
        let response_stream = stream! {
            match client.execute(reqwest_request).await {
                Ok(response) => {
                    let status = response.status();
                    let headers = response.headers().clone();
                    
                    // Record response time
                    let response_time = start_time.elapsed().as_nanos() as u64;
                    stats.total_response_time_nanos.fetch_add(response_time, Ordering::Relaxed);
                    
                    // Record success/failure based on status code
                    if status.is_success() {
                        stats.successful_requests.fetch_add(1, Ordering::Relaxed);
                    } else {
                        stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                    }
                    
                    log::debug!("HTTP Response: {} {}", status.as_u16(), status.canonical_reason().unwrap_or(""));
                    yield Ok(HttpChunk::Head(status, headers));

                    let mut byte_stream = response.bytes_stream();
                    while let Some(item) = byte_stream.next().await {
                        match item {
                            Ok(bytes) => {
                                // Record bytes received
                                stats.total_bytes_received.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                                yield Ok(HttpChunk::Body(bytes));
                            },
                            Err(e) => {
                                yield Err(HttpError::from(e));
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    // Record failed request and response time
                    let response_time = start_time.elapsed().as_nanos() as u64;
                    stats.total_response_time_nanos.fetch_add(response_time, Ordering::Relaxed);
                    stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                    yield Err(HttpError::from(e));
                }
            }
        };

        HttpStream::new(Box::pin(response_stream))
    }

    /// Executes a download request and returns a stream of `DownloadChunk`s.
    ///
    /// Optimized for file downloads with progress tracking. Each chunk includes
    /// download progress information including bytes downloaded and total size.
    pub fn download_file(&self, request: HttpRequest) -> DownloadStream {
        use std::sync::atomic::Ordering;
        use std::time::Instant;
        
        // Record download request start
        let start_time = Instant::now();
        self.stats.request_count.fetch_add(1, Ordering::Relaxed);
        
        // Record request body size if present
        if let Some(body) = request.body() {
            self.stats.total_bytes_sent.fetch_add(body.len() as u64, Ordering::Relaxed);
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
                // Record failed request
                self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                let error_stream = stream! { yield Err(HttpError::from(e)); };
                return DownloadStream::new(Box::pin(error_stream));
            }
        };

        let stats = self.stats.clone();
        let download_stream = stream! {
            match client.execute(reqwest_request).await {
                Ok(response) => {
                    let status = response.status();
                    let total_size = response.content_length();
                    let mut bytes_downloaded = 0;
                    let mut chunk_number = 0;
                    
                    // Record response time and success/failure
                    let response_time = start_time.elapsed().as_nanos() as u64;
                    stats.total_response_time_nanos.fetch_add(response_time, Ordering::Relaxed);
                    
                    if status.is_success() {
                        stats.successful_requests.fetch_add(1, Ordering::Relaxed);
                    } else {
                        stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                    }

                    let mut byte_stream = response.bytes_stream();
                    while let Some(item) = byte_stream.next().await {
                        match item {
                            Ok(bytes) => {
                                bytes_downloaded += bytes.len() as u64;
                                chunk_number += 1;
                                
                                // Record bytes received for downloads
                                stats.total_bytes_received.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                                
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
                    // Record failed request and response time for downloads
                    let response_time = start_time.elapsed().as_nanos() as u64;
                    stats.total_response_time_nanos.fetch_add(response_time, Ordering::Relaxed);
                    stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                    yield Err(HttpError::from(e));
                }
            }
        };

        DownloadStream::new(Box::pin(download_stream))
    }
}
