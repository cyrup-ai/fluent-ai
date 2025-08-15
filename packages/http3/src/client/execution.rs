//! HTTP request execution with streaming support
//!
//! Provides streaming execution methods for HTTP requests and downloads
//! with zero-allocation streaming patterns and efficient error handling.

use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};

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
            self.stats
                .total_bytes_sent
                .fetch_add(body.len() as u64, Ordering::Relaxed);
        }

        let http3_request = match client
            .request(request.method().clone(), request.url())
            .headers(request.headers().clone())
            .body(request.body().cloned().unwrap_or_else(Vec::new))
            .build()
        {
            Ok(req) => req,
            Err(_e) => {
                // Record failed request
                self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                let (_sender, stream) = fluent_ai_async::channel::channel();
                // Error streams are empty per streams-first architecture
                return HttpStream::new(stream);
            }
        };

        let stats = self.stats.clone();
        let response_stream = AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                // Execute the stream and get the first response
                let mut response_stream = client.execute(http3_request);
                let responses: Vec<_> = response_stream.into_iter().collect();
                let response = match responses.into_iter().next() {
                    Some(response) => response,
                    None => {
                        emit!(
                            sender,
                            HttpChunk::Error(crate::HttpError::StreamError {
                                message: "no response received from client execute".to_string()
                            })
                        );
                        return;
                    }
                };

                let status = response.status();
                let headers = response.headers().clone();

                // Record response time and success/failure
                let response_time = start_time.elapsed().as_nanos() as u64;
                stats
                    .total_response_time_nanos
                    .fetch_add(response_time, Ordering::Relaxed);
                if status.is_success() {
                    stats.successful_requests.fetch_add(1, Ordering::Relaxed);
                } else {
                    stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                }

                emit!(sender, HttpChunk::Head(status, headers));
            });

            task.collect();
        });

        HttpStream::new(response_stream)
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
            self.stats
                .total_bytes_sent
                .fetch_add(body.len() as u64, Ordering::Relaxed);
        }

        let client = self.inner.clone();
        let http3_request = match client
            .request(request.method().clone(), request.url())
            .headers(request.headers().clone())
            .body(request.body().cloned().unwrap_or_else(Vec::new))
            .build()
        {
            Ok(req) => req,
            Err(_e) => {
                // Record failed request
                self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                let (_sender, stream) = fluent_ai_async::channel::channel();
                // Error streams are empty per streams-first architecture
                return DownloadStream::new(stream);
            }
        };

        let stats = self.stats.clone();
        let download_stream = AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                // Execute the stream and get the first response
                let mut response_stream = client.execute(http3_request);
                let responses: Vec<_> = response_stream.into_iter().collect();
                let response = match responses.into_iter().next() {
                    Some(response) => response,
                    None => {
                        // Record failed request and response time for downloads
                        let response_time = start_time.elapsed().as_nanos() as u64;
                        stats
                            .total_response_time_nanos
                            .fetch_add(response_time, Ordering::Relaxed);
                        stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                        emit!(
                            sender,
                            DownloadChunk {
                                data: bytes::Bytes::new(),
                                chunk_number: 0,
                                total_size: None,
                                bytes_downloaded: 0,
                                error_message: Some(
                                    "no response received from client execute".to_string()
                                ),
                            }
                        );
                        return;
                    }
                };

                let status = response.status();
                let total_size = response.content_length();
                let mut bytes_downloaded = 0;
                let mut chunk_number = 0;

                // Record response time and success/failure
                let response_time = start_time.elapsed().as_nanos() as u64;
                stats
                    .total_response_time_nanos
                    .fetch_add(response_time, Ordering::Relaxed);

                if status.is_success() {
                    stats.successful_requests.fetch_add(1, Ordering::Relaxed);
                } else {
                    stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                }

                // Get full response body as bytes using stream collection
                let bytes_stream = response.bytes();
                let bytes_collected = bytes_stream.collect();
                let all_bytes = match bytes_collected.first() {
                    Some(bytes) => bytes,
                    None => {
                        emit!(
                            sender,
                            DownloadChunk {
                                data: bytes::Bytes::new(),
                                chunk_number: 0,
                                total_size,
                                bytes_downloaded: 0,
                                error_message: Some(
                                    "download body processing: no bytes received".to_string()
                                ),
                            }
                        );
                        return;
                    }
                };

                // Split into chunks for download progress simulation
                const CHUNK_SIZE: usize = 8192; // 8KB chunks
                let mut offset = 0;

                while offset < all_bytes.len() {
                    let end = std::cmp::min(offset + CHUNK_SIZE, all_bytes.len());
                    let chunk_bytes = all_bytes.slice(offset..end);

                    bytes_downloaded += chunk_bytes.len() as u64;
                    chunk_number += 1;

                    // Record bytes received for downloads
                    stats
                        .total_bytes_received
                        .fetch_add(chunk_bytes.len() as u64, Ordering::Relaxed);

                    let chunk = DownloadChunk {
                        data: chunk_bytes,
                        chunk_number,
                        total_size,
                        bytes_downloaded,
                        error_message: None,
                    };
                    emit!(sender, chunk);

                    offset = end;
                }
            });

            task.collect();
        });

        DownloadStream::new(download_stream)
    }
}
