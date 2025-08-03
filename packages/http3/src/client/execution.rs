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
    ///
    /// Optimized for file downloads with progress tracking. Each chunk includes
    /// download progress information including bytes downloaded and total size.
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
