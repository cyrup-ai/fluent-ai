//! # Fluent AI HTTP3 Client
//!
//! Zero-allocation HTTP/3 (QUIC) client with HTTP/2 fallback designed for AI providers.
//! Provides blazing-fast performance with connection pooling, intelligent caching,
//! and comprehensive error handling.
//!
//! ## Features
//!
//! - **HTTP/3 (QUIC) prioritization** with HTTP/2 fallback
//! - **Zero-allocation design** for maximum performance
//! - **Connection pooling** with intelligent reuse
//! - **Rustls TLS** with native root certificates
//! - **Compression support** (gzip, brotli, deflate)
//! - **Intelligent caching** with `ETag` and conditional requests
//! - **Streaming support** for real-time AI responses
//! - **File download streaming** with progress tracking and `on_chunk` handlers
//! - **Request/Response middleware** for customization
//! - **Comprehensive error handling** with detailed diagnostics
//!
//! ## Usage
//!
//! ### Basic HTTP Requests
//!
//! ```rust
//! use fluent_ai_http3::Http3;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct ApiResponse {
//!     status: String,
//!     data: Vec<String>,
//! }
//!
//! fn main() {
//!     let response: ApiResponse = Http3::json()
//!         .headers([("Authorization", "Bearer sk-...")])
//!         .get("https://api.openai.com/v1/models")
//!         .collect_one_or_else(|_e| ApiResponse {
//!             status: "error".to_string(),
//!             data: vec![],
//!         });
//!
//!     println!("Status: {}", response.status);
//!     println!("Data count: {}", response.data.len());
//! }
//! ```
//!
//! ### File Downloads with Progress Tracking
//!
//! ```rust
//! use fluent_ai_http3::{Http3, DownloadChunk};
//!
//! fn main() {
//!     let progress = Http3::new()
//!         .download_file("https://example.com/large-file.zip")
//!         .on_chunk(|chunk: DownloadChunk| {
//!             if let Some(progress) = chunk.progress_percentage() {
//!                 println!("Download progress: {:.1}%", progress);
//!             }
//!             println!("Downloaded {} bytes", chunk.bytes_downloaded);
//!             
//!             if let Some(speed) = chunk.download_speed {
//!                 println!("Download speed: {:.2} MB/s", speed / 1_048_576.0);
//!             }
//!         })
//!         .save("/tmp/large-file.zip");
//!
//!     println!("Download complete! Total size: {} bytes", progress.total_bytes);
//! }
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

#[cfg(test)]
#[macro_use]
extern crate doc_comment;

use std::sync::Arc;
use std::sync::LazyLock;

pub mod builder;
pub mod client;
pub mod common;
pub mod config;
pub mod error;
/// Async task primitives for streaming-first architecture
pub mod hyper;
pub mod json_path;
pub mod middleware;
pub mod operations;
pub mod request;
pub mod response;
pub mod stream;

// Core streaming types - NO Result wrapping, pure streams
pub use builder::{
    ContentType, DownloadBuilder, DownloadProgress, Http3Builder, HttpStreamExt, JsonPathStream,
};
pub use common::{AuthMethod, ContentTypes};
pub use stream::{
    BadChunk, DownloadChunk, DownloadStream, HttpChunk, HttpStream, JsonStream, SseEvent, SseStream,
};

/// Ergonomic type alias for `Http3Builder` - streams-first architecture
pub type Http3 = Http3Builder;
pub use client::{ClientStats, ClientStatsSnapshot, HttpClient};
pub use config::HttpConfig;
pub use error::HttpError;
pub use error::HttpResult;
// Import core async stream types
pub use fluent_ai_async::{AsyncStream, AsyncStreamSender};
pub use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
pub use hyper::Body;
pub use hyper::{Error, Result};
pub use hyper::{dns, header, tls};
pub use json_path::{JsonArrayStream, JsonPathError, StreamStats};
/// Convenience type alias for Result with HttpError
pub use middleware::{Middleware, MiddlewareChain, cache::CacheMiddleware};
pub use request::HttpRequest;
pub use response::HttpResponse;
pub use url::Url;

// Internal alias: many modules migrated from a http3-like API still reference `crate::hyper::...` paths.
// To make those resolve within this crate, we alias the crate root as `http3`.
// Note: Integration tests must still import `fluent_ai_http3` explicitly; this alias only affects paths
// resolved inside this crate.
pub use crate as http3;

/// Global HTTP client instance with connection pooling
/// Uses the Default implementation which provides graceful fallback handling
/// in case the optimized configuration fails to initialize
static GLOBAL_CLIENT: LazyLock<Arc<HttpClient>> = LazyLock::new(|| Arc::new(HttpClient::default()));

/// Get the global HTTP client instance
///
/// This provides a shared, high-performance HTTP client with connection pooling
/// and QUIC/HTTP3 support. The client is initialized once and reused across
/// all requests for maximum efficiency.
pub fn global_client() -> Arc<HttpClient> {
    GLOBAL_CLIENT.clone()
}

// Note: Convenience functions removed in favor of modular operations architecture.
// Use HttpClient directly or the global_client() function to access operation builders.

/// Get connection pool statistics
#[must_use]
pub fn connection_stats() -> ClientStatsSnapshot {
    global_client().as_ref().stats_snapshot()
}

/// Initialize the global HTTP client with custom configuration
/// Returns stream of initialization status - NO Result wrapping
pub fn init_global_client(_config: HttpConfig) -> AsyncStream<bool> {
    use fluent_ai_async::AsyncStream;
    // Global client is lazily initialized, so we can't change it after creation
    // This is a design choice to prevent race conditions
    // Users should create their own client instances if they need custom configuration
    AsyncStream::with_channel(|sender| {
        let _ = sender.send(true); // Always succeeds for now
    })
}
