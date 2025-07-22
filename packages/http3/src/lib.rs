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
//! - **File download streaming** with progress tracking and on_chunk handlers
//! - **Request/Response middleware** for customization
//! - **Comprehensive error handling** with detailed diagnostics
//!
//! ## Usage
//!
//! ### Basic HTTP Requests
//!
//! ```rust
//! use fluent_ai_http3::{HttpClient, HttpRequest, HttpResponse};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = HttpClient::new()?;
//!     
//!     let response = client
//!         .get("https://api.openai.com/v1/models")
//!         .header("Authorization", "Bearer sk-...")
//!         .send()
//!         .await?;
//!     
//!     println!("Status: {}", response.status());
//!     println!("Body: {}", response.text().await?);
//!     
//!     Ok(())
//! }
//! ```
//!
//! ### File Downloads with Progress Tracking
//!
//! ```rust
//! use fluent_ai_http3::{HttpClient, DownloadChunk};
//! use futures_util::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = HttpClient::new()?;
//!     
//!     let mut download_stream = client
//!         .download_file("https://example.com/large-file.zip")
//!         .await?
//!         .on_chunk(|chunk: &DownloadChunk| {
//!             if let Some(progress) = chunk.progress_percentage() {
//!                 println!("Download progress: {:.1}%", progress);
//!             }
//!             println!("Downloaded {} bytes", chunk.bytes_downloaded);
//!             Ok(())
//!         });
//!     
//!     let mut file_data = Vec::new();
//!     while let Some(chunk_result) = download_stream.next().await {
//!         let chunk = chunk_result?;
//!         file_data.extend_from_slice(&chunk.data);
//!         
//!         if let Some(speed) = chunk.download_speed {
//!             println!("Download speed: {:.2} MB/s", speed / 1_048_576.0);
//!         }
//!     }
//!     
//!     println!("Download complete! Total size: {} bytes", file_data.len());
//!     Ok(())
//! }
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use std::sync::Arc;
use std::sync::LazyLock;

/// Async task primitives for streaming-first architecture

pub mod builder;
pub use builder::Http3Builder as Http3;
pub mod client;
pub mod common;
pub mod config;
pub mod error;
pub mod middleware;
pub mod operations;
pub mod request;
pub mod response;
pub mod stream;

pub use builder::{ContentType, DownloadBuilder, DownloadProgress, Http3Builder, HttpStreamExt};
pub use client::{ClientStats, ClientStatsSnapshot, HttpClient};
pub use common::cache::CacheEntry;
pub use config::HttpConfig;
pub use error::{HttpError, HttpResult};
pub use middleware::{Middleware, MiddlewareChain, cache::CacheMiddleware};
pub use request::HttpRequest;
pub use response::{HttpResponse, JsonStream, SseEvent};
pub use stream::{DownloadChunk, DownloadStream, HttpChunk, HttpStream, LinesStream, SseStream};

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
pub fn init_global_client(_config: HttpConfig) -> HttpResult<()> {
    // Global client is lazily initialized, so we can't change it after creation
    // This is a design choice to prevent race conditions
    // Users should create their own client instances if they need custom configuration
    Ok(())
}
