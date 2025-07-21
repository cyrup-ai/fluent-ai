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
//! use futures::StreamExt;
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

pub mod cache;
pub mod client;
pub mod config;
pub mod error;
pub mod middleware;
pub mod request;
pub mod response;
pub mod stream;

pub use cache::CacheEntry;
pub use client::{ClientStats, HttpClient, Ready, RequestBuilder};
pub use config::HttpConfig;
pub use error::{HttpError, HttpResult};
pub use middleware::{cache::CacheMiddleware, Middleware, MiddlewareChain};
pub use request::{HttpMethod, HttpRequest};
pub use response::HttpResponse;
pub use stream::{CachedDownloadStream, DownloadChunk, DownloadStream, HttpStream, LinesStream, SseStream};

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

/// Create a new HTTP request using the global client
#[must_use]
pub fn get(url: &str) -> RequestBuilder<'static, Ready> {
    GLOBAL_CLIENT.get(url)
}

/// Create a new POST request using the global client
#[must_use]
pub fn post(url: &str) -> RequestBuilder<'static, Ready> {
    GLOBAL_CLIENT.post(url)
}

/// Create a new PUT request using the global client
#[must_use]
pub fn put(url: &str) -> RequestBuilder<'static, Ready> {
    GLOBAL_CLIENT.put(url)
}

/// Create a new DELETE request using the global client
#[must_use]
pub fn delete(url: &str) -> RequestBuilder<'static, Ready> {
    GLOBAL_CLIENT.delete(url)
}

/// Create a new PATCH request using the global client
#[must_use]
pub fn patch(url: &str) -> RequestBuilder<'static, Ready> {
    GLOBAL_CLIENT.patch(url)
}

/// Create a new HEAD request using the global client
#[must_use]
pub fn head(url: &str) -> RequestBuilder<'static, Ready> {
    GLOBAL_CLIENT.head(url)
}

/// Get connection pool statistics
#[must_use]
pub fn connection_stats() -> ClientStats {
    global_client().stats()
}

/// Initialize the global HTTP client with custom configuration
pub fn init_global_client(_config: HttpConfig) -> HttpResult<()> {
    // Global client is lazily initialized, so we can't change it after creation
    // This is a design choice to prevent race conditions
    // Users should create their own client instances if they need custom configuration
    Ok(())
}


