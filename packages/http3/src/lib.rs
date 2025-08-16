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
//! - **File download streaming** with progress tracking and `on_chunk` Result handlers
//! - **Request/Response middleware** for customization
//! - **Comprehensive error handling** with detailed diagnostics
//!
//! ## Usage

// Removed orphan rule violation - MessageChunk implementations moved to appropriate modules
// ### Basic HTTP Requests
//
// ```rust
// use fluent_ai_http3::Http3;
// use serde::Deserialize;
//
// #[derive(Deserialize)]
// struct ApiResponse {
//     status: String,
//     data: Vec<String>,
// }
//
// fn main() {
//     let response: ApiResponse = Http3::json()
//         .headers([("Authorization", "Bearer sk-...")])
//         .get("https://api.openai.com/v1/models")
//         .collect_one_or_else(|_e| ApiResponse {
//             status: "error".to_string(),
//             data: vec![],
//         });
//
//     println!("Status: {}", response.status);
//     println!("Data count: {}", response.data.len());
// }
// ```
//
// ### File Downloads with Progress Tracking
//
// ```rust
// use fluent_ai_http3::{Http3, DownloadChunk};
//
// fn main() {
//     let progress = Http3::new()
//         .download_file("https://example.com/large-file.zip")
//         .on_chunk(|result| match result {
//             Ok(chunk) => {
//                 if let Some(progress) = chunk.progress_percentage() {
//                     println!("Download progress: {:.1}%", progress);
//                 }
//                 println!("Downloaded {} bytes", chunk.bytes_downloaded);
//                 chunk
//             },
//             Err(error) => {
//                 println!("Download error: {}", error);
//                 DownloadChunk::bad_chunk(error.to_string())
//             }
//         })
//         .save("/tmp/large-file.zip");
//
//     println!("Download complete! Total size: {} bytes", progress.total_bytes);
// }
// ```

// Crate-level attributes moved to appropriate location
// Unsafe code denial temporarily disabled during migration
// #[deny(unsafe_code)]
#[warn(missing_docs)]
#[warn(clippy::all)]
#[warn(clippy::pedantic)]
#[allow(clippy::missing_errors_doc)]
#[allow(clippy::missing_panics_doc)]
#[cfg(test)]
#[macro_use]
extern crate doc_comment;

use std::sync::Arc;
use std::sync::OnceLock;

// Re-export core error types for crate-wide usage
pub use crate::error::{HttpError, HttpResult};

/// Crate-level Error type alias for compatibility
pub type Error = HttpError;

/// Crate-level Result type alias for compatibility  
pub type Result<T> = HttpResult<T>;

// Import MessageChunk for trait implementations

pub mod builder;
pub mod client;
pub mod common;
pub mod config;
pub mod error;
/// Async task primitives for streaming-first architecture
pub mod hyper;
pub mod json_path;
pub mod middleware;
pub mod util;
// mod bytes_impl; // Removed due to orphan rule violations - use BytesWrapper instead
pub mod operations;
pub mod request;
pub mod response;
pub mod stream;
pub mod wrappers;

// Core streaming types - NO Result wrapping, pure streams
pub use builder::{
    ContentType, Http3Builder, JsonPathStreaming,
};
pub use common::{AuthMethod, ContentTypes};
pub use response::HttpResponseChunk;
pub use stream::{
    BadChunk, DownloadChunk, DownloadStream, HttpChunk, HttpStream, JsonStream, SseEvent, SseStream,
};
pub use wrappers::{BoxBodyWrapper, BytesWrapper, FrameWrapper, ResultWrapper};

/// Ergonomic type alias for `Http3Builder` - streams-first architecture
pub type Http3 = Http3Builder;
pub use client::{ClientStats, ClientStatsSnapshot, HttpClient};
pub use config::HttpConfig;
// Import core async stream types
pub use fluent_ai_async::{AsyncStream, AsyncStreamSender};
pub use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
pub use hyper::Body;
// Removed hyper Result types - using fluent_ai_async only
pub use hyper::{dns, tls};
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
static GLOBAL_CLIENT: OnceLock<Arc<HttpClient>> = OnceLock::new();

/// Get the global HTTP client instance
/// This provides a shared, high-performance HTTP client with connection pooling
/// and QUIC/HTTP3 support. The client is initialized once and reused across
/// all requests for maximum efficiency.
pub fn global_client() -> Arc<HttpClient> {
    GLOBAL_CLIENT
        .get_or_init(|| Arc::new(HttpClient::default()))
        .clone()
}

// Note: Convenience functions removed in favor of modular operations architecture.
// Use HttpClient directly or the global_client() function to access operation builders.

/// Get connection pool statistics
#[must_use]
pub fn connection_stats() -> ClientStatsSnapshot {
    global_client().as_ref().stats_snapshot()
}

/// Initialize the global HTTP client with custom configuration
/// Returns Result following standard Rust patterns for initialization
pub fn init_global_client(
    config: HttpConfig,
) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    validate_and_init_client(config)
}

/// Internal validation and initialization
fn validate_and_init_client(
    config: HttpConfig,
) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Validate the provided configuration
    validate_http_config(&config)?;

    // Initialize global client with the provided configuration
    initialize_global_client_internal(config)
}

/// Validate HTTP configuration before initialization
fn validate_http_config(config: &HttpConfig) -> std::result::Result<(), String> {
    // Validate timeout values
    if config.timeout.as_secs() == 0 {
        return Err("Timeout must be greater than zero".to_string());
    }
    if config.timeout.as_secs() > 3600 {
        return Err("Timeout must not exceed 1 hour".to_string());
    }

    // Validate connection timeout
    if config.connect_timeout.as_secs() == 0 {
        return Err("Connect timeout must be greater than zero".to_string());
    }
    if config.connect_timeout.as_secs() > 300 {
        return Err("Connect timeout must not exceed 5 minutes".to_string());
    }

    // Validate pool configuration
    if config.pool_max_idle_per_host == 0 {
        return Err("Pool max idle per host must be greater than zero".to_string());
    }
    if config.pool_max_idle_per_host > 1000 {
        return Err("Pool max idle per host must not exceed 1000".to_string());
    }

    // Validate user agent
    if config.user_agent.is_empty() {
        return Err("User agent cannot be empty".to_string());
    }
    if config.user_agent.len() > 1000 {
        return Err("User agent must not exceed 1000 characters".to_string());
    }

    Ok(())
}

/// Internal function to perform the actual global client initialization
fn initialize_global_client_internal(
    config: HttpConfig,
) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use crate::hyper::ClientBuilder;

    // Build a new client with the provided configuration
    let mut client_builder = ClientBuilder::new();

    // Apply configuration settings
    client_builder = client_builder.timeout(config.timeout);
    client_builder = client_builder.connect_timeout(config.connect_timeout);
    client_builder = client_builder.user_agent(&config.user_agent);

    // Configure TLS if specified
    #[cfg(feature = "__tls")]
    {
        // TLS configuration handled by default settings
    }

    // Build the client
    let hyper_client = client_builder.build()?;
    let stats = crate::client::ClientStats::default();
    let new_client = crate::client::HttpClient::new(hyper_client, config, stats);

    // Replace the global client using atomic operations for thread safety
    // This is a one-time initialization operation with proper error handling
    // Set the global client - OnceLock allows one-time initialization
    GLOBAL_CLIENT
        .set(Arc::new(new_client))
        .map_err(|_| "Global client already initialized")?;

    Ok(())
}
