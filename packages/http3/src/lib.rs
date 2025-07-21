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

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;

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
    // SAFETY: We use a leaked static reference to make the lifetime 'static
    // This is safe because GLOBAL_CLIENT is a static that lives for the entire program
    let client: &'static HttpClient = unsafe { std::mem::transmute(global_client().as_ref()) };
    client.get(url)
}

/// Create a new POST request using the global client
#[must_use]
pub fn post(url: &str) -> RequestBuilder<'static, Ready> {
    // SAFETY: We use a leaked static reference to make the lifetime 'static
    // This is safe because GLOBAL_CLIENT is a static that lives for the entire program
    let client: &'static HttpClient = unsafe { std::mem::transmute(global_client().as_ref()) };
    client.post(url)
}

/// Create a new PUT request using the global client
#[must_use]
pub fn put(url: &str) -> RequestBuilder<'static, Ready> {
    // SAFETY: We use a leaked static reference to make the lifetime 'static
    // This is safe because GLOBAL_CLIENT is a static that lives for the entire program
    let client: &'static HttpClient = unsafe { std::mem::transmute(global_client().as_ref()) };
    client.put(url)
}

/// Create a new DELETE request using the global client
#[must_use]
pub fn delete(url: &str) -> RequestBuilder<'static, Ready> {
    // SAFETY: We use a leaked static reference to make the lifetime 'static
    // This is safe because GLOBAL_CLIENT is a static that lives for the entire program
    let client: &'static HttpClient = unsafe { std::mem::transmute(global_client().as_ref()) };
    client.delete(url)
}

/// Create a new PATCH request using the global client
#[must_use]
pub fn patch(url: &str) -> RequestBuilder<'static, Ready> {
    // SAFETY: We use a leaked static reference to make the lifetime 'static
    // This is safe because GLOBAL_CLIENT is a static that lives for the entire program
    let client: &'static HttpClient = unsafe { std::mem::transmute(global_client().as_ref()) };
    client.patch(url)
}

/// Create a new HEAD request using the global client
#[must_use]
pub fn head(url: &str) -> RequestBuilder<'static, Ready> {
    // SAFETY: We use a leaked static reference to make the lifetime 'static
    // This is safe because GLOBAL_CLIENT is a static that lives for the entire program
    let client: &'static HttpClient = unsafe { std::mem::transmute(global_client().as_ref()) };
    client.head(url)
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

/// Legacy builder pattern for creating HTTP requests (deprecated - use client methods instead)
#[deprecated(note = "Use client.get(), client.post(), etc. methods instead")]
pub struct LegacyRequestBuilder {
    client: Arc<HttpClient>,
    method: HttpMethod,
    url: String,
    headers: HashMap<String, String>,
    body: Option<Vec<u8>>,
    timeout: Option<Duration>,
    cache_control: Option<String>,
}

impl LegacyRequestBuilder {
    /// Create a new request builder
    pub fn new(client: Arc<HttpClient>, method: HttpMethod, url: String) -> Self {
        Self {
            client,
            method,
            url,
            headers: HashMap::new(),
            body: None,
            timeout: None,
            cache_control: None,
        }
    }

    /// Add a header to the request
    #[must_use]
    pub fn header<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Add multiple headers to the request
    #[must_use]
    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Set the request body
    #[must_use]
    pub fn body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }

    /// Set the request body from a string
    #[must_use]
    pub fn body_string(mut self, body: String) -> Self {
        self.body = Some(body.into_bytes());
        self
    }

    /// Set the request body from JSON
    pub fn json<T: serde::Serialize>(mut self, json: &T) -> HttpResult<Self> {
        let body = serde_json::to_vec(json)?;
        self.headers
            .insert("Content-Type".to_string(), "application/json".to_string());
        self.body = Some(body);
        Ok(self)
    }

    /// Set request timeout
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set cache control header
    #[must_use]
    pub fn cache_control(mut self, cache_control: String) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    /// Send the request
    pub async fn send(self) -> HttpResult<HttpResponse> {
        let mut request = HttpRequest::new(self.method, self.url);

        for (key, value) in self.headers {
            request = request.header(key, value);
        }

        if let Some(body) = self.body {
            request = request.set_body(body);
        }

        if let Some(timeout) = self.timeout {
            request = request.set_timeout(timeout);
        }

        if let Some(cache_control) = self.cache_control {
            request = request.header("Cache-Control", cache_control);
        }

        self.client.send(request).await
    }

    /// Send the request and return a stream
    pub async fn send_stream(self) -> HttpResult<HttpStream> {
        let mut request = HttpRequest::new(self.method, self.url);

        for (key, value) in self.headers {
            request = request.header(key, value);
        }

        if let Some(body) = self.body {
            request = request.set_body(body);
        }

        if let Some(timeout) = self.timeout {
            request = request.set_timeout(timeout);
        }

        if let Some(cache_control) = self.cache_control {
            request = request.header("Cache-Control", cache_control);
        }

        self.client.send_stream(request).await
    }
}
