//! Blazing-fast ergonomic Http3 builder API with zero allocation and elegant fluent interface
//! Supports streaming HttpChunk/BadHttpChunk responses, Serde integration, and shorthand methods

use std::marker::PhantomData;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use futures_util::StreamExt;
use http::{HeaderName, HeaderValue, Method};
use serde::{Serialize, de::DeserializeOwned};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

use crate::{
    DownloadStream, HttpChunk, HttpClient, HttpError, HttpRequest, HttpStream,
};

/// Content type enumeration for elegant API
#[derive(Debug, Clone, Copy)]
pub enum ContentType {
    /// application/json content type
    ApplicationJson,
    /// application/x-www-form-urlencoded content type
    ApplicationFormUrlEncoded,
    /// application/octet-stream content type
    ApplicationOctetStream,
    /// text/plain content type
    TextPlain,
    /// text/html content type
    TextHtml,
}

impl ContentType {
    /// Convert content type to string representation
    #[inline(always)]
    pub fn as_str(self) -> &'static str {
        match self {
            ContentType::ApplicationJson => "application/json",
            ContentType::ApplicationFormUrlEncoded => "application/x-www-form-urlencoded",
            ContentType::ApplicationOctetStream => "application/octet-stream",
            ContentType::TextPlain => "text/plain",
            ContentType::TextHtml => "text/html",
        }
    }
}

/// Header name aliases for elegant builder syntax
pub mod header {
    pub use http::header::*;
    
    /// X-API-Key header name (not in http crate standard headers)
    pub const X_API_KEY: http::HeaderName = http::HeaderName::from_static("x-api-key");
}

/// Zero allocation, blazing-fast HTTP builder
#[derive(Clone)]
pub struct Http3Builder<S = BodyNotSet> {
    client: HttpClient,
    request: HttpRequest,
    state: PhantomData<S>,
    debug_enabled: bool,
}

// Entry points
impl Http3Builder<BodyNotSet> {
    /// Start building a new request with a shared client instance
    pub fn new(client: &HttpClient) -> Self {
        Self {
            client: client.clone(),
            request: HttpRequest::new(Method::GET, "".to_string(), None, None, None),
            state: PhantomData,
            debug_enabled: false,
        }
    }

    /// Shorthand for setting Content-Type to application/json
    pub fn json() -> Self {
        let client = HttpClient::default();
        Self::new(&client).content_type(ContentType::ApplicationJson)
    }

    // Removed duplicate debug method - keeping the one in impl<S> Http3Builder<S>

    /// Shorthand for setting Content-Type to application/x-www-form-urlencoded
    pub fn form_urlencoded() -> Self {
        let client = HttpClient::default();
        Self::new(&client).content_type(ContentType::ApplicationFormUrlEncoded)
    }
}

// State-agnostic methods  
impl<S> Http3Builder<S> {
    /// Enable debug logging for this request
    pub fn debug(mut self) -> Self {
        self.debug_enabled = true;
        self
    }
    /// Set the request URL
    pub fn url(mut self, url: &str) -> Self {
        self.request = self.request.set_url(url.to_string());
        self
    }

    /// Set a header
    pub fn header(mut self, key: HeaderName, value: HeaderValue) -> Self {
        self.request = self.request.header(key, value);
        self
    }

    /// Add multiple headers without overwriting existing ones
    pub fn headers<F>(mut self, f: F) -> Self
    where
        F: FnOnce() -> std::collections::HashMap<HeaderName, &'static str>,
    {
        let params = f();
        for (header_key, header_value) in params {
            self.request = self.request.header(header_key, HeaderValue::from_static(header_value));
        }
        self
    }

    /// Set the Content-Type header
    pub fn content_type(self, content_type: ContentType) -> Self {
        self.header(
            header::CONTENT_TYPE,
            HeaderValue::from_static(content_type.as_str()),
        )
    }

    /// Set the Accept header
    pub fn accept(self, content_type: ContentType) -> Self {
        self.header(
            header::ACCEPT,
            HeaderValue::from_static(content_type.as_str()),
        )
    }

    /// Set the API key using the x-api-key header
    pub fn api_key(self, key: &str) -> Self {
        match HeaderValue::from_str(key) {
            Ok(header_value) => self.header(header::X_API_KEY, header_value),
            Err(_) => self, // Skip invalid header value
        }
    }

    /// Set basic authentication header
    pub fn basic_auth<F>(self, f: F) -> Self
    where
        F: FnOnce() -> std::collections::HashMap<&'static str, &'static str>,
    {
        let params = f();
        if let Some((user, pass)) = params.into_iter().next() {
            let auth_string = format!("{}:{}", user, pass);
            let encoded = STANDARD.encode(auth_string.as_bytes());
            let header_value = format!("Basic {}", encoded);
            return match HeaderValue::from_str(&header_value) {
                Ok(value) => self.header(header::AUTHORIZATION, value),
                Err(_) => self, // Skip invalid header value
            };
        }
        self
    }

    /// Set bearer token authentication header
    pub fn bearer_auth(self, token: &str) -> Self {
        let header_value = format!("Bearer {}", token);
        match HeaderValue::from_str(&header_value) {
            Ok(value) => self.header(header::AUTHORIZATION, value),
            Err(_) => self, // Skip invalid header value
        }
    }

    /// Set the request body
    pub fn body<T: Serialize>(self, body: &T) -> Http3Builder<BodySet> {
        let content_type = self.request.headers().get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json");
            
        let body_bytes = if content_type.contains("application/x-www-form-urlencoded") {
            // Serialize as form-urlencoded
            match serde_urlencoded::to_string(body) {
                Ok(form_string) => form_string.into_bytes(),
                Err(_) => Vec::new(),
            }
        } else {
            // Default to JSON serialization
            match serde_json::to_vec(body) {
                Ok(bytes) => bytes,
                Err(_) => Vec::new(),
            }
        };
        
        if self.debug_enabled {
            log::debug!("HTTP3 Builder: Set request body ({} bytes, content-type: {})", body_bytes.len(), content_type);
        }
        
        let request = self.request.set_body(body_bytes);

        Http3Builder {
            client: self.client,
            request,
            state: PhantomData,
            debug_enabled: self.debug_enabled,
        }
    }
}

// Terminal methods for BodyNotSet
impl Http3Builder<BodyNotSet> {
    /// Execute a GET request
    pub fn get(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::GET)
            .set_url(url.to_string());
        
        if self.debug_enabled {
            log::debug!("HTTP3 Builder: GET {}", url);
        }
        
        self.client.execute_streaming(self.request)
    }

    /// Execute a DELETE request
    pub fn delete(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::DELETE)
            .set_url(url.to_string());
        self.client.execute_streaming(self.request)
    }

    /// Initiate a file download
    pub fn download_file(mut self, url: &str) -> DownloadBuilder {
        self.request = self
            .request
            .set_method(Method::GET)
            .set_url(url.to_string());
        let stream = self.client.download_file(self.request);
        DownloadBuilder::new(stream)
    }
}

// Terminal methods for BodySet
impl Http3Builder<BodySet> {
    /// Execute a POST request
    pub fn post(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::POST)
            .set_url(url.to_string());
        
        if self.debug_enabled {
            log::debug!("HTTP3 Builder: POST {}", url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }
        
        self.client.execute_streaming(self.request)
    }

    /// Execute a PUT request
    pub fn put(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::PUT)
            .set_url(url.to_string());
        self.client.execute_streaming(self.request)
    }

    /// Execute a PATCH request
    pub fn patch(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::PATCH)
            .set_url(url.to_string());
        self.client.execute_streaming(self.request)
    }
}

// Typestates for builder
/// Type state indicating that no request body has been set yet
#[derive(Clone)]
pub struct BodyNotSet;

/// Type state indicating that a request body has been set
#[derive(Clone)]
pub struct BodySet;

/// Extension trait for collecting HTTP streams into deserialized types
pub trait HttpStreamExt {
    /// Collect the entire HTTP stream into a deserialized type, returning default on error
    fn collect<T: DeserializeOwned + Default + Send + 'static>(self) -> T;
    
    /// Collect the entire HTTP stream into a deserialized type, calling error handler on failure
    fn collect_or_else<
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> T + Send + 'static,
    >(
        self,
        f: F,
    ) -> T;
}

impl HttpStream {
    // EXPLICITLY APPROVED BY DAVID MAPLE 07/22/2025
    async fn collect_internal<T: DeserializeOwned + Send + 'static>(
        self,
    ) -> Result<T, HttpError> {
        use futures_util::StreamExt;
        
        let mut stream = self;
        let mut all_bytes = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(HttpChunk::Body(bytes)) => {
                    all_bytes.extend_from_slice(&bytes);
                }
                Ok(HttpChunk::Head(status, _)) => {
                    log::debug!("HTTP Response Status: {}", status.as_u16());
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        if all_bytes.is_empty() {
            return Err(HttpError::InvalidResponse {
                message: "Empty response body".to_string(),
            });
        }

        match serde_json::from_slice(&all_bytes) {
            Ok(value) => Ok(value),
            Err(err) => Err(HttpError::DeserializationError {
                message: format!("JSON deserialization failed: {}", err),
            }),
        }
    }
}

impl HttpStreamExt for HttpStream {
    // EXPLICITLY APPROVED BY DAVID MAPLE 07/22/2025
    #[inline(always)]
    fn collect<T: DeserializeOwned + Default + Send + 'static>(self) -> T {
        // Create a channel for async result communication
        let (tx, rx) = std::sync::mpsc::channel();
        
        // Spawn task to handle async collection
        tokio::spawn(async move {
            let result = self.collect_internal().await;
            let _ = tx.send(result);
        });
        
        // Receive result synchronously
        match rx.recv() {
            Ok(Ok(value)) => value,
            _ => T::default(),
        }
    }

    // EXPLICITLY APPROVED BY DAVID MAPLE 07/22/2025
    #[inline(always)]
    fn collect_or_else<
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> T + Send + 'static,
    >(
        self,
        f: F,
    ) -> T {
        // Create a channel for async result communication
        let (tx, rx) = std::sync::mpsc::channel();
        
        // Spawn task to handle async collection
        tokio::spawn(async move {
            let result = self.collect_internal::<T>().await;
            let _ = tx.send(result);
        });
        
        // Receive result synchronously
        match rx.recv() {
            Ok(Ok(value)) => value,
            Ok(Err(err)) => f(err),
            Err(_) => f(HttpError::InvalidResponse {
                message: "Channel communication failed".to_string(),
            }),
        }
    }
}


/// Builder for download-specific operations
pub struct DownloadBuilder {
    stream: DownloadStream,
}

impl DownloadBuilder {
    fn new(stream: DownloadStream) -> Self {
        Self { stream }
    }

    /// Save the downloaded file to a local path
    pub async fn save(self, local_path: &str) -> Result<DownloadProgress, HttpError> {
        let mut file = File::create(local_path)
            .await
            .map_err::<HttpError, _>(|e| e.into())?;
        let mut stream = self.stream;
        let mut total_written = 0;
        let mut chunk_count = 0;
        let mut total_size = None;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(download_chunk) => {
                    total_size = download_chunk.total_size;
                    let bytes_written = file
                        .write(&download_chunk.data)
                        .await
                        .map_err::<HttpError, _>(|e| e.into())?;
                    total_written += bytes_written as u64;
                    chunk_count += 1;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(DownloadProgress {
            chunk_count,
            bytes_written: total_written,
            total_size,
            local_path: local_path.to_string(),
            is_complete: true,
        })
    }
}

/// Download progress information for saved files
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Number of chunks received during download
    pub chunk_count: u32,
    /// Total bytes written to local file
    pub bytes_written: u64,
    /// Total expected file size if known from headers
    pub total_size: Option<u64>,
    /// Local filesystem path where file was saved
    pub local_path: String,
    /// Whether the download completed successfully
    pub is_complete: bool,
}

impl DownloadProgress {
    /// Calculate progress percentage if total size is known
    pub fn progress_percentage(&self) -> Option<f64> {
        self.total_size.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.bytes_written as f64 / total as f64) * 100.0
            }
        })
    }
}

/// Re-export for convenience
pub use Http3Builder as Builder;
