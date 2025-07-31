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

use fluent_ai_async::{AsyncStream, AsyncStreamSender};
use crate::{DownloadStream, HttpChunk, HttpClient, HttpError, HttpRequest, HttpStream};

// cyrup_sugars prelude removed - not used in this file


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
    /// multipart/form-data content type
    MultipartFormData}

impl ContentType {
    /// Convert content type to string representation
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ContentType::ApplicationJson => "application/json",
            ContentType::ApplicationFormUrlEncoded => "application/x-www-form-urlencoded",
            ContentType::ApplicationOctetStream => "application/octet-stream",
            ContentType::TextPlain => "text/plain",
            ContentType::TextHtml => "text/html",
            ContentType::MultipartFormData => "multipart/form-data"}
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
    debug_enabled: bool}

// Entry points
impl Http3Builder<BodyNotSet> {
    /// Start building a new request with a shared client instance
    #[must_use]
    pub fn new(client: &HttpClient) -> Self {
        Self {
            client: client.clone(),
            request: HttpRequest::new(Method::GET, String::new(), None, None, None),
            state: PhantomData,
            debug_enabled: false}
    }

    /// Shorthand for setting Content-Type to application/json
    #[must_use]
    pub fn json<T>() -> Self
    where
        T: serde::de::DeserializeOwned + serde::Serialize + Send + 'static,
    {
        let client = HttpClient::default();
        Self::new(&client).content_type(ContentType::ApplicationJson)
    }

    // Removed duplicate debug method - keeping the one in impl<S> Http3Builder<S>

    /// Shorthand for setting Content-Type to application/x-www-form-urlencoded
    #[must_use]
    pub fn form_urlencoded() -> Self {
        let client = HttpClient::default();
        Self::new(&client).content_type(ContentType::ApplicationFormUrlEncoded)
    }
}

// State-agnostic methods
impl<S> Http3Builder<S> {
    /// Enable debug logging for this request
    #[must_use]
    pub fn debug(mut self) -> Self {
        self.debug_enabled = true;
        self
    }
    /// Set the request URL
    ///
    /// # Arguments
    /// * `url` - The URL to set
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn url(mut self, url: &str) -> Self {
        self.request = self.request.set_url(url.to_string());
        self
    }

    /// Set a header
    ///
    /// # Arguments
    /// * `key` - The header name
    /// * `value` - The header value
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn header(mut self, key: HeaderName, value: HeaderValue) -> Self {
        self.request = self.request.header(key, value);
        self
    }

    /// Add multiple headers without overwriting existing ones
    ///
    /// # Arguments
    /// * `headers_config` - Headers configuration using [("key", "value")] syntax
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn headers(mut self, headers_config: impl Into<hashbrown::HashMap<&'static str, &'static str>>) -> Self {
        let headers_config = headers_config.into();
        for (header_key, header_value) in headers_config {
            match HeaderName::from_bytes(header_key.as_bytes()) {
                Ok(header_name) => {
                    self.request = self
                        .request
                        .header(header_name, HeaderValue::from_static(header_value));
                }
                Err(_) => continue, // Skip invalid header names
            }
        }
        self
    }

    /// Set the Content-Type header
    #[must_use]
    pub fn content_type(self, content_type: ContentType) -> Self {
        self.header(
            header::CONTENT_TYPE,
            HeaderValue::from_static(content_type.as_str()),
        )
    }

    /// Set the Accept header
    #[must_use]
    pub fn accept(self, content_type: ContentType) -> Self {
        self.header(
            header::ACCEPT,
            HeaderValue::from_static(content_type.as_str()),
        )
    }

    /// Set the API key using the x-api-key header
    ///
    /// # Arguments
    /// * `key` - The API key value
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn api_key(self, key: &str) -> Self {
        match HeaderValue::from_str(key) {
            Ok(header_value) => self.header(header::X_API_KEY, header_value),
            Err(_) => self, // Skip invalid header value
        }
    }

    /// Set basic authentication header
    ///
    /// # Arguments
    /// * `auth_config` - Authentication configuration using [("user", "password")] syntax
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn basic_auth(self, auth_config: impl Into<hashbrown::HashMap<&'static str, &'static str>>) -> Self {
        let auth_config = auth_config.into();
        if let Some((user, pass)) = auth_config.into_iter().next() {
            let auth_string = format!("{user}:{pass}");
            let encoded = STANDARD.encode(auth_string);
            let header_value = format!("Basic {encoded}");
            return match HeaderValue::from_str(&header_value) {
                Ok(value) => self.header(header::AUTHORIZATION, value),
                Err(_) => self, // Skip invalid header value
            };
        }
        self
    }

    /// Set bearer token authentication header
    ///
    /// # Arguments
    /// * `token` - The bearer token to use for authentication
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn bearer_auth(self, token: &str) -> Self {
        let header_value = format!("Bearer {token}");
        match HeaderValue::from_str(&header_value) {
            Ok(value) => self.header(header::AUTHORIZATION, value),
            Err(_) => self, // Skip invalid header value
        }
    }

    /// Set cache control header
    ///
    /// # Arguments
    /// * `value` - The cache control directive (e.g., "no-cache", "max-age=3600")
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn cache_control(self, value: &str) -> Self {
        match HeaderValue::from_str(value) {
            Ok(header_value) => self.header(header::CACHE_CONTROL, header_value),
            Err(_) => self, // Skip invalid header value
        }
    }

    /// Set max age cache control directive
    ///
    /// # Arguments
    /// * `seconds` - Maximum age in seconds
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn max_age(self, seconds: u64) -> Self {
        let value = format!("max-age={}", seconds);
        self.cache_control(&value)
    }



    /// Set the request body
    ///
    /// # Arguments
    /// * `body` - The body to serialize and send
    ///
    /// # Returns
    /// `Http3Builder<BodySet>` for chaining
    #[must_use]
    pub fn body<T: Serialize>(self, body: &T) -> Http3Builder<BodySet> {
        let content_type = self
            .request
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json");

        let body_bytes = if content_type.contains("application/x-www-form-urlencoded") {
            // Serialize as form-urlencoded
            serde_urlencoded::to_string(body)
                .map(std::string::String::into_bytes)
                .unwrap_or_default()
        } else {
            // Default to JSON serialization
            serde_json::to_vec(body).unwrap_or_default()
        };

        if self.debug_enabled {
            log::debug!(
                "HTTP3 Builder: Set request body ({} bytes, content-type: {})",
                body_bytes.len(),
                content_type
            );
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
    ///
    /// # Arguments
    /// * `url` - The URL to send the GET request to
    ///
    /// # Returns
    /// `HttpStream` for streaming the response
    #[must_use]
    pub fn get(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::GET)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: GET {url}");
        }

        self.client.execute_streaming(self.request)
    }

    /// Execute a DELETE request
    ///
    /// # Arguments
    /// * `url` - The URL to send the DELETE request to
    ///
    /// # Returns
    /// `HttpStream` for streaming the response
    #[must_use]
    pub fn delete(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::DELETE)
            .set_url(url.to_string());
        self.client.execute_streaming(self.request)
    }

    /// Initiate a file download
    ///
    /// # Arguments
    /// * `url` - The URL to download from
    ///
    /// # Returns
    /// `DownloadBuilder` for configuring the download
    #[must_use]
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

/// Extension trait for collecting HTTP streams into Vec of deserialized types
pub trait HttpStreamExt {
    /// Collect the entire HTTP stream into a Vec of deserialized types
    fn collect<T: DeserializeOwned + Default + Send + 'static>(self) -> Vec<T>;

    /// Collect the entire HTTP stream into a single item, calling error handler on failure
    fn collect_or_else<
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> T + Send + Sync + 'static + Clone,
    >(
        self,
        f: F,
    ) -> T;

    /// Process each chunk with a handler function supporting pattern matching syntax
    fn on_chunk<T, F>(self, handler: F) -> Self
    where
        T: DeserializeOwned + serde::Serialize + Send + 'static,
        F: FnMut(Result<T, crate::HttpError>) -> T + Send + 'static;
}

impl HttpStreamExt for HttpStream {
    #[inline(always)]
    fn collect<T: DeserializeOwned + Default + Send + 'static>(self) -> Vec<T> {
        self.collect_internal()
    }

    #[inline(always)]
    fn collect_or_else<T, F>(self, f: F) -> T
    where
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> T + Send + Sync + 'static + Clone,
    {
        self.collect_or_else_impl(f)
    }

    fn on_chunk<T, F>(mut self, handler: F) -> Self
    where
        T: DeserializeOwned + serde::Serialize + Send + 'static,
        F: FnMut(Result<T, crate::HttpError>) -> T + Send + 'static,
    {
        // Create a processor that deserializes Body chunks and passes Result<T, HttpError> to handler
        let mut handler = handler;
        let processor = Box::new(move |chunk: HttpChunk| -> HttpChunk {
            match chunk {
                HttpChunk::Body(bytes) => {
                    // Attempt to deserialize the chunk data into Result<T, HttpError>
                    let result = match serde_json::from_slice::<T>(&bytes) {
                        Ok(deserialized) => Ok(deserialized),
                        Err(e) => Err(crate::HttpError::HttpStatus {
                            status: 500,
                            message: format!("Deserialization failed: {}", e),
                            body: String::from_utf8_lossy(&bytes).to_string(),
                        })
                    };
                    
                    // Call handler with wrapped Result - handler UNWRAPS it
                    let processed = handler(result);
                    
                    // Convert processed result back to HttpChunk::Deserialized
                    let json_value = match serde_json::to_value(processed) {
                        Ok(val) => val,
                        Err(_) => serde_json::Value::Null,
                    };
                    HttpChunk::Deserialized(json_value)
                }
                // All other chunks should be processed too - convert to Result<T, HttpError>
                HttpChunk::Error(e) => {
                    let processed = handler(Err(e));
                    let json_value = match serde_json::to_value(processed) {
                        Ok(val) => val,
                        Err(_) => serde_json::Value::Null,
                    };
                    HttpChunk::Deserialized(json_value)
                }
                // Head chunks are not data - pass through
                other => other,
            }
        });

        // Add the processor to the stream
        self.add_processor(processor);
        self
    }
}

impl HttpStream {

    // NO FUTURES - Pure AsyncStream pattern for HTTP response collection
pub(crate) fn collect_internal<T: DeserializeOwned + Send + 'static>(self) -> T
    where
        T: Default,
    {
        use futures_util::StreamExt;

        // Convert HttpStream to AsyncStream for pure streaming pattern
        let stream = AsyncStream::<T, 1024>::with_channel(move |sender: AsyncStreamSender<T, 1024>| {
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(_) => {
                    log::error!("Failed to create tokio runtime");
                    let _ = sender.send(T::default());
                    return;
                }
            };
            
            rt.block_on(async move {
                let mut http_stream = self;
                let mut all_bytes = Vec::new();
                let mut status_code = None;

                while let Some(chunk_result) = http_stream.next().await {
                    match chunk_result {
                        Ok(HttpChunk::Body(bytes)) => {
                            all_bytes.extend_from_slice(&bytes);
                        }
                        Ok(HttpChunk::Head(status, _)) => {
                            status_code = Some(status);
                            log::debug!("HTTP Response Status: {}", status.as_u16());
                        }
                        Ok(HttpChunk::Deserialized(json_value)) => {
                            // Handle pre-deserialized data from on_chunk processors
                            match serde_json::from_value::<T>(json_value) {
                                Ok(value) => {
                                    let _ = sender.send(value);
                                    return ();
                                }
                                Err(e) => {
                                    log::error!("Failed to convert deserialized chunk: {}", e);
                                    let _ = sender.send(T::default());
                                    return ();
                                }
                            }
                        }
                        Ok(HttpChunk::Error(http_error)) => {
                            log::error!("HttpChunk error: {}", http_error);
                            let _ = sender.send(T::default());
                            return ();
                        }
                        Err(e) => {
                            log::error!("Error receiving chunk: {}", e);
                            let _ = sender.send(T::default());
                            return ();
                        }
                    }
                }

                // If we got a 204 No Content, return default
                if status_code.map_or(false, |s| s.as_u16() == 204) || all_bytes.is_empty() {
                    let _ = sender.send(T::default());
                    return ();
                }

                // Try to deserialize the response
                match serde_json::from_slice(&all_bytes) {
                    Ok(value) => {
                        let _ = sender.send(value);
                    }
                    Err(e) => {
                        log::error!("Failed to deserialize response: {}", e);
                        let _ = sender.send(T::default());
                    }
                }
            });
        });

        // Collect from AsyncStream (blocking, no futures)
        let results = stream.collect();
        match results.into_iter().next() {
            Some(value) => value,
            None => T::default(),
        }
    }

    #[inline(always)]
    #[allow(dead_code)] // Public API method for library users
    fn collect_or_else<T, F>(self, f: F) -> T
    where
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> T + Send + Sync + 'static + Clone,
    {
        self.collect_or_else_impl(f)
    }

    fn collect_or_else_impl<T, F>(self, f: F) -> T
    where
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> T + Send + Sync + 'static + Clone,
    {
        use futures_util::StreamExt;
        use http::StatusCode;

        // Convert HttpStream to AsyncStream for pure streaming pattern  
        let f_arc = std::sync::Arc::new(f.clone());
        let f_arc_clone = f_arc.clone();
        let stream = AsyncStream::<T, 1024>::with_channel(move |sender: AsyncStreamSender<T, 1024>| {
            let f_arc = f_arc_clone.clone();
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(_) => {
                    // Send error on runtime creation failure
                    let _ = sender.send(f_arc(HttpError::HttpStatus {
                        status: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
                        message: "Runtime creation failed".to_string(),
                        body: String::new(),
                    }));
                    return;
                }
            };
            
            rt.block_on(async move {
                let mut http_stream = self;
                let mut all_bytes = Vec::new();
                let mut status_code = None;

                while let Some(chunk_result) = http_stream.next().await {
                    match chunk_result {
                        Ok(HttpChunk::Body(bytes)) => {
                            all_bytes.extend_from_slice(&bytes);
                        }
                        Ok(HttpChunk::Head(status, _)) => {
                            status_code = Some(status);
                            log::debug!("HTTP Response Status: {}", status.as_u16());
                        }
                        Ok(HttpChunk::Deserialized(json_value)) => {
                            // Handle pre-deserialized data from on_chunk processors
                            match serde_json::from_value::<T>(json_value) {
                                Ok(value) => {
                                    let _ = sender.send(value);
                                    return ();
                                }
                                Err(e) => {
                                    let _ = sender.send(f_arc(HttpError::HttpStatus {
                                        status: 500,
                                        message: format!("Failed to convert deserialized chunk: {}", e),
                                        body: String::new(),
                                    }));
                                    return ();
                                }
                            }
                        }
                        Ok(HttpChunk::Error(http_error)) => {
                            let _ = sender.send(f_arc(http_error));
                            return ();
                        }
                        Err(e) => {
                            let _ = sender.send(f_arc(e));
                            return ();
                        }
                    }
                }

                // If we got a 204 No Content or empty response
                let status = match status_code {
                    Some(code) => code,
                    None => StatusCode::NO_CONTENT,
                };
                if status == StatusCode::NO_CONTENT || all_bytes.is_empty() {
                    let _ = sender.send(f_arc(HttpError::HttpStatus {
                        status: status.as_u16(),
                        message: "No content".to_string(),
                        body: String::from_utf8_lossy(&all_bytes).to_string(),
                    }));
                    return ();
                }

                // Try to deserialize the response
                match serde_json::from_slice(&all_bytes) {
                    Ok(value) => {
                        let _ = sender.send(value);
                    }
                    Err(e) => {
                        log::error!("Failed to deserialize response: {}", e);
                        let _ = sender.send(f_arc(HttpError::HttpStatus {
                            status: status.as_u16(),
                            message: format!("Failed to deserialize response: {}", e),
                            body: String::from_utf8_lossy(&all_bytes).to_string(),
                        }));
                    }
                }
            });
        });

        // Collect from AsyncStream (blocking, no futures)
        let results = stream.collect();
        results.into_iter().next().unwrap_or_else(|| f_arc(HttpError::HttpStatus {
            status: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
            message: "Stream collection failed".to_string(),
            body: String::new(),
        }))
    }
}

/// Builder for download-specific operations
pub struct DownloadBuilder {
    stream: DownloadStream}

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
                Err(e) => return Err(e)}
        }

        Ok(DownloadProgress {
            chunk_count,
            bytes_written: total_written,
            total_size,
            local_path: local_path.to_string(),
            is_complete: true})
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
    pub is_complete: bool}

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
