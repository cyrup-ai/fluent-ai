//! Blazing-fast ergonomic Http3 builder API with zero allocation and elegant fluent interface
//! Supports streaming HttpChunk/BadHttpChunk responses, Serde integration, and shorthand methods

use std::marker::PhantomData;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_async::AsyncStream;
use futures_util::{StreamExt, TryStreamExt};
use http::{HeaderMap, HeaderName, HeaderValue, Method};
use serde::{Serialize, de::DeserializeOwned};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

use crate::{
    DownloadChunk, DownloadStream, HttpChunk, HttpClient, HttpError, HttpRequest, HttpStream,
};

/// Alias for http header names - no custom enum duplication
pub use http::header::*;


/// Content type enumeration for elegant API
#[derive(Debug, Clone, Copy)]
pub enum ContentType {
    ApplicationJson,
    ApplicationFormUrlEncoded,
    ApplicationOctetStream,
    TextPlain,
    TextHtml,
}

impl ContentType {
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



/// Zero allocation, blazing-fast HTTP builder
#[derive(Clone)]
pub struct Http3Builder<S = BodyNotSet> {
    client: HttpClient,
    request: HttpRequest,
    state: PhantomData<S>,
}

// Entry points
impl Http3Builder<BodyNotSet> {
    /// Start building a new request with a shared client instance
    pub fn new(client: &HttpClient) -> Self {
        Self {
            client: client.clone(),
            request: HttpRequest::new(Method::GET, "".to_string(), None, None, None),
            state: PhantomData,
        }
    }

    /// Shorthand for setting Content-Type to application/json
    pub fn json() -> Self {
        let client = HttpClient::default();
        Self::new(&client).content_type(ContentType::ApplicationJson)
    }

    /// Shorthand for setting Content-Type to application/x-www-form-urlencoded
    pub fn form_urlencoded() -> Self {
        let client = HttpClient::default();
        Self::new(&client).content_type(ContentType::ApplicationFormUrlEncoded)
    }
}


// State-agnostic methods
impl<S> Http3Builder<S> {
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

    pub fn headers(mut self, headers: ZeroOneOrMany<(HeaderName, HeaderValue)>) -> Self {
        let mut header_map = HeaderMap::new();
        match headers {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One((name, value)) => {
                header_map.insert(name, value);
            }
            ZeroOneOrMany::Many(header_list) => {
                for (name, value) in header_list {
                    header_map.insert(name, value);
                }
            }
        }
        self.request = self.request.with_headers(header_map);
        self
    }

    /// Set the Content-Type header
    pub fn content_type(self, content_type: ContentType) -> Self {
        self.header(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static(content_type.as_str()),
        )
    }

    /// Set the Accept header
    pub fn accept(self, content_type: ContentType) -> Self {
        self.header(
            HeaderName::from_static("accept"),
            HeaderValue::from_static(content_type.as_str()),
        )
    }

    /// Set the API key using the x-api-key header
    pub fn api_key(self, key: &str) -> Self {
        self.header(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(key).unwrap(),
        )
    }

    /// Set basic authentication header
    pub fn basic_auth(self, user: &str, pass: Option<&str>) -> Self {
        let auth_string = format!("{}:{}", user, pass.unwrap_or(""));
        let encoded = STANDARD.encode(auth_string.as_bytes());
        let header_value = format!("Basic {}", encoded);
        self.header(
            HeaderName::from_static("authorization"),
            HeaderValue::from_str(&header_value).unwrap(),
        )
    }

    /// Set basic authentication using HashMap syntax
    pub fn basic_auth_map<F>(self, f: F) -> Self
    where 
        F: FnOnce() -> std::collections::HashMap<&'static str, &'static str>,
    {
        let auth_map = f();
        if let (Some(user), Some(pass)) = (auth_map.get("user"), auth_map.get("password")) {
            self.basic_auth(user, Some(pass))
        } else {
            self
        }
    }

    /// Set bearer token authentication header
    pub fn bearer_auth(self, token: &str) -> Self {
        let header_value = format!("Bearer {}", token);
        self.header(
            HeaderName::from_static("authorization"),
            HeaderValue::from_str(&header_value).unwrap(),
        )
    }

    /// Set the request body
    pub fn body<T: Serialize>(self, body: &T) -> Http3Builder<BodySet> {
        let body_bytes = serde_json::to_vec(body).unwrap();
        let request = self.request.set_body(body_bytes);

        Http3Builder {
            client: self.client,
            request,
            state: PhantomData,
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
#[derive(Clone)]
pub struct BodyNotSet;
#[derive(Clone)]
pub struct BodySet;

// Extension trait for collecting stream into a Serde type
pub trait HttpStreamExt {
    fn collect<T: DeserializeOwned + Send + 'static>(self) -> AsyncStream<Result<T, HttpError>>;
    fn collect_or_else<
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> T + Send + 'static,
    >(
        self,
        f: F,
    ) -> AsyncStream<T>;
}

impl HttpStreamExt for HttpStream {
    fn collect<T: DeserializeOwned + Send + 'static>(self) -> AsyncStream<Result<T, HttpError>> {
        AsyncStream::with_channel(move |mut sender| {
            tokio::spawn(async move {
                let body_bytes_result = self
                    .try_fold(Vec::new(), |mut acc, chunk| async move {
                        if let HttpChunk::Body(bytes) = chunk {
                            acc.extend_from_slice(&bytes);
                        }
                        Ok(acc)
                    })
                    .await;

                let result = match body_bytes_result {
                    Ok(body_bytes) => serde_json::from_slice(&body_bytes).map_err(|e| {
                        HttpError::ChunkProcessingError {
                            source: std::sync::Arc::new(e),
                            body: body_bytes,
                        }
                    }),
                    Err(e) => Err(e),
                };

                sender.send(result).ok();
            });
        })
    }

    fn collect_or_else<
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> T + Send + 'static,
    >(
        self,
        f: F,
    ) -> AsyncStream<T> {
        AsyncStream::with_channel(move |mut sender| {
            tokio::spawn(async move {
                let mut stream = HttpStreamExt::collect::<T>(self);
                if let Some(result) = stream.next().await {
                    let value = result.unwrap_or_else(f);
                    sender.send(value).ok();
                }
            });
        })
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
    pub chunk_count: u32,
    pub bytes_written: u64,
    pub total_size: Option<u64>,
    pub local_path: String,
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
