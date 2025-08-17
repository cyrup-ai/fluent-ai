//! HTTP request types and builders
//!
//! This module provides the CANONICAL HttpRequest implementation that consolidates
//! all previous Request variants into a single, comprehensive request type.

use std::collections::HashMap;
use std::time::Duration;

use bytes::Bytes;
use fluent_ai_async::prelude::*;
use http::{HeaderMap, HeaderName, HeaderValue, Method, Version};
use url::Url;

use crate::streaming::chunks::HttpChunk;

/// HTTP request structure with comprehensive functionality
///
/// This is the CANONICAL HttpRequest implementation that consolidates all
/// previous Request variants into a single, feature-complete request type.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    method: Method,
    url: Url,
    headers: HeaderMap,
    body: Option<RequestBody>,
    timeout: Option<Duration>,
    retry_attempts: Option<u32>,
    version: Version,

    /// Request configuration
    pub cors: bool,
    pub follow_redirects: bool,
    pub max_redirects: u32,
    pub compress: bool,

    /// Authentication
    pub auth: Option<RequestAuth>,

    /// Caching
    pub cache_control: Option<String>,
    pub etag: Option<String>,

    /// Request metadata
    pub user_agent: Option<String>,
    pub referer: Option<String>,

    /// Protocol-specific options
    pub h2_prior_knowledge: bool,
    pub h3_alt_svc: bool,
}

/// Request body types
#[derive(Debug, Clone)]
pub enum RequestBody {
    /// Raw bytes
    Bytes(Bytes),
    /// Text content
    Text(String),
    /// JSON data
    Json(serde_json::Value),
    /// Form data
    Form(HashMap<String, String>),
    /// Multipart form data
    Multipart(Vec<MultipartField>),
    /// Streaming body
    Stream(AsyncStream<HttpChunk, 1024>),
}

/// Multipart form field
#[derive(Debug, Clone)]
pub struct MultipartField {
    pub name: String,
    pub value: MultipartValue,
    pub content_type: Option<String>,
    pub filename: Option<String>,
}

/// Multipart field value
#[derive(Debug, Clone)]
pub enum MultipartValue {
    Text(String),
    Bytes(Bytes),
}

/// Authentication methods
#[derive(Debug, Clone)]
pub enum RequestAuth {
    Basic { username: String, password: String },
    Bearer(String),
    ApiKey { key: String, value: String },
    Custom(HeaderMap),
}

impl HttpRequest {
    /// Creates a new `HttpRequest`
    #[inline]
    pub fn new(method: Method, url: Url) -> Self {
        Self {
            method,
            url,
            headers: HeaderMap::new(),
            body: None,
            timeout: Some(Duration::from_secs(30)),
            retry_attempts: Some(3),
            version: Version::HTTP_3,
            cors: true,
            follow_redirects: true,
            max_redirects: 10,
            compress: true,
            auth: None,
            cache_control: None,
            etag: None,
            user_agent: Some("fluent-ai-http3/1.0".to_string()),
            referer: None,
            h2_prior_knowledge: false,
            h3_alt_svc: true,
        }
    }

    /// Create GET request
    #[inline]
    pub fn get<U: TryInto<Url>>(url: U) -> Result<Self, U::Error> {
        let url = url.try_into()?;
        Ok(Self::new(Method::GET, url))
    }

    /// Create POST request
    #[inline]
    pub fn post<U: TryInto<Url>>(url: U) -> Result<Self, U::Error> {
        let url = url.try_into()?;
        Ok(Self::new(Method::POST, url))
    }

    /// Create PUT request
    #[inline]
    pub fn put<U: TryInto<Url>>(url: U) -> Result<Self, U::Error> {
        let url = url.try_into()?;
        Ok(Self::new(Method::PUT, url))
    }

    /// Create DELETE request
    #[inline]
    pub fn delete<U: TryInto<Url>>(url: U) -> Result<Self, U::Error> {
        let url = url.try_into()?;
        Ok(Self::new(Method::DELETE, url))
    }

    /// Create PATCH request
    #[inline]
    pub fn patch<U: TryInto<Url>>(url: U) -> Result<Self, U::Error> {
        let url = url.try_into()?;
        Ok(Self::new(Method::PATCH, url))
    }

    /// Create HEAD request
    #[inline]
    pub fn head<U: TryInto<Url>>(url: U) -> Result<Self, U::Error> {
        let url = url.try_into()?;
        Ok(Self::new(Method::HEAD, url))
    }

    /// Create OPTIONS request
    #[inline]
    pub fn options<U: TryInto<Url>>(url: U) -> Result<Self, U::Error> {
        let url = url.try_into()?;
        Ok(Self::new(Method::OPTIONS, url))
    }

    // Getters

    /// Get the HTTP method
    #[inline]
    pub fn method(&self) -> &Method {
        &self.method
    }

    /// Get the URL
    #[inline]
    pub fn url(&self) -> &Url {
        &self.url
    }

    /// Get the headers
    #[inline]
    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    /// Get mutable reference to headers
    #[inline]
    pub fn headers_mut(&mut self) -> &mut HeaderMap {
        &mut self.headers
    }

    /// Get the body
    #[inline]
    pub fn body(&self) -> Option<&RequestBody> {
        self.body.as_ref()
    }

    /// Get the timeout
    #[inline]
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Get retry attempts
    #[inline]
    pub fn retry_attempts(&self) -> Option<u32> {
        self.retry_attempts
    }

    /// Get HTTP version
    #[inline]
    pub fn version(&self) -> Version {
        self.version
    }

    // Setters (builder pattern)

    /// Set the HTTP method
    #[inline]
    pub fn method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    /// Set the URL
    #[inline]
    pub fn url(mut self, url: Url) -> Self {
        self.url = url;
        self
    }

    /// Set HTTP version
    #[inline]
    pub fn version(mut self, version: Version) -> Self {
        self.version = version;
        self
    }

    /// Add a header
    #[inline]
    pub fn header<K, V>(mut self, key: K, value: V) -> Self
    where
        K: TryInto<HeaderName>,
        V: TryInto<HeaderValue>,
    {
        if let (Ok(name), Ok(val)) = (key.try_into(), value.try_into()) {
            self.headers.insert(name, val);
        }
        self
    }

    /// Extend headers
    #[inline]
    pub fn headers(mut self, headers: HeaderMap) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Set body as bytes
    #[inline]
    pub fn body_bytes<B: Into<Bytes>>(mut self, body: B) -> Self {
        self.body = Some(RequestBody::Bytes(body.into()));
        self
    }

    /// Set body as text
    #[inline]
    pub fn body_text<S: Into<String>>(mut self, body: S) -> Self {
        self.body = Some(RequestBody::Text(body.into()));
        self
    }

    /// Set body as JSON
    #[inline]
    pub fn json<T: serde::Serialize>(mut self, json: &T) -> Result<Self, serde_json::Error> {
        let value = serde_json::to_value(json)?;
        self.body = Some(RequestBody::Json(value));
        self.headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        Ok(self)
    }

    /// Set body as form data
    #[inline]
    pub fn form(mut self, form: HashMap<String, String>) -> Self {
        self.body = Some(RequestBody::Form(form));
        self.headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/x-www-form-urlencoded"),
        );
        self
    }

    /// Set body as multipart form
    #[inline]
    pub fn multipart(mut self, fields: Vec<MultipartField>) -> Self {
        self.body = Some(RequestBody::Multipart(fields));
        // Content-Type with boundary will be set during serialization
        self
    }

    /// Set streaming body
    #[inline]
    pub fn body_stream(mut self, stream: AsyncStream<HttpChunk, 1024>) -> Self {
        self.body = Some(RequestBody::Stream(stream));
        self
    }

    /// Set timeout
    #[inline]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set retry attempts
    #[inline]
    pub fn retry_attempts(mut self, attempts: u32) -> Self {
        self.retry_attempts = Some(attempts);
        self
    }

    /// Enable/disable CORS
    #[inline]
    pub fn cors(mut self, enable: bool) -> Self {
        self.cors = enable;
        self
    }

    /// Enable/disable redirect following
    #[inline]
    pub fn follow_redirects(mut self, follow: bool) -> Self {
        self.follow_redirects = follow;
        self
    }

    /// Set maximum redirects
    #[inline]
    pub fn max_redirects(mut self, max: u32) -> Self {
        self.max_redirects = max;
        self
    }

    /// Enable/disable compression
    #[inline]
    pub fn compress(mut self, enable: bool) -> Self {
        self.compress = enable;
        self
    }

    /// Set basic authentication
    #[inline]
    pub fn basic_auth<U, P>(mut self, username: U, password: P) -> Self
    where
        U: Into<String>,
        P: Into<String>,
    {
        self.auth = Some(RequestAuth::Basic {
            username: username.into(),
            password: password.into(),
        });
        self
    }

    /// Set bearer token authentication
    #[inline]
    pub fn bearer_auth<T: Into<String>>(mut self, token: T) -> Self {
        self.auth = Some(RequestAuth::Bearer(token.into()));
        self
    }

    /// Set API key authentication
    #[inline]
    pub fn api_key<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.auth = Some(RequestAuth::ApiKey {
            key: key.into(),
            value: value.into(),
        });
        self
    }

    /// Set custom authentication headers
    #[inline]
    pub fn custom_auth(mut self, headers: HeaderMap) -> Self {
        self.auth = Some(RequestAuth::Custom(headers));
        self
    }

    /// Set user agent
    #[inline]
    pub fn user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Set referer
    #[inline]
    pub fn referer<S: Into<String>>(mut self, referer: S) -> Self {
        self.referer = Some(referer.into());
        self
    }

    /// Add query parameters
    #[inline]
    pub fn query<K, V>(mut self, params: &[(K, V)]) -> Self
    where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut query_pairs = self.url.query_pairs_mut();
        for (key, value) in params {
            query_pairs.append_pair(key.as_ref(), value.as_ref());
        }
        drop(query_pairs);
        self
    }

    /// Enable HTTP/2 prior knowledge
    #[inline]
    pub fn h2_prior_knowledge(mut self, enable: bool) -> Self {
        self.h2_prior_knowledge = enable;
        self
    }

    /// Enable HTTP/3 Alt-Svc
    #[inline]
    pub fn h3_alt_svc(mut self, enable: bool) -> Self {
        self.h3_alt_svc = enable;
        self
    }

    // Utility methods

    /// Check if request has body
    #[inline]
    pub fn has_body(&self) -> bool {
        self.body.is_some()
    }

    /// Get content length if known
    pub fn content_length(&self) -> Option<u64> {
        match &self.body {
            Some(RequestBody::Bytes(bytes)) => Some(bytes.len() as u64),
            Some(RequestBody::Text(text)) => Some(text.len() as u64),
            Some(RequestBody::Json(json)) => serde_json::to_vec(json).ok().map(|v| v.len() as u64),
            Some(RequestBody::Form(form)) => {
                let encoded = serde_urlencoded::to_string(form).ok()?;
                Some(encoded.len() as u64)
            }
            _ => None,
        }
    }

    /// Apply authentication to headers
    pub fn apply_auth(&mut self) {
        if let Some(auth) = &self.auth {
            match auth {
                RequestAuth::Basic { username, password } => {
                    let credentials = base64::encode(format!("{}:{}", username, password));
                    if let Ok(value) = HeaderValue::from_str(&format!("Basic {}", credentials)) {
                        self.headers.insert(http::header::AUTHORIZATION, value);
                    }
                }
                RequestAuth::Bearer(token) => {
                    if let Ok(value) = HeaderValue::from_str(&format!("Bearer {}", token)) {
                        self.headers.insert(http::header::AUTHORIZATION, value);
                    }
                }
                RequestAuth::ApiKey { key, value } => {
                    if let (Ok(header_name), Ok(header_value)) = (
                        HeaderName::from_bytes(key.as_bytes()),
                        HeaderValue::from_str(value),
                    ) {
                        self.headers.insert(header_name, header_value);
                    }
                }
                RequestAuth::Custom(auth_headers) => {
                    self.headers.extend(auth_headers.clone());
                }
            }
        }

        // Apply user agent
        if let Some(user_agent) = &self.user_agent {
            if let Ok(value) = HeaderValue::from_str(user_agent) {
                self.headers.insert(http::header::USER_AGENT, value);
            }
        }

        // Apply referer
        if let Some(referer) = &self.referer {
            if let Ok(value) = HeaderValue::from_str(referer) {
                self.headers.insert(http::header::REFERER, value);
            }
        }
    }

    /// Convert to streaming chunks
    pub fn to_chunks(self) -> AsyncStream<HttpChunk, 1024> {
        AsyncStream::with_channel(move |sender| {
            // Emit request line
            emit!(
                sender,
                HttpChunk::RequestLine {
                    method: self.method.clone(),
                    uri: self.url.to_string(),
                    version: self.version,
                }
            );

            // Emit headers
            for (name, value) in &self.headers {
                emit!(
                    sender,
                    HttpChunk::Header {
                        name: name.clone(),
                        value: value.clone(),
                    }
                );
            }

            // Emit body if present
            if let Some(body) = self.body {
                match body {
                    RequestBody::Bytes(bytes) => {
                        emit!(sender, HttpChunk::Body(bytes));
                    }
                    RequestBody::Text(text) => {
                        emit!(sender, HttpChunk::Body(text.into_bytes().into()));
                    }
                    RequestBody::Json(json) => {
                        if let Ok(bytes) = serde_json::to_vec(&json) {
                            emit!(sender, HttpChunk::Body(bytes.into()));
                        }
                    }
                    RequestBody::Form(form) => {
                        if let Ok(encoded) = serde_urlencoded::to_string(&form) {
                            emit!(sender, HttpChunk::Body(encoded.into_bytes().into()));
                        }
                    }
                    RequestBody::Stream(stream) => {
                        for chunk in stream {
                            emit!(sender, chunk);
                        }
                    }
                    RequestBody::Multipart(_fields) => {
                        // TODO: Implement multipart encoding
                        emit!(
                            sender,
                            HttpChunk::Error("Multipart encoding not yet implemented".to_string())
                        );
                    }
                }
            }

            emit!(sender, HttpChunk::Complete);
        })
    }
}

impl MultipartField {
    /// Create text field
    #[inline]
    pub fn text<N: Into<String>, V: Into<String>>(name: N, value: V) -> Self {
        Self {
            name: name.into(),
            value: MultipartValue::Text(value.into()),
            content_type: Some("text/plain".to_string()),
            filename: None,
        }
    }

    /// Create file field
    #[inline]
    pub fn file<N: Into<String>, F: Into<String>>(
        name: N,
        filename: F,
        content_type: Option<String>,
        data: Bytes,
    ) -> Self {
        Self {
            name: name.into(),
            value: MultipartValue::Bytes(data),
            content_type,
            filename: Some(filename.into()),
        }
    }
}
