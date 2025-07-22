//! HTTP request types and builders

use std::collections::HashMap;
use std::time::Duration;

/// HTTP request methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpMethod {
    /// GET method
    Get,
    /// POST method
    Post,
    /// PUT method
    Put,
    /// DELETE method
    Delete,
    /// PATCH method
    Patch,
    /// HEAD method
    Head,
}

/// HTTP request structure
#[derive(Debug, Clone)]
pub struct HttpRequest {
    method: HttpMethod,
    url: String,
    headers: HashMap<String, String>,
    body: Option<Vec<u8>>,
    timeout: Option<Duration>,
}

impl HttpRequest {
    /// Create a new HTTP request with default optimized configuration
    pub fn new(method: HttpMethod, url: String) -> Self {
        Self {
            method,
            url,
            headers: HashMap::new(),
            body: Some(Vec::new()), // Default to empty body so users don't have to handle None
            timeout: None,
        }
    }

    /// Get the HTTP method
    pub fn method(&self) -> &HttpMethod {
        &self.method
    }

    /// Get the URL
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Get the headers
    pub fn headers(&self) -> &HashMap<String, String> {
        &self.headers
    }

    /// Get the body
    pub fn body(&self) -> Option<&Vec<u8>> {
        self.body.as_ref()
    }

    /// Get the timeout
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Set the body
    pub fn set_body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }

    /// Set the timeout
    pub fn set_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Add a header
    pub fn header<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Add multiple headers
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Set the request body
    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }

    /// Set the request body from a string
    pub fn body_string(mut self, body: String) -> Self {
        self.body = Some(body.into_bytes());
        self
    }

    /// Set the request body from JSON
    pub fn json<T: serde::Serialize>(mut self, json: &T) -> crate::HttpResult<Self> {
        let body = serde_json::to_vec(json)?;
        self.headers
            .insert("Content-Type".to_string(), "application/json".to_string());
        self.body = Some(body);
        Ok(self)
    }

    /// Set request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set timeout in seconds
    pub fn timeout_seconds(mut self, seconds: u64) -> Self {
        self.timeout = Some(Duration::from_secs(seconds));
        self
    }

    /// Set timeout in milliseconds
    pub fn timeout_millis(mut self, millis: u64) -> Self {
        self.timeout = Some(Duration::from_millis(millis));
        self
    }

    /// Add authorization header
    pub fn authorization(mut self, auth: String) -> Self {
        self.headers.insert("Authorization".to_string(), auth);
        self
    }

    /// Add bearer token authorization
    pub fn bearer_token(mut self, token: String) -> Self {
        self.headers
            .insert("Authorization".to_string(), format!("Bearer {}", token));
        self
    }

    /// Add If-None-Match header for conditional requests (ETag-based)
    /// This will return 304 Not Modified if the resource hasn't changed
    pub fn if_none_match(mut self, etag: String) -> Self {
        self.headers.insert("If-None-Match".to_string(), etag);
        self
    }

    /// Add If-Modified-Since header for conditional requests (date-based)
    /// This will return 304 Not Modified if the resource hasn't been modified since the given date
    pub fn if_modified_since(mut self, date: String) -> Self {
        self.headers.insert("If-Modified-Since".to_string(), date);
        self
    }

    /// Add both ETag and Last-Modified conditional headers
    /// This provides the most robust conditional request validation
    pub fn conditional(mut self, etag: Option<String>, last_modified: Option<String>) -> Self {
        if let Some(etag) = etag {
            self.headers.insert("If-None-Match".to_string(), etag);
        }
        if let Some(last_modified) = last_modified {
            self.headers
                .insert("If-Modified-Since".to_string(), last_modified);
        }
        self
    }

    /// Check if this is a conditional request
    pub fn is_conditional(&self) -> bool {
        self.headers.contains_key("If-None-Match") || self.headers.contains_key("If-Modified-Since")
    }

    /// Add content type header
    pub fn content_type(mut self, content_type: String) -> Self {
        self.headers
            .insert("Content-Type".to_string(), content_type);
        self
    }

    /// Add user agent header
    pub fn user_agent(mut self, user_agent: String) -> Self {
        self.headers.insert("User-Agent".to_string(), user_agent);
        self
    }

    /// Add cache control header
    pub fn cache_control(mut self, cache_control: String) -> Self {
        self.headers
            .insert("Cache-Control".to_string(), cache_control);
        self
    }

    /// Add accept header
    pub fn accept(mut self, accept: String) -> Self {
        self.headers.insert("Accept".to_string(), accept);
        self
    }

    /// Add accept encoding header
    pub fn accept_encoding(mut self, accept_encoding: String) -> Self {
        self.headers
            .insert("Accept-Encoding".to_string(), accept_encoding);
        self
    }

    /// Add connection header
    pub fn connection(mut self, connection: String) -> Self {
        self.headers.insert("Connection".to_string(), connection);
        self
    }

    /// Add keep-alive connection header
    pub fn keep_alive(mut self) -> Self {
        self.headers
            .insert("Connection".to_string(), "keep-alive".to_string());
        self
    }

    /// Add close connection header
    pub fn close(mut self) -> Self {
        self.headers
            .insert("Connection".to_string(), "close".to_string());
        self
    }

    /// Add streaming-specific headers
    pub fn streaming(mut self) -> Self {
        self.headers
            .insert("Accept".to_string(), "text/event-stream".to_string());
        self.headers
            .insert("Cache-Control".to_string(), "no-cache".to_string());
        self.headers
            .insert("Connection".to_string(), "keep-alive".to_string());
        self
    }

    /// Convert to reqwest RequestBuilder
    #[allow(dead_code)]
    pub(crate) fn to_reqwest(&self, client: &reqwest::Client) -> reqwest::RequestBuilder {
        let method = match self.method {
            HttpMethod::Get => reqwest::Method::GET,
            HttpMethod::Post => reqwest::Method::POST,
            HttpMethod::Put => reqwest::Method::PUT,
            HttpMethod::Delete => reqwest::Method::DELETE,
            HttpMethod::Patch => reqwest::Method::PATCH,
            HttpMethod::Head => reqwest::Method::HEAD,
        };

        let mut builder = client.request(method, &self.url);

        // Add headers
        for (key, value) in &self.headers {
            builder = builder.header(key, value);
        }

        // Add body if present
        if let Some(body) = &self.body {
            builder = builder.body(body.clone());
        }

        // Add timeout if specified
        if let Some(timeout) = self.timeout {
            builder = builder.timeout(timeout);
        }

        builder
    }
}

impl std::hash::Hash for HttpMethod {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            HttpMethod::Get => 0_u8.hash(state),
            HttpMethod::Post => 1_u8.hash(state),
            HttpMethod::Put => 2_u8.hash(state),
            HttpMethod::Delete => 3_u8.hash(state),
            HttpMethod::Patch => 4_u8.hash(state),
            HttpMethod::Head => 5_u8.hash(state),
        }
    }
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpMethod::Get => write!(f, "GET"),
            HttpMethod::Post => write!(f, "POST"),
            HttpMethod::Put => write!(f, "PUT"),
            HttpMethod::Delete => write!(f, "DELETE"),
            HttpMethod::Patch => write!(f, "PATCH"),
            HttpMethod::Head => write!(f, "HEAD"),
        }
    }
}
