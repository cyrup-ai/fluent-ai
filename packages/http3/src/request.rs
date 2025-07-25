//! HTTP request types and builders

use std::time::Duration;

use http::{HeaderMap, HeaderName, HeaderValue, Method};

/// HTTP request structure, designed to be built by a fluent builder.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    method: Method,
    url: String,
    headers: HeaderMap,
    body: Option<Vec<u8>>,
    timeout: Option<Duration>}

impl HttpRequest {
    /// Creates a new `HttpRequest`.
    pub fn new(
        method: Method,
        url: String,
        headers: Option<HeaderMap>,
        body: Option<Vec<u8>>,
        timeout: Option<Duration>,
    ) -> Self {
        Self {
            method,
            url,
            headers: headers.unwrap_or_else(HeaderMap::new),
            body,
            timeout}
    }

    /// Returns the HTTP method.
    pub fn method(&self) -> &Method {
        &self.method
    }

    /// Returns the URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Returns a reference to the headers.
    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    /// Returns a reference to the body.
    pub fn body(&self) -> Option<&Vec<u8>> {
        self.body.as_ref()
    }

    /// Returns the timeout.
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Sets the body, consuming the request and returning a new one.
    pub fn set_body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }

    /// Sets the timeout, consuming the request and returning a new one.
    pub fn set_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the HTTP method, consuming the request and returning a new one.
    pub fn set_method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    /// Sets the URL, consuming the request and returning a new one.
    pub fn set_url(mut self, url: String) -> Self {
        self.url = url;
        self
    }

    /// Adds a header, consuming the request and returning a new one.
    pub fn header(mut self, key: HeaderName, value: HeaderValue) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// Extends the headers with a `HeaderMap`, consuming the request and returning a new one.
    pub fn with_headers(mut self, headers: HeaderMap) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Adds query parameters to the URL.
    pub fn with_query_params(mut self, params: &[(&str, &str)]) -> Self {
        if !params.is_empty() {
            let mut url = match url::Url::parse(&self.url) {
                Ok(url) => url,
                Err(_) => return self, // Skip query params if URL parsing fails
            };
            for (key, value) in params {
                url.query_pairs_mut().append_pair(key, value);
            }
            self.url = url.to_string();
        }
        self
    }
}
