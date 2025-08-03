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
    timeout: Option<Duration>,
}

impl HttpRequest {
    /// Creates a new `HttpRequest`.
    #[must_use]
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
            headers: headers.unwrap_or_default(),
            body,
            timeout,
        }
    }

    /// Returns the HTTP method.
    #[must_use]
    pub fn method(&self) -> &Method {
        &self.method
    }

    /// Returns the URL.
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Returns a reference to the headers.
    #[must_use]
    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    /// Returns a reference to the body.
    #[must_use]
    pub fn body(&self) -> Option<&Vec<u8>> {
        self.body.as_ref()
    }

    /// Returns the timeout.
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Sets the body, consuming the request and returning a new one.
    #[must_use = "returns a new `HttpRequest` with the updated body"]
    pub fn set_body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }

    /// Sets the timeout, consuming the request and returning a new one.
    #[must_use = "returns a new `HttpRequest` with the updated timeout"]
    pub fn set_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the HTTP method, consuming the request and returning a new one.
    #[must_use = "returns a new `HttpRequest` with the updated method"]
    pub fn set_method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    /// Sets the URL, consuming the request and returning a new one.
    #[must_use = "returns a new `HttpRequest` with the updated URL"]
    pub fn set_url(mut self, url: String) -> Self {
        self.url = url;
        self
    }

    /// Adds a header, consuming the request and returning a new one.
    #[must_use = "returns a new `HttpRequest` with the added header"]
    pub fn header(mut self, key: HeaderName, value: HeaderValue) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// Extends the headers with a `HeaderMap`, consuming the request and returning a new one.
    #[must_use = "returns a new `HttpRequest` with the extended headers"]
    pub fn with_headers(mut self, headers: HeaderMap) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Adds query parameters to the URL.
    #[must_use = "returns a new `HttpRequest` with the added query parameters"]
    pub fn with_query_params(mut self, params: &[(&str, &str)]) -> Self {
        if params.is_empty() {
            return self;
        }

        let Ok(mut url) = url::Url::parse(&self.url) else {
            return self; // Skip query params if URL parsing fails
        };

        for (key, value) in params {
            url.query_pairs_mut().append_pair(key, value);
        }

        self.url = url.to_string();
        self
    }

    /// Sets the Content-Type header, consuming the request and returning a new one.
    ///
    /// # Arguments
    ///
    /// * `content_type` - The content type string (e.g., "application/json", "text/plain")
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fluent_ai_http3::HttpRequest;
    /// use http::Method;
    ///
    /// let request = HttpRequest::new(Method::POST, "https://api.example.com".to_string(), None, None, None)
    ///     .content_type("application/json");
    /// ```
    #[must_use = "returns a new `HttpRequest` with the updated Content-Type header"]
    pub fn content_type(mut self, content_type: &str) -> Self {
        let header_name = HeaderName::from_static("content-type");
        if let Ok(header_value) = HeaderValue::from_str(content_type) {
            self.headers.insert(header_name, header_value);
        }
        self
    }
}
