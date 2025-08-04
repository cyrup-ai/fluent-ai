//! Header management and manipulation functionality
//!
//! Provides methods for setting and managing HTTP headers including
//! common headers like Content-Type, Accept, and custom headers.

use http::{HeaderName, HeaderValue};

use crate::builder::core::Http3Builder;

/// Header constants for common HTTP headers
pub mod header {
    pub use http::header::*;

    /// Custom X-API-Key header for API authentication
    pub const X_API_KEY: &str = "x-api-key";
}

impl<S> Http3Builder<S> {
    /// Add a custom header to the request
    ///
    /// # Arguments
    /// * `key` - The header name
    /// * `value` - The header value
    ///
    /// # Returns
    /// `Self` for method chaining
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use http::{HeaderName, HeaderValue};
    ///
    /// let response = Http3Builder::json()
    ///     .header(
    ///         HeaderName::from_static("x-custom-header"),
    ///         HeaderValue::from_static("custom-value")
    ///     )
    ///     .get("https://api.example.com/data");
    /// ```
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
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use hashbrown::HashMap;
    ///
    /// let headers = HashMap::from([
    ///     ("user-agent", "MyApp/1.0"),
    ///     ("x-api-version", "v1"),
    /// ]);
    ///
    /// let response = Http3Builder::json()
    ///     .headers(headers)
    ///     .get("https://api.example.com/data");
    /// ```
    #[must_use]
    pub fn headers(
        mut self,
        headers_config: impl Into<hashbrown::HashMap<&'static str, &'static str>>,
    ) -> Self {
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

    /// Set cache control header
    ///
    /// # Arguments
    /// * `value` - The cache control directive (e.g., "no-cache", "max-age=3600")
    ///
    /// # Returns
    /// `Self` for method chaining
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .cache_control("no-cache")
    ///     .get("https://api.example.com/data");
    /// ```
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
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .max_age(3600) // Cache for 1 hour
    ///     .get("https://api.example.com/data");
    /// ```
    #[must_use]
    pub fn max_age(self, seconds: u64) -> Self {
        let value = format!("max-age={}", seconds);
        self.cache_control(&value)
    }

    /// Set User-Agent header
    ///
    /// # Arguments
    /// * `user_agent` - The user agent string
    ///
    /// # Returns
    /// `Self` for method chaining
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .user_agent("MyApp/1.0")
    ///     .get("https://api.example.com/data");
    /// ```
    #[must_use]
    pub fn user_agent(self, user_agent: &str) -> Self {
        match HeaderValue::from_str(user_agent) {
            Ok(header_value) => self.header(header::USER_AGENT, header_value),
            Err(_) => self, // Skip invalid header value
        }
    }

    /// Set Accept header  
    ///
    /// # Arguments
    /// * `accept` - The accept header value (e.g., "application/json", "text/html")
    ///
    /// # Returns
    /// `Self` for method chaining
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .accept("application/json")
    ///     .get("https://api.example.com/data");
    /// ```
    #[must_use]
    pub fn accept(self, accept: &str) -> Self {
        match HeaderValue::from_str(accept) {
            Ok(header_value) => self.header(header::ACCEPT, header_value),
            Err(_) => self, // Skip invalid header value
        }
    }

    /// Set Accept header using ContentType enum
    ///
    /// # Arguments
    /// * `content_type` - The content type to accept
    ///
    /// # Returns
    /// `Self` for method chaining
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::{Http3Builder, ContentType};
    ///
    /// let response = Http3Builder::json()
    ///     .accept_content_type(ContentType::ApplicationJson)
    ///     .get("https://api.example.com/data");
    /// ```
    #[must_use]
    pub fn accept_content_type(self, content_type: crate::builder::core::ContentType) -> Self {
        self.accept(content_type.as_str())
    }
}
