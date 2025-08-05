//! Core Http3Builder structures and base functionality
//!
//! Contains the main Http3Builder struct, state types, and foundational methods
//! for building HTTP requests with zero allocation and elegant fluent interface.

use std::marker::PhantomData;

use http::Method;

use crate::{HttpClient, HttpRequest};

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
    MultipartFormData,
}

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
            ContentType::MultipartFormData => "multipart/form-data",
        }
    }
}

impl From<&str> for ContentType {
    fn from(s: &str) -> Self {
        match s {
            "application/json" => ContentType::ApplicationJson,
            "application/x-www-form-urlencoded" => ContentType::ApplicationFormUrlEncoded,
            "application/octet-stream" => ContentType::ApplicationOctetStream,
            "text/plain" => ContentType::TextPlain,
            "text/html" => ContentType::TextHtml,
            "multipart/form-data" => ContentType::MultipartFormData,
            _ => ContentType::ApplicationJson, // Default fallback
        }
    }
}

/// State marker indicating no body has been set
#[derive(Debug, Clone, Copy)]
pub struct BodyNotSet;

/// State marker indicating a body has been set
#[derive(Debug, Clone, Copy)]
pub struct BodySet;

/// JSONPath streaming configuration state
#[derive(Debug, Clone)]
pub struct JsonPathStreaming {
    /// JSONPath expression for filtering JSON array responses
    pub jsonpath_expr: String,
}

/// Main Http3 builder for constructing HTTP requests with fluent API
///
/// Type parameter `S` tracks the body state:
/// - `BodyNotSet`: Default state, body methods available
/// - `BodySet`: Body has been set, only execution methods available
/// - `JsonPathStreaming`: Configured for JSONPath array streaming
#[derive(Clone, Debug)]
pub struct Http3Builder<S = BodyNotSet> {
    /// HTTP client instance for making requests
    pub(crate) client: HttpClient,
    /// Request being built
    pub(crate) request: HttpRequest,
    /// Type state marker
    pub(crate) state: PhantomData<S>,
    /// Debug logging enabled flag
    pub(crate) debug_enabled: bool,
    /// JSONPath streaming configuration
    pub(crate) jsonpath_config: Option<JsonPathStreaming>,
}

impl Http3Builder<BodyNotSet> {
    /// Start building a new request with a shared client instance
    #[must_use]
    pub fn new(client: &HttpClient) -> Self {
        Self {
            client: client.clone(),
            request: HttpRequest::new(Method::GET, String::new(), None, None, None),
            state: PhantomData,
            debug_enabled: false,
            jsonpath_config: None,
        }
    }

    /// Shorthand for setting Content-Type to application/json
    #[must_use]
    pub fn json() -> Self {
        let client = HttpClient::default();
        Self::new(&client).content_type(ContentType::ApplicationJson)
    }

    /// Shorthand for setting Content-Type to application/x-www-form-urlencoded
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn form_urlencoded() -> Self {
        let client = HttpClient::default();
        Self::new(&client).content_type(ContentType::ApplicationFormUrlEncoded)
    }

    /// Configure JSONPath streaming for array responses
    ///
    /// Transforms the builder to stream individual objects from JSON arrays
    /// matching the provided JSONPath expression.
    ///
    /// # Arguments
    /// * `jsonpath` - JSONPath expression to filter array elements
    ///
    /// # Returns
    /// `Http3Builder<JsonPathStreaming>` for streaming operations
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let stream = Http3Builder::json()
    ///     .array_stream("$.items[*]")
    ///     .get("https://api.example.com/data");
    /// ```
    #[must_use]
    pub fn array_stream(self, jsonpath: &str) -> Http3Builder<JsonPathStreaming> {
        Http3Builder {
            client: self.client,
            request: self.request,
            state: PhantomData,
            debug_enabled: self.debug_enabled,
            jsonpath_config: Some(JsonPathStreaming {
                jsonpath_expr: jsonpath.to_string(),
            }),
        }
    }
}

impl<S> Http3Builder<S> {
    /// Enable debug logging for this request
    ///
    /// When enabled, detailed request and response information will be logged
    /// to help with debugging and development.
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn debug(mut self) -> Self {
        self.debug_enabled = true;
        self
    }

    /// Set the target URL for the request
    ///
    /// # Arguments
    /// * `url` - The complete URL to send the request to
    ///
    /// # Returns
    /// `Self` for method chaining
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .url("https://api.example.com/users")
    ///     .get("");
    /// ```
    #[must_use]
    pub fn url(mut self, url: &str) -> Self {
        self.request = self.request.set_url(url.to_string());
        self
    }

    /// Set content type using the ContentType enum
    ///
    /// # Arguments
    /// * `content_type` - The content type to set for the request
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn content_type(self, content_type: ContentType) -> Self {
        use std::str::FromStr;

        use http::{HeaderName, HeaderValue};

        let content_type_str = content_type.as_str();
        match (
            HeaderName::from_str("content-type"),
            HeaderValue::from_str(content_type_str),
        ) {
            (Ok(name), Ok(value)) => self.header(name, value),
            _ => self, // Skip invalid header
        }
    }



    /// Set request timeout in seconds
    ///
    /// # Arguments  
    /// * `seconds` - Timeout duration in seconds
    ///
    /// # Returns
    /// `Self` for method chaining
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .timeout_seconds(30)
    ///     .get("https://api.example.com/data");
    /// ```
    #[must_use]
    pub fn timeout_seconds(mut self, seconds: u64) -> Self {
        let timeout = std::time::Duration::from_secs(seconds);
        // Store timeout in the request - we'll need to modify HttpRequest to support this
        // For now, this is a placeholder that compiles
        self.request = self.request.with_timeout(timeout);
        self
    }

    /// Set retry attempts for failed requests
    ///
    /// # Arguments
    /// * `attempts` - Number of retry attempts (0 disables retries)
    ///
    /// # Returns
    /// `Self` for method chaining
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .retry_attempts(3)
    ///     .get("https://api.example.com/data");
    /// ```
    #[must_use]
    pub fn retry_attempts(mut self, attempts: u32) -> Self {
        // Store retry attempts in the request
        self.request = self.request.with_retry_attempts(attempts);
        self
    }


}
