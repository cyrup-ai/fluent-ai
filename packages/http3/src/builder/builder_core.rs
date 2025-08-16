//! Core Http3Builder struct and foundational methods
//!
//! Contains the main Http3Builder struct definition and basic construction methods
//! for building HTTP requests with zero allocation and elegant fluent interface.

use std::marker::PhantomData;
use std::sync::Arc;

use fluent_ai_async::prelude::ChunkHandler;
use http::Method;

use super::content_type::ContentType;
use super::state_types::{BodyNotSet, JsonPathStreaming};
use crate::{HttpChunk, HttpClient, HttpError, HttpRequest};

/// Main Http3 builder for constructing HTTP requests with fluent API
///
/// Type parameter `S` tracks the body state:
/// - `BodyNotSet`: Default state, body methods available
/// - `BodySet`: Body has been set, only execution methods available
/// - `JsonPathStreaming`: Configured for JSONPath array streaming
#[derive(Clone)]
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
    /// Chunk handler for error handling in streaming
    pub(crate) chunk_handler:
        Option<Arc<dyn Fn(Result<HttpChunk, HttpError>) -> HttpChunk + Send + Sync + 'static>>,
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
            chunk_handler: None,
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
            chunk_handler: self.chunk_handler,
        }
    }
}

impl<S> Http3Builder<S> {
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

    /// Set a header on the request
    ///
    /// # Arguments
    /// * `name` - Header name
    /// * `value` - Header value
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn header(mut self, name: http::HeaderName, value: http::HeaderValue) -> Self {
        self.request = self.request.add_header(name, value);
        self
    }
}
