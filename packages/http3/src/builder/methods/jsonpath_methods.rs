//! JSONPath streaming HTTP methods
//!
//! This module contains terminal methods for HTTP requests with JSONPath streaming:
//! GET and POST operations for the JsonPathStreaming builder state that return
//! streams of deserialized objects matching JSONPath expressions.

use cyrup_sugars::prelude::MessageChunk;
use fluent_ai_async::prelude::MessageChunk as FluentMessageChunk;
use http::Method;
use serde::de::DeserializeOwned;

use crate::builder::core::{Http3Builder, JsonPathStreaming};
use crate::builder::streaming::JsonPathStream;

impl Http3Builder<JsonPathStreaming> {
    /// Execute a GET request with JSONPath streaming
    ///
    /// Returns a specialized stream that yields individual JSON objects matching
    /// the configured JSONPath expression instead of raw HTTP chunks.
    ///
    /// # Arguments
    /// * `url` - The URL to send the GET request to
    ///
    /// # Returns
    /// `JsonPathStream` for streaming individual deserialized objects
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let stream = Http3Builder::json()
    ///     .array_stream("$.users[*]")
    ///     .get::<User>("https://api.example.com/users");
    ///
    /// for user in stream.collect() {
    ///     println!("User: {}", user.name);
    /// }
    /// ```
    #[must_use]
    pub fn get<T>(mut self, url: &str) -> JsonPathStream<T>
    where
        T: DeserializeOwned + Send + 'static + MessageChunk + FluentMessageChunk + Default,
    {
        self.request = self
            .request
            .set_method(Method::GET)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: JSONPath GET {url}");
            if let Some(config) = &self.jsonpath_config {
                log::debug!(
                    "HTTP3 Builder: JSONPath expression: {}",
                    config.jsonpath_expr
                );
            }
        }

        let http_stream = self.client.execute_streaming(self.request);
        let jsonpath_expr = if let Some(config) = self.jsonpath_config {
            config.jsonpath_expr
        } else {
            log::error!("JsonPathStreaming state missing jsonpath_config, using default");
            "$".to_string()
        };

        JsonPathStream::new(http_stream, jsonpath_expr)
    }

    /// Execute a POST request with JSONPath streaming
    ///
    /// Sends a POST request and returns a stream that yields individual JSON objects
    /// matching the configured JSONPath expression.
    ///
    /// # Arguments
    /// * `url` - The URL to send the POST request to
    ///
    /// # Returns
    /// `JsonPathStream` for streaming individual deserialized objects
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Serialize)]
    /// struct Query {
    ///     filter: String,
    /// }
    ///
    /// #[derive(Deserialize)]
    /// struct Result {
    ///     id: u64,
    ///     title: String,
    /// }
    ///
    /// let query = Query {
    ///     filter: "active".to_string(),
    /// };
    ///
    /// let stream = Http3Builder::json()
    ///     .body(&query)
    ///     .array_stream("$.results[*]")
    ///     .post::<Result>("https://api.example.com/search");
    /// ```
    #[must_use]
    pub fn post<T>(mut self, url: &str) -> JsonPathStream<T>
    where
        T: DeserializeOwned + Send + 'static + MessageChunk + FluentMessageChunk + Default,
    {
        self.request = self
            .request
            .set_method(Method::POST)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: JSONPath POST {url}");
            if let Some(config) = &self.jsonpath_config {
                log::debug!(
                    "HTTP3 Builder: JSONPath expression: {}",
                    config.jsonpath_expr
                );
            }
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        let http_stream = self.client.execute_streaming(self.request);
        let jsonpath_expr = if let Some(config) = self.jsonpath_config {
            config.jsonpath_expr
        } else {
            log::error!("JsonPathStreaming state missing jsonpath_config, using default");
            "$".to_string()
        };

        JsonPathStream::new(http_stream, jsonpath_expr)
    }
}
