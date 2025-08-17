//! JSONPath streaming methods for HTTP responses
//!
//! This module provides JSONPath filtering and streaming functionality for HttpResponse,
//! enabling extraction of specific objects from JSON arrays and nested structures.

use bytes::Bytes;
use fluent_ai_async::AsyncStream;
use fluent_ai_async::prelude::MessageChunk;
use serde::de::DeserializeOwned;

use crate::jsonpath::JsonStreamProcessor;
use crate::response::core::HttpResponse;

impl HttpResponse {
    /// Stream individual objects from JSON arrays using JSONPath filtering
    ///
    /// Processes the response body through JSONPath expressions to extract
    /// individual objects from JSON arrays and nested structures.
    ///
    /// # Arguments
    /// * `jsonpath_expr` - JSONPath expression for filtering (e.g., "$.data[*]", "$.results[?(@.active)]")
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    ///
    /// # Returns
    /// AsyncStream of successfully deserialized objects of type T
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::HttpResponse;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// // Parse OpenAI-style {"data": [...]} response
    /// let users: Vec<User> = response
    ///     .jsonpath_stream("$.data[*]")
    ///     .collect();
    /// ```
    #[must_use]
    pub fn jsonpath_stream<T>(&self, jsonpath_expr: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + MessageChunk + MessageChunk + Default + Send + 'static,
    {
        let stream_processor = JsonStreamProcessor::<T>::new(jsonpath_expr);
        let response_bytes = Bytes::copy_from_slice(&self.body);

        // Process the entire response body through JSONPath filtering
        stream_processor.process_bytes(response_bytes)
    }

    /// Extract the first object matching a JSONPath expression
    ///
    /// Convenience method for getting a single object from JSON responses
    /// using JSONPath filtering.
    ///
    /// # Arguments
    /// * `jsonpath_expr` - JSONPath expression for filtering
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize the matching JSON object into
    ///
    /// # Returns
    /// Option containing the first matching object, or None if no matches
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::HttpResponse;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// // Get first user from {"data": [...]} response
    /// let first_user: Option<User> = response
    ///     .jsonpath_first("$.data[0]");
    /// ```
    #[must_use]
    pub fn jsonpath_first<T>(&self, jsonpath_expr: &str) -> Option<T>
    where
        T: DeserializeOwned + Send + 'static + MessageChunk + MessageChunk + Default,
    {
        self.jsonpath_stream(jsonpath_expr)
            .collect()
            .into_iter()
            .next()
    }

    /// Extract all objects matching a JSONPath expression into a Vec
    ///
    /// Convenience method for collecting all matching objects from JSON responses
    /// using JSONPath filtering.
    ///
    /// # Arguments
    /// * `jsonpath_expr` - JSONPath expression for filtering
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    ///
    /// # Returns
    /// Vec containing all matching objects
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::HttpResponse;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// // Get all active users from response
    /// let active_users: Vec<User> = response
    ///     .jsonpath_collect("$.users[?(@.active == true)]");
    /// ```
    #[must_use]
    pub fn jsonpath_collect<T>(&self, jsonpath_expr: &str) -> Vec<T>
    where
        T: DeserializeOwned + Send + 'static + MessageChunk + MessageChunk + Default,
    {
        self.jsonpath_stream(jsonpath_expr).collect()
    }
}
