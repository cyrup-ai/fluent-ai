//! JSONPath processing and filtering functionality
//!
//! Provides JSONPath-based streaming, filtering, and extraction methods
//! for processing JSON response bodies with zero-allocation patterns.

use bytes::Bytes;
use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::AsyncStream;
use serde::de::DeserializeOwned;
use std::marker::PhantomData;

use crate::json_path::JsonStreamProcessor;
use crate::response::core::HttpResponse;
use super::types::JsonStream;

impl HttpResponse {
    /// Parse the body as JSON stream - returns unwrapped T chunks
    /// Only available for JSON content-type responses
    /// Zero futures, error handling via user `on_chunk` handlers, users call `.collect()` for await-like behavior
    #[must_use]
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Option<JsonStream<T>> {
        // Only provide JSON parsing for JSON content types
        if self.is_json_content() {
            Some(JsonStream {
                body: self.body().to_vec(),
                _phantom: PhantomData,
                handler: None,
            })
        } else {
            None
        }
    }

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
        let response_bytes = Bytes::from(self.body().to_vec());

        // Process the entire response body through JSONPath filtering
        stream_processor.process_body(response_bytes)
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