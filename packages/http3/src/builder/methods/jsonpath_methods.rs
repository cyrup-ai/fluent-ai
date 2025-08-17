//! This module contains terminal methods for HTTP requests with JSONPath streaming:
//! GET and POST operations for the JsonPathStreaming builder state that return
//! streams of deserialized objects matching JSONPath expressions.

use fluent_ai_async::prelude::MessageChunk;
use http::Method;
use serde::de::DeserializeOwned;

use crate::builder::core::{Http3Builder, JsonPathStreaming};
use crate::builder::streaming::JsonPathStream;

/// Trait for HTTP methods that support JSONPath streaming
pub trait JsonPathMethods {
    /// Execute a GET request with JSONPath streaming
    fn get<T: DeserializeOwned + MessageChunk + MessageChunk>(self, url: &str)
    -> JsonPathStream<T>;

    /// Execute a POST request with JSONPath streaming
    fn post<T: DeserializeOwned + MessageChunk + MessageChunk>(
        self,
        url: &str,
    ) -> JsonPathStream<T>;
}

/// GET method implementation for JSONPath streaming
pub struct JsonPathGetMethod;

/// POST method implementation for JSONPath streaming
pub struct JsonPathPostMethod;

impl JsonPathMethods for Http3Builder<JsonPathStreaming> {
    #[inline]
    fn get<T: DeserializeOwned + MessageChunk + MessageChunk>(
        self,
        url: &str,
    ) -> JsonPathStream<T> {
        self.execute_jsonpath_with_method::<T>(Method::GET, url)
    }

    #[inline]
    fn post<T: DeserializeOwned + MessageChunk + MessageChunk>(
        self,
        url: &str,
    ) -> JsonPathStream<T> {
        self.execute_jsonpath_with_method::<T>(Method::POST, url)
    }
}

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
    /// use fluent_ai_http3::Http3;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let users_stream = Http3::jsonpath("$.users[*]")
    ///     .get::<User>("https://api.example.com/data");
    /// ```
    #[inline]
    pub fn get<T: DeserializeOwned + MessageChunk + MessageChunk>(
        mut self,
        url: &str,
    ) -> JsonPathStream<T> {
        *self.request.method_mut() = Method::GET;
        *self.request.uri_mut() = url.parse().unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            "http://invalid".parse().unwrap()
        });

        if self.debug_enabled {
            log::debug!(
                "HTTP3 Builder: GET {} (JSONPath: {})",
                url,
                self.jsonpath_expression
            );
        }

        JsonPathStream::new(
            self.client.execute_streaming(self.request),
            self.jsonpath_expression,
        )
    }

    /// Execute a POST request with JSONPath streaming
    ///
    /// Returns a specialized stream that yields individual JSON objects matching
    /// the configured JSONPath expression instead of raw HTTP chunks.
    ///
    /// # Arguments
    /// * `url` - The URL to send the POST request to
    ///
    /// # Returns
    /// `JsonPathStream` for streaming individual deserialized objects
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each matching JSON object into
    #[inline]
    pub fn post<T: DeserializeOwned + MessageChunk + MessageChunk>(
        mut self,
        url: &str,
    ) -> JsonPathStream<T> {
        *self.request.method_mut() = Method::POST;
        *self.request.uri_mut() = url.parse().unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            "http://invalid".parse().unwrap()
        });

        if self.debug_enabled {
            log::debug!(
                "HTTP3 Builder: POST {} (JSONPath: {})",
                url,
                self.jsonpath_expression
            );
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        JsonPathStream::new(
            self.client.execute_streaming(self.request),
            self.jsonpath_expression,
        )
    }

    /// Internal method to execute JSONPath request with specified HTTP method
    #[inline]
    fn execute_jsonpath_with_method<T: DeserializeOwned + MessageChunk + MessageChunk>(
        mut self,
        method: Method,
        url: &str,
    ) -> JsonPathStream<T> {
        *self.request.method_mut() = method;
        *self.request.uri_mut() = url.parse().unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            "http://invalid".parse().unwrap()
        });

        if self.debug_enabled {
            log::debug!(
                "HTTP3 Builder: {} {} (JSONPath: {})",
                method,
                url,
                self.jsonpath_expression
            );
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        JsonPathStream::new(
            self.client.execute_streaming(self.request),
            self.jsonpath_expression,
        )
    }
}
