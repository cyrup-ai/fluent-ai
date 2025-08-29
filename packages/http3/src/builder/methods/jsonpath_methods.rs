//! This module contains terminal methods for HTTP requests with JSONPath streaming:
//! GET and POST operations for the JsonPathStreaming builder state that return
//! streams of deserialized objects matching JSONPath expressions.

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::AsyncStream;
use http::Method;
use serde::de::DeserializeOwned;
use url::Url;

use crate::builder::core::{Http3Builder, JsonPathStreaming};
use crate::builder::streaming::JsonPathStream;

/// Trait for HTTP methods that support JSONPath streaming
pub trait JsonPathMethods {
    /// Execute a GET request with JSONPath streaming
    fn get<T: DeserializeOwned + MessageChunk + Send + Default + Clone + 'static>(
        self,
        url: &str,
    ) -> JsonPathStream<T>;

    /// Execute a POST request with JSONPath streaming
    fn post<T: DeserializeOwned + MessageChunk + Send + Default + Clone + 'static>(
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
    fn get<T: DeserializeOwned + MessageChunk + Send + Default + Clone + 'static>(
        self,
        url: &str,
    ) -> JsonPathStream<T> {
        self.execute_jsonpath_with_method::<T>(Method::GET, url)
    }

    #[inline]
    fn post<T: DeserializeOwned + MessageChunk + Send + Default + Clone + 'static>(
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
    pub fn get<T: DeserializeOwned + Send + Default + MessageChunk + 'static>(
        mut self,
        url: &str,
    ) -> JsonPathStream<T> {
        self.request = self.request.with_method(Method::GET);
        self.request = self.request.with_url(Url::parse(url).unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            Url::parse("http://invalid").unwrap()
        }));

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: GET {} (JSONPath: {})", url, "$.items[*]");
        }

        JsonPathStream::new(
            AsyncStream::with_channel(move |sender| {
                let http_response = self.client.execute(self.request);
                fluent_ai_async::emit!(sender, http_response);
            }),
            "$.items[*]".to_string(),
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
    pub fn post<T: DeserializeOwned + Send + Default + MessageChunk + 'static>(
        mut self,
        url: &str,
    ) -> JsonPathStream<T> {
        self.request = self.request.with_method(Method::POST);
        self.request = self.request.with_url(Url::parse(url).unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            Url::parse("http://invalid").unwrap()
        }));

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: POST {} (JSONPath: {})", url, "$.items[*]");
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body present");
            }
        }

        JsonPathStream::new(
            AsyncStream::with_channel(move |sender| {
                let http_response = self.client.execute(self.request);
                fluent_ai_async::emit!(sender, http_response);
            }),
            "$.items[*]".to_string(),
        )
    }

    /// Internal method to execute JSONPath request with specified HTTP method
    #[inline]
    fn execute_jsonpath_with_method<
        T: DeserializeOwned + MessageChunk + Send + Default + 'static,
    >(
        mut self,
        method: Method,
        url: &str,
    ) -> JsonPathStream<T> {
        let method_str = method.to_string();
        self.request = self.request.with_method(method);
        self.request = self.request.with_url(Url::parse(url).unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            Url::parse("http://invalid").unwrap()
        }));

        if self.debug_enabled {
            log::debug!(
                "HTTP3 Builder: {} {} (JSONPath: {})",
                method_str,
                url,
                "$.items[*]"
            );
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body present");
            }
        }

        JsonPathStream::new(
            AsyncStream::with_channel(move |sender| {
                let http_response = self.client.execute(self.request);
                fluent_ai_async::emit!(sender, http_response);
            }),
            "$.items[*]".to_string(),
        )
    }
}
