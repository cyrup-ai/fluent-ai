//! HTTP method implementations
//!
//! Terminal methods for executing HTTP requests (GET, POST, PUT, PATCH, DELETE)
//! with appropriate request configurations and response streaming.

use http::Method;
use serde::de::DeserializeOwned;

use crate::builder::core::{BodyNotSet, BodySet, Http3Builder, JsonPathStreaming};
use crate::builder::streaming::JsonPathStream;
use crate::{DownloadBuilder, HttpStream};

// Terminal methods for BodyNotSet (no body required)
impl Http3Builder<BodyNotSet> {
    /// Execute a GET request
    ///
    /// # Arguments
    /// * `url` - The URL to send the GET request to
    ///
    /// # Returns
    /// `HttpStream` for streaming the response
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .get("https://api.example.com/users");
    /// ```
    #[must_use]
    pub fn get(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::GET)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: GET {url}");
        }

        self.client.execute_streaming(self.request)
    }

    /// Execute a DELETE request
    ///
    /// # Arguments
    /// * `url` - The URL to send the DELETE request to
    ///
    /// # Returns
    /// `HttpStream` for streaming the response
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let response = Http3Builder::json()
    ///     .delete("https://api.example.com/users/123");
    /// ```
    #[must_use]
    pub fn delete(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::DELETE)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: DELETE {url}");
        }

        self.client.execute_streaming(self.request)
    }

    /// Initiate a file download
    ///
    /// Creates a specialized download stream with progress tracking and
    /// file writing capabilities.
    ///
    /// # Arguments
    /// * `url` - The URL to download from
    ///
    /// # Returns
    /// `DownloadBuilder` for configuring the download
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    ///
    /// let download = Http3Builder::new(&client)
    ///     .download_file("https://example.com/large-file.zip")
    ///     .destination("/tmp/downloaded-file.zip")
    ///     .start();
    /// ```
    #[must_use]
    pub fn download_file(mut self, url: &str) -> DownloadBuilder {
        self.request = self
            .request
            .set_method(Method::GET)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: DOWNLOAD {url}");
        }

        let stream = self.client.download_file(self.request);
        DownloadBuilder::new(stream)
    }
}

// Terminal methods for BodySet (body has been set)
impl Http3Builder<BodySet> {
    /// Execute a POST request
    ///
    /// # Arguments
    /// * `url` - The URL to send the POST request to
    ///
    /// # Returns
    /// `HttpStream` for streaming the response
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::Serialize;
    ///
    /// #[derive(Serialize)]
    /// struct CreateUser {
    ///     name: String,
    ///     email: String,
    /// }
    ///
    /// let user = CreateUser {
    ///     name: "John Doe".to_string(),
    ///     email: "john@example.com".to_string(),
    /// };
    ///
    /// let response = Http3Builder::json()
    ///     .body(&user)
    ///     .post("https://api.example.com/users");
    /// ```
    pub fn post(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::POST)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: POST {}", url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        self.client.execute_streaming(self.request)
    }

    /// Execute a PUT request
    ///
    /// # Arguments
    /// * `url` - The URL to send the PUT request to
    ///
    /// # Returns
    /// `HttpStream` for streaming the response
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::Serialize;
    ///
    /// #[derive(Serialize)]
    /// struct UpdateUser {
    ///     name: String,
    ///     email: String,
    /// }
    ///
    /// let user = UpdateUser {
    ///     name: "Jane Doe".to_string(),
    ///     email: "jane@example.com".to_string(),
    /// };
    ///
    /// let response = Http3Builder::json()
    ///     .body(&user)
    ///     .put("https://api.example.com/users/123");
    /// ```
    pub fn put(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::PUT)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: PUT {}", url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        self.client.execute_streaming(self.request)
    }

    /// Execute a PATCH request
    ///
    /// # Arguments
    /// * `url` - The URL to send the PATCH request to
    ///
    /// # Returns
    /// `HttpStream` for streaming the response
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::Serialize;
    ///
    /// #[derive(Serialize)]
    /// struct PatchUser {
    ///     email: String,
    /// }
    ///
    /// let update = PatchUser {
    ///     email: "newemail@example.com".to_string(),
    /// };
    ///
    /// let response = Http3Builder::json()
    ///     .body(&update)
    ///     .patch("https://api.example.com/users/123");
    /// ```
    pub fn patch(mut self, url: &str) -> HttpStream {
        self.request = self
            .request
            .set_method(Method::PATCH)
            .set_url(url.to_string());

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: PATCH {}", url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        self.client.execute_streaming(self.request)
    }
}

// Terminal methods for JSONPath streaming
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
        T: DeserializeOwned + Send + 'static,
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
        T: DeserializeOwned + Send + 'static,
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
