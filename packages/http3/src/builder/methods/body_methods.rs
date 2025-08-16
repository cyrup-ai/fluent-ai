//! HTTP methods for requests with body
//!
//! This module contains terminal methods for HTTP requests that require a body:
//! POST, PUT, and PATCH operations for the BodySet builder state.

use http::Method;

use crate::HttpStream;
use crate::builder::core::{BodySet, Http3Builder};

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
