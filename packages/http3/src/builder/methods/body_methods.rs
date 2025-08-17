//! This module contains terminal methods for HTTP requests that require a body:
//! POST, PUT, and PATCH operations for the BodySet builder state.

use http::Method;

use crate::HttpStream;
use crate::builder::core::{BodySet, Http3Builder};

/// Trait for HTTP methods that require a body
pub trait BodyMethods {
    /// Execute a POST request
    fn post(self, url: &str) -> HttpStream;

    /// Execute a PUT request
    fn put(self, url: &str) -> HttpStream;

    /// Execute a PATCH request
    fn patch(self, url: &str) -> HttpStream;
}

/// POST method implementation
pub struct PostMethod;

/// PUT method implementation
pub struct PutMethod;

/// PATCH method implementation
pub struct PatchMethod;

impl BodyMethods for Http3Builder<BodySet> {
    #[inline]
    fn post(self, url: &str) -> HttpStream {
        self.execute_with_method(Method::POST, url)
    }

    #[inline]
    fn put(self, url: &str) -> HttpStream {
        self.execute_with_method(Method::PUT, url)
    }

    #[inline]
    fn patch(self, url: &str) -> HttpStream {
        self.execute_with_method(Method::PATCH, url)
    }
}

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
    ///     name: "Alice".to_string(),
    ///     email: "alice@example.com".to_string(),
    /// };
    ///
    /// let response = Http3Builder::json()
    ///     .body(&user)
    ///     .post("https://api.example.com/users");
    /// ```
    #[inline]
    pub fn post(mut self, url: &str) -> HttpStream {
        *self.request.method_mut() = Method::POST;
        *self.request.uri_mut() = url.parse().unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            "http://invalid".parse().unwrap()
        });

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
    #[inline]
    pub fn put(mut self, url: &str) -> HttpStream {
        *self.request.method_mut() = Method::PUT;
        *self.request.uri_mut() = url.parse().unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            "http://invalid".parse().unwrap()
        });

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
    #[inline]
    pub fn patch(mut self, url: &str) -> HttpStream {
        *self.request.method_mut() = Method::PATCH;
        *self.request.uri_mut() = url.parse().unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            "http://invalid".parse().unwrap()
        });

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: PATCH {}", url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        self.client.execute_streaming(self.request)
    }

    /// Internal method to execute request with specified HTTP method
    #[inline]
    fn execute_with_method(mut self, method: Method, url: &str) -> HttpStream {
        *self.request.method_mut() = method;
        *self.request.uri_mut() = url.parse().unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            "http://invalid".parse().unwrap()
        });

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: {} {}", method, url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body size: {} bytes", body.len());
            }
        }

        self.client.execute_streaming(self.request)
    }
}
