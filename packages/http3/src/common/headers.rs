//! Headers Management Module - Using standard http crate types

use base64::{Engine as _, engine::general_purpose};
use http::{HeaderMap, HeaderName, HeaderValue, header};
use thiserror::Error;

/// A wrapper around `http::HeaderMap` to provide fluent, application-specific helpers.
#[derive(Debug, Clone, Default)]
pub struct HeaderManager {
    headers: HeaderMap,
}

impl HeaderManager {
    /// Creates a new, empty `HeaderManager`.
    pub fn new() -> Self {
        HeaderManager {
            headers: HeaderMap::new(),
        }
    }

    /// Sets a header, consuming the manager and returning a new one.
    pub fn set(mut self, key: HeaderName, value: HeaderValue) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// Sets the Content-Type header.
    pub fn content_type(self, content_type: &str) -> Result<Self, HeaderError> {
        let value = HeaderValue::from_str(content_type)?;
        Ok(self.set(header::CONTENT_TYPE, value))
    }

    /// Sets the Authorization header with a bearer token.
    pub fn bearer_token(self, token: &str) -> Result<Self, HeaderError> {
        let value = HeaderValue::from_str(&format!("Bearer {}", token))?;
        Ok(self.set(header::AUTHORIZATION, value))
    }

    /// Sets basic authentication.
    pub fn basic_auth(self, user: &str, pass: Option<&str>) -> Result<Self, HeaderError> {
        let credentials = format!("{}:{}", user, pass.unwrap_or_default());
        let encoded = general_purpose::STANDARD.encode(credentials);
        let value = HeaderValue::from_str(&format!("Basic {}", encoded))?;
        Ok(self.set(header::AUTHORIZATION, value))
    }

    /// Returns the underlying `HeaderMap`.
    pub fn build(self) -> HeaderMap {
        self.headers
    }
}

/// Header-related errors.
#[derive(Debug, Clone, Error)]
pub enum HeaderError {
    /// Represents an error when a header value is invalid.
    #[error("Invalid header value: {message}")]
    InvalidHeaderValue {
        /// Error message describing the invalid header value
        message: String,
    },
}

impl From<http::header::InvalidHeaderValue> for HeaderError {
    fn from(err: http::header::InvalidHeaderValue) -> Self {
        HeaderError::InvalidHeaderValue {
            message: err.to_string(),
        }
    }
}
