//! HTTP status validation methods
//!
//! This module contains methods for checking and validating HTTP status codes.

use hyper::StatusCode;

use super::types::Response;

impl Response {
    /// Returns `true` if the status code is in the 2xx range.
    #[inline]
    pub fn is_success(&self) -> bool {
        self.status().is_success()
    }

    /// Returns `true` if the status code is in the 1xx range.
    #[inline]
    pub fn is_informational(&self) -> bool {
        self.status().is_informational()
    }

    /// Returns `true` if the status code is in the 3xx range.
    #[inline]
    pub fn is_redirection(&self) -> bool {
        self.status().is_redirection()
    }

    /// Returns `true` if the status code is in the 4xx range.
    #[inline]
    pub fn is_client_error(&self) -> bool {
        self.status().is_client_error()
    }

    /// Returns `true` if the status code is in the 5xx range.
    #[inline]
    pub fn is_server_error(&self) -> bool {
        self.status().is_server_error()
    }
}
