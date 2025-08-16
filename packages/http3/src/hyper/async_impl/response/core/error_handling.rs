//! Error conversion functionality
//!
//! This module contains error handling and conversion methods for Response.

use super::types::Response;

impl Response {
    /// Convert response into an error if the status indicates failure
    pub fn error_for_status(self) -> Result<Self, crate::Error> {
        if self.status().is_client_error() || self.status().is_server_error() {
            Err(crate::Error::status_code(Some(self.status())))
        } else {
            Ok(self)
        }
    }

    /// Convert response reference into an error if the status indicates failure
    pub fn error_for_status_ref(&self) -> Result<&Self, crate::Error> {
        if self.status().is_client_error() || self.status().is_server_error() {
            Err(crate::Error::status_code(Some(self.status())))
        } else {
            Ok(self)
        }
    }
}
