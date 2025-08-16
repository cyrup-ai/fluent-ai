//! Utility methods and error handling for HTTP Response
//!
//! This module contains utility methods for response processing including
//! error handling and status code validation.

use super::types::Response;

impl Response {
    /// Turn a response into an error if the server returned an error.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::hyper::Response;
    /// fn on_response(res: Response) {
    ///     match res.error_for_status() {
    ///         Ok(_res) => (),
    ///         Err(err) => {
    ///             // asserting a 400 as an example
    ///             // it could be any status between 400...599
    ///             assert_eq!(
    ///                 err.status(),
    ///                 Some(crate::hyper::StatusCode::BAD_REQUEST)
    ///             );
    ///         }
    ///     }
    /// }
    /// # fn main() {}
    /// ```
    pub fn error_for_status(self) -> crate::Result<Self> {
        let status = self.status();
        let reason = self.extensions().get::<hyper::ext::ReasonPhrase>().cloned();
        if status.is_client_error() || status.is_server_error() {
            Err(crate::HttpError::HttpStatus {
                status: status.as_u16(),
                message: format!(
                    "HTTP error at {}: {}",
                    self.url,
                    reason
                        .map(|r| format!("{:?}", r))
                        .unwrap_or_else(|| "Unknown".to_string())
                ),
                body: String::new(),
            })
        } else {
            Ok(self)
        }
    }

    /// Turn a reference to a response into an error if the server returned an error.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::hyper::Response;
    /// fn on_response(res: &Response) {
    ///     match res.error_for_status_ref() {
    ///         Ok(_res) => (),
    ///         Err(err) => {
    ///             // asserting a 400 as an example
    ///             // it could be any status between 400...599
    ///             assert_eq!(
    ///                 err.status(),
    ///                 Some(crate::hyper::StatusCode::BAD_REQUEST)
    ///             );
    ///         }
    ///     }
    /// }
    /// # fn main() {}
    /// ```
    pub fn error_for_status_ref(&self) -> crate::Result<&Self> {
        let status = self.status();
        let reason = self.extensions().get::<hyper::ext::ReasonPhrase>().cloned();
        if status.is_client_error() || status.is_server_error() {
            Err(crate::Error::from(crate::hyper::error::status_code(
                *self.url.clone(),
                status,
                reason,
            )))
        } else {
            Ok(self)
        }
    }
}
