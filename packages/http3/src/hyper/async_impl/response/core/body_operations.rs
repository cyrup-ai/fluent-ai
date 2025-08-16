//! Body-related utilities
//!
//! This module contains methods for working with response bodies and content.

use super::types::Response;

impl Response {
    /// Get the content length of the response, if it is known.
    ///
    /// This value does not directly represents the value of the `Content-Length`
    /// header, but rather the size of the response's body. To read the header's
    /// value, please use the [`Response::headers`] method instead.
    ///
    /// Reasons it may not be known:
    ///
    /// - The response does not include a body (e.g. it responds to a `HEAD`
    ///   request).
    /// - The response is gzipped and automatically decoded (thus changing the
    ///   actual decoded length).
    pub fn content_length(&self) -> Option<u64> {
        // Get content-length from headers since Decoder doesn't implement Body
        self.headers()
            .get("content-length")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok())
    }

    /// Retrieve the cookies contained in the response.
    ///
    /// Note that invalid 'Set-Cookie' headers will be ignored.
    ///
    /// # Optional
    ///
    /// This requires the optional `cookies` feature to be enabled.
    #[cfg(feature = "cookies")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cookies")))]
    pub fn cookies<'a>(&'a self) -> impl Iterator<Item = crate::cookie::Cookie<'a>> + 'a {
        crate::cookie::extract_response_cookies(self.res.headers()).filter_map(Result::ok)
    }
}
