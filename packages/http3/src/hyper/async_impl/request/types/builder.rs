//! RequestBuilder implementation with builder pattern
//!
//! Provides the RequestBuilder struct for constructing HTTP requests with a fluent API.

use std::convert::TryFrom;

use super::super::super::client::Client;
use super::request::Request;
use http::header::{HeaderName, HeaderValue};

/// A builder to construct the properties of a `Request`.
///
/// To construct a `RequestBuilder`, refer to the `Client` documentation.
#[must_use = "RequestBuilder does nothing until you 'send' it"]
pub struct RequestBuilder {
    pub(super) client: Client,
    pub(super) request: crate::Result<Request>,
}

impl RequestBuilder {
    pub(super) fn new(client: Client, request: crate::Result<Request>) -> RequestBuilder {
        RequestBuilder { client, request }
    }

    /// Add a `Header` to this Request.
    pub fn header<K, V>(mut self, key: K, value: V) -> RequestBuilder
    where
        HeaderName: TryFrom<K>,
        <HeaderName as TryFrom<K>>::Error: Into<http::Error>,
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.header_sensitive(key, value, false)
    }

    /// Add a `Header` to this Request with ability to define if `header_value` is sensitive.
    pub(super) fn header_sensitive<K, V>(
        mut self,
        key: K,
        value: V,
        sensitive: bool,
    ) -> RequestBuilder
    where
        HeaderName: TryFrom<K>,
        <HeaderName as TryFrom<K>>::Error: Into<http::Error>,
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        let mut error = None;
        if let Ok(ref mut req) = self.request {
            match <HeaderName as TryFrom<K>>::try_from(key) {
                Ok(key) => match <HeaderValue as TryFrom<V>>::try_from(value) {
                    Ok(mut value) => {
                        // We want to potentially make an non-sensitive header
                        // to be sensitive, not the reverse. So, don't turn off
                        // a previously sensitive header.
                        if sensitive {
                            value.set_sensitive(true);
                        }
                        req.headers_mut().append(key, value);
                    }
                    Err(_) => {
                        error = Some(crate::HttpError::builder("Header value error".to_string()))
                    }
                },
                Err(_) => error = Some(crate::HttpError::builder("Header name error".to_string())),
            };
        }
        if let Some(err) = error {
            self.request = Err(err);
        }
        self
    }
}
