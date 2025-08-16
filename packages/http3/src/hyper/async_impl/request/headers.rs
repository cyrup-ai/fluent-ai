use std::convert::TryFrom;

use super::types::RequestBuilder;
use http::header::{HeaderMap, HeaderName, HeaderValue};

impl RequestBuilder {
    /// Add a set of Headers to the existing ones on this Request.
    ///
    /// The headers will be merged in to any already set.
    pub fn headers(mut self, headers: http::HeaderMap) -> RequestBuilder {
        if let Ok(ref mut req) = self.request {
            crate::util::replace_headers(req.headers_mut(), headers);
        }
        self
    }

    /// Set the `User-Agent` header to be used by this request.
    pub fn user_agent<V>(self, value: V) -> RequestBuilder
    where
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.header(http::header::USER_AGENT, value)
    }

    /// Set the `Accept` header to be used by this request.
    pub fn accept<V>(self, value: V) -> RequestBuilder
    where
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.header(http::header::ACCEPT, value)
    }

    /// Set the `Accept-Encoding` header to be used by this request.
    pub fn accept_encoding<V>(self, value: V) -> RequestBuilder
    where
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.header(http::header::ACCEPT_ENCODING, value)
    }

    /// Set the `Content-Type` header to be used by this request.
    pub fn content_type<V>(self, value: V) -> RequestBuilder
    where
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.header(http::header::CONTENT_TYPE, value)
    }

    /// Set the `Content-Length` header to be used by this request.
    pub fn content_length(self, len: u64) -> RequestBuilder {
        self.header(http::header::CONTENT_LENGTH, len.to_string())
    }

    /// Set the `Cache-Control` header to be used by this request.
    pub fn cache_control<V>(self, value: V) -> RequestBuilder
    where
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.header(http::header::CACHE_CONTROL, value)
    }

    /// Set the `If-None-Match` header to be used by this request.
    pub fn if_none_match<V>(self, value: V) -> RequestBuilder
    where
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.header(http::header::IF_NONE_MATCH, value)
    }

    /// Set the `If-Modified-Since` header to be used by this request.
    pub fn if_modified_since<V>(self, value: V) -> RequestBuilder
    where
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.header(http::header::IF_MODIFIED_SINCE, value)
    }

    /// Set the `Range` header to be used by this request.
    pub fn range<V>(self, value: V) -> RequestBuilder
    where
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        self.header(http::header::RANGE, value)
    }

    /// Set a custom header to be used by this request.
    ///
    /// This is a convenience method that allows setting headers with string keys.
    pub fn header_str(self, key: &str, value: &str) -> RequestBuilder {
        self.header(key, value)
    }

    /// Remove a header from this request.
    pub fn remove_header<K>(mut self, key: K) -> RequestBuilder
    where
        HeaderName: TryFrom<K>,
        <HeaderName as TryFrom<K>>::Error: Into<http::Error>,
    {
        if let Ok(ref mut req) = self.request {
            if let Ok(key) = <HeaderName as TryFrom<K>>::try_from(key) {
                req.headers_mut().remove(&key);
            }
        }
        self
    }

    /// Clear all headers from this request.
    pub fn clear_headers(mut self) -> RequestBuilder {
        if let Ok(ref mut req) = self.request {
            req.headers_mut().clear();
        }
        self
    }

    /// Check if a header exists in this request.
    pub fn has_header<K>(&self, key: K) -> bool
    where
        HeaderName: TryFrom<K>,
        <HeaderName as TryFrom<K>>::Error: Into<http::Error>,
    {
        if let Ok(ref req) = self.request {
            if let Ok(key) = <HeaderName as TryFrom<K>>::try_from(key) {
                return req.headers().contains_key(&key);
            }
        }
        false
    }

    /// Get the value of a header from this request.
    pub fn get_header<K>(&self, key: K) -> Option<&HeaderValue>
    where
        HeaderName: TryFrom<K>,
        <HeaderName as TryFrom<K>>::Error: Into<http::Error>,
    {
        if let Ok(ref req) = self.request {
            if let Ok(key) = <HeaderName as TryFrom<K>>::try_from(key) {
                return req.headers().get(&key);
            }
        }
        None
    }
}
