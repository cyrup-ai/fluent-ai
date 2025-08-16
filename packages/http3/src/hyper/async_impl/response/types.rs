//! Type definitions and data structures for HTTP Response
//!
//! This module contains the core data structures used by the HTTP response
//! implementation including the main Response struct and helper types.

use std::fmt;
use std::net::SocketAddr;
use std::pin::Pin;
use std::time::Duration;

use bytes::Bytes;
use fluent_ai_async::prelude::MessageChunk;
use hyper::rt::Sleep;
use hyper::{HeaderMap, StatusCode, Version};
use hyper_util::client::legacy::connect::HttpInfo;
#[cfg(feature = "json")]
use serde::de::DeserializeOwned;
use url::Url;

use super::super::body::{Body, ResponseBody};
use super::super::decoder::{Accepts, Decoder};

/// String wrapper to implement MessageChunk for fluent_ai_async patterns
#[derive(Debug, Clone, Default)]
pub struct StringChunk(pub String);

impl MessageChunk for StringChunk {
    fn bad_chunk(error: String) -> Self {
        StringChunk(format!("ERROR: {}", error))
    }

    fn is_error(&self) -> bool {
        self.0.starts_with("ERROR:")
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some(&self.0[7..]) // Skip "ERROR: " prefix
        } else {
            None
        }
    }
}

impl From<String> for StringChunk {
    fn from(s: String) -> Self {
        StringChunk(s)
    }
}

impl From<StringChunk> for String {
    fn from(chunk: StringChunk) -> Self {
        chunk.0
    }
}

/// A Response to a submitted `Request`.
pub struct Response {
    pub(super) res: hyper::Response<Decoder>,
    // Boxed to save space (11 words to 1 word), and it's not accessed
    // frequently internally.
    pub(super) url: Box<Url>,
    #[cfg(feature = "cookies")]
    pub(super) cookie_jar: Option<crate::cookie::CookieJar>,
}

impl fmt::Debug for Response {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Response")
            .field("url", &self.url().as_str())
            .field("status", &self.status())
            .field("headers", self.headers())
            .finish()
    }
}
