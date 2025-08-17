//! Core type definitions and re-exports

pub use std::collections::HashMap;
pub use std::time::Duration;

pub use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Version};
pub use url::Url;

/// HTTP protocol version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpVersion {
    Http1_1,
    Http2,
    Http3,
}

impl Default for HttpVersion {
    fn default() -> Self {
        Self::Http3
    }
}

/// Connection timeout configuration
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    pub connect: Duration,
    pub read: Duration,
    pub write: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(30),
            read: Duration::from_secs(60),
            write: Duration::from_secs(60),
        }
    }
}

/// Request/response metadata
#[derive(Debug, Clone, Default)]
pub struct RequestMetadata {
    pub start_time: Option<std::time::Instant>,
    pub end_time: Option<std::time::Instant>,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub redirects: u32,
}
