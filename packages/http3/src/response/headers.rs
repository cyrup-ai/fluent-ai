//! HTTP response header management and caching utilities
//!
//! Provides methods for accessing response headers, content type detection,
//! and cache-related functionality with zero-allocation design.

use crate::response::core::HttpResponse;

impl HttpResponse {
    /// Get header value by name (case-insensitive)
    #[must_use]
    pub fn header(&self, key: &str) -> Option<&String> {
        &self.headers.get(key)
    }

    /// Get `Content-Type` header value
    #[must_use]
    pub fn content_type(&self) -> Option<&String> {
        self.header("content-type")
    }

    /// Check if response has JSON content type
    #[must_use]
    pub fn is_json_content(&self) -> bool {
        self.content_type()
            .is_some_and(|ct| ct.contains("application/json") || ct.contains("text/json"))
    }

    /// Get `ETag` header value
    #[must_use]
    pub fn etag(&self) -> Option<&String> {
        self.header("etag")
    }

    /// Get `Last-Modified` header value
    #[must_use]
    pub fn last_modified(&self) -> Option<&String> {
        self.header("last-modified")
    }

    /// Get `Cache-Control` header value
    #[must_use]
    pub fn cache_control(&self) -> Option<&String> {
        self.header("cache-control")
    }

    /// Get content length
    #[must_use]
    pub fn content_length(&self) -> Option<u64> {
        self.header("content-length").and_then(|v| v.parse().ok())
    }

    /// Get `Expires` header value
    #[must_use]
    pub fn expires(&self) -> Option<&String> {
        self.header("expires")
    }

    /// Get computed expires timestamp (Unix timestamp)
    /// This is set by `CacheMiddleware` and represents the effective cache expiration
    #[must_use]
    pub fn computed_expires(&self) -> Option<u64> {
        self.header("x-computed-expires")
            .and_then(|v| v.parse().ok())
    }

    /// Check if response is cacheable based on computed expires
    #[must_use]
    pub fn is_cacheable(&self) -> bool {
        self.computed_expires().is_some() && self.is_success()
    }

    /// Get time until expires in seconds
    #[must_use]
    pub fn seconds_until_expires(&self) -> Option<u64> {
        self.computed_expires().and_then(|expires| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_secs();

            if expires > now {
                Some(expires - now)
            } else {
                Some(0) // Already expired
            }
        })
    }

    /// Get Server header value
    #[must_use]
    pub fn server(&self) -> Option<&String> {
        self.header("server")
    }

    /// Get Date header value
    #[must_use]
    pub fn date(&self) -> Option<&String> {
        self.header("date")
    }

    /// Get Location header value (for redirects)
    #[must_use]
    pub fn location(&self) -> Option<&String> {
        self.header("location")
    }
}
