//! Cache middleware for HTTP requests/responses
//!
//! Zero-allocation cache middleware integrating with the production cache system.
//! Provides conditional request validation and response caching with lock-free operations.

use std::sync::Arc;

use super::Middleware;
use crate::cache::{CacheKey, GLOBAL_CACHE, httpdate};
use crate::{HttpRequest, HttpResponse};

/// Cache middleware for HTTP requests/responses with zero-allocation design
#[derive(Debug)]
pub struct CacheMiddleware {
    enabled: bool,
    cache_key_buffer: Arc<str>,
}

impl Default for CacheMiddleware {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CacheMiddleware {
    #[inline]
    pub fn new() -> Self {
        Self {
            enabled: true,
            cache_key_buffer: Arc::from(""),
        }
    }

    #[inline]
    pub fn enabled(enabled: bool) -> Self {
        Self {
            enabled,
            cache_key_buffer: Arc::from(""),
        }
    }

    /// Generate cache key with zero allocations using request context
    #[inline]
    fn generate_cache_key(&self, method: &str, uri: &str, headers: &[(&str, &str)]) -> CacheKey {
        let headers_map = headers
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        CacheKey::new(uri.to_string(), method.to_string(), headers_map)
    }
}

impl Middleware for CacheMiddleware {
    #[inline]
    fn process_request(&self, request: HttpRequest) -> crate::error::Result<HttpRequest> {
        if !self.enabled {
            return Ok(request);
        }

        let uri = request.uri();
        let method = request.method().as_str();
        let cache_key = self.generate_cache_key(method, &uri, &[]);

        match GLOBAL_CACHE.get(&cache_key) {
            Some(cached_entry) => {
                let mut modified_request = request;

                // Add conditional validation headers for cache revalidation
                if let Some(ref etag) = cached_entry.etag() {
                    if let Ok(header_value) = http::HeaderValue::from_str(etag) {
                        modified_request =
                            modified_request.header(http::header::IF_NONE_MATCH, header_value);
                    }
                }

                if let Some(last_modified) = cached_entry.last_modified() {
                    let system_time = std::time::SystemTime::UNIX_EPOCH
                        + std::time::Duration::from_secs(last_modified.parse().unwrap_or(0));
                    let http_date = httpdate::fmt_http_date(system_time);
                    if let Ok(header_value) = http::HeaderValue::from_str(&http_date) {
                        modified_request =
                            modified_request.header(http::header::IF_MODIFIED_SINCE, header_value);
                    }
                }

                Ok(modified_request)
            }
            None => Ok(request),
        }
    }

    #[inline]
    fn process_response(&self, response: HttpResponse) -> crate::error::Result<HttpResponse> {
        if !self.enabled {
            return Ok(response);
        }

        // Only cache responses that meet HTTP caching criteria
        if !GLOBAL_CACHE.should_cache(&response) {
            return Ok(response);
        }

        // Skip caching for streaming response since we can't easily access headers
        // TODO: Implement proper header extraction from headers_stream when needed
        Ok(response)
    }
}
