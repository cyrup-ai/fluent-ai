//! Cache middleware for HTTP requests/responses
//!
//! Zero-allocation cache middleware integrating with the production cache system.
//! Provides conditional request validation and response caching with lock-free operations.

use std::sync::Arc;

use super::Middleware;
use crate::cache::{CacheKey, GLOBAL_CACHE, httpdate};
use crate::{HttpRequest, HttpResponse, HttpResult};

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
        CacheKey::new(method, uri, headers)
    }
}

impl Middleware for CacheMiddleware {
    #[inline]
    fn process_request(&self, request: HttpRequest) -> HttpResult<HttpRequest> {
        if !self.enabled {
            return HttpResult::Ok(request);
        }

        let uri = request.uri();
        let method = request.method().as_str();
        let cache_key = self.generate_cache_key(method, uri.path(), &[]);

        match GLOBAL_CACHE.get(&cache_key) {
            Some(cached_entry) => {
                let mut modified_request = request;

                // Add conditional validation headers for cache revalidation
                if let Some(ref etag) = cached_entry.etag {
                    if let Ok(header_value) = http::HeaderValue::from_str(etag) {
                        modified_request =
                            modified_request.header(http::header::IF_NONE_MATCH, header_value);
                    }
                }

                if let Some(last_modified) = cached_entry.last_modified {
                    match httpdate::format_http_date(last_modified) {
                        Ok(http_date) => {
                            if let Ok(header_value) = http::HeaderValue::from_str(&http_date) {
                                modified_request = modified_request
                                    .header(http::header::IF_MODIFIED_SINCE, header_value);
                            }
                        }
                        Err(_) => {
                            // Invalid date format, skip header
                        }
                    }
                }

                HttpResult::Ok(modified_request)
            }
            None => HttpResult::Ok(request),
        }
    }

    #[inline]
    fn process_response(&self, response: HttpResponse) -> HttpResult<HttpResponse> {
        if !self.enabled {
            return HttpResult::Ok(response);
        }

        // Only cache responses that meet HTTP caching criteria
        if !GLOBAL_CACHE.should_cache(&response) {
            return HttpResult::Ok(response);
        }

        // Extract request context from response headers (set by client)
        let uri_path = match response.headers().get("x-request-uri") {
            Some(header_value) => match header_value.to_str() {
                Ok(uri_str) => uri_str,
                Err(_) => return HttpResult::Ok(response), // Invalid UTF-8, skip caching
            },
            None => return HttpResult::Ok(response), // No URI context, skip caching
        };

        let method = match response.headers().get("x-request-method") {
            Some(header_value) => match header_value.to_str() {
                Ok(method_str) => method_str,
                Err(_) => "GET", // Default to GET on invalid UTF-8
            },
            None => "GET", // Default method
        };

        // Generate cache key and store response
        let cache_key = self.generate_cache_key(method, uri_path, &[]);
        GLOBAL_CACHE.put(cache_key, response.clone());

        HttpResult::Ok(response)
    }
}
