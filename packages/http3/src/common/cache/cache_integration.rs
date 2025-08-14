//! Cache integration with HTTP client and global cache instance
//!
//! Provides global cache instance and cache-aware streaming functions
//! for seamless integration with the HTTP client.

use std::collections::HashMap;

use fluent_ai_async::{AsyncStream, emit};

use super::{cache_key::CacheKey, response_cache::ResponseCache};
use crate::HttpResponse;
use crate::HttpResult;

lazy_static::lazy_static! {
    /// Global cache instance for use across the HTTP client
    pub static ref GLOBAL_CACHE: ResponseCache = ResponseCache::default();
}

/// Cache-aware HTTP stream that checks cache before making requests using AsyncStream
pub fn cached_stream<F>(cache_key: CacheKey, operation: F) -> AsyncStream<HttpResult<HttpResponse>>
where
    F: Fn() -> HttpResult<HttpResponse> + Send + Sync + 'static,
{
    AsyncStream::with_channel(move |sender| {
        // Check cache first
        if let Some(cached_response) = GLOBAL_CACHE.get(&cache_key) {
            emit!(sender, Ok(cached_response));
            return;
        }

        // Cache miss - execute operation
        match operation() {
            Ok(response) => {
                // Check if response should be cached
                if GLOBAL_CACHE.should_cache(&response) {
                    GLOBAL_CACHE.put(cache_key, response.clone());
                }
                emit!(sender, Ok(response));
            }
            Err(error) => {
                emit!(sender, Err(error));
            }
        }
    })
}

/// Helper to create conditional request headers for cache validation
pub fn conditional_headers_for_key(cache_key: &CacheKey) -> HashMap<String, String> {
    GLOBAL_CACHE
        .get_validation_headers(cache_key)
        .unwrap_or_default()
}
