//! Cache integration with HTTP client and global cache instance
//!
//! Provides global cache instance and cache-aware streaming functions
//! for seamless integration with the HTTP client.

use std::collections::HashMap;

use fluent_ai_async::{AsyncStream, prelude::MessageChunk};

use super::{cache_key::CacheKey, response_cache::ResponseCache};
use crate::prelude::*;

lazy_static::lazy_static! {
    /// Global cache instance for use across the HTTP client
    pub static ref GLOBAL_CACHE: ResponseCache = ResponseCache::default();
}

/// Cache-aware HTTP stream that checks cache before making requests using AsyncStream
pub fn cached_stream<F>(cache_key: CacheKey, operation: F) -> AsyncStream<HttpResponse, 1024>
where
    F: Fn() -> AsyncStream<HttpResponse, 1024> + Send + Sync + 'static,
{
    AsyncStream::with_channel(move |sender| {
        // Check cache first
        if let Some(cached_response) = GLOBAL_CACHE.get(&cache_key) {
            fluent_ai_async::emit!(sender, cached_response);
            return;
        }

        // Cache miss - execute operation stream
        let operation_stream = operation();
        for response in operation_stream {
            if response.is_error() {
                // Forward error chunks as-is
                fluent_ai_async::emit!(sender, response);
            } else {
                // Check if response should be cached
                // Note: Clone not available for streaming HttpResponse - cache integration needs redesign
                // For now, just forward the response without caching
                // TODO: Implement proper cache integration for streaming responses
                fluent_ai_async::emit!(sender, response);
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
