//! Cache integration with HTTP client and global cache instance
//!
//! Provides global cache instance and cache-aware streaming functions
//! for seamless integration with the HTTP client.

use std::{collections::HashMap, pin::Pin};

use futures_util::StreamExt;
use futures_util::stream::{self, Stream};

use super::{cache_key::CacheKey, response_cache::ResponseCache};
use crate::HttpResponse;
use crate::HttpResult;

lazy_static::lazy_static! {
    /// Global cache instance for use across the HTTP client
    pub static ref GLOBAL_CACHE: ResponseCache = ResponseCache::default();
}

/// Cache-aware HTTP stream that checks cache before making requests using native Streams
pub fn cached_stream<F>(
    cache_key: CacheKey,
    operation: F,
) -> Pin<Box<dyn Stream<Item = HttpResult<HttpResponse>> + Send>>
where
    F: Fn() -> Pin<Box<dyn Stream<Item = HttpResult<HttpResponse>> + Send>> + Send + Sync + 'static,
{
    Box::pin(stream::unfold(
        (Some(cache_key), operation, false),
        move |(cache_key_opt, operation, cache_checked)| async move {
            if let Some(cache_key) = cache_key_opt {
                if !cache_checked {
                    // Check cache first
                    if let Some(cached_response) = GLOBAL_CACHE.get(&cache_key) {
                        return Some((Ok(cached_response), (None, operation, true)));
                    }

                    // Cache miss - execute operation
                    let mut operation_stream = operation();
                    if let Some(result) = operation_stream.next().await {
                        match result {
                            Ok(response) => {
                                // Check if response should be cached
                                if GLOBAL_CACHE.should_cache(&response) {
                                    GLOBAL_CACHE.put(cache_key.clone(), response.clone());
                                }
                                Some((Ok(response), (None, operation, true)))
                            }
                            Err(error) => Some((Err(error), (None, operation, true)))}
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        },
    ))
}

/// Helper to create conditional request headers for cache validation
pub fn conditional_headers_for_key(cache_key: &CacheKey) -> HashMap<String, String> {
    GLOBAL_CACHE
        .get_validation_headers(cache_key)
        .unwrap_or_default()
}
