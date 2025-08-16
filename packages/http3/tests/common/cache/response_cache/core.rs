//! Tests for response cache core functionality
//!
//! Tests for ResponseCache creation, configuration, and basic operations

use std::sync::atomic::Ordering;
use fluent_ai_http3::common::cache::response_cache::core::{ResponseCache, CacheConfig};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_cache_new() {
        let config = CacheConfig::default();
        let cache = ResponseCache::new(config);

        let (entries, memory, _) = cache.size_info();
        assert_eq!(entries, 0);
        assert_eq!(memory, 0);
    }

    #[test]
    fn test_response_cache_default() {
        let cache = ResponseCache::default();
        let stats = cache.stats();

        assert_eq!(stats.hits.load(Ordering::Relaxed), 0);
        assert_eq!(stats.misses.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_response_cache_clear() {
        let cache = ResponseCache::default();
        cache.clear();

        let (entries, memory, _) = cache.size_info();
        assert_eq!(entries, 0);
        assert_eq!(memory, 0);
    }
}