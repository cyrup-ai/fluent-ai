//! Cache operations for get, put, and validation
//!
//! Core caching operations with HTTP semantics, cache-control handling,
//! and conditional request validation using zero-allocation patterns.

use std::{collections::HashMap, sync::atomic::Ordering};

use super::super::{cache_entry::CacheEntry, cache_key::CacheKey, http_date::httpdate};
use super::core::ResponseCache;
use crate::HttpResponse;

impl ResponseCache {
    /// Get cached response if available and valid
    pub fn get(&self, key: &CacheKey) -> Option<HttpResponse> {
        let hash_key = key.hash_key();

        if let Some(entry_ref) = self.entries.get(&hash_key) {
            let mut entry = entry_ref.value().clone();

            // Check if expired
            if entry.is_expired() {
                // Remove expired entry
                self.entries.remove(&hash_key);
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                self.entry_count.fetch_sub(1, Ordering::Relaxed);
                self.memory_usage
                    .fetch_sub(entry.size_bytes, Ordering::Relaxed);
                return None;
            }

            // Record hit and update access time
            entry.record_hit();
            self.stats.hits.fetch_add(1, Ordering::Relaxed);

            // Update entry in cache with new access time
            self.entries.insert(hash_key, entry.clone());

            Some(entry.response)
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Store response in cache
    pub fn put(&self, key: CacheKey, response: HttpResponse) {
        if self.config.max_entries == 0 {
            return; // Caching disabled
        }

        let entry = CacheEntry::new(response);
        let hash_key = key.hash_key();

        // Check memory limits
        let current_memory = self.memory_usage.load(Ordering::Relaxed);
        if current_memory + entry.size_bytes > self.config.max_memory_bytes {
            let evicted = self.evict_lru_entries();
            if evicted > 0 {
                eprintln!("Cache evicted {} entries due to memory limit", evicted);
            }
        }

        // Check entry count limits
        let current_entries = self.entry_count.load(Ordering::Relaxed) as usize;
        if current_entries >= self.config.max_entries {
            let evicted = self.evict_lru_entries();
            if evicted > 0 {
                eprintln!("Cache evicted {} entries due to count limit", evicted);
            }
        }

        // Check if entry already exists
        let had_existing = self.entries.contains_key(&hash_key);
        let old_size = if had_existing {
            // Get existing entry size before replacement
            self.entries
                .get(&hash_key)
                .map(|e| e.value().size_bytes)
                .unwrap_or(0)
        } else {
            0
        };

        // Insert new entry (always succeeds)
        self.entries.insert(hash_key, entry.clone());

        if !had_existing {
            // New entry
            self.entry_count.fetch_add(1, Ordering::Relaxed);
            self.memory_usage
                .fetch_add(entry.size_bytes, Ordering::Relaxed);
        } else {
            // Replaced existing entry - adjust memory usage
            self.memory_usage
                .fetch_add(entry.size_bytes, Ordering::Relaxed);
            self.memory_usage.fetch_sub(old_size, Ordering::Relaxed);
        }
    }

    /// Check if response should be cached
    pub fn should_cache(&self, response: &HttpResponse) -> bool {
        // Don't cache error responses
        if response.status().is_client_error() || response.status().is_server_error() {
            return false;
        }

        // Check Cache-Control directives
        if let Some(cache_control) = response.headers().get("cache-control") {
            let cache_control = cache_control.to_str().unwrap_or("").to_lowercase();

            // Explicit no-cache or no-store
            if cache_control.contains("no-cache") || cache_control.contains("no-store") {
                return false;
            }

            // Private responses shouldn't be cached
            if cache_control.contains("private") {
                return false;
            }
        }

        // Check for ETag or Last-Modified for validation
        let has_validators = response.headers().contains_key("etag")
            || response.headers().contains_key("last-modified");

        // Check for explicit expiration
        let has_expiration = response.headers().contains_key("expires")
            || response
                .headers()
                .get("cache-control")
                .and_then(|cc| cc.to_str().ok())
                .map_or(false, |cc| cc.contains("max-age"));

        // Cache if it has validators or expiration info
        has_validators || has_expiration
    }

    /// Check if entry exists and get validation headers
    pub fn get_validation_headers(&self, key: &CacheKey) -> Option<HashMap<String, String>> {
        let hash_key = key.hash_key();

        if let Some(entry_ref) = self.entries.get(&hash_key) {
            let entry = entry_ref.value();
            let mut headers = HashMap::new();

            if let Some(etag) = &entry.etag {
                headers.insert("If-None-Match".to_string(), etag.clone());
            }

            if let Some(last_modified) = entry.last_modified {
                headers.insert(
                    "If-Modified-Since".to_string(),
                    httpdate::fmt_http_date(last_modified),
                );
            }

            if !headers.is_empty() {
                self.stats.validations.fetch_add(1, Ordering::Relaxed);
                Some(headers)
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use http::{Response, StatusCode};

    use super::*;

    fn create_test_response() -> HttpResponse {
        Response::builder()
            .status(StatusCode::OK)
            .header("etag", "\"test-etag\"")
            .body(())
            .unwrap()
    }

    #[test]
    fn test_should_cache_with_etag() {
        let cache = ResponseCache::default();
        let response = create_test_response();

        assert!(cache.should_cache(&response));
    }

    #[test]
    fn test_should_not_cache_error() {
        let cache = ResponseCache::default();
        let response = Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(())
            .unwrap();

        assert!(!cache.should_cache(&response));
    }

    #[test]
    fn test_should_not_cache_no_store() {
        let cache = ResponseCache::default();
        let response = Response::builder()
            .status(StatusCode::OK)
            .header("cache-control", "no-store")
            .body(())
            .unwrap();

        assert!(!cache.should_cache(&response));
    }
}
