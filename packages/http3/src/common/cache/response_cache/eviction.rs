//! Cache eviction and cleanup operations
//!
//! LRU eviction, expired entry cleanup, and memory management operations
//! using lock-free patterns for concurrent cache maintenance.

use std::{sync::atomic::Ordering, time::Instant};

use super::core::ResponseCache;

impl ResponseCache {
    /// Evict least recently used entries to free space
    /// Returns the number of entries actually evicted
    pub(super) fn evict_lru_entries(&self) -> u32 {
        let mut candidates: Vec<(String, Instant, u64)> = Vec::new();

        // Collect candidates for eviction (key, last_accessed, size)
        for entry_ref in self.entries.iter() {
            let key = entry_ref.key().clone();
            let entry = entry_ref.value();
            candidates.push((key, entry.last_accessed, entry.size_bytes));
        }

        // Sort by last accessed (oldest first)
        candidates.sort_by_key(|(_, last_accessed, _)| *last_accessed);

        // Evict oldest 25% of entries or until under limits
        let target_evictions = (candidates.len() / 4).max(1);
        let mut evicted_count = 0;

        for (key, _, size) in candidates.iter().take(target_evictions) {
            if let Some(_) = self.entries.remove(key) {
                self.entry_count.fetch_sub(1, Ordering::Relaxed);
                self.memory_usage.fetch_sub(*size, Ordering::Relaxed);
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                evicted_count += 1;

                // Stop if under limits
                let current_memory = self.memory_usage.load(Ordering::Relaxed);
                let current_entries = self.entry_count.load(Ordering::Relaxed) as usize;

                if current_memory < self.config.max_memory_bytes
                    && current_entries < self.config.max_entries
                {
                    break;
                }
            }
        }

        evicted_count
    }

    /// Clean up expired entries
    pub fn cleanup_expired(&self) {
        if !self
            .cleanup_running
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return; // Cleanup already running
        }

        let mut expired_keys = Vec::new();

        for entry_ref in self.entries.iter() {
            if entry_ref.value().is_expired() {
                expired_keys.push(entry_ref.key().clone());
            }
        }

        for key in expired_keys {
            if let Some(entry_ref) = self.entries.get(&key) {
                let size_bytes = entry_ref.value().size_bytes;
                self.entries.remove(&key);
                self.entry_count.fetch_sub(1, Ordering::Relaxed);
                self.memory_usage.fetch_sub(size_bytes, Ordering::Relaxed);
            }
        }

        self.cleanup_running.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use http::{Response, StatusCode};

    use super::super::super::{cache_config::CacheConfig, cache_entry::CacheEntry};
    use super::*;
    use crate::HttpResponse;

    fn create_test_response() -> HttpResponse {
        Response::builder().status(StatusCode::OK).body(()).unwrap()
    }

    #[test]
    fn test_cleanup_expired_no_entries() {
        let cache = ResponseCache::default();
        cache.cleanup_expired();

        let (entries, _, _) = cache.size_info();
        assert_eq!(entries, 0);
    }

    #[test]
    fn test_evict_lru_entries_empty_cache() {
        let cache = ResponseCache::default();
        let evicted = cache.evict_lru_entries();

        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_cleanup_running_flag() {
        let cache = ResponseCache::default();

        // Set cleanup running flag manually for test
        cache.cleanup_running.store(true, Ordering::Relaxed);

        // This should return early due to flag
        cache.cleanup_expired();

        // Reset flag
        cache.cleanup_running.store(false, Ordering::Relaxed);
    }
}
