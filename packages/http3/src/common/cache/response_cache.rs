//! Main HTTP response cache implementation with lock-free storage
//!
//! Provides ResponseCache using crossbeam SkipMap for concurrent operations
//! with LRU eviction, TTL expiration, and memory management.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant};

use crossbeam_skiplist::SkipMap;

use super::{
    cache_config::CacheConfig, cache_entry::CacheEntry, cache_key::CacheKey,
    cache_stats::CacheStats, http_date::httpdate};
use crate::HttpResponse;

/// Lock-free HTTP response cache using crossbeam skiplist
pub struct ResponseCache {
    /// Main cache storage (key -> entry)
    entries: SkipMap<String, CacheEntry>,
    /// Configuration
    config: CacheConfig,
    /// Current memory usage estimate
    memory_usage: AtomicU64,
    /// Entry count
    entry_count: AtomicU64,
    /// Cache statistics
    stats: CacheStats,
    /// Cleanup task running flag
    cleanup_running: AtomicBool}

impl ResponseCache {
    /// Create new response cache with configuration
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: SkipMap::new(),
            config,
            memory_usage: AtomicU64::new(0),
            entry_count: AtomicU64::new(0),
            stats: CacheStats::default(),
            cleanup_running: AtomicBool::new(false)}
    }

    /// Create cache with default configuration
    pub fn default() -> Self {
        Self::new(CacheConfig::default())
    }

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
            let cache_control = cache_control.to_lowercase();

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
                .map_or(false, |cc| cc.contains("max-age"));

        // Cache if it has validators or expiration info
        has_validators || has_expiration
    }

    /// Evict least recently used entries to free space
    /// Returns the number of entries actually evicted
    fn evict_lru_entries(&self) -> u32 {
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

    /// Clear all cached entries
    pub fn clear(&self) {
        self.entries.clear();
        self.entry_count.store(0, Ordering::Relaxed);
        self.memory_usage.store(0, Ordering::Relaxed);
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get current cache size information
    pub fn size_info(&self) -> (usize, u64, f64) {
        let entries = self.entry_count.load(Ordering::Relaxed) as usize;
        let memory = self.memory_usage.load(Ordering::Relaxed);
        let memory_pct = (memory as f64 / self.config.max_memory_bytes as f64) * 100.0;

        (entries, memory, memory_pct)
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
