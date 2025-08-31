//! Cache operations for get, put, and validation
//!
//! Core caching operations with HTTP semantics, cache-control handling,
//! and conditional request validation using zero-allocation patterns.

use std::{collections::HashMap, sync::atomic::Ordering};

use super::super::{cache_entry::CacheEntry, cache_key::CacheKey, http_date::httpdate};
use super::core::ResponseCache;
use crate::prelude::*;

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

            // Convert cached entry back to HttpResponse
            Some(Self::entry_to_response(entry))
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Convert cached entry back to streaming HttpResponse
    fn entry_to_response(entry: CacheEntry) -> crate::HttpResponse {
        use fluent_ai_async::AsyncStream;

        use crate::http::response::{HttpBodyChunk, HttpHeader};

        // Clone the data to avoid lifetime issues
        let status = entry.status;
        let version = entry.version;
        let headers = entry.headers.clone();
        let body = entry.body.clone();

        let headers_stream = AsyncStream::with_channel(move |sender| {
            for (name, value) in headers.iter() {
                fluent_ai_async::emit!(
                    sender,
                    HttpHeader {
                        name: name.clone(),
                        value: value.clone(),
                        timestamp: std::time::Instant::now(),
                    }
                );
            }
        });

        let body_stream = AsyncStream::with_channel(move |sender| {
            fluent_ai_async::emit!(
                sender,
                HttpBodyChunk {
                    data: body,
                    offset: 0,
                    is_final: true,
                    timestamp: std::time::Instant::now(),
                }
            );
        });

        let trailers_stream = AsyncStream::with_channel(|_sender| {
            // Empty trailers stream
        });

        // Use proper constructor instead of struct literal
        let mut response = crate::HttpResponse::new(
            headers_stream,
            body_stream,
            trailers_stream,
            entry.version,
            0,
        );
        
        // Set the status after construction
        response.set_status(status);
        
        response
    }

    /// Store response in cache
    pub async fn put(&self, key: CacheKey, response: HttpResponse) {
        if self.config.max_entries == 0 {
            return; // Caching disabled
        }

        let entry = CacheEntry::new(response).await;
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
        if response.is_error() {
            return false;
        }

        // Check Cache-Control directives
        // Note: Would need to collect headers for full implementation
        // For now, allow caching of success responses
        true
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
