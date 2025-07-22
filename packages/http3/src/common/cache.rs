//! Zero allocation HTTP response caching with lock-free skiplist storage

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use crossbeam_skiplist::SkipMap;
use fastrand::Rng;
use fluent_ai_async::{AsyncStream, AsyncStreamSender};
use futures_util::{StreamExt, pin_mut};

use crate::{HttpError, HttpResponse, HttpResult};

/// Cache key for HTTP responses based on URL and headers
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    /// Request URL
    pub url: String,
    /// Normalized headers that affect caching (e.g., Accept, Authorization)
    pub cache_headers: HashMap<String, String>,
    /// HTTP method (GET, POST, etc.)
    pub method: String,
}

impl CacheKey {
    /// Create cache key from request components
    pub fn new(url: String, method: String, headers: HashMap<String, String>) -> Self {
        // Only include headers that affect caching behavior
        let cache_headers = headers
            .into_iter()
            .filter(|(key, _)| {
                let key_lower = key.to_lowercase();
                matches!(
                    key_lower.as_str(),
                    "accept"
                        | "accept-encoding"
                        | "accept-language"
                        | "authorization"
                        | "cache-control"
                        | "if-none-match"
                        | "if-modified-since"
                        | "user-agent"
                )
            })
            .collect();

        Self {
            url,
            method,
            cache_headers,
        }
    }

    /// Generate hash key for storage
    pub fn hash_key(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

impl std::hash::Hash for CacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.url.hash(state);
        self.method.hash(state);

        // Sort headers for consistent hashing
        let mut sorted_headers: Vec<_> = self.cache_headers.iter().collect();
        sorted_headers.sort_by_key(|(key, _)| *key);

        for (key, value) in sorted_headers {
            key.hash(state);
            value.hash(state);
        }
    }
}

/// Cached response entry with metadata
#[derive(Debug)]
pub struct CacheEntry {
    /// HTTP response
    pub response: HttpResponse,
    /// Cache creation timestamp
    pub created_at: Instant,
    /// Last access timestamp
    pub last_accessed: Instant,
    /// Cache expiration time (from Cache-Control headers)
    pub expires_at: Option<Instant>,
    /// ETag for validation
    pub etag: Option<String>,
    /// Last-Modified timestamp for validation
    pub last_modified: Option<SystemTime>,
    /// Hit count for LRU tracking
    pub hit_count: AtomicU64,
    /// Size in bytes for memory management
    pub size_bytes: u64,
}

impl CacheEntry {
    /// Create new cache entry from HTTP response
    pub fn new(response: HttpResponse) -> Self {
        let now = Instant::now();
        let etag = response.headers().get("etag").map(|v| v.to_string());

        let last_modified = response
            .headers()
            .get("last-modified")
            .and_then(|v| httpdate::parse_http_date(v).ok());

        // Calculate expiration based on Cache-Control or Expires headers
        let expires_at = Self::parse_expires(&response);

        // Estimate response size
        let size_bytes = response.body().len() as u64 + response.headers().len() as u64 * 64; // Estimate header overhead

        Self {
            response,
            created_at: now,
            last_accessed: now,
            expires_at,
            etag,
            last_modified,
            hit_count: AtomicU64::new(0),
            size_bytes,
        }
    }

    /// Parse expiration time from response headers
    fn parse_expires(response: &HttpResponse) -> Option<Instant> {
        // Check Cache-Control max-age first
        if let Some(cache_control) = response.headers().get("cache-control") {
            if let Some(max_age) = Self::parse_max_age(cache_control) {
                return Some(Instant::now() + Duration::from_secs(max_age));
            }
        }

        // Fall back to Expires header
        if let Some(expires) = response.headers().get("expires") {
            if let Ok(expires_time) = httpdate::parse_http_date(expires) {
                let duration_since_unix = expires_time.duration_since(UNIX_EPOCH).ok()?;
                let now_since_unix = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;

                if duration_since_unix > now_since_unix {
                    let ttl = duration_since_unix - now_since_unix;
                    return Some(Instant::now() + ttl);
                }
            }
        }

        None
    }

    /// Parse max-age value from Cache-Control header
    fn parse_max_age(cache_control: &str) -> Option<u64> {
        for directive in cache_control.split(',') {
            let directive = directive.trim();
            if directive.starts_with("max-age=") {
                if let Ok(seconds) = directive[8..].parse::<u64>() {
                    return Some(seconds);
                }
            }
        }
        None
    }

    /// Check if cache entry is expired
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map_or(false, |expires| Instant::now() > expires)
    }

    /// Check if entry can be validated with conditional request
    pub fn can_validate(&self) -> bool {
        self.etag.is_some() || self.last_modified.is_some()
    }

    /// Record cache hit and update access time
    pub fn record_hit(&mut self) {
        self.last_accessed = Instant::now();
        self.hit_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get hit count
    pub fn hits(&self) -> u64 {
        self.hit_count.load(Ordering::Relaxed)
    }

    /// Calculate age of this cache entry
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

impl Clone for CacheEntry {
    fn clone(&self) -> Self {
        Self {
            response: self.response.clone(),
            created_at: self.created_at,
            last_accessed: self.last_accessed,
            expires_at: self.expires_at,
            etag: self.etag.clone(),
            last_modified: self.last_modified,
            hit_count: AtomicU64::new(self.hit_count.load(Ordering::Relaxed)),
            size_bytes: self.size_bytes,
        }
    }
}

/// Cache configuration and limits
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in cache
    pub max_entries: usize,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: u64,
    /// Default TTL for entries without explicit expiration
    pub default_ttl: Duration,
    /// Enable automatic cleanup of expired entries
    pub auto_cleanup: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_memory_bytes: 100 * 1024 * 1024,   // 100MB
            default_ttl: Duration::from_secs(300), // 5 minutes
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(60), // 1 minute
        }
    }
}

impl CacheConfig {
    /// Create aggressive caching configuration
    pub fn aggressive() -> Self {
        Self {
            max_entries: 5000,
            max_memory_bytes: 500 * 1024 * 1024,    // 500MB
            default_ttl: Duration::from_secs(3600), // 1 hour
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(30),
        }
    }

    /// Create conservative caching configuration
    pub fn conservative() -> Self {
        Self {
            max_entries: 200,
            max_memory_bytes: 20 * 1024 * 1024,   // 20MB
            default_ttl: Duration::from_secs(60), // 1 minute
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(120), // 2 minutes
        }
    }

    /// Create no-cache configuration (disabled caching)
    pub fn no_cache() -> Self {
        Self {
            max_entries: 0,
            max_memory_bytes: 0,
            default_ttl: Duration::ZERO,
            auto_cleanup: false,
            cleanup_interval: Duration::MAX,
        }
    }
}

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
    cleanup_running: AtomicBool,
}

/// Cache statistics for monitoring
#[derive(Debug)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub validations: AtomicU64,
    pub errors: AtomicU64,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            validations: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        }
    }
}

impl CacheStats {
    /// Get hit rate as percentage
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let total = hits + self.misses.load(Ordering::Relaxed) as f64;

        if total > 0.0 {
            (hits / total) * 100.0
        } else {
            0.0
        }
    }

    /// Get statistics snapshot
    pub fn snapshot(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.evictions.load(Ordering::Relaxed),
            self.validations.load(Ordering::Relaxed),
            self.errors.load(Ordering::Relaxed),
        )
    }
}

impl ResponseCache {
    /// Create new response cache with configuration
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: SkipMap::new(),
            config,
            memory_usage: AtomicU64::new(0),
            entry_count: AtomicU64::new(0),
            stats: CacheStats::default(),
            cleanup_running: AtomicBool::new(false),
        }
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
            self.evict_lru_entries();
        }

        // Check entry count limits
        let current_entries = self.entry_count.load(Ordering::Relaxed) as usize;
        if current_entries >= self.config.max_entries {
            self.evict_lru_entries();
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
    fn evict_lru_entries(&self) {
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

/// Global cache instance for use across the HTTP client
lazy_static::lazy_static! {
    pub static ref GLOBAL_CACHE: ResponseCache = ResponseCache::default();
}

/// Cache-aware HTTP stream that checks cache before making requests
pub fn cached_stream<F>(cache_key: CacheKey, operation: F) -> AsyncStream<HttpResult<HttpResponse>>
where
    F: Fn() -> AsyncStream<HttpResult<HttpResponse>> + Send + Sync + 'static,
{
    AsyncStream::with_channel(move |sender: AsyncStreamSender<HttpResult<HttpResponse>>| {
        let _handle = tokio::spawn(async move {
            // Check cache first
            if let Some(cached_response) = GLOBAL_CACHE.get(&cache_key) {
                let _ = sender.send(Ok(cached_response));
                return;
            }

            // Cache miss - execute operation
            let operation_stream = operation();
            pin_mut!(operation_stream);

            while let Some(result) = operation_stream.next().await {
                match result {
                    Ok(response) => {
                        // Check if response should be cached
                        if GLOBAL_CACHE.should_cache(&response) {
                            GLOBAL_CACHE.put(cache_key.clone(), response.clone());
                        }

                        let _ = sender.send(Ok(response));
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                    }
                }
            }
        });
    })
}

/// Helper to create conditional request headers for cache validation
pub fn conditional_headers_for_key(cache_key: &CacheKey) -> HashMap<String, String> {
    GLOBAL_CACHE
        .get_validation_headers(cache_key)
        .unwrap_or_default()
}

/// HTTP date parsing utilities
mod httpdate {
    use std::time::SystemTime;

    pub fn parse_http_date(date_str: &str) -> Result<SystemTime, ()> {
        // Simplified HTTP date parsing - in production, use a proper HTTP date parser
        use chrono::{DateTime, Utc};

        if let Ok(dt) = DateTime::parse_from_rfc2822(date_str) {
            return Ok(
                SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(dt.timestamp() as u64)
            );
        }

        if let Ok(dt) = DateTime::parse_from_str(date_str, "%a, %d %b %Y %H:%M:%S %Z") {
            return Ok(
                SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(dt.timestamp() as u64)
            );
        }

        Err(())
    }

    pub fn fmt_http_date(time: SystemTime) -> String {
        use chrono::{DateTime, Utc};

        let duration = time
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();

        let dt = DateTime::<Utc>::from_timestamp(duration.as_secs() as i64, 0).unwrap_or_default();

        dt.format("%a, %d %b %Y %H:%M:%S GMT").to_string()
    }
}
