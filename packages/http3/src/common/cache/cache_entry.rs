//! Cache entry management with TTL and validation support
//!
//! Provides CacheEntry for storing HTTP responses with metadata like
//! expiration times, ETags, and hit tracking for LRU eviction.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::{HttpResponse, common::cache::http_date::httpdate};

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
