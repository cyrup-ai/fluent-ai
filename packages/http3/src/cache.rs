//! HTTP caching utilities

use std::time::{Duration, Instant};
use hashbrown::HashMap;
use chrono::{DateTime, Utc};
use crate::HttpResponse;

/// Cache entry for HTTP responses
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cached response
    pub response: HttpResponse,
    /// When the entry was created
    pub created_at: Instant,
    /// ETag for validation
    pub etag: Option<String>,
    /// Last-Modified header for validation
    pub last_modified: Option<String>,
    /// Expires header
    pub expires: Option<Instant>,
    /// Cache-Control max-age
    pub max_age: Option<Duration>,
    /// Whether the entry is stale
    pub is_stale: bool,
}

impl CacheEntry {
    /// Create a new cache entry
    pub fn new(response: HttpResponse) -> Self {
        let etag = response.etag().cloned();
        let last_modified = response.last_modified().cloned();
        let expires = response.expires().and_then(|e| parse_http_date(e));
        let max_age = response.cache_control()
            .and_then(|cc| parse_max_age(cc));
        
        Self {
            response,
            created_at: Instant::now(),
            etag,
            last_modified,
            expires,
            max_age,
            is_stale: false,
        }
    }
    
    /// Check if the cache entry is expired
    pub fn is_expired(&self) -> bool {
        // Check expires header
        if let Some(expires) = self.expires {
            if Instant::now() > expires {
                return true;
            }
        }
        
        // Check max-age
        if let Some(max_age) = self.max_age {
            if self.created_at.elapsed() > max_age {
                return true;
            }
        }
        
        // Default expiration (1 hour)
        if self.created_at.elapsed() > Duration::from_secs(3600) {
            return true;
        }
        
        false
    }
    
    /// Check if the entry is fresh
    pub fn is_fresh(&self) -> bool {
        !self.is_expired() && !self.is_stale
    }
    
    /// Mark the entry as stale
    pub fn mark_stale(&mut self) {
        self.is_stale = true;
    }
    
    /// Get the age of the entry
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
    
    /// Check if the entry can be validated
    pub fn can_validate(&self) -> bool {
        self.etag.is_some() || self.last_modified.is_some()
    }
    
    /// Get validation headers
    pub fn validation_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        
        if let Some(etag) = &self.etag {
            headers.insert("If-None-Match".to_string(), etag.clone());
        }
        
        if let Some(last_modified) = &self.last_modified {
            headers.insert("If-Modified-Since".to_string(), last_modified.clone());
        }
        
        headers
    }
}

/// Simple in-memory cache
pub struct MemoryCache {
    entries: HashMap<String, CacheEntry>,
    max_size: usize,
}

impl MemoryCache {
    /// Create a new memory cache
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_size,
        }
    }
    
    /// Get an entry from the cache
    pub fn get(&self, key: &str) -> Option<&CacheEntry> {
        self.entries.get(key)
    }
    
    /// Put an entry into the cache
    pub fn put(&mut self, key: String, entry: CacheEntry) {
        // Clean up expired entries if cache is full
        if self.entries.len() >= self.max_size {
            self.cleanup();
        }
        
        // If still full, remove oldest entry
        if self.entries.len() >= self.max_size {
            if let Some(oldest_key) = self.find_oldest_key() {
                self.entries.remove(&oldest_key);
            }
        }
        
        self.entries.insert(key, entry);
    }
    
    /// Remove an entry from the cache
    pub fn remove(&mut self, key: &str) -> Option<CacheEntry> {
        self.entries.remove(key)
    }
    
    /// Clear all entries from the cache
    pub fn clear(&mut self) {
        self.entries.clear();
    }
    
    /// Get the number of entries in the cache
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    
    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    
    /// Clean up expired entries
    pub fn cleanup(&mut self) {
        self.entries.retain(|_, entry| !entry.is_expired());
    }
    
    /// Find the oldest entry key
    fn find_oldest_key(&self) -> Option<String> {
        self.entries
            .iter()
            .min_by_key(|(_, entry)| entry.created_at)
            .map(|(key, _)| key.clone())
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let expired = self.entries.values().filter(|e| e.is_expired()).count();
        let stale = self.entries.values().filter(|e| e.is_stale).count();
        let fresh = self.entries.len() - expired - stale;
        
        CacheStats {
            total_entries: self.entries.len(),
            fresh_entries: fresh,
            stale_entries: stale,
            expired_entries: expired,
            max_size: self.max_size,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Number of fresh entries
    pub fresh_entries: usize,
    /// Number of stale entries
    pub stale_entries: usize,
    /// Number of expired entries
    pub expired_entries: usize,
    /// Maximum cache size
    pub max_size: usize,
}

impl Default for MemoryCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Parse HTTP date string to Instant
/// Supports RFC 1123, RFC 850, and ANSI C date formats
fn parse_http_date(date_str: &str) -> Option<Instant> {
    // Try RFC 1123 format first: "Sun, 06 Nov 1994 08:49:37 GMT"
    if let Some(instant) = parse_rfc1123(date_str) {
        return Some(instant);
    }
    
    // Try RFC 850 format: "Sunday, 06-Nov-94 08:49:37 GMT"
    if let Some(instant) = parse_rfc850(date_str) {
        return Some(instant);
    }
    
    // Try ANSI C format: "Sun Nov  6 08:49:37 1994"
    if let Some(instant) = parse_ansi_c(date_str) {
        return Some(instant);
    }
    
    None
}

/// Parse RFC 1123 format: "Sun, 06 Nov 1994 08:49:37 GMT"
fn parse_rfc1123(date_str: &str) -> Option<Instant> {
    let parts: Vec<&str> = date_str.split_whitespace().collect();
    if parts.len() != 6 {
        return None;
    }
    
    let day = parts[1].parse::<u32>().ok()?;
    let month = month_to_number(parts[2])?;
    let year = parts[3].parse::<u32>().ok()?;
    let time_parts: Vec<&str> = parts[4].split(':').collect();
    if time_parts.len() != 3 {
        return None;
    }
    
    let hour = time_parts[0].parse::<u32>().ok()?;
    let minute = time_parts[1].parse::<u32>().ok()?;
    let second = time_parts[2].parse::<u32>().ok()?;
    
    timestamp_to_instant(year, month, day, hour, minute, second)
}

/// Parse RFC 850 format: "Sunday, 06-Nov-94 08:49:37 GMT"
fn parse_rfc850(date_str: &str) -> Option<Instant> {
    let parts: Vec<&str> = date_str.split_whitespace().collect();
    if parts.len() != 4 {
        return None;
    }
    
    let date_part = parts[1];
    let date_components: Vec<&str> = date_part.split('-').collect();
    if date_components.len() != 3 {
        return None;
    }
    
    let day = date_components[0].parse::<u32>().ok()?;
    let month = month_to_number(date_components[1])?;
    let mut year = date_components[2].parse::<u32>().ok()?;
    
    // Convert 2-digit year to 4-digit
    if year < 70 {
        year += 2000;
    } else if year < 100 {
        year += 1900;
    }
    
    let time_parts: Vec<&str> = parts[2].split(':').collect();
    if time_parts.len() != 3 {
        return None;
    }
    
    let hour = time_parts[0].parse::<u32>().ok()?;
    let minute = time_parts[1].parse::<u32>().ok()?;
    let second = time_parts[2].parse::<u32>().ok()?;
    
    timestamp_to_instant(year, month, day, hour, minute, second)
}

/// Parse ANSI C format: "Sun Nov  6 08:49:37 1994"
fn parse_ansi_c(date_str: &str) -> Option<Instant> {
    let parts: Vec<&str> = date_str.split_whitespace().collect();
    if parts.len() != 5 {
        return None;
    }
    
    let month = month_to_number(parts[1])?;
    let day = parts[2].parse::<u32>().ok()?;
    let year = parts[4].parse::<u32>().ok()?;
    let time_parts: Vec<&str> = parts[3].split(':').collect();
    if time_parts.len() != 3 {
        return None;
    }
    
    let hour = time_parts[0].parse::<u32>().ok()?;
    let minute = time_parts[1].parse::<u32>().ok()?;
    let second = time_parts[2].parse::<u32>().ok()?;
    
    timestamp_to_instant(year, month, day, hour, minute, second)
}

/// Convert month name to number
fn month_to_number(month: &str) -> Option<u32> {
    match month {
        "Jan" => Some(1),
        "Feb" => Some(2),
        "Mar" => Some(3),
        "Apr" => Some(4),
        "May" => Some(5),
        "Jun" => Some(6),
        "Jul" => Some(7),
        "Aug" => Some(8),
        "Sep" => Some(9),
        "Oct" => Some(10),
        "Nov" => Some(11),
        "Dec" => Some(12),
        _ => None,
    }
}

/// Convert timestamp components to Instant
fn timestamp_to_instant(year: u32, month: u32, day: u32, hour: u32, minute: u32, second: u32) -> Option<Instant> {
    // Basic validation
    if year < 1970 || year > 2100 || month < 1 || month > 12 || day < 1 || day > 31 ||
       hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    
    // Calculate days since Unix epoch (1970-01-01)
    let mut days = 0u64;
    
    // Add days for years
    for y in 1970..year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }
    
    // Add days for months in current year
    for m in 1..month {
        days += days_in_month(m, year);
    }
    
    // Add days in current month
    days += (day - 1) as u64;
    
    // Convert to seconds
    let total_seconds = days * 86400 + hour as u64 * 3600 + minute as u64 * 60 + second as u64;
    
    // Create Duration from Unix epoch
    let duration = Duration::from_secs(total_seconds);
    
    // Convert to proper timestamp using chrono
    let unix_timestamp = DateTime::from_timestamp(total_seconds as i64, 0)?;
    let duration_since_epoch = unix_timestamp
        .signed_duration_since(DateTime::UNIX_EPOCH);
    
    // Convert to Instant by calculating offset from current time
    let now = Instant::now();
    let system_now = std::time::SystemTime::now();
    let current_unix_duration = system_now.duration_since(std::time::UNIX_EPOCH).ok()?;
    let current_unix_secs = current_unix_duration.as_secs() as i64;
    let target_unix_secs = duration_since_epoch.num_seconds();
    
    if target_unix_secs <= current_unix_secs {
        let offset = Duration::from_secs((current_unix_secs - target_unix_secs) as u64);
        Some(now - offset)
    } else {
        let offset = Duration::from_secs((target_unix_secs - current_unix_secs) as u64);
        Some(now + offset)
    }
}

/// Check if year is a leap year
fn is_leap_year(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Get number of days in a month
fn days_in_month(month: u32, year: u32) -> u64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if is_leap_year(year) { 29 } else { 28 },
        _ => 0,
    }
}

/// Parse max-age from Cache-Control header
fn parse_max_age(cache_control: &str) -> Option<Duration> {
    // Look for max-age directive
    for directive in cache_control.split(',') {
        let directive = directive.trim();
        if directive.starts_with("max-age=") {
            if let Some(age_str) = directive.strip_prefix("max-age=") {
                if let Ok(age) = age_str.parse::<u64>() {
                    return Some(Duration::from_secs(age));
                }
            }
        }
    }
    None
}

/// Check if response is cacheable
pub fn is_cacheable(response: &HttpResponse) -> bool {
    // Don't cache non-GET requests
    // (This would need to be checked at the request level)
    
    // Don't cache error responses
    if !response.is_success() {
        return false;
    }
    
    // Check Cache-Control header
    if let Some(cache_control) = response.cache_control() {
        if cache_control.contains("no-cache") || 
           cache_control.contains("no-store") ||
           cache_control.contains("private") {
            return false;
        }
    }
    
    // Check for explicit caching headers
    if response.etag().is_some() || 
       response.last_modified().is_some() ||
       response.expires().is_some() ||
       response.cache_control().is_some() {
        return true;
    }
    
    // Default to cacheable for successful responses
    true
}

/// Generate cache key from request
pub fn generate_cache_key(method: &str, url: &str, headers: &HashMap<String, String>) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    method.hash(&mut hasher);
    url.hash(&mut hasher);
    
    // Include relevant headers in cache key
    for (key, value) in headers {
        let key_lower = key.to_lowercase();
        if key_lower == "accept" || 
           key_lower == "accept-encoding" ||
           key_lower == "accept-language" {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }
    }
    
    format!("{:x}", hasher.finish())
}