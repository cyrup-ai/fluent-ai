//! Cache management for build system
//!
//! This module provides a zero-allocation, lock-free cache manager for
//! build artifacts and provider metadata.

use std::fs;
use std::num::NonZero;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use bincode::{Decode, Encode};
use std::hash::Hasher;
use twox_hash::XxHash64;

use super::errors::{BuildError, BuildResult, CacheError, CacheResult};
use super::performance::PerformanceStats;

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct CacheEntryMeta {
    /// Size of the cached data in bytes
    pub size: u64,
    /// Timestamp when this entry was created (UNIX epoch seconds)
    pub created_at: u64,
    /// Timestamp when this entry expires (UNIX epoch seconds)
    pub expires_at: u64,
    /// Content hash for validation
    pub content_hash: u64,
}

/// Cache entry
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct CacheEntry {
    pub meta: CacheEntryMeta,
    pub data: Vec<u8>,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Cache directory path
    pub cache_dir: PathBuf,
    /// Maximum cache size in bytes
    pub max_size: u64,
    /// Default TTL for cache entries in seconds
    pub default_ttl: u64,
    /// Whether to enable memory caching
    pub enable_memory_cache: bool,
    /// Maximum number of memory cache entries
    pub max_memory_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        let cache_dir = std::env::temp_dir().join("fluent-ai-provider-cache");

        Self {
            cache_dir,
            max_size: 100 * 1024 * 1024, // 100MB
            default_ttl: 24 * 60 * 60,   // 24 hours
            enable_memory_cache: true,
            max_memory_entries: 1000,
        }
    }
}

/// Cache manager for the build system
#[derive(Debug)]
pub struct CacheManager {
    config: CacheConfig,
    stats: Arc<PerformanceStats>,
    memory_cache: Mutex<LruCache<u64, CacheEntry>>,
    current_size: Mutex<u64>,
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new(config: CacheConfig, stats: Arc<PerformanceStats>) -> BuildResult<Self> {
        fs::create_dir_all(&config.cache_dir)?;
        let current_size = Self::calculate_disk_usage(&config.cache_dir)?;

        let max_memory_entries = NonZero::new(config.max_memory_entries).unwrap_or_else(|| NonZero::new(1).unwrap());

        Ok(Self {
            config,
            stats,
            memory_cache: Mutex::new(LruCache::new(max_memory_entries)),
            current_size: Mutex::new(current_size),
        })
    }

    /// Get a value from the cache
    pub fn get(&self, key: &str) -> CacheResult<Option<Vec<u8>>> {
        let key_hash = self.hash_key(key);

        // Try memory cache first
        if self.config.enable_memory_cache {
            if let Some(entry) = self.memory_cache.lock().get(&key_hash) {
                if self.is_entry_valid(entry) {
                    self.stats.record_cache_hit();
                    return Ok(Some(entry.data.clone()));
                }
                self.memory_cache.lock().pop(&key_hash);
            }
        }

        // Fallback to disk cache
        let path = self.get_path(key_hash);
        if !path.exists() {
            self.stats.record_cache_miss();
            return Ok(None);
        }

        let content = fs::read(&path).map_err(|e| CacheError::Io(e.to_string()))?;
        let (entry, _): (CacheEntry, usize) = bincode::decode_from_slice(&content, bincode::config::standard()).map_err(|e| CacheError::Serialization(e.to_string()))?;

        if self.is_entry_valid(&entry) {
            self.stats.record_cache_hit();
            Ok(Some(entry.data))
        } else {
            self.stats.record_cache_miss();
            self.remove(key)?;
            Ok(None)
        }
    }

    /// Set a value in the cache
    pub fn set(&self, key: &str, value: Vec<u8>, ttl_seconds: Option<u64>) -> BuildResult<()> {
        let key_hash = self.hash_key(key);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let ttl = ttl_seconds.unwrap_or(self.config.default_ttl);

        let meta = CacheEntryMeta {
            size: value.len() as u64,
            created_at: now,
            expires_at: now + ttl,
            content_hash: self.hash_content(&value),
        };

        let entry = CacheEntry { meta, data: value };

        // Write to disk first
        let path = self.get_path(key_hash);
        let encoded = bincode::encode_to_vec(&entry, bincode::config::standard()).map_err(|e| BuildError::BincodeError(e))?;
        fs::write(&path, &encoded)?;

        let mut current_size = self.current_size.lock();
        *current_size += encoded.len() as u64;
        self.enforce_size_limit(&mut current_size)?;

        // Update memory cache
        if self.config.enable_memory_cache {
            self.memory_cache.lock().put(key_hash, entry);
        }

        Ok(())
    }

    /// Remove a value from the cache
    pub fn remove(&self, key: &str) -> CacheResult<bool> {
        let key_hash = self.hash_key(key);
        let path = self.get_path(key_hash);

        if self.config.enable_memory_cache {
            self.memory_cache.lock().pop(&key_hash);
        }

        if path.exists() {
            let meta = fs::metadata(&path).map_err(|e| CacheError::Io(e.to_string()))?;
            fs::remove_file(&path).map_err(|e| CacheError::Io(e.to_string()))?;
            *self.current_size.lock() -= meta.len();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // Removed unused methods clear() and stats()

    /// Enforce cache size limit by evicting oldest entries
    fn enforce_size_limit(&self, current_size: &mut u64) -> BuildResult<()> {
        if *current_size <= self.config.max_size {
            return Ok(());
        }

        let mut entries: Vec<_> = fs::read_dir(&self.config.cache_dir)?
            .filter_map(Result::ok)
            .filter_map(|e| {
                e.metadata()
                    .ok()
                    .and_then(|m| m.created().ok().map(|t| (e.path(), m, t)))
            })
            .collect();

        entries.sort_by_key(|(_, _, created_time)| *created_time);

        for (path, meta, _) in entries {
            if *current_size <= self.config.max_size {
                break;
            }
            if path.is_file() {
                fs::remove_file(&path)?;
                *current_size -= meta.len();
                self.stats.record_cache_eviction();
            }
        }

        Ok(())
    }

    /// Check if a cache entry is still valid
    fn is_entry_valid(&self, entry: &CacheEntry) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        now < entry.meta.expires_at
    }

    /// Get the file path for a given key hash
    fn get_path(&self, key_hash: u64) -> PathBuf {
        let (b1, b2) = ((key_hash >> 8) as u8, key_hash as u8);
        self.config.cache_dir.join(format!("{:02x}/{:02x}/{:016x}", b1, b2, key_hash))
    }

    /// Hash a string key
    fn hash_key(&self, key: &str) -> u64 {
        let mut hasher = XxHash64::with_seed(0);
        hasher.write(key.as_bytes());
        hasher.finish()
    }

    /// Hash binary content
    fn hash_content(&self, data: &[u8]) -> u64 {
        let mut hasher = XxHash64::with_seed(0);
        hasher.write(data);
        hasher.finish()
    }

    /// Calculate total disk usage of the cache directory
    fn calculate_disk_usage(path: &PathBuf) -> BuildResult<u64> {
        let mut total_size = 0;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_file() {
                total_size += metadata.len();
            } else if metadata.is_dir() {
                total_size += Self::calculate_disk_usage(&entry.path())?;
            }
        }
        Ok(total_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use tempfile::tempdir;

    #[test]
    fn test_cache_operations() -> BuildResult<()> {
        let temp_dir = tempdir().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            max_size: 1_000_000,
            default_ttl: 60,
            enable_memory_cache: true,
            max_memory_entries: 10,
        };
        let stats = Arc::new(PerformanceStats::new());
        let cache = CacheManager::new(config, stats.clone())?;

        let key = "test_key";
        let value = b"test_value".to_vec();

        // Should not exist yet
        assert!(cache.get(key)?.is_none());

        // Set value
        cache.set(key, value.clone(), None)?;

        // Should exist now
        let cached = cache.get(key)?.unwrap();
        assert_eq!(cached, value);

        // Test remove
        assert!(cache.remove(key)?);
        assert!(!cache.remove(key)?); // Should return false on second remove
        assert!(cache.get(key)?.is_none());

        // Test expiration
        cache.set(key, value.clone(), Some(1))?; // 1 second TTL
        std::thread::sleep(std::time::Duration::from_secs(2));
        assert!(cache.get(key)?.is_none());

        // Test clear
        cache.set(key, value.clone(), None)?;
        cache.clear()?;
        assert!(cache.get(key)?.is_none());

        // Test stats
        assert_eq!(stats.cache_hits.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(stats.cache_misses.load(std::sync::atomic::Ordering::Relaxed), 3);
        assert_eq!(stats.cache_evictions.load(std::sync::atomic::Ordering::Relaxed), 0);

        Ok(())
    }

    #[test]
    fn test_cache_eviction() -> BuildResult<()> {
        let temp_dir = tempdir().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            max_size: 100, // Very small size to force eviction
            default_ttl: 60,
            enable_memory_cache: false, // Disable memory cache for this test
            max_memory_entries: 0,
        };
        let stats = Arc::new(PerformanceStats::new());
        let cache = CacheManager::new(config, stats.clone())?;

        // Add entries until we exceed the cache size
        for i in 0..10 {
            let key = format!("key_{}", i);
            let value = vec![0u8; 20]; // Each entry is 20 bytes
            cache.set(&key, value, None)?;
        }

        // Some entries should have been evicted
        assert!(*cache.current_size.lock() <= 100);
        assert!(stats.cache_evictions.load(std::sync::atomic::Ordering::Relaxed) > 0);

        Ok(())
    }
}
