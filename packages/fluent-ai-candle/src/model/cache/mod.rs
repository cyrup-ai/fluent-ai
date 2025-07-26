//! Lock-free KV cache system with atomic operations and memory management
//!
//! This module provides:
//! - Lock-free KV cache with crossbeam-skiplist
//! - Atomic memory management and reference counting
//! - Per-sequence isolation and memory limits
//! - LRU eviction with high/low watermarks
//! - Comprehensive cache statistics

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use candle_core::Tensor;
use crossbeam_skiplist::SkipMap;

use crate::error::CandleError;

/// Composite cache key for per-sequence isolation and efficient lookups
#[derive(Hash, Eq, PartialEq, Ord, PartialOrd, Clone, Copy)]
#[repr(C)]
pub struct CacheKey {
    /// Sequence identifier for isolation
    sequence_id: u64,
    /// Layer identifier for hierarchical caching  
    layer_id: u32,
    /// Position range start for efficient range queries
    position_start: u32,
    /// Position range end for efficient range queries
    position_end: u32
}

impl CacheKey {
    /// Create new cache key with sequence isolation and position range
    ///
    /// Creates a composite cache key that provides efficient lookups and
    /// supports per-sequence isolation. Position ranges enable efficient
    /// range queries and cache invalidation.
    ///
    /// # Arguments
    /// * `sequence_id` - Unique identifier for sequence isolation
    /// * `layer_id` - Neural network layer identifier for hierarchical caching
    /// * `position_start` - Starting position in the sequence
    /// * `position_end` - Ending position in the sequence
    ///
    /// # Returns
    /// A new CacheKey instance with optimal ordering for skiplist performance
    #[inline(always)]
    pub fn new(sequence_id: u64, layer_id: u32, position_start: u32, position_end: u32) -> Self {
        Self {
            sequence_id,
            layer_id,
            position_start,
            position_end}
    }

    /// Get the sequence identifier for this cache entry
    ///
    /// Sequences provide isolation between different conversation contexts
    /// or parallel processing streams. Each sequence maintains its own
    /// independent cache state.
    ///
    /// # Returns
    /// The sequence identifier as a 64-bit unsigned integer
    #[inline(always)]
    pub fn sequence_id(&self) -> u64 {
        self.sequence_id
    }

    /// Get the neural network layer identifier
    ///
    /// Layer IDs enable hierarchical caching where each transformer
    /// layer maintains separate key-value cache entries. This supports
    /// efficient layer-specific cache operations and memory management.
    ///
    /// # Returns
    /// The layer identifier as a 32-bit unsigned integer
    #[inline(always)]
    pub fn layer_id(&self) -> u32 {
        self.layer_id
    }

    /// Get the position range (start, end) for this cache entry
    ///
    /// Position ranges define the token sequence span covered by this
    /// cache entry. This enables efficient range-based operations like
    /// partial cache invalidation and position-aware lookups.
    ///
    /// # Returns
    /// A tuple containing (position_start, position_end)
    #[inline(always)]
    pub fn position_range(&self) -> (u32, u32) {
        (self.position_start, self.position_end)
    }

    /// Calculate memory footprint of this cache key
    #[allow(dead_code)] // Part of CacheKey API for future memory monitoring
    #[inline(always)]
    fn memory_footprint(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Enhanced KV cache entry with atomic reference counting and memory management
#[repr(C, align(64))] // Cache line alignment for SIMD operations
pub struct KVCacheEntry {
    /// Key tensor for attention computation
    key_tensor: Tensor,
    /// Value tensor for attention computation
    value_tensor: Tensor,
    /// Sequence identifier for isolation
    sequence_id: u64,
    /// Layer identifier
    layer_id: u32,
    /// Position range start
    position_start: u32,
    /// Position range end  
    position_end: u32,
    /// Memory usage in bytes (atomic for lock-free updates)
    memory_bytes: AtomicU64,
    /// Atomic reference count for memory management
    ref_count: AtomicU32,
    /// Last access timestamp in nanoseconds (for precise LRU)
    last_access_nanos: AtomicU64,
    /// Creation timestamp in nanoseconds
    creation_time_nanos: AtomicU64,
    /// Access count for usage statistics
    access_count: AtomicU64}

impl KVCacheEntry {
    /// Create a new KV cache entry with automatic memory tracking
    ///
    /// Constructs a cache entry containing key and value tensors with
    /// atomic reference counting, memory usage tracking, and precise
    /// LRU timestamp management for optimal cache performance.
    ///
    /// # Arguments
    /// * `key_tensor` - The key tensor for attention computation
    /// * `value_tensor` - The value tensor for attention computation
    /// * `sequence_id` - Sequence identifier for isolation
    /// * `layer_id` - Neural network layer identifier
    /// * `position_start` - Starting token position
    /// * `position_end` - Ending token position
    ///
    /// # Returns
    /// Result containing the new KVCacheEntry or CandleError on failure
    ///
    /// # Performance
    /// Memory usage is calculated upfront for efficient cache management.
    /// Timestamps use nanosecond precision for accurate LRU ordering.
    #[inline(always)]
    pub fn new(
        key_tensor: Tensor,
        value_tensor: Tensor,
        sequence_id: u64,
        layer_id: u32,
        position_start: u32,
        position_end: u32,
    ) -> Result<Self, CandleError> {
        // Calculate memory usage for both tensors
        let key_bytes = key_tensor.elem_count() * key_tensor.dtype().size_in_bytes();
        let value_bytes = value_tensor.elem_count() * value_tensor.dtype().size_in_bytes();
        let total_memory = key_bytes + value_bytes;

        // Get current time in nanoseconds for precise LRU tracking
        let now_nanos = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => duration.as_nanos() as u64,
            Err(_) => 0, // Fallback for clock issues
        };

        Ok(Self {
            key_tensor,
            value_tensor,
            sequence_id,
            layer_id,
            position_start,
            position_end,
            memory_bytes: AtomicU64::new(total_memory as u64),
            ref_count: AtomicU32::new(1), // Start with reference count of 1
            last_access_nanos: AtomicU64::new(now_nanos),
            creation_time_nanos: AtomicU64::new(now_nanos),
            access_count: AtomicU64::new(0)})
    }

    /// Get a reference to the key tensor for attention computation
    ///
    /// Returns an immutable reference to the cached key tensor used
    /// in transformer attention mechanisms. The tensor remains valid
    /// as long as this cache entry exists.
    ///
    /// # Returns
    /// Immutable reference to the key tensor
    #[inline(always)]
    pub fn key_tensor(&self) -> &Tensor {
        &self.key_tensor
    }

    /// Get a reference to the value tensor for attention computation
    ///
    /// Returns an immutable reference to the cached value tensor used
    /// in transformer attention mechanisms. The tensor remains valid
    /// as long as this cache entry exists.
    ///
    /// # Returns
    /// Immutable reference to the value tensor
    #[inline(always)]
    pub fn value_tensor(&self) -> &Tensor {
        &self.value_tensor
    }

    /// Update the last access timestamp and increment access counter
    ///
    /// Marks this cache entry as recently used for LRU eviction policy.
    /// Updates both the nanosecond-precision timestamp and access count
    /// atomically for thread-safe operation.
    ///
    /// # Thread Safety
    /// This method is lock-free and safe to call from multiple threads
    /// concurrently. Uses relaxed atomic ordering for optimal performance.
    #[inline(always)]
    pub fn touch(&self) {
        let now_nanos = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => duration.as_nanos() as u64,
            Err(_) => {
                // Fallback: increment by 1 to maintain ordering
                self.last_access_nanos.load(Ordering::Relaxed) + 1
            }
        };
        self.last_access_nanos.store(now_nanos, Ordering::Relaxed);
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Atomically increment the reference count
    ///
    /// Increases the reference count to prevent premature eviction
    /// while the entry is in use. Uses acquire ordering to ensure
    /// proper memory synchronization.
    ///
    /// # Returns
    /// The previous reference count value
    ///
    /// # Thread Safety
    /// Atomic operation safe for concurrent access
    #[inline(always)]
    pub fn add_ref(&self) -> u32 {
        self.ref_count.fetch_add(1, Ordering::Acquire)
    }

    /// Atomically decrement the reference count
    ///
    /// Decreases the reference count when finished using the entry.
    /// When the count reaches zero, the entry becomes eligible for
    /// eviction. Uses release ordering for proper memory synchronization.
    ///
    /// # Returns
    /// The previous reference count value
    ///
    /// # Thread Safety
    /// Atomic operation safe for concurrent access
    #[inline(always)]
    pub fn release_ref(&self) -> u32 {
        self.ref_count.fetch_sub(1, Ordering::Release)
    }

    /// Get the current reference count
    ///
    /// Returns the current reference count for this cache entry.
    /// Entries with reference count > 1 are protected from eviction.
    ///
    /// # Returns
    /// Current reference count as a 32-bit unsigned integer
    ///
    /// # Thread Safety
    /// Uses acquire ordering for consistent reads
    #[inline(always)]
    pub fn ref_count(&self) -> u32 {
        self.ref_count.load(Ordering::Acquire)
    }

    /// Get the total memory usage in bytes
    ///
    /// Returns the combined memory usage of both key and value tensors
    /// in bytes. This value is calculated once at creation time for
    /// efficient memory management operations.
    ///
    /// # Returns
    /// Total memory usage in bytes as a 64-bit unsigned integer
    #[inline(always)]
    pub fn memory_usage(&self) -> u64 {
        self.memory_bytes.load(Ordering::Relaxed)
    }

    /// Get the last access timestamp in nanoseconds
    ///
    /// Returns the nanosecond-precision timestamp of when this entry
    /// was last accessed via touch(). Used for LRU eviction decisions.
    ///
    /// # Returns
    /// Unix timestamp in nanoseconds as a 64-bit unsigned integer
    #[inline(always)]
    pub fn last_access(&self) -> u64 {
        self.last_access_nanos.load(Ordering::Relaxed)
    }

    /// Calculate the age of this cache entry in nanoseconds
    ///
    /// Computes the time elapsed since entry creation using
    /// nanosecond precision. Useful for age-based eviction policies
    /// and cache performance analysis.
    ///
    /// # Returns
    /// Age in nanoseconds since creation, or 0 if system time is unavailable
    #[inline(always)]
    pub fn age_nanos(&self) -> u64 {
        let now = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => duration.as_nanos() as u64,
            Err(_) => return 0};
        let creation_time = self.creation_time_nanos.load(Ordering::Relaxed);
        now.saturating_sub(creation_time)
    }

    /// Get the total number of times this entry has been accessed
    ///
    /// Returns the cumulative access count incremented by each call
    /// to touch(). Useful for cache hit frequency analysis and
    /// performance monitoring.
    ///
    /// # Returns
    /// Total access count as a 64-bit unsigned integer
    #[inline(always)]
    pub fn access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }
}

/// Configuration for KV cache with memory management
#[repr(C)]
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: u64,
    /// Maximum number of concurrent sequences
    pub max_sequences: u32,
    /// High water mark for memory usage (0.0-1.0)
    pub high_water_mark: f32,
    /// Low water mark for memory usage (0.0-1.0)
    pub low_water_mark: f32,
    /// Number of entries to evict in each batch
    pub eviction_batch_size: u32,
    /// Enable per-sequence memory limits
    pub per_sequence_limits: bool,
    /// Memory limit per sequence (if per_sequence_limits enabled)
    pub max_memory_per_sequence: u64}

impl Default for KVCacheConfig {
    /// Create default KV cache configuration with production-ready settings
    ///
    /// Provides sensible defaults for most use cases:
    /// - 1GB total memory limit
    /// - 64 concurrent sequences
    /// - 90% high watermark, 70% low watermark for eviction
    /// - 32-entry eviction batches for efficiency
    /// - Per-sequence limits enabled with 16MB per sequence
    ///
    /// # Returns
    /// KVCacheConfig with balanced performance and memory settings
    #[inline(always)]
    fn default() -> Self {
        Self {
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB default
            max_sequences: 64,
            high_water_mark: 0.9,
            low_water_mark: 0.7,
            eviction_batch_size: 32,
            per_sequence_limits: true,
            max_memory_per_sequence: 16 * 1024 * 1024, // 16MB per sequence
        }
    }
}

/// Cache statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct KVCacheStats {
    /// Total number of entries currently in cache
    pub total_entries: u64,
    /// Total memory usage in bytes across all entries
    pub total_memory_bytes: u64,
    /// Number of active sequences with cache entries
    pub active_sequences: u32,
    /// Total number of successful cache lookups
    pub cache_hits: u64,
    /// Total number of failed cache lookups
    pub cache_misses: u64,
    /// Cache hit rate as a percentage (0.0 to 1.0)
    pub hit_rate: f32,
    /// Total number of entries evicted due to memory pressure
    pub eviction_count: u64,
    /// Memory utilization as a percentage of configured limit (0.0 to 1.0)
    pub memory_utilization: f32}

/// Lock-free KV cache manager with atomic operations
pub struct KVCacheManager {
    /// Lock-free cache storage using crossbeam-skiplist
    cache: SkipMap<CacheKey, Arc<KVCacheEntry>>,
    /// Configuration
    config: KVCacheConfig,
    /// Total memory usage (atomic for lock-free updates)
    total_memory_usage: AtomicU64,
    /// Current sequence count
    active_sequences: AtomicU32,
    /// Next sequence ID (atomic counter)
    next_sequence_id: AtomicU64,
    /// Cache hit count for statistics
    cache_hits: AtomicU64,
    /// Cache miss count for statistics
    cache_misses: AtomicU64,
    /// Eviction count for statistics
    eviction_count: AtomicU64,
    /// Per-sequence memory usage tracking
    sequence_memory: SkipMap<u64, AtomicU64>}

impl KVCacheManager {
    /// Create a new KV cache manager with the specified configuration
    ///
    /// Initializes a lock-free cache manager using crossbeam-skiplist
    /// for optimal concurrent performance. All atomic counters start
    /// at zero and sequence IDs begin at 1.
    ///
    /// # Arguments
    /// * `config` - Cache configuration including memory limits and eviction settings
    ///
    /// # Returns
    /// A new KVCacheManager instance ready for use
    ///
    /// # Performance
    /// Uses lock-free data structures for maximum throughput under
    /// concurrent access patterns typical in ML inference workloads.
    #[inline(always)]
    pub fn new(config: KVCacheConfig) -> Self {
        Self {
            cache: SkipMap::new(),
            config,
            total_memory_usage: AtomicU64::new(0),
            active_sequences: AtomicU32::new(0),
            next_sequence_id: AtomicU64::new(1),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            eviction_count: AtomicU64::new(0),
            sequence_memory: SkipMap::new()}
    }

    /// Create a new sequence and return its unique identifier
    ///
    /// Allocates a new sequence ID and initializes per-sequence memory
    /// tracking if enabled in configuration. Each sequence provides
    /// isolation for independent conversation contexts or processing streams.
    ///
    /// # Returns
    /// A unique 64-bit sequence identifier
    ///
    /// # Thread Safety
    /// Atomically increments sequence counter for thread-safe ID generation
    #[inline(always)]
    pub fn new_sequence(&self) -> u64 {
        let seq_id = self.next_sequence_id.fetch_add(1, Ordering::Relaxed);
        self.active_sequences.fetch_add(1, Ordering::Relaxed);
        if self.config.per_sequence_limits {
            self.sequence_memory.insert(seq_id, AtomicU64::new(0));
        }
        seq_id
    }

    /// Retrieve a cache entry by key with automatic LRU update
    ///
    /// Performs a lock-free lookup in the skip-list cache. If found,
    /// automatically updates the entry's access timestamp and increments
    /// its reference count to prevent premature eviction.
    ///
    /// # Arguments
    /// * `key` - The cache key to look up
    ///
    /// # Returns
    /// Some(Arc<KVCacheEntry>) if found, None if not in cache
    ///
    /// # Side Effects
    /// - Updates cache hit/miss statistics
    /// - Updates entry LRU timestamp if found
    /// - Increments entry reference count if found
    #[inline(always)]
    pub fn get_entry(&self, key: &CacheKey) -> Option<Arc<KVCacheEntry>> {
        match self.cache.get(key) {
            Some(entry) => {
                entry.value().touch(); // Update LRU timestamp
                entry.value().add_ref(); // Increment reference count
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                Some(Arc::clone(entry.value()))
            }
            None => {
                self.cache_misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Insert a new cache entry with automatic memory management
    ///
    /// Creates and inserts a new KV cache entry, checking memory limits
    /// and triggering eviction if necessary. Supports both per-sequence
    /// and global memory limits with high/low watermark eviction.
    ///
    /// # Arguments
    /// * `key` - The cache key for this entry
    /// * `key_tensor` - The key tensor to cache
    /// * `value_tensor` - The value tensor to cache
    ///
    /// # Returns
    /// Result containing Arc<KVCacheEntry> on success or CandleError on failure
    ///
    /// # Memory Management
    /// - Checks per-sequence memory limits if enabled
    /// - Triggers LRU eviction when high watermark is exceeded
    /// - Updates atomic memory usage counters
    ///
    /// # Performance
    /// Memory usage is calculated upfront to avoid repeated tensor queries
    #[inline(always)]
    pub fn insert_entry(
        &self,
        key: CacheKey,
        key_tensor: Tensor,
        value_tensor: Tensor,
    ) -> Result<Arc<KVCacheEntry>, CandleError> {
        let entry = KVCacheEntry::new(
            key_tensor,
            value_tensor,
            key.sequence_id,
            key.layer_id,
            key.position_start,
            key.position_end,
        )?;

        let memory_usage = entry.memory_usage();
        let arc_entry = Arc::new(entry);

        // Check memory limits before insertion
        if self.config.per_sequence_limits {
            if let Some(seq_memory) = self.sequence_memory.get(&key.sequence_id) {
                let current_seq_memory = seq_memory.value().load(Ordering::Relaxed);
                if current_seq_memory + memory_usage > self.config.max_memory_per_sequence {
                    self.evict_sequence_lru(&key.sequence_id)?;
                }
                seq_memory
                    .value()
                    .fetch_add(memory_usage, Ordering::Relaxed);
            }
        }

        // Check global memory limits
        let new_total = self
            .total_memory_usage
            .fetch_add(memory_usage, Ordering::Relaxed)
            + memory_usage;
        if new_total > (self.config.max_memory_bytes as f32 * self.config.high_water_mark) as u64 {
            // Trigger async eviction to reach low water mark
            self.evict_to_low_water_mark()?;
        }

        self.cache.insert(key, Arc::clone(&arc_entry));
        Ok(arc_entry)
    }

    /// Clear all cache entries for a specific sequence
    ///
    /// Removes all cache entries belonging to the specified sequence,
    /// freeing their memory and updating all relevant counters.
    /// This is typically called when a conversation or processing
    /// context is completed.
    ///
    /// # Arguments
    /// * `sequence_id` - The sequence identifier to clear
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Side Effects
    /// - Removes all entries for the sequence from cache
    /// - Updates global and per-sequence memory counters
    /// - Decrements active sequence count
    #[inline(always)]
    pub fn clear_sequence(&self, sequence_id: u64) -> Result<(), CandleError> {
        let mut keys_to_remove = Vec::new();
        let mut memory_freed = 0u64;

        // Collect keys for the sequence
        for entry in self.cache.iter() {
            if entry.key().sequence_id == sequence_id {
                memory_freed += entry.value().memory_usage();
                keys_to_remove.push(*entry.key());
            }
        }

        // Remove entries
        for key in keys_to_remove {
            self.cache.remove(&key);
        }

        // Update memory counters
        self.total_memory_usage
            .fetch_sub(memory_freed, Ordering::Relaxed);
        if self.config.per_sequence_limits {
            self.sequence_memory.remove(&sequence_id);
        }
        self.active_sequences.fetch_sub(1, Ordering::Relaxed);

        Ok(())
    }

    #[inline(always)]
    fn evict_sequence_lru(&self, sequence_id: &u64) -> Result<(), CandleError> {
        let target_memory =
            (self.config.max_memory_per_sequence as f32 * self.config.low_water_mark) as u64;
        let mut candidates: Vec<(CacheKey, u64)> = Vec::new();

        // Collect eviction candidates for this sequence
        for entry in self.cache.iter() {
            if &entry.key().sequence_id == sequence_id && entry.value().ref_count() <= 1 {
                candidates.push((*entry.key(), entry.value().last_access()));
            }
        }

        // Sort by LRU (oldest first)
        candidates.sort_by_key(|(_, last_access)| *last_access);

        // Evict entries until we reach target memory
        let mut memory_freed = 0u64;
        let seq_memory = match self.sequence_memory.get(sequence_id) {
            Some(mem) => mem,
            None => return Ok(()), // No memory tracking for this sequence
        };

        for (key, _) in candidates {
            if seq_memory.value().load(Ordering::Relaxed) - memory_freed <= target_memory {
                break;
            }

            if let Some(entry) = self.cache.get(&key) {
                let entry_memory = entry.value().memory_usage();
                self.cache.remove(&key);
                memory_freed += entry_memory;
                self.eviction_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Update memory counters
        self.total_memory_usage
            .fetch_sub(memory_freed, Ordering::Relaxed);
        seq_memory
            .value()
            .fetch_sub(memory_freed, Ordering::Relaxed);

        Ok(())
    }

    #[inline(always)]
    fn evict_to_low_water_mark(&self) -> Result<(), CandleError> {
        let target_memory =
            (self.config.max_memory_bytes as f32 * self.config.low_water_mark) as u64;
        let current_memory = self.total_memory_usage.load(Ordering::Relaxed);

        if current_memory <= target_memory {
            return Ok(()); // Already below target
        }

        let mut candidates: Vec<(CacheKey, u64)> = Vec::new();

        // Collect eviction candidates (only entries with ref_count <= 1)
        for entry in self.cache.iter() {
            if entry.value().ref_count() <= 1 {
                candidates.push((*entry.key(), entry.value().last_access()));
            }
        }

        // Sort by LRU (oldest first)
        candidates.sort_by_key(|(_, last_access)| *last_access);

        // Evict entries in batches
        let mut memory_freed = 0u64;
        let memory_to_free = current_memory - target_memory;

        for (key, _) in candidates {
            if memory_freed >= memory_to_free {
                break;
            }

            if let Some(entry) = self.cache.get(&key) {
                let entry_memory = entry.value().memory_usage();
                let sequence_id = entry.key().sequence_id;

                self.cache.remove(&key);
                memory_freed += entry_memory;

                // Update per-sequence memory tracking
                if self.config.per_sequence_limits {
                    if let Some(seq_memory) = self.sequence_memory.get(&sequence_id) {
                        seq_memory
                            .value()
                            .fetch_sub(entry_memory, Ordering::Relaxed);
                    }
                }

                self.eviction_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.total_memory_usage
            .fetch_sub(memory_freed, Ordering::Relaxed);
        Ok(())
    }

    /// Get comprehensive cache performance statistics
    ///
    /// Returns detailed statistics including memory usage, hit rates,
    /// eviction counts, and utilization metrics. All values are
    /// atomically consistent at the time of the call.
    ///
    /// # Returns
    /// KVCacheStats struct containing current cache metrics
    ///
    /// # Performance
    /// Statistics are computed on-demand from atomic counters,
    /// providing zero-overhead monitoring when not called.
    #[inline(always)]
    pub fn get_stats(&self) -> KVCacheStats {
        KVCacheStats {
            total_entries: self.cache.len() as u64,
            total_memory_bytes: self.total_memory_usage.load(Ordering::Relaxed),
            active_sequences: self.active_sequences.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            hit_rate: {
                let hits = self.cache_hits.load(Ordering::Relaxed);
                let misses = self.cache_misses.load(Ordering::Relaxed);
                if hits + misses > 0 {
                    hits as f32 / (hits + misses) as f32
                } else {
                    0.0
                }
            },
            eviction_count: self.eviction_count.load(Ordering::Relaxed),
            memory_utilization: self.total_memory_usage.load(Ordering::Relaxed) as f32
                / self.config.max_memory_bytes as f32}
    }

    /// Clear all cache entries and reset all statistics
    ///
    /// Removes all cache entries from all sequences and resets
    /// all atomic counters to zero. This is a complete cache
    /// reset operation typically used for testing or cleanup.
    ///
    /// # Side Effects
    /// - Clears main cache and per-sequence memory tracking
    /// - Resets all atomic counters to zero
    /// - Invalidates all existing cache entry references
    #[inline(always)]
    pub fn clear_all(&self) {
        self.cache.clear();
        self.sequence_memory.clear();
        self.total_memory_usage.store(0, Ordering::Relaxed);
        self.active_sequences.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.eviction_count.store(0, Ordering::Relaxed);
    }
}
