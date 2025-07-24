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
    position_end: u32,
}

impl CacheKey {
    #[inline(always)]
    pub fn new(sequence_id: u64, layer_id: u32, position_start: u32, position_end: u32) -> Self {
        Self {
            sequence_id,
            layer_id,
            position_start,
            position_end,
        }
    }

    #[inline(always)]
    pub fn sequence_id(&self) -> u64 {
        self.sequence_id
    }

    #[inline(always)]
    pub fn layer_id(&self) -> u32 {
        self.layer_id
    }

    #[inline(always)]
    pub fn position_range(&self) -> (u32, u32) {
        (self.position_start, self.position_end)
    }

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
    access_count: AtomicU64,
}

impl KVCacheEntry {
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
            access_count: AtomicU64::new(0),
        })
    }

    #[inline(always)]
    pub fn key_tensor(&self) -> &Tensor {
        &self.key_tensor
    }

    #[inline(always)]
    pub fn value_tensor(&self) -> &Tensor {
        &self.value_tensor
    }

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

    #[inline(always)]
    pub fn add_ref(&self) -> u32 {
        self.ref_count.fetch_add(1, Ordering::Acquire)
    }

    #[inline(always)]
    pub fn release_ref(&self) -> u32 {
        self.ref_count.fetch_sub(1, Ordering::Release)
    }

    #[inline(always)]
    pub fn ref_count(&self) -> u32 {
        self.ref_count.load(Ordering::Acquire)
    }

    #[inline(always)]
    pub fn memory_usage(&self) -> u64 {
        self.memory_bytes.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn last_access(&self) -> u64 {
        self.last_access_nanos.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn age_nanos(&self) -> u64 {
        let now = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => duration.as_nanos() as u64,
            Err(_) => return 0,
        };
        let creation_time = self.creation_time_nanos.load(Ordering::Relaxed);
        now.saturating_sub(creation_time)
    }

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
    pub max_memory_per_sequence: u64,
}

impl Default for KVCacheConfig {
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
    pub total_entries: u64,
    pub total_memory_bytes: u64,
    pub active_sequences: u32,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_rate: f32,
    pub eviction_count: u64,
    pub memory_utilization: f32,
}

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
    sequence_memory: SkipMap<u64, AtomicU64>,
}

impl KVCacheManager {
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
            sequence_memory: SkipMap::new(),
        }
    }

    #[inline(always)]
    pub fn new_sequence(&self) -> u64 {
        let seq_id = self.next_sequence_id.fetch_add(1, Ordering::Relaxed);
        self.active_sequences.fetch_add(1, Ordering::Relaxed);
        if self.config.per_sequence_limits {
            self.sequence_memory.insert(seq_id, AtomicU64::new(0));
        }
        seq_id
    }

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
                / self.config.max_memory_bytes as f32,
        }
    }

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
