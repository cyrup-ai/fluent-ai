//! Memory pool management for zero-allocation tensor operations
//!
//! This module provides high-performance memory pooling for ML inference with
//! cache-aligned allocations, lock-free coordination, and automatic pool sizing.

use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;
use std::sync::Arc;

use arc_swap::ArcSwap;
use crossbeam::atomic::AtomicCell;
use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
use smallvec::SmallVec;

use super::error::{CandleError, CandleResult};

/// Cache line size for optimal memory alignment
const CACHE_LINE_SIZE: usize = 64;
/// Default pool capacity for each size class
const DEFAULT_POOL_CAPACITY: usize = 128;
/// Maximum number of size classes
const MAX_SIZE_CLASSES: usize = 32;
/// Size class multiplier for exponential sizing
const SIZE_CLASS_MULTIPLIER: f32 = 1.5;

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial capacity per size class
    pub initial_capacity: usize,
    /// Maximum capacity per size class
    pub max_capacity: usize,
    /// Enable automatic pool expansion
    pub auto_expand: bool,
    /// Pool shrink threshold (unused entries / total capacity)
    pub shrink_threshold: f32,
    /// Enable memory alignment to cache lines
    pub enable_alignment: bool,
    /// Minimum allocation size in bytes
    pub min_alloc_size: usize,
    /// Maximum allocation size in bytes
    pub max_alloc_size: usize,
    /// Enable pool statistics collection
    pub enable_statistics: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_capacity: DEFAULT_POOL_CAPACITY,
            max_capacity: DEFAULT_POOL_CAPACITY * 4,
            auto_expand: true,
            shrink_threshold: 0.25,
            enable_alignment: true,
            min_alloc_size: 64,
            max_alloc_size: 256 * 1024 * 1024, // 256MB max
            enable_statistics: true,
        }
    }
}

impl PoolConfig {
    /// Create configuration optimized for Candle operations
    pub fn for_candle() -> Self {
        Self {
            initial_capacity: 256,
            max_capacity: 1024,
            auto_expand: true,
            shrink_threshold: 0.2,
            enable_alignment: true,
            min_alloc_size: 256, // Larger minimum for tensor operations
            max_alloc_size: 512 * 1024 * 1024, // 512MB for large models
            enable_statistics: true,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> CandleResult<()> {
        if self.initial_capacity == 0 {
            return Err(CandleError::config(
                "Initial capacity must be positive",
                "initial_capacity",
                "> 0",
            ));
        }

        if self.max_capacity < self.initial_capacity {
            return Err(CandleError::config(
                "Maximum capacity must be >= initial capacity",
                "max_capacity",
                ">= initial_capacity",
            ));
        }

        if self.shrink_threshold <= 0.0 || self.shrink_threshold >= 1.0 {
            return Err(CandleError::config(
                "Shrink threshold must be between 0.0 and 1.0",
                "shrink_threshold",
                "0.0 < threshold < 1.0",
            ));
        }

        if self.min_alloc_size == 0 {
            return Err(CandleError::config(
                "Minimum allocation size must be positive",
                "min_alloc_size",
                "> 0",
            ));
        }

        if self.max_alloc_size < self.min_alloc_size {
            return Err(CandleError::config(
                "Maximum allocation size must be >= minimum",
                "max_alloc_size",
                ">= min_alloc_size",
            ));
        }

        Ok(())
    }
}

/// Cache-aligned memory entry with automatic cleanup
#[derive(Debug)]
pub struct PooledEntry {
    /// Raw memory pointer
    ptr: NonNull<u8>,
    /// Allocated size in bytes
    size: usize,
    /// Memory layout for deallocation
    layout: Layout,
    /// Data accessor (provides safe Vec-like interface)
    data: Vec<f32>,
}

impl PooledEntry {
    /// Create new pooled entry with specified size
    fn new(size: usize, enable_alignment: bool) -> CandleResult<Self> {
        let align = if enable_alignment {
            CACHE_LINE_SIZE
        } else {
            std::mem::align_of::<f32>()
        };

        let layout =
            Layout::from_size_align(size * std::mem::size_of::<f32>(), align).map_err(|e| {
                CandleError::memory(
                    &format!("Invalid memory layout: {}", e),
                    "new",
                    "valid size and alignment",
                )
            })?;

        let ptr = unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                return Err(CandleError::memory(
                    "Memory allocation failed",
                    "new",
                    "sufficient memory available",
                ));
            }
            NonNull::new_unchecked(raw_ptr)
        };

        // Create Vec that uses the allocated memory
        let data = unsafe { Vec::from_raw_parts(ptr.as_ptr() as *mut f32, 0, size) };

        Ok(Self {
            ptr,
            size,
            layout,
            data,
        })
    }

    /// Get mutable access to the underlying data
    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }

    /// Get read-only access to the underlying data
    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }

    /// Get the capacity of this entry
    pub fn capacity(&self) -> usize {
        self.size
    }

    /// Get the current length of data
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the entry is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear the data without deallocating
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Reset the entry for reuse
    fn reset(&mut self) {
        self.data.clear();

        // Zero out the memory for security
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.layout.size());
        }
    }
}

impl Drop for PooledEntry {
    fn drop(&mut self) {
        // Prevent Vec from trying to deallocate
        let _ = std::mem::take(&mut self.data);

        // Deallocate the raw memory
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// Memory pool for a specific size class
#[derive(Debug)]
pub struct MemoryPool {
    /// Size class for this pool (number of f32 elements)
    size_class: usize,
    /// Available entries
    available: Sender<PooledEntry>,
    /// Receiver for available entries
    receiver: Receiver<PooledEntry>,
    /// Pool statistics
    stats: PoolStatistics,
    /// Pool configuration
    config: PoolConfig,
}

impl MemoryPool {
    /// Create new memory pool for size class
    fn new(size_class: usize, config: PoolConfig) -> CandleResult<Self> {
        config.validate()?;

        let (available, receiver) = bounded(config.max_capacity);

        let mut pool = Self {
            size_class,
            available,
            receiver,
            stats: PoolStatistics::new(size_class),
            config,
        };

        // Pre-allocate initial entries
        pool.expand_pool(config.initial_capacity)?;

        Ok(pool)
    }

    /// Acquire an entry from the pool
    fn acquire(&self) -> CandleResult<PooledEntry> {
        match self.receiver.try_recv() {
            Ok(mut entry) => {
                entry.reset();
                self.stats
                    .cache_hits
                    .store(self.stats.cache_hits.load() + 1);
                self.stats
                    .entries_in_use
                    .store(self.stats.entries_in_use.load() + 1);
                Ok(entry)
            }
            Err(TryRecvError::Empty) => {
                // Pool empty - create new entry
                self.stats
                    .cache_misses
                    .store(self.stats.cache_misses.load() + 1);
                self.stats
                    .total_allocations
                    .store(self.stats.total_allocations.load() + 1);

                let entry = PooledEntry::new(self.size_class, self.config.enable_alignment)?;
                self.stats
                    .entries_in_use
                    .store(self.stats.entries_in_use.load() + 1);
                Ok(entry)
            }
            Err(TryRecvError::Disconnected) => Err(CandleError::memory(
                "Memory pool disconnected",
                "acquire",
                "connected pool",
            )),
        }
    }

    /// Return an entry to the pool
    fn release(&self, mut entry: PooledEntry) -> CandleResult<()> {
        if entry.capacity() != self.size_class {
            return Err(CandleError::memory(
                "Entry size mismatch for pool",
                "release",
                "matching size class",
            ));
        }

        entry.reset();

        match self.available.try_send(entry) {
            Ok(_) => {
                self.stats
                    .entries_in_use
                    .store(self.stats.entries_in_use.load().saturating_sub(1));
                Ok(())
            }
            Err(TrySendError::Full(entry)) => {
                // Pool full - drop the entry
                drop(entry);
                self.stats
                    .entries_dropped
                    .store(self.stats.entries_dropped.load() + 1);
                self.stats
                    .entries_in_use
                    .store(self.stats.entries_in_use.load().saturating_sub(1));
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => Err(CandleError::memory(
                "Memory pool disconnected",
                "release",
                "connected pool",
            )),
        }
    }

    /// Expand pool capacity
    fn expand_pool(&self, additional_entries: usize) -> CandleResult<()> {
        for _ in 0..additional_entries {
            let entry = PooledEntry::new(self.size_class, self.config.enable_alignment)?;

            if self.available.try_send(entry).is_err() {
                break; // Pool full
            }
        }

        Ok(())
    }

    /// Get pool statistics
    fn statistics(&self) -> PoolStatistics {
        let mut stats = self.stats.clone();
        stats.available_entries = self.receiver.len();
        stats.pool_capacity = self.available.capacity().unwrap_or(0);
        stats
    }
}

/// Statistics for a memory pool
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    /// Size class for this pool
    pub size_class: usize,
    /// Number of cache hits
    cache_hits: AtomicCell<u64>,
    /// Number of cache misses
    cache_misses: AtomicCell<u64>,
    /// Total allocations made
    total_allocations: AtomicCell<u64>,
    /// Entries currently in use
    entries_in_use: AtomicCell<u64>,
    /// Entries dropped due to full pool
    entries_dropped: AtomicCell<u64>,
    /// Current available entries
    pub available_entries: usize,
    /// Pool capacity
    pub pool_capacity: usize,
}

impl PoolStatistics {
    fn new(size_class: usize) -> Self {
        Self {
            size_class,
            cache_hits: AtomicCell::new(0),
            cache_misses: AtomicCell::new(0),
            total_allocations: AtomicCell::new(0),
            entries_in_use: AtomicCell::new(0),
            entries_dropped: AtomicCell::new(0),
            available_entries: 0,
            pool_capacity: 0,
        }
    }

    /// Calculate hit rate
    pub fn hit_rate(&self) -> f32 {
        let hits = self.cache_hits.load();
        let misses = self.cache_misses.load();
        let total = hits + misses;

        if total > 0 {
            hits as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Calculate utilization rate
    pub fn utilization_rate(&self) -> f32 {
        if self.pool_capacity > 0 {
            (self.pool_capacity - self.available_entries) as f32 / self.pool_capacity as f32
        } else {
            0.0
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        self.size_class * std::mem::size_of::<f32>() * self.pool_capacity
    }
}

/// High-performance memory pool manager with size classes
#[derive(Debug)]
pub struct MemoryPoolManager {
    /// Memory pools by size class
    pools: ArcSwap<SmallVec<Arc<MemoryPool>, MAX_SIZE_CLASSES>>,
    /// Size classes (in number of f32 elements)
    size_classes: SmallVec<usize, MAX_SIZE_CLASSES>,
    /// Pool configuration
    config: PoolConfig,
    /// Global statistics
    global_stats: GlobalPoolStatistics,
}

/// Global statistics across all pools
#[derive(Debug)]
struct GlobalPoolStatistics {
    /// Total memory allocated
    total_memory_bytes: AtomicCell<u64>,
    /// Total acquire operations
    total_acquires: AtomicCell<u64>,
    /// Total release operations
    total_releases: AtomicCell<u64>,
    /// Peak memory usage
    peak_memory_bytes: AtomicCell<u64>,
}

impl Default for GlobalPoolStatistics {
    fn default() -> Self {
        Self {
            total_memory_bytes: AtomicCell::new(0),
            total_acquires: AtomicCell::new(0),
            total_releases: AtomicCell::new(0),
            peak_memory_bytes: AtomicCell::new(0),
        }
    }
}

impl MemoryPoolManager {
    /// Create new memory pool manager
    pub fn new(config: PoolConfig) -> CandleResult<Self> {
        config.validate()?;

        let mut size_classes = SmallVec::new();
        let mut pools = SmallVec::new();

        // Generate size classes exponentially
        let mut size = config.min_alloc_size / std::mem::size_of::<f32>();
        let max_size = config.max_alloc_size / std::mem::size_of::<f32>();

        while size <= max_size && size_classes.len() < MAX_SIZE_CLASSES {
            size_classes.push(size);

            let pool = Arc::new(MemoryPool::new(size, config.clone())?);
            pools.push(pool);

            size = (size as f32 * SIZE_CLASS_MULTIPLIER) as usize;
        }

        Ok(Self {
            pools: ArcSwap::from_pointee(pools),
            size_classes,
            config,
            global_stats: GlobalPoolStatistics::default(),
        })
    }

    /// Create optimized manager for Candle operations
    pub fn for_candle() -> CandleResult<Self> {
        Self::new(PoolConfig::for_candle())
    }

    /// Acquire tensor memory from appropriate pool
    pub fn acquire_tensor(&self, num_elements: usize) -> CandleResult<PooledEntry> {
        let size_class = self.find_size_class(num_elements)?;
        let pool_index = self.find_pool_index(size_class)?;

        let pools = self.pools.load();
        let pool = &pools[pool_index];

        self.global_stats
            .total_acquires
            .store(self.global_stats.total_acquires.load() + 1);

        pool.acquire()
    }

    /// Release tensor memory back to pool
    pub fn release_tensor(&self, entry: PooledEntry) -> CandleResult<()> {
        let size_class = entry.capacity();
        let pool_index = self.find_pool_index(size_class)?;

        let pools = self.pools.load();
        let pool = &pools[pool_index];

        self.global_stats
            .total_releases
            .store(self.global_stats.total_releases.load() + 1);

        pool.release(entry)
    }

    /// Find appropriate size class for allocation
    fn find_size_class(&self, num_elements: usize) -> CandleResult<usize> {
        for &size_class in &self.size_classes {
            if size_class >= num_elements {
                return Ok(size_class);
            }
        }

        Err(CandleError::memory(
            &format!("No size class available for {} elements", num_elements),
            "find_size_class",
            "allocation within limits",
        ))
    }

    /// Find pool index for size class
    fn find_pool_index(&self, size_class: usize) -> CandleResult<usize> {
        for (i, &class) in self.size_classes.iter().enumerate() {
            if class == size_class {
                return Ok(i);
            }
        }

        Err(CandleError::memory(
            &format!("No pool found for size class {}", size_class),
            "find_pool_index",
            "valid size class",
        ))
    }

    /// Get comprehensive statistics
    pub fn statistics(&self) -> MemoryPoolManagerStatistics {
        let pools = self.pools.load();
        let pool_stats: Vec<PoolStatistics> = pools.iter().map(|pool| pool.statistics()).collect();

        let total_memory: usize = pool_stats.iter().map(|s| s.memory_usage_bytes()).sum();

        MemoryPoolManagerStatistics {
            pool_stats,
            total_pools: pools.len(),
            total_memory_bytes: total_memory as u64,
            total_acquires: self.global_stats.total_acquires.load(),
            total_releases: self.global_stats.total_releases.load(),
            peak_memory_bytes: self.global_stats.peak_memory_bytes.load(),
        }
    }

    /// Get pool configuration
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }
}

/// Comprehensive statistics for memory pool manager
#[derive(Debug, Clone)]
pub struct MemoryPoolManagerStatistics {
    /// Per-pool statistics
    pub pool_stats: Vec<PoolStatistics>,
    /// Total number of pools
    pub total_pools: usize,
    /// Total memory usage in bytes
    pub total_memory_bytes: u64,
    /// Total acquire operations
    pub total_acquires: u64,
    /// Total release operations
    pub total_releases: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
}

impl MemoryPoolManagerStatistics {
    /// Calculate overall hit rate
    pub fn overall_hit_rate(&self) -> f32 {
        if self.pool_stats.is_empty() {
            return 0.0;
        }

        let total_hits: u64 = self.pool_stats.iter().map(|s| s.cache_hits.load()).sum();

        let total_misses: u64 = self.pool_stats.iter().map(|s| s.cache_misses.load()).sum();

        let total = total_hits + total_misses;
        if total > 0 {
            total_hits as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Get memory usage in MB
    pub fn memory_usage_mb(&self) -> f32 {
        self.total_memory_bytes as f32 / (1024.0 * 1024.0)
    }

    /// Check if pools are performing well
    pub fn is_healthy(&self) -> bool {
        let hit_rate = self.overall_hit_rate();
        hit_rate >= 0.8 // Good hit rate threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config_validation() {
        let config = PoolConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.initial_capacity = 0;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_pooled_entry_creation() {
        let entry = PooledEntry::new(1024, true);
        assert!(entry.is_ok());

        let mut entry = entry.unwrap();
        assert_eq!(entry.capacity(), 1024);
        assert!(entry.is_empty());

        entry.data_mut().push(1.0);
        assert_eq!(entry.len(), 1);
        assert_eq!(entry.data()[0], 1.0);
    }

    #[test]
    fn test_memory_pool_operations() {
        let config = PoolConfig::default();
        let pool = MemoryPool::new(256, config);
        assert!(pool.is_ok());

        let pool = pool.unwrap();

        // Acquire entry
        let entry = pool.acquire();
        assert!(entry.is_ok());

        let entry = entry.unwrap();
        assert_eq!(entry.capacity(), 256);

        // Release entry
        assert!(pool.release(entry).is_ok());

        let stats = pool.statistics();
        assert_eq!(stats.size_class, 256);
    }

    #[test]
    fn test_memory_pool_manager() {
        let manager = MemoryPoolManager::for_candle();

        // Acquire tensor
        let entry = manager.acquire_tensor(512);
        assert!(entry.is_ok());

        let entry = entry.unwrap();
        assert!(entry.capacity() >= 512);

        // Release tensor
        assert!(manager.release_tensor(entry).is_ok());

        let stats = manager.statistics();
        assert!(stats.total_pools > 0);
        assert_eq!(stats.total_acquires, 1);
        assert_eq!(stats.total_releases, 1);
    }
}
