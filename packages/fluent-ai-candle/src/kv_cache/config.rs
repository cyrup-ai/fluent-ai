//! KV Cache Configuration and Builder Pattern
//!
//! Ultra-compact configuration management with:
//! - Bit-packed flags for minimal memory overhead
//! - Const fn builders for compile-time optimization
//! - Fluent interface for elegant configuration
//! - Zero-allocation validation and defaults

use super::eviction::EvictionStrategy;

/// KV cache configuration with ultra-compact storage
///
/// Uses bit-packed flags and optimized layouts for maximum performance.
/// All configuration is validated at creation time for safety.
#[repr(C, align(32))]
#[derive(Clone, Debug)]
pub struct KVCacheConfig {
    /// Number of attention heads
    num_heads: u16,

    /// Head dimension size
    head_dim: u16,

    /// Maximum sequence length
    max_sequence_length: u32,

    /// Memory pool size per pool
    memory_pool_size: u32,

    /// Eviction batch size
    eviction_batch_size: u16,

    /// Configuration flags (bit-packed)
    /// Bit 0: enable_compression
    /// Bit 1: enable_prefetch
    /// Bit 2: enable_statistics
    /// Bit 3: enable_memory_pooling
    /// Bit 4: enable_batch_operations
    /// Bits 5-15: Reserved
    flags: u16,

    /// Eviction strategy
    eviction_strategy: EvictionStrategy,
}

impl KVCacheConfig {
    /// Create new configuration with safe defaults
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            num_heads: 32,
            head_dim: 128,
            max_sequence_length: 8192,
            memory_pool_size: 1024,
            eviction_batch_size: 64,
            flags: 0b11100, // Enable statistics, memory pooling, and batch operations by default
            eviction_strategy: EvictionStrategy::AdaptiveLRU,
        }
    }

    /// Set number of attention heads
    #[inline(always)]
    pub const fn with_num_heads(mut self, heads: usize) -> Self {
        if heads <= u16::MAX as usize {
            self.num_heads = heads as u16;
        }
        self
    }

    /// Set head dimension
    #[inline(always)]
    pub const fn with_head_dim(mut self, dim: usize) -> Self {
        if dim <= u16::MAX as usize {
            self.head_dim = dim as u16;
        }
        self
    }

    /// Set maximum sequence length
    #[inline(always)]
    pub const fn with_max_sequence_length(mut self, length: usize) -> Self {
        if length <= u32::MAX as usize {
            self.max_sequence_length = length as u32;
        }
        self
    }

    /// Set memory pool size
    #[inline(always)]
    pub const fn with_memory_pool_size(mut self, size: usize) -> Self {
        if size <= u32::MAX as usize {
            self.memory_pool_size = size as u32;
        }
        self
    }

    /// Set eviction batch size
    #[inline(always)]
    pub const fn with_eviction_batch_size(mut self, size: usize) -> Self {
        if size <= u16::MAX as usize {
            self.eviction_batch_size = size as u16;
        }
        self
    }

    /// Set eviction strategy
    #[inline(always)]
    pub const fn with_eviction_strategy(mut self, strategy: EvictionStrategy) -> Self {
        self.eviction_strategy = strategy;
        self
    }

    /// Enable compression
    #[inline(always)]
    pub const fn enable_compression(mut self) -> Self {
        self.flags |= 1;
        self
    }

    /// Enable prefetching
    #[inline(always)]
    pub const fn enable_prefetch(mut self) -> Self {
        self.flags |= 2;
        self
    }

    /// Enable statistics collection
    #[inline(always)]
    pub const fn enable_statistics(mut self) -> Self {
        self.flags |= 4;
        self
    }

    /// Enable memory pooling
    #[inline(always)]
    pub const fn enable_memory_pooling(mut self) -> Self {
        self.flags |= 8;
        self
    }

    /// Enable batch operations
    #[inline(always)]
    pub const fn enable_batch_operations(mut self) -> Self {
        self.flags |= 16;
        self
    }

    /// Enable all optimizations
    #[inline(always)]
    pub const fn enable_all_optimizations(mut self) -> Self {
        self.flags = 0b11111;
        self
    }

    /// Disable compression
    #[inline(always)]
    pub const fn disable_compression(mut self) -> Self {
        self.flags &= !1;
        self
    }

    /// Disable prefetching
    #[inline(always)]
    pub const fn disable_prefetch(mut self) -> Self {
        self.flags &= !2;
        self
    }

    /// Disable statistics collection
    #[inline(always)]
    pub const fn disable_statistics(mut self) -> Self {
        self.flags &= !4;
        self
    }

    /// Disable memory pooling
    #[inline(always)]
    pub const fn disable_memory_pooling(mut self) -> Self {
        self.flags &= !8;
        self
    }

    /// Disable batch operations
    #[inline(always)]
    pub const fn disable_batch_operations(mut self) -> Self {
        self.flags &= !16;
        self
    }

    /// Get number of heads
    #[inline(always)]
    pub const fn num_heads(&self) -> usize {
        self.num_heads as usize
    }

    /// Get head dimension
    #[inline(always)]
    pub const fn head_dim(&self) -> usize {
        self.head_dim as usize
    }

    /// Get maximum sequence length
    #[inline(always)]
    pub const fn max_sequence_length(&self) -> usize {
        self.max_sequence_length as usize
    }

    /// Get memory pool size
    #[inline(always)]
    pub const fn memory_pool_size(&self) -> usize {
        self.memory_pool_size as usize
    }

    /// Get eviction batch size
    #[inline(always)]
    pub const fn eviction_batch_size(&self) -> usize {
        self.eviction_batch_size as usize
    }

    /// Get eviction strategy
    #[inline(always)]
    pub const fn eviction_strategy(&self) -> EvictionStrategy {
        self.eviction_strategy
    }

    /// Check if compression is enabled
    #[inline(always)]
    pub const fn compression_enabled(&self) -> bool {
        (self.flags & 1) != 0
    }

    /// Check if prefetch is enabled
    #[inline(always)]
    pub const fn prefetch_enabled(&self) -> bool {
        (self.flags & 2) != 0
    }

    /// Check if statistics are enabled
    #[inline(always)]
    pub const fn statistics_enabled(&self) -> bool {
        (self.flags & 4) != 0
    }

    /// Check if memory pooling is enabled
    #[inline(always)]
    pub const fn memory_pooling_enabled(&self) -> bool {
        (self.flags & 8) != 0
    }

    /// Check if batch operations are enabled
    #[inline(always)]
    pub const fn batch_operations_enabled(&self) -> bool {
        (self.flags & 16) != 0
    }

    /// Get all flags as bitmask
    #[inline(always)]
    pub const fn flags(&self) -> u16 {
        self.flags
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_heads == 0 {
            return Err("Number of heads must be greater than zero");
        }

        if self.head_dim == 0 {
            return Err("Head dimension must be greater than zero");
        }

        if self.max_sequence_length == 0 {
            return Err("Maximum sequence length must be greater than zero");
        }

        if self.memory_pool_size == 0 && self.memory_pooling_enabled() {
            return Err("Memory pool size must be greater than zero when pooling is enabled");
        }

        if self.eviction_batch_size == 0 {
            return Err("Eviction batch size must be greater than zero");
        }

        // Check reasonable limits
        if self.num_heads > 256 {
            return Err("Number of heads exceeds reasonable limit (256)");
        }

        if self.head_dim > 8192 {
            return Err("Head dimension exceeds reasonable limit (8192)");
        }

        if self.max_sequence_length > 1_048_576 {
            return Err("Maximum sequence length exceeds reasonable limit (1M)");
        }

        Ok(())
    }

    /// Get memory footprint estimate in bytes
    #[inline(always)]
    pub const fn estimated_memory_bytes(&self) -> usize {
        let entry_size = std::mem::size_of::<super::types::KVCacheEntry>();
        let max_entries = self.num_heads as usize * 2048; // MAX_CACHE_ENTRIES_PER_HEAD
        let entries_memory = max_entries * entry_size;

        let head_table_size = std::mem::size_of::<super::types::HeadTable>();
        let head_tables_memory = self.num_heads as usize * head_table_size;

        let pools_memory = if self.memory_pooling_enabled() {
            self.memory_pool_size as usize * 16 // MAX_MEMORY_POOLS
        } else {
            0
        };

        entries_memory + head_tables_memory + pools_memory + 1024 // Base overhead
    }

    /// Create configuration optimized for small models
    #[inline(always)]
    pub const fn small_model() -> Self {
        Self::new()
            .with_num_heads(12)
            .with_head_dim(64)
            .with_max_sequence_length(2048)
            .with_memory_pool_size(256)
            .with_eviction_batch_size(16)
            .enable_all_optimizations()
    }

    /// Create configuration optimized for medium models
    #[inline(always)]
    pub const fn medium_model() -> Self {
        Self::new()
            .with_num_heads(32)
            .with_head_dim(128)
            .with_max_sequence_length(8192)
            .with_memory_pool_size(512)
            .with_eviction_batch_size(32)
            .enable_all_optimizations()
    }

    /// Create configuration optimized for large models
    #[inline(always)]
    pub const fn large_model() -> Self {
        Self::new()
            .with_num_heads(64)
            .with_head_dim(256)
            .with_max_sequence_length(32768)
            .with_memory_pool_size(2048)
            .with_eviction_batch_size(128)
            .enable_all_optimizations()
    }

    /// Create configuration optimized for inference
    #[inline(always)]
    pub const fn inference_optimized() -> Self {
        Self::new()
            .enable_prefetch()
            .enable_batch_operations()
            .enable_memory_pooling()
            .disable_statistics() // Reduce overhead during inference
            .with_eviction_strategy(EvictionStrategy::AdaptiveLRU)
    }

    /// Create configuration optimized for training
    #[inline(always)]
    pub const fn training_optimized() -> Self {
        Self::new()
            .enable_statistics()
            .enable_compression()
            .disable_prefetch() // Training patterns are less predictable
            .with_eviction_strategy(EvictionStrategy::AdaptiveLFU)
    }
}

impl Default for KVCacheConfig {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration validation errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    /// Invalid number of heads
    InvalidHeads,
    /// Invalid head dimension
    InvalidHeadDim,
    /// Invalid sequence length
    InvalidSequenceLength,
    /// Invalid memory pool size
    InvalidMemoryPoolSize,
    /// Invalid eviction batch size
    InvalidEvictionBatchSize,
    /// Configuration exceeds memory limits
    MemoryLimitExceeded,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::InvalidHeads => write!(f, "Invalid number of attention heads"),
            ConfigError::InvalidHeadDim => write!(f, "Invalid head dimension"),
            ConfigError::InvalidSequenceLength => write!(f, "Invalid maximum sequence length"),
            ConfigError::InvalidMemoryPoolSize => write!(f, "Invalid memory pool size"),
            ConfigError::InvalidEvictionBatchSize => write!(f, "Invalid eviction batch size"),
            ConfigError::MemoryLimitExceeded => write!(f, "Configuration exceeds memory limits"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Configuration presets for common use cases
pub struct ConfigPresets;

impl ConfigPresets {
    /// Configuration for GPT-2 style models
    #[inline(always)]
    pub const fn gpt2() -> KVCacheConfig {
        KVCacheConfig::new()
            .with_num_heads(12)
            .with_head_dim(64)
            .with_max_sequence_length(1024)
            .enable_all_optimizations()
    }

    /// Configuration for GPT-3 style models
    #[inline(always)]
    pub const fn gpt3() -> KVCacheConfig {
        KVCacheConfig::new()
            .with_num_heads(96)
            .with_head_dim(128)
            .with_max_sequence_length(2048)
            .enable_all_optimizations()
    }

    /// Configuration for BERT style models
    #[inline(always)]
    pub const fn bert() -> KVCacheConfig {
        KVCacheConfig::new()
            .with_num_heads(12)
            .with_head_dim(64)
            .with_max_sequence_length(512)
            .enable_compression()
            .enable_statistics()
    }

    /// Configuration for T5 style models
    #[inline(always)]
    pub const fn t5() -> KVCacheConfig {
        KVCacheConfig::new()
            .with_num_heads(32)
            .with_head_dim(64)
            .with_max_sequence_length(512)
            .enable_batch_operations()
            .enable_memory_pooling()
    }

    /// Configuration for LLaMA style models
    #[inline(always)]
    pub const fn llama() -> KVCacheConfig {
        KVCacheConfig::large_model()
            .with_eviction_strategy(EvictionStrategy::AdaptiveLFU)
    }

    /// Configuration for memory-constrained environments
    #[inline(always)]
    pub const fn memory_constrained() -> KVCacheConfig {
        KVCacheConfig::small_model()
            .enable_compression()
            .with_eviction_batch_size(8)
            .with_memory_pool_size(128)
    }

    /// Configuration for high-throughput serving
    #[inline(always)]
    pub const fn high_throughput() -> KVCacheConfig {
        KVCacheConfig::medium_model()
            .enable_prefetch()
            .enable_batch_operations()
            .disable_statistics()
            .with_eviction_strategy(EvictionStrategy::Random) // Fastest eviction
    }
}