//! Ultra-High-Performance KV Cache System for Multi-Head Attention
//!
//! Zero-allocation, lock-free KV cache optimized for transformer attention mechanisms with:
//! - Pre-allocated memory pools with intelligent overflow handling
//! - Lock-free multi-head attention support with atomic operations  
//! - Cache-friendly memory layout with SIMD-optimized operations
//! - Sophisticated eviction strategies without blocking operations
//! - Comprehensive performance monitoring with zero overhead
//! - Automatic cleanup with configurable memory pressure thresholds
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
//! │   Attention     │ -> │     KVCache      │ -> │   Cached Tensors    │
//! │  (Multi-Head)   │    │  (Lock-Free)     │    │ (Memory Pooled)     │
//! └─────────────────┘    └──────────────────┘    └─────────────────────┘
//!                               │
//!                        ┌──────────────────┐
//!                        │ EvictionStrategy │
//!                        │ (Intelligent)    │
//!                        └──────────────────┘
//! ```
//!
//! ## Performance Features
//!
//! - **Zero Allocation**: Pre-allocated pools, stack-based metadata
//! - **Lock-Free**: Atomic operations, wait-free data structures
//! - **Cache-Friendly**: Aligned layouts, hot/cold separation
//! - **SIMD-Optimized**: Vectorized operations for batch processing
//! - **Memory Efficient**: Compact encoding, intelligent compression
//! - **Eviction Smart**: Predictive algorithms, usage-based cleanup
//!
//! ## Usage Examples
//!
//! ### Basic KV Caching
//!
//! ```rust
//! use fluent_ai_candle::kv_cache::{KVCache, KVCacheBuilder, EvictionStrategy};
//!
//! // Create high-performance cache
//! let cache = KVCacheBuilder::new()
//!     .num_heads(32)
//!     .max_sequence_length(8192)
//!     .eviction_strategy(EvictionStrategy::AdaptiveLRU)
//!     .enable_compression()
//!     .build()?;
//!
//! // Store key-value pairs for attention
//! cache.store(head_idx, seq_pos, key_tensor, value_tensor)?;
//!
//! // Retrieve with zero-copy access
//! if let Some((key, value)) = cache.get(head_idx, seq_pos) {
//!     // Use cached tensors directly
//! }
//! ```
//!
//! ### Advanced Multi-Head Usage
//!
//! ```rust
//! use fluent_ai_candle::kv_cache::*;
//!
//! // Configure for large transformer model
//! let config = KVCacheConfig::new()
//!     .with_num_heads(64)
//!     .with_head_dim(128)
//!     .with_max_sequence_length(32768)
//!     .with_eviction_strategy(EvictionStrategy::AdaptiveLFU)
//!     .enable_all_optimizations();
//!
//! let cache = KVCache::with_config(config)?;
//!
//! // Batch operations for multiple heads
//! cache.store_batch(&head_indices, &seq_positions, &keys, &values)?;
//! ```

// Module declarations
pub mod types;
pub mod config;
pub mod stats;
pub mod eviction;
pub mod memory;
pub mod builder;

// Re-export core types and traits
pub use types::{
    KVCache, KVCacheEntry, HeadTable, CacheKey,
    MAX_ATTENTION_HEADS, MAX_CACHE_ENTRIES_PER_HEAD, MAX_SEQUENCE_LENGTH,
    CACHE_LINE_SIZE, MEMORY_POOL_BLOCK_SIZE, MAX_MEMORY_POOLS,
    ERR_CACHE_FULL, ERR_INVALID_HEAD, ERR_INVALID_POSITION, 
    ERR_TENSOR_MISMATCH, ERR_DEVICE_MISMATCH,
};

pub use config::{
    KVCacheConfig, ConfigError, ConfigPresets,
};

pub use stats::{
    CacheStats, StatsSummary, StatsBenchmark, BenchmarkResult,
};

pub use eviction::{
    EvictionStrategy, EvictionManager, EvictionComplexity,
    AccessTracker, AccessStats,
};

pub use memory::{
    MemoryPool, MemoryPoolStats, MemoryPoolCollection, 
    PoolCollectionStats, PoolHealth,
};

pub use builder::{
    KVCacheBuilder, BuilderPresets, ConfigWizard,
    ModelSize, WorkloadType, PerformancePriority,
};

/// Version information
pub const KV_CACHE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information  
pub const KV_CACHE_BUILD_INFO: &str = concat!(
    "fluent_ai_candle::kv_cache v",
    env!("CARGO_PKG_VERSION"),
    " - Ultra-high-performance KV cache with zero allocation"
);

/// Module feature flags (compile-time configuration)
pub mod features {
    /// SIMD optimization support
    pub const SIMD_ENABLED: bool = cfg!(target_feature = "avx2") || cfg!(target_feature = "neon");
    
    /// Compression support
    pub const COMPRESSION_ENABLED: bool = true;
    
    /// Statistics collection support
    pub const STATISTICS_ENABLED: bool = true;
    
    /// Memory pooling support
    pub const MEMORY_POOLING_ENABLED: bool = true;
    
    /// Batch operations support
    pub const BATCH_OPERATIONS_ENABLED: bool = true;
    
    /// Prefetching support
    pub const PREFETCH_ENABLED: bool = cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64");
}

/// Performance tuning constants
pub mod tuning {
    /// Default cache line size for alignment
    pub const DEFAULT_CACHE_LINE_SIZE: usize = 64;
    
    /// Default memory pool block size
    pub const DEFAULT_MEMORY_POOL_BLOCK_SIZE: usize = 4096;
    
    /// Default eviction batch size
    pub const DEFAULT_EVICTION_BATCH_SIZE: usize = 64;
    
    /// Maximum SIMD batch size for vectorized operations
    pub const MAX_SIMD_BATCH_SIZE: usize = 8;
    
    /// Prefetch distance for memory access patterns
    pub const PREFETCH_DISTANCE: usize = 64;
    
    /// Hash table load factor threshold for resizing
    pub const HASH_LOAD_FACTOR_THRESHOLD: f64 = 0.75;
    
    /// Memory pressure threshold for aggressive eviction
    pub const MEMORY_PRESSURE_THRESHOLD: f64 = 0.9;
    
    /// Statistics collection interval in operations
    pub const STATS_COLLECTION_INTERVAL: usize = 1000;
}

/// Utility functions for cache management
pub mod utils {
    use super::*;
    
    /// Calculate optimal cache size for given model parameters
    pub fn calculate_optimal_cache_size(
        num_heads: usize,
        head_dim: usize,
        max_seq_length: usize,
        dtype_size: usize,
    ) -> usize {
        // Calculate memory needed for key and value tensors
        let tensor_size = head_dim * dtype_size;
        let max_entries_per_head = max_seq_length;
        let total_entries = num_heads * max_entries_per_head;
        
        // Account for both key and value tensors plus metadata overhead
        let entry_overhead = std::mem::size_of::<types::KVCacheEntry>();
        total_entries * (tensor_size * 2 + entry_overhead)
    }
    
    /// Get recommended eviction strategy based on usage pattern
    pub fn recommend_eviction_strategy(
        access_pattern: AccessPattern,
        memory_pressure: MemoryPressure,
    ) -> EvictionStrategy {
        match (access_pattern, memory_pressure) {
            (AccessPattern::Sequential, _) => EvictionStrategy::FIFO,
            (AccessPattern::Random, MemoryPressure::Low) => EvictionStrategy::Random,
            (AccessPattern::Random, _) => EvictionStrategy::LRU,
            (AccessPattern::Temporal, _) => EvictionStrategy::AdaptiveLRU,
            (AccessPattern::Frequency, _) => EvictionStrategy::AdaptiveLFU,
            (AccessPattern::Mixed, MemoryPressure::High) => EvictionStrategy::TTL,
            (AccessPattern::Mixed, _) => EvictionStrategy::AdaptiveLRU,
        }
    }
    
    /// Estimate cache performance for given configuration
    pub fn estimate_cache_performance(config: &KVCacheConfig) -> PerformanceEstimate {
        let memory_mb = config.estimated_memory_bytes() / (1024 * 1024);
        
        // Simple heuristic-based performance estimation
        let throughput_ops_per_sec = match config.eviction_strategy() {
            EvictionStrategy::Random | EvictionStrategy::FIFO => 1_000_000.0,
            EvictionStrategy::LRU | EvictionStrategy::LFU => 800_000.0,
            EvictionStrategy::AdaptiveLRU | EvictionStrategy::AdaptiveLFU => 600_000.0,
            EvictionStrategy::TTL => 700_000.0,
            EvictionStrategy::Clock | EvictionStrategy::SecondChance => 850_000.0,
        };
        
        let expected_hit_ratio = match config.eviction_strategy() {
            EvictionStrategy::Random => 0.6,
            EvictionStrategy::FIFO => 0.7,
            EvictionStrategy::LRU => 0.85,
            EvictionStrategy::LFU => 0.8,
            EvictionStrategy::AdaptiveLRU => 0.9,
            EvictionStrategy::AdaptiveLFU => 0.88,
            EvictionStrategy::TTL => 0.75,
            EvictionStrategy::Clock => 0.82,
            EvictionStrategy::SecondChance => 0.84,
        };
        
        PerformanceEstimate {
            memory_usage_mb: memory_mb,
            throughput_ops_per_sec,
            expected_hit_ratio,
            latency_percentile_95_ns: 1000.0 / throughput_ops_per_sec * 1_000_000_000.0 * 1.5,
        }
    }
    
    /// Access pattern classification
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AccessPattern {
        /// Sequential access pattern
        Sequential,
        /// Random access pattern
        Random,
        /// Temporal locality (recent items accessed more)
        Temporal,
        /// Frequency-based (popular items accessed more)
        Frequency,
        /// Mixed access pattern
        Mixed,
    }
    
    /// Memory pressure level
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum MemoryPressure {
        /// Low memory pressure (< 70% utilization)
        Low,
        /// Medium memory pressure (70-90% utilization)
        Medium,
        /// High memory pressure (> 90% utilization)
        High,
    }
    
    /// Performance estimation results
    #[derive(Debug, Clone)]
    pub struct PerformanceEstimate {
        /// Estimated memory usage in MB
        pub memory_usage_mb: usize,
        /// Estimated throughput in operations per second
        pub throughput_ops_per_sec: f64,
        /// Expected cache hit ratio (0.0-1.0)
        pub expected_hit_ratio: f64,
        /// 95th percentile latency in nanoseconds
        pub latency_percentile_95_ns: f64,
    }
    
    impl std::fmt::Display for PerformanceEstimate {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Performance Estimate: {} MB, {:.0} ops/sec, {:.1}% hit rate, {:.0}ns p95",
                self.memory_usage_mb,
                self.throughput_ops_per_sec,
                self.expected_hit_ratio * 100.0,
                self.latency_percentile_95_ns
            )
        }
    }
}

/// Testing utilities (conditionally compiled)
#[cfg(test)]
pub mod test_utils {
    use super::*;
    use candle_core::{Tensor, Device, DType};
    
    /// Create test tensor with given shape
    pub fn create_test_tensor(shape: &[usize]) -> Tensor {
        Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
    }
    
    /// Create test cache with default configuration
    pub fn create_test_cache() -> KVCache {
        KVCacheBuilder::new()
            .small_model()
            .disable_statistics() // Reduce overhead in tests
            .build()
            .unwrap()
    }
    
    /// Create test cache with custom configuration
    pub fn create_test_cache_with_config(config: KVCacheConfig) -> KVCache {
        KVCache::with_config(config).unwrap()
    }
    
    /// Fill cache with test data
    pub fn fill_cache_with_test_data(cache: &mut KVCache, num_entries: usize) {
        let key_tensor = create_test_tensor(&[64, 128]);
        let value_tensor = create_test_tensor(&[64, 128]);
        
        for i in 0..num_entries {
            let head_idx = i % 8;
            let seq_pos = i / 8;
            let _ = cache.store(head_idx, seq_pos, key_tensor.clone(), value_tensor.clone());
        }
    }
    
    /// Benchmark cache operations
    pub fn benchmark_cache_operations(cache: &mut KVCache, num_operations: usize) -> std::time::Duration {
        let key_tensor = create_test_tensor(&[64, 128]);
        let value_tensor = create_test_tensor(&[64, 128]);
        
        let start = std::time::Instant::now();
        
        for i in 0..num_operations {
            let head_idx = i % 8;
            let seq_pos = i % 1024;
            
            if i % 2 == 0 {
                let _ = cache.store(head_idx, seq_pos, key_tensor.clone(), value_tensor.clone());
            } else {
                let _ = cache.get(head_idx, seq_pos);
            }
        }
        
        start.elapsed()
    }
}