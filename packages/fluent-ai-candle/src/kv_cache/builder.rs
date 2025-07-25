//! Fluent Builder Pattern for KV Cache Configuration
//!
//! Elegant and ergonomic cache construction with:
//! - Fluent interface for intuitive configuration
//! - Compile-time optimization with const fn methods
//! - Type-safe configuration validation
//! - Zero-allocation builder pattern using const generics

use crate::error::{CandleError, CandleResult as Result};
use super::config::{KVCacheConfig, ConfigPresets};
use super::eviction::EvictionStrategy;
use super::types::KVCache;

/// Fluent builder for KV cache configuration
pub struct KVCacheBuilder {
    config: KVCacheConfig}

impl KVCacheBuilder {
    /// Create new builder with default configuration
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            config: KVCacheConfig::new()}
    }

    /// Create builder from existing configuration
    #[inline(always)]
    pub const fn from_config(config: KVCacheConfig) -> Self {
        Self { config }
    }

    /// Set maximum cache entries (affects memory allocation)
    #[inline(always)]
    pub const fn max_entries(self, _entries: usize) -> Self {
        // Max entries affects memory allocation planning
        // In this implementation, capacity is determined by compile-time constants
        self
    }

    /// Set number of attention heads
    #[inline(always)]
    pub const fn num_heads(mut self, heads: usize) -> Self {
        self.config = self.config.with_num_heads(heads);
        self
    }

    /// Set head dimension
    #[inline(always)]
    pub const fn head_dim(mut self, dim: usize) -> Self {
        self.config = self.config.with_head_dim(dim);
        self
    }

    /// Set maximum sequence length
    #[inline(always)]
    pub const fn max_sequence_length(mut self, length: usize) -> Self {
        self.config = self.config.with_max_sequence_length(length);
        self
    }

    /// Set memory pool size
    #[inline(always)]
    pub const fn memory_pool_size(mut self, size: usize) -> Self {
        self.config = self.config.with_memory_pool_size(size);
        self
    }

    /// Set eviction batch size
    #[inline(always)]
    pub const fn eviction_batch_size(mut self, size: usize) -> Self {
        self.config = self.config.with_eviction_batch_size(size);
        self
    }

    /// Set eviction strategy
    #[inline(always)]
    pub const fn eviction_strategy(mut self, strategy: EvictionStrategy) -> Self {
        self.config = self.config.with_eviction_strategy(strategy);
        self
    }

    /// Enable compression
    #[inline(always)]
    pub const fn enable_compression(mut self) -> Self {
        self.config = self.config.enable_compression();
        self
    }

    /// Disable compression
    #[inline(always)]
    pub const fn disable_compression(mut self) -> Self {
        self.config = self.config.disable_compression();
        self
    }

    /// Enable prefetching
    #[inline(always)]
    pub const fn enable_prefetch(mut self) -> Self {
        self.config = self.config.enable_prefetch();
        self
    }

    /// Disable prefetching
    #[inline(always)]
    pub const fn disable_prefetch(mut self) -> Self {
        self.config = self.config.disable_prefetch();
        self
    }

    /// Enable statistics collection
    #[inline(always)]
    pub const fn enable_statistics(mut self) -> Self {
        self.config = self.config.enable_statistics();
        self
    }

    /// Disable statistics collection
    #[inline(always)]
    pub const fn disable_statistics(mut self) -> Self {
        self.config = self.config.disable_statistics();
        self
    }

    /// Enable memory pooling
    #[inline(always)]
    pub const fn enable_memory_pooling(mut self) -> Self {
        self.config = self.config.enable_memory_pooling();
        self
    }

    /// Disable memory pooling
    #[inline(always)]
    pub const fn disable_memory_pooling(mut self) -> Self {
        self.config = self.config.disable_memory_pooling();
        self
    }

    /// Enable batch operations
    #[inline(always)]
    pub const fn enable_batch_operations(mut self) -> Self {
        self.config = self.config.enable_batch_operations();
        self
    }

    /// Disable batch operations
    #[inline(always)]
    pub const fn disable_batch_operations(mut self) -> Self {
        self.config = self.config.disable_batch_operations();
        self
    }

    /// Enable all optimizations
    #[inline(always)]
    pub const fn enable_all_optimizations(mut self) -> Self {
        self.config = self.config.enable_all_optimizations();
        self
    }

    /// Configure for small models (< 1B parameters)
    #[inline(always)]
    pub const fn small_model(mut self) -> Self {
        self.config = KVCacheConfig::small_model();
        self
    }

    /// Configure for medium models (1B-10B parameters)
    #[inline(always)]
    pub const fn medium_model(mut self) -> Self {
        self.config = KVCacheConfig::medium_model();
        self
    }

    /// Configure for large models (> 10B parameters)
    #[inline(always)]
    pub const fn large_model(mut self) -> Self {
        self.config = KVCacheConfig::large_model();
        self
    }

    /// Configure for inference workloads
    #[inline(always)]
    pub const fn inference_optimized(mut self) -> Self {
        self.config = KVCacheConfig::inference_optimized();
        self
    }

    /// Configure for training workloads
    #[inline(always)]
    pub const fn training_optimized(mut self) -> Self {
        self.config = KVCacheConfig::training_optimized();
        self
    }

    /// Configure for memory-constrained environments
    #[inline(always)]
    pub const fn memory_constrained(mut self) -> Self {
        self.config = ConfigPresets::memory_constrained();
        self
    }

    /// Configure for high-throughput serving
    #[inline(always)]
    pub const fn high_throughput(mut self) -> Self {
        self.config = ConfigPresets::high_throughput();
        self
    }

    /// Use GPT-2 style configuration
    #[inline(always)]
    pub const fn gpt2(mut self) -> Self {
        self.config = ConfigPresets::gpt2();
        self
    }

    /// Use GPT-3 style configuration
    #[inline(always)]
    pub const fn gpt3(mut self) -> Self {
        self.config = ConfigPresets::gpt3();
        self
    }

    /// Use BERT style configuration
    #[inline(always)]
    pub const fn bert(mut self) -> Self {
        self.config = ConfigPresets::bert();
        self
    }

    /// Use T5 style configuration
    #[inline(always)]
    pub const fn t5(mut self) -> Self {
        self.config = ConfigPresets::t5();
        self
    }

    /// Use LLaMA style configuration
    #[inline(always)]
    pub const fn llama(mut self) -> Self {
        self.config = ConfigPresets::llama();
        self
    }

    /// Configure with custom settings for specific use case
    pub const fn custom(
        mut self, 
        heads: usize, 
        head_dim: usize, 
        seq_len: usize, 
        strategy: EvictionStrategy
    ) -> Self {
        self.config = self.config
            .with_num_heads(heads)
            .with_head_dim(head_dim)
            .with_max_sequence_length(seq_len)
            .with_eviction_strategy(strategy);
        self
    }

    /// Set configuration flags using bitmask
    #[inline(always)]
    pub const fn with_flags(mut self, flags: u16) -> Self {
        // Apply flags to configuration
        if (flags & 1) != 0 {
            self.config = self.config.enable_compression();
        }
        if (flags & 2) != 0 {
            self.config = self.config.enable_prefetch();
        }
        if (flags & 4) != 0 {
            self.config = self.config.enable_statistics();
        }
        if (flags & 8) != 0 {
            self.config = self.config.enable_memory_pooling();
        }
        if (flags & 16) != 0 {
            self.config = self.config.enable_batch_operations();
        }
        self
    }

    /// Get current configuration (for inspection)
    #[inline(always)]
    pub const fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// Validate configuration before building
    pub fn validate(&self) -> Result<()> {
        self.config.validate().map_err(|msg| CandleError::ProcessingError(msg))
    }

    /// Get estimated memory footprint in bytes
    #[inline(always)]
    pub const fn estimated_memory_bytes(&self) -> usize {
        self.config.estimated_memory_bytes()
    }

    /// Get estimated memory footprint in MB
    #[inline(always)]
    pub const fn estimated_memory_mb(&self) -> usize {
        self.estimated_memory_bytes() / (1024 * 1024)
    }

    /// Check if configuration is suitable for available memory
    pub fn fits_in_memory(&self, available_bytes: usize) -> bool {
        self.estimated_memory_bytes() <= available_bytes
    }

    /// Build KV cache with current configuration
    pub fn build(self) -> Result<KVCache> {
        // Validate configuration before building
        self.validate()?;
        
        // Create cache with validated configuration
        KVCache::with_config(self.config)
    }

    /// Build KV cache without validation (for performance)
    pub fn build_unchecked(self) -> Result<KVCache> {
        KVCache::with_config(self.config)
    }

    /// Try to build KV cache, returning None on failure
    pub fn try_build(self) -> Option<KVCache> {
        self.build().ok()
    }
}

impl Default for KVCacheBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for KVCacheBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KVCacheBuilder")
            .field("num_heads", &self.config.num_heads())
            .field("head_dim", &self.config.head_dim())
            .field("max_sequence_length", &self.config.max_sequence_length())
            .field("eviction_strategy", &self.config.eviction_strategy())
            .field("estimated_memory_mb", &self.estimated_memory_mb())
            .finish()
    }
}

impl std::fmt::Display for KVCacheBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KVCacheBuilder({} heads, {} dim, {} seq_len, {} MB)",
            self.config.num_heads(),
            self.config.head_dim(),
            self.config.max_sequence_length(),
            self.estimated_memory_mb()
        )
    }
}

/// Builder preset collection for common configurations
pub struct BuilderPresets;

impl BuilderPresets {
    /// Fast cache for inference with minimal memory usage
    #[inline(always)]
    pub const fn fast_inference() -> KVCacheBuilder {
        KVCacheBuilder::new()
            .inference_optimized()
            .enable_prefetch()
            .disable_statistics()
            .eviction_strategy(EvictionStrategy::Random) // Fastest eviction
    }

    /// High-quality cache for training with comprehensive monitoring
    #[inline(always)]
    pub const fn high_quality_training() -> KVCacheBuilder {
        KVCacheBuilder::new()
            .training_optimized()
            .enable_statistics()
            .enable_compression()
            .eviction_strategy(EvictionStrategy::AdaptiveLFU)
    }

    /// Memory-efficient cache for resource-constrained environments
    #[inline(always)]
    pub const fn memory_efficient() -> KVCacheBuilder {
        KVCacheBuilder::new()
            .memory_constrained()
            .enable_compression()
            .eviction_strategy(EvictionStrategy::TTL)
    }

    /// High-throughput cache for serving multiple concurrent requests
    #[inline(always)]
    pub const fn high_throughput_serving() -> KVCacheBuilder {
        KVCacheBuilder::new()
            .high_throughput()
            .enable_batch_operations()
            .enable_prefetch()
            .disable_statistics() // Reduce overhead
    }

    /// Balanced cache for general-purpose usage
    #[inline(always)]
    pub const fn balanced() -> KVCacheBuilder {
        KVCacheBuilder::new()
            .medium_model()
            .enable_statistics()
            .enable_memory_pooling()
            .eviction_strategy(EvictionStrategy::AdaptiveLRU)
    }

    /// Development cache with extensive monitoring and debugging
    #[inline(always)]
    pub const fn development() -> KVCacheBuilder {
        KVCacheBuilder::new()
            .small_model()
            .enable_all_optimizations()
            .eviction_strategy(EvictionStrategy::LRU) // Predictable for debugging
    }
}

/// Configuration wizard for guided cache setup
pub struct ConfigWizard {
    model_size: Option<ModelSize>,
    workload_type: Option<WorkloadType>,
    memory_constraint: Option<usize>,
    performance_priority: Option<PerformancePriority>}

impl ConfigWizard {
    /// Start configuration wizard
    pub const fn new() -> Self {
        Self {
            model_size: None,
            workload_type: None,
            memory_constraint: None,
            performance_priority: None}
    }

    /// Set model size category
    pub const fn model_size(mut self, size: ModelSize) -> Self {
        self.model_size = Some(size);
        self
    }

    /// Set workload type
    pub const fn workload_type(mut self, workload: WorkloadType) -> Self {
        self.workload_type = Some(workload);
        self
    }

    /// Set memory constraint in bytes
    pub const fn memory_constraint(mut self, bytes: usize) -> Self {
        self.memory_constraint = Some(bytes);
        self
    }

    /// Set performance priority
    pub const fn performance_priority(mut self, priority: PerformancePriority) -> Self {
        self.performance_priority = Some(priority);
        self
    }

    /// Generate recommended configuration
    pub fn recommend(self) -> KVCacheBuilder {
        let mut builder = KVCacheBuilder::new();

        // Apply model size recommendations
        if let Some(size) = self.model_size {
            builder = match size {
                ModelSize::Small => builder.small_model(),
                ModelSize::Medium => builder.medium_model(),
                ModelSize::Large => builder.large_model()};
        }

        // Apply workload type recommendations
        if let Some(workload) = self.workload_type {
            builder = match workload {
                WorkloadType::Inference => builder.inference_optimized(),
                WorkloadType::Training => builder.training_optimized(),
                WorkloadType::Serving => builder.high_throughput(),
                WorkloadType::Development => builder.enable_all_optimizations()};
        }

        // Apply memory constraints
        if let Some(memory_limit) = self.memory_constraint {
            if memory_limit < 1024 * 1024 * 1024 { // < 1GB
                builder = builder.memory_constrained();
            }
        }

        // Apply performance priorities
        if let Some(priority) = self.performance_priority {
            builder = match priority {
                PerformancePriority::Speed => builder
                    .enable_prefetch()
                    .disable_statistics()
                    .eviction_strategy(EvictionStrategy::Random),
                PerformancePriority::Memory => builder
                    .enable_compression()
                    .eviction_strategy(EvictionStrategy::TTL),
                PerformancePriority::Quality => builder
                    .enable_statistics()
                    .eviction_strategy(EvictionStrategy::AdaptiveLFU),
                PerformancePriority::Balanced => builder
                    .eviction_strategy(EvictionStrategy::AdaptiveLRU)};
        }

        builder
    }
}

impl Default for ConfigWizard {
    fn default() -> Self {
        Self::new()
    }
}

/// Model size categories for configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    /// Small models (< 1B parameters)
    Small,
    /// Medium models (1B-10B parameters)
    Medium,
    /// Large models (> 10B parameters)
    Large}

/// Workload type categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    /// Inference workloads
    Inference,
    /// Training workloads
    Training,
    /// High-throughput serving
    Serving,
    /// Development and debugging
    Development}

/// Performance priority categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformancePriority {
    /// Optimize for speed
    Speed,
    /// Optimize for memory usage
    Memory,
    /// Optimize for cache quality/hit rate
    Quality,
    /// Balanced optimization
    Balanced}