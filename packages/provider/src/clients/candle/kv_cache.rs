//! Lock-free KV cache implementation for transformer attention mechanisms
//!
//! This module provides memory-efficient key-value caching for transformer models
//! with zero-allocation patterns, cache-aligned data structures, and atomic operations.

use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwap;
use arrayvec::ArrayVec;
use crossbeam::atomic::AtomicCell;
use smallvec::SmallVec;

use super::error::{CandleError, CandleResult};
use super::models::CandleModel;

/// Maximum number of attention layers in supported models
const MAX_LAYERS: usize = 80;
/// Maximum sequence length for efficient caching
const MAX_SEQUENCE_LENGTH: usize = 32768;
/// Cache line size for optimal memory alignment
const CACHE_LINE_SIZE: usize = 64;
/// Maximum batch size for efficient processing
const MAX_BATCH_SIZE: usize = 32;

/// Cache-aligned KV tensor data for optimal memory access
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct AlignedKvData {
    /// Key tensor data (flattened)
    keys: SmallVec<[f32; 4096], 4096>,
    /// Value tensor data (flattened)
    values: SmallVec<[f32; 4096], 4096>,
    /// Sequence length this data represents
    sequence_length: u32,
    /// Attention head dimension
    head_dim: u32,
}

impl AlignedKvData {
    /// Create new aligned KV data
    pub fn new(sequence_length: u32, head_dim: u32) -> Self {
        let capacity = (sequence_length * head_dim) as usize;

        Self {
            keys: SmallVec::with_capacity(capacity),
            values: SmallVec::with_capacity(capacity),
            sequence_length,
            head_dim,
        }
    }

    /// Create with pre-allocated data
    pub fn with_data(
        keys: SmallVec<[f32; 4096], 4096>,
        values: SmallVec<[f32; 4096], 4096>,
        sequence_length: u32,
        head_dim: u32,
    ) -> Self {
        Self {
            keys,
            values,
            sequence_length,
            head_dim,
        }
    }

    /// Get key data
    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    /// Get value data
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Get mutable key data
    pub fn keys_mut(&mut self) -> &mut SmallVec<[f32; 4096], 4096> {
        &mut self.keys
    }

    /// Get mutable value data
    pub fn values_mut(&mut self) -> &mut SmallVec<[f32; 4096], 4096> {
        &mut self.values
    }

    /// Get sequence length
    pub fn sequence_length(&self) -> u32 {
        self.sequence_length
    }

    /// Get head dimension
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// Check if data is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty() && self.values.is_empty()
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        (self.keys.len() + self.values.len()) * std::mem::size_of::<f32>()
    }

    /// Append new KV data for incremental generation
    pub fn append_incremental(&mut self, new_keys: &[f32], new_values: &[f32]) -> CandleResult<()> {
        if new_keys.len() != new_values.len() {
            return Err(CandleError::cache(
                "Key and value lengths must match",
                "kv_append",
                "equal lengths",
            ));
        }

        // Check capacity limits
        let new_total_len = self.keys.len() + new_keys.len();
        if new_total_len > MAX_SEQUENCE_LENGTH * self.head_dim as usize {
            return Err(CandleError::cache(
                "KV cache would exceed maximum sequence length",
                "kv_append",
                "within sequence limits",
            ));
        }

        self.keys.extend_from_slice(new_keys);
        self.values.extend_from_slice(new_values);

        // Update sequence length
        self.sequence_length = (self.keys.len() / self.head_dim as usize) as u32;

        Ok(())
    }

    /// Clear all cached data
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.sequence_length = 0;
    }
}

/// Per-layer KV cache with lock-free access patterns
#[derive(Debug)]
pub struct LayerCache {
    /// Layer index in the model
    layer_index: u32,
    /// Number of attention heads
    num_heads: u32,
    /// Dimension per attention head
    head_dim: u32,
    /// Cached KV data per batch element
    batch_cache: ArcSwap<ArrayVec<AlignedKvData, MAX_BATCH_SIZE>>,
    /// Cache statistics for this layer
    stats: LayerCacheStats,
}

/// Statistics for monitoring layer cache performance
#[derive(Debug)]
struct LayerCacheStats {
    /// Total cache hits
    cache_hits: AtomicCell<u64>,
    /// Total cache misses
    cache_misses: AtomicCell<u64>,
    /// Total memory allocated
    total_memory_bytes: AtomicCell<u64>,
    /// Number of cache evictions
    evictions: AtomicCell<u64>,
    /// Peak memory usage
    peak_memory_bytes: AtomicCell<u64>,
}

impl Default for LayerCacheStats {
    fn default() -> Self {
        Self {
            cache_hits: AtomicCell::new(0),
            cache_misses: AtomicCell::new(0),
            total_memory_bytes: AtomicCell::new(0),
            evictions: AtomicCell::new(0),
            peak_memory_bytes: AtomicCell::new(0),
        }
    }
}

impl LayerCache {
    /// Create a new layer cache
    pub fn new(layer_index: u32, num_heads: u32, head_dim: u32) -> Self {
        Self {
            layer_index,
            num_heads,
            head_dim,
            batch_cache: ArcSwap::from_pointee(ArrayVec::new()),
            stats: LayerCacheStats::default(),
        }
    }

    /// Get cached KV data for a batch element
    pub fn get_kv_data(&self, batch_index: usize) -> Option<AlignedKvData> {
        let cache = self.batch_cache.load();

        if batch_index < cache.len() {
            self.stats
                .cache_hits
                .store(self.stats.cache_hits.load() + 1);
            Some(cache[batch_index].clone())
        } else {
            self.stats
                .cache_misses
                .store(self.stats.cache_misses.load() + 1);
            None
        }
    }

    /// Set KV data for a batch element
    pub fn set_kv_data(&self, batch_index: usize, kv_data: AlignedKvData) -> CandleResult<()> {
        let mut new_cache = (**self.batch_cache.load()).clone();

        // Ensure the cache is large enough
        while new_cache.len() <= batch_index {
            if new_cache
                .try_push(AlignedKvData::new(0, self.head_dim))
                .is_err()
            {
                return Err(CandleError::cache(
                    "Batch cache capacity exceeded",
                    "set_kv_data",
                    "within batch limits",
                ));
            }
        }

        // Calculate memory usage
        let memory_usage = kv_data.memory_usage() as u64;
        let new_total = self.stats.total_memory_bytes.load() + memory_usage;

        // Update peak memory if necessary
        let current_peak = self.stats.peak_memory_bytes.load();
        if new_total > current_peak {
            self.stats.peak_memory_bytes.store(new_total);
        }

        new_cache[batch_index] = kv_data;
        self.batch_cache.store(Arc::new(new_cache));

        // Update memory statistics
        self.stats.total_memory_bytes.store(new_total);

        Ok(())
    }

    /// Append incremental KV data for autoregressive generation
    pub fn append_incremental_kv(
        &self,
        batch_index: usize,
        new_keys: &[f32],
        new_values: &[f32],
    ) -> CandleResult<()> {
        let cache = self.batch_cache.load();

        if batch_index >= cache.len() {
            return Err(CandleError::cache(
                "Batch index out of range",
                "append_incremental_kv",
                "valid batch index",
            ));
        }

        let mut new_cache = (**cache).clone();
        new_cache[batch_index].append_incremental(new_keys, new_values)?;

        self.batch_cache.store(Arc::new(new_cache));

        Ok(())
    }

    /// Clear cache for a specific batch element
    pub fn clear_batch_element(&self, batch_index: usize) -> CandleResult<()> {
        let cache = self.batch_cache.load();

        if batch_index >= cache.len() {
            return Ok(()); // Nothing to clear
        }

        let mut new_cache = (**cache).clone();
        let old_memory = new_cache[batch_index].memory_usage() as u64;
        new_cache[batch_index].clear();

        self.batch_cache.store(Arc::new(new_cache));

        // Update memory statistics
        let current_memory = self.stats.total_memory_bytes.load();
        if current_memory >= old_memory {
            self.stats
                .total_memory_bytes
                .store(current_memory - old_memory);
        }

        Ok(())
    }

    /// Clear all cached data for this layer
    pub fn clear_all(&self) {
        let empty_cache = ArrayVec::new();
        self.batch_cache.store(Arc::new(empty_cache));

        // Reset statistics
        self.stats.total_memory_bytes.store(0);
        self.stats.evictions.store(self.stats.evictions.load() + 1);
    }

    /// Get layer index
    pub fn layer_index(&self) -> u32 {
        self.layer_index
    }

    /// Get number of attention heads
    pub fn num_heads(&self) -> u32 {
        self.num_heads
    }

    /// Get head dimension
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// Get cache statistics for this layer
    pub fn statistics(&self) -> LayerCacheStatistics {
        LayerCacheStatistics {
            layer_index: self.layer_index,
            cache_hits: self.stats.cache_hits.load(),
            cache_misses: self.stats.cache_misses.load(),
            hit_rate: {
                let hits = self.stats.cache_hits.load();
                let misses = self.stats.cache_misses.load();
                let total = hits + misses;
                if total > 0 {
                    hits as f32 / total as f32
                } else {
                    0.0
                }
            },
            total_memory_bytes: self.stats.total_memory_bytes.load(),
            peak_memory_bytes: self.stats.peak_memory_bytes.load(),
            evictions: self.stats.evictions.load(),
        }
    }
}

/// Configuration for the model KV cache
#[derive(Debug, Clone)]
pub struct ModelCacheConfig {
    /// Target model for this cache
    pub model: CandleModel,
    /// Number of transformer layers
    pub num_layers: u32,
    /// Number of attention heads per layer
    pub num_heads: u32,
    /// Dimension per attention head
    pub head_dim: u32,
    /// Maximum sequence length to cache
    pub max_sequence_length: u32,
    /// Maximum batch size to support
    pub max_batch_size: u32,
    /// Memory limit in bytes (0 = no limit)
    pub memory_limit_bytes: u64,
    /// Enable automatic cache eviction
    pub enable_eviction: bool,
    /// Cache eviction threshold (fraction of memory limit)
    pub eviction_threshold: f32,
}

impl ModelCacheConfig {
    /// Create cache configuration for a specific model
    pub fn for_model(model: CandleModel) -> Self {
        let (num_layers, num_heads, head_dim) = match model {
            CandleModel::Devstral_22B => {
                (48, 64, 128) // 48 layers, 64 heads, 128 head dim (8192 / 64)
            }
            CandleModel::Llama2_7B | CandleModel::Mistral_7B | CandleModel::CodeLlama_7B => {
                (32, 32, 128) // 32 layers, 32 heads, 128 head dim (4096 / 32)
            }
            CandleModel::Llama2_13B => {
                (40, 40, 128) // 40 layers, 40 heads, 128 head dim (5120 / 40)
            }
            CandleModel::Phi3_Mini => {
                (32, 32, 96) // 32 layers, 32 heads, 96 head dim (3072 / 32)
            }
            CandleModel::Gemma_2B => {
                (18, 8, 256) // 18 layers, 8 heads, 256 head dim (2048 / 8)
            }
            CandleModel::Gemma_7B => {
                (28, 16, 192) // 28 layers, 16 heads, 192 head dim (3072 / 16)
            }
        };

        let max_seq_len = match model {
            CandleModel::Devstral_22B => 32768,
            CandleModel::CodeLlama_7B => 16384,
            CandleModel::Mistral_7B | CandleModel::Gemma_2B | CandleModel::Gemma_7B => 8192,
            _ => 4096,
        };

        Self {
            model,
            num_layers,
            num_heads,
            head_dim,
            max_sequence_length: max_seq_len,
            max_batch_size: 8, // Conservative default for local inference
            memory_limit_bytes: 2 * 1024 * 1024 * 1024, // 2GB default limit
            enable_eviction: true,
            eviction_threshold: 0.8, // Evict when 80% of memory limit reached
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> CandleResult<()> {
        if self.num_layers == 0 {
            return Err(CandleError::config(
                "Number of layers must be positive",
                "num_layers",
                "> 0",
            ));
        }

        if self.num_layers > MAX_LAYERS as u32 {
            return Err(CandleError::config(
                "Number of layers exceeds maximum",
                "num_layers",
                &format!("<= {}", MAX_LAYERS),
            ));
        }

        if self.num_heads == 0 {
            return Err(CandleError::config(
                "Number of heads must be positive",
                "num_heads",
                "> 0",
            ));
        }

        if self.head_dim == 0 {
            return Err(CandleError::config(
                "Head dimension must be positive",
                "head_dim",
                "> 0",
            ));
        }

        if self.max_sequence_length > MAX_SEQUENCE_LENGTH as u32 {
            return Err(CandleError::config(
                "Maximum sequence length exceeds limit",
                "max_sequence_length",
                &format!("<= {}", MAX_SEQUENCE_LENGTH),
            ));
        }

        if self.max_batch_size > MAX_BATCH_SIZE as u32 {
            return Err(CandleError::config(
                "Maximum batch size exceeds limit",
                "max_batch_size",
                &format!("<= {}", MAX_BATCH_SIZE),
            ));
        }

        if self.eviction_threshold <= 0.0 || self.eviction_threshold > 1.0 {
            return Err(CandleError::config(
                "Eviction threshold must be between 0 and 1",
                "eviction_threshold",
                "0.0 < threshold <= 1.0",
            ));
        }

        Ok(())
    }

    /// Estimate memory usage for this configuration
    pub fn estimate_memory_usage(&self) -> u64 {
        let kv_size_per_token = (self.num_heads * self.head_dim * 2) as u64; // Keys + Values
        let memory_per_layer =
            kv_size_per_token * self.max_sequence_length as u64 * self.max_batch_size as u64;
        let total_kv_memory = memory_per_layer * self.num_layers as u64;

        // Add overhead for data structures (approximately 20%)
        let overhead = total_kv_memory / 5;

        total_kv_memory + overhead
    }
}

/// Main KV cache for transformer models with lock-free design
#[derive(Debug)]
pub struct ModelKvCache {
    /// Cache configuration
    config: ModelCacheConfig,
    /// Per-layer caches
    layer_caches: ArrayVec<LayerCache, MAX_LAYERS>,
    /// Global cache statistics
    global_stats: GlobalCacheStats,
    /// Memory pressure monitoring
    memory_pressure: AtomicCell<MemoryPressure>,
}

/// Memory pressure levels for adaptive cache management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// Normal operation, no pressure
    Low,
    /// Approaching memory limits
    Medium,
    /// Near memory limit, aggressive eviction needed
    High,
    /// Memory limit exceeded, emergency eviction
    Critical,
}

impl Default for MemoryPressure {
    fn default() -> Self {
        MemoryPressure::Low
    }
}

/// Global cache statistics across all layers
#[derive(Debug)]
struct GlobalCacheStats {
    /// Total memory usage across all layers
    total_memory_usage: AtomicCell<u64>,
    /// Total cache operations
    total_operations: AtomicCell<u64>,
    /// Total cache hits across all layers
    total_hits: AtomicCell<u64>,
    /// Total cache misses across all layers
    total_misses: AtomicCell<u64>,
    /// Number of automatic evictions
    automatic_evictions: AtomicCell<u64>,
}

impl Default for GlobalCacheStats {
    fn default() -> Self {
        Self {
            total_memory_usage: AtomicCell::new(0),
            total_operations: AtomicCell::new(0),
            total_hits: AtomicCell::new(0),
            total_misses: AtomicCell::new(0),
            automatic_evictions: AtomicCell::new(0),
        }
    }
}

impl ModelKvCache {
    /// Create a new model KV cache
    pub fn new(config: ModelCacheConfig) -> CandleResult<Self> {
        config.validate()?;

        let mut layer_caches = ArrayVec::new();

        // Initialize layer caches
        for layer_idx in 0..config.num_layers {
            let layer_cache = LayerCache::new(layer_idx, config.num_heads, config.head_dim);

            if layer_caches.try_push(layer_cache).is_err() {
                return Err(CandleError::cache(
                    "Failed to initialize layer caches",
                    "new",
                    "layer count within limits",
                ));
            }
        }

        Ok(Self {
            config,
            layer_caches,
            global_stats: GlobalCacheStats::default(),
            memory_pressure: AtomicCell::new(MemoryPressure::Low),
        })
    }

    /// Get KV data for a specific layer and batch element
    pub fn get_layer_kv(&self, layer_index: usize, batch_index: usize) -> Option<AlignedKvData> {
        if layer_index >= self.layer_caches.len() {
            return None;
        }

        let result = self.layer_caches[layer_index].get_kv_data(batch_index);

        // Update global statistics
        self.global_stats
            .total_operations
            .store(self.global_stats.total_operations.load() + 1);

        if result.is_some() {
            self.global_stats
                .total_hits
                .store(self.global_stats.total_hits.load() + 1);
        } else {
            self.global_stats
                .total_misses
                .store(self.global_stats.total_misses.load() + 1);
        }

        result
    }

    /// Set KV data for a specific layer and batch element
    pub fn set_layer_kv(
        &self,
        layer_index: usize,
        batch_index: usize,
        kv_data: AlignedKvData,
    ) -> CandleResult<()> {
        if layer_index >= self.layer_caches.len() {
            return Err(CandleError::cache(
                "Layer index out of range",
                "set_layer_kv",
                "valid layer index",
            ));
        }

        // Check memory pressure before setting
        self.check_memory_pressure()?;

        let memory_before = self.total_memory_usage();
        self.layer_caches[layer_index].set_kv_data(batch_index, kv_data)?;
        let memory_after = self.total_memory_usage();

        // Update global memory usage
        self.global_stats.total_memory_usage.store(memory_after);

        // Update memory pressure based on new usage
        self.update_memory_pressure();

        Ok(())
    }

    /// Append incremental KV data for autoregressive generation
    pub fn append_incremental(
        &self,
        layer_index: usize,
        batch_index: usize,
        new_keys: &[f32],
        new_values: &[f32],
    ) -> CandleResult<()> {
        if layer_index >= self.layer_caches.len() {
            return Err(CandleError::cache(
                "Layer index out of range",
                "append_incremental",
                "valid layer index",
            ));
        }

        self.layer_caches[layer_index].append_incremental_kv(batch_index, new_keys, new_values)?;

        // Update global memory tracking
        let new_memory_usage = self.total_memory_usage();
        self.global_stats.total_memory_usage.store(new_memory_usage);
        self.update_memory_pressure();

        Ok(())
    }

    /// Clear cache for a specific layer
    pub fn clear_layer(&self, layer_index: usize) -> CandleResult<()> {
        if layer_index >= self.layer_caches.len() {
            return Err(CandleError::cache(
                "Layer index out of range",
                "clear_layer",
                "valid layer index",
            ));
        }

        self.layer_caches[layer_index].clear_all();

        // Update global memory usage
        let new_memory_usage = self.total_memory_usage();
        self.global_stats.total_memory_usage.store(new_memory_usage);
        self.update_memory_pressure();

        Ok(())
    }

    /// Clear all cached data across all layers
    pub fn clear_all(&self) {
        for layer_cache in &self.layer_caches {
            layer_cache.clear_all();
        }

        // Reset global statistics
        self.global_stats.total_memory_usage.store(0);
        self.memory_pressure.store(MemoryPressure::Low);
    }

    /// Check memory pressure and perform eviction if necessary
    fn check_memory_pressure(&self) -> CandleResult<()> {
        if !self.config.enable_eviction {
            return Ok(());
        }

        let current_memory = self.total_memory_usage();
        let memory_limit = self.config.memory_limit_bytes;

        if memory_limit > 0 && current_memory > memory_limit {
            // Emergency eviction - clear oldest layers
            let layers_to_clear = self.layer_caches.len() / 4; // Clear 25% of layers

            for i in 0..layers_to_clear {
                self.layer_caches[i].clear_all();
            }

            self.global_stats
                .automatic_evictions
                .store(self.global_stats.automatic_evictions.load() + 1);

            // Update memory usage after eviction
            let new_memory_usage = self.total_memory_usage();
            self.global_stats.total_memory_usage.store(new_memory_usage);
        }

        Ok(())
    }

    /// Update memory pressure based on current usage
    fn update_memory_pressure(&self) {
        let current_memory = self.total_memory_usage();
        let memory_limit = self.config.memory_limit_bytes;

        if memory_limit == 0 {
            self.memory_pressure.store(MemoryPressure::Low);
            return;
        }

        let usage_ratio = current_memory as f64 / memory_limit as f64;

        let pressure = if usage_ratio >= 1.0 {
            MemoryPressure::Critical
        } else if usage_ratio >= 0.9 {
            MemoryPressure::High
        } else if usage_ratio >= 0.7 {
            MemoryPressure::Medium
        } else {
            MemoryPressure::Low
        };

        self.memory_pressure.store(pressure);
    }

    /// Calculate total memory usage across all layers
    fn total_memory_usage(&self) -> u64 {
        self.layer_caches
            .iter()
            .map(|layer| layer.statistics().total_memory_bytes)
            .sum()
    }

    /// Get cache configuration
    pub fn config(&self) -> &ModelCacheConfig {
        &self.config
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layer_caches.len()
    }

    /// Get current memory pressure
    pub fn memory_pressure(&self) -> MemoryPressure {
        self.memory_pressure.load()
    }

    /// Get comprehensive cache statistics
    pub fn statistics(&self) -> KvCacheStatistics {
        let layer_stats: Vec<LayerCacheStatistics> = self
            .layer_caches
            .iter()
            .map(|layer| layer.statistics())
            .collect();

        let total_memory = self.total_memory_usage();
        let total_hits = self.global_stats.total_hits.load();
        let total_misses = self.global_stats.total_misses.load();
        let total_ops = total_hits + total_misses;

        KvCacheStatistics {
            layer_stats,
            total_memory_bytes: total_memory,
            memory_limit_bytes: self.config.memory_limit_bytes,
            memory_usage_ratio: if self.config.memory_limit_bytes > 0 {
                total_memory as f32 / self.config.memory_limit_bytes as f32
            } else {
                0.0
            },
            global_hit_rate: if total_ops > 0 {
                total_hits as f32 / total_ops as f32
            } else {
                0.0
            },
            total_operations: self.global_stats.total_operations.load(),
            automatic_evictions: self.global_stats.automatic_evictions.load(),
            memory_pressure: self.memory_pressure.load(),
        }
    }
}

/// Statistics for a single layer cache
#[derive(Debug, Clone)]
pub struct LayerCacheStatistics {
    /// Layer index
    pub layer_index: u32,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f32,
    /// Total memory usage in bytes
    pub total_memory_bytes: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Number of evictions
    pub evictions: u64,
}

/// Comprehensive KV cache statistics
#[derive(Debug, Clone)]
pub struct KvCacheStatistics {
    /// Per-layer statistics
    pub layer_stats: Vec<LayerCacheStatistics>,
    /// Total memory usage across all layers
    pub total_memory_bytes: u64,
    /// Memory limit in bytes
    pub memory_limit_bytes: u64,
    /// Memory usage ratio (0.0 to 1.0+)
    pub memory_usage_ratio: f32,
    /// Global cache hit rate
    pub global_hit_rate: f32,
    /// Total cache operations
    pub total_operations: u64,
    /// Number of automatic evictions
    pub automatic_evictions: u64,
    /// Current memory pressure level
    pub memory_pressure: MemoryPressure,
}

impl KvCacheStatistics {
    /// Calculate average hit rate across all layers
    pub fn avg_layer_hit_rate(&self) -> f32 {
        if self.layer_stats.is_empty() {
            return 0.0;
        }

        let sum: f32 = self.layer_stats.iter().map(|s| s.hit_rate).sum();
        sum / self.layer_stats.len() as f32
    }

    /// Get memory usage in MB
    pub fn memory_usage_mb(&self) -> f32 {
        self.total_memory_bytes as f32 / (1024.0 * 1024.0)
    }

    /// Check if cache is under memory pressure
    pub fn is_under_pressure(&self) -> bool {
        matches!(
            self.memory_pressure,
            MemoryPressure::High | MemoryPressure::Critical
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_kv_data_creation() {
        let kv_data = AlignedKvData::new(10, 64);
        assert_eq!(kv_data.sequence_length(), 10);
        assert_eq!(kv_data.head_dim(), 64);
        assert!(kv_data.is_empty());
    }

    #[test]
    fn test_model_cache_config() {
        let config = ModelCacheConfig::for_model(CandleModel::Mistral_7B);
        assert_eq!(config.model, CandleModel::Mistral_7B);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_layer_cache_operations() {
        let layer_cache = LayerCache::new(0, 32, 128);
        assert_eq!(layer_cache.layer_index(), 0);
        assert_eq!(layer_cache.num_heads(), 32);
        assert_eq!(layer_cache.head_dim(), 128);

        // Test cache miss
        assert!(layer_cache.get_kv_data(0).is_none());

        let stats = layer_cache.statistics();
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 0);
    }

    #[test]
    fn test_memory_pressure_calculation() {
        let config = ModelCacheConfig::for_model(CandleModel::Gemma_2B);
        let cache = ModelKvCache::new(config).unwrap();

        assert_eq!(cache.memory_pressure(), MemoryPressure::Low);

        let stats = cache.statistics();
        assert_eq!(stats.total_memory_bytes, 0);
        assert!(!stats.is_under_pressure());
    }
}
