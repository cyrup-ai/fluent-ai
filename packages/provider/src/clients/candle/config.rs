//! Global configuration and metrics collection for Candle client
//!
//! This module provides comprehensive configuration management and real-time
//! metrics collection for optimal performance monitoring and tuning.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use crossbeam::atomic::AtomicCell;

use super::error::{CandleError, CandleResult};
use super::models::CandleModel;

/// Global configuration for Candle operations
#[derive(Debug, Clone)]
pub struct CandleGlobalConfig {
    /// Model-specific configurations
    pub model_configs: HashMap<CandleModel, ModelSpecificConfig>,
    /// Cache configuration
    pub cache_config: CacheConfig,
    /// Compute configuration
    pub compute_config: ComputeConfig,
    /// Metrics collection settings
    pub metrics_config: MetricsConfig,
    /// Global performance settings
    pub performance_config: PerformanceSettings,
}

impl Default for CandleGlobalConfig {
    fn default() -> Self {
        let mut model_configs = HashMap::new();

        // Add default configurations for each model
        for model in [
            CandleModel::Llama2_7B,
            CandleModel::Llama2_13B,
            CandleModel::Mistral_7B,
            CandleModel::CodeLlama_7B,
            CandleModel::Phi3_Mini,
            CandleModel::Gemma_2B,
            CandleModel::Gemma_7B,
        ] {
            model_configs.insert(model, ModelSpecificConfig::for_model(model));
        }

        Self {
            model_configs,
            cache_config: CacheConfig::default(),
            compute_config: ComputeConfig::default(),
            metrics_config: MetricsConfig::default(),
            performance_config: PerformanceSettings::default(),
        }
    }
}

impl CandleGlobalConfig {
    /// Create optimized configuration for production
    pub fn production() -> Self {
        let mut config = Self::default();
        config.cache_config = CacheConfig::high_performance();
        config.compute_config = ComputeConfig::optimized();
        config.metrics_config = MetricsConfig::production();
        config.performance_config = PerformanceSettings::aggressive();
        config
    }

    /// Create configuration optimized for specific model
    pub fn for_model(model: CandleModel) -> CandleResult<Self> {
        let mut config = Self::default();

        // Get model-specific configuration
        let model_config = ModelSpecificConfig::for_model(model);
        config.model_configs.insert(model, model_config.clone());

        // Adjust global settings based on model requirements
        match model {
            CandleModel::Llama2_13B => {
                // Large model - conservative settings
                config.cache_config.model_cache_mb = 8192;
                config.cache_config.kv_cache_limit_mb = 4096;
                config.compute_config.memory_optimization = MemoryOptimization::Aggressive;
            }
            CandleModel::Phi3_Mini | CandleModel::Gemma_2B => {
                // Small models - aggressive optimization
                config.cache_config.aggressive_caching = true;
                config.performance_config.optimization_level = 3;
                config.compute_config.enable_fp16 = true;
            }
            CandleModel::CodeLlama_7B => {
                // Code model - balanced with focus on accuracy
                config.performance_config.optimization_level = 2;
                config.compute_config.precision = ComputePrecision::FP32;
            }
            _ => {
                // Default optimized settings for other models
                config.performance_config.optimization_level = 2;
                config.compute_config.enable_fp16 = true;
            }
        }

        config.validate()?;
        Ok(config)
    }

    /// Create configuration for development/debugging
    pub fn development() -> Self {
        let mut config = Self::default();
        config.cache_config = CacheConfig::development();
        config.compute_config = ComputeConfig::debug();
        config.metrics_config = MetricsConfig::verbose();
        config.performance_config = PerformanceSettings::conservative();
        config
    }

    /// Get configuration for specific model
    pub fn get_model_config(&self, model: CandleModel) -> Option<&ModelSpecificConfig> {
        self.model_configs.get(&model)
    }

    /// Update model configuration
    pub fn set_model_config(&mut self, model: CandleModel, config: ModelSpecificConfig) {
        self.model_configs.insert(model, config);
    }

    /// Validate all configurations
    pub fn validate(&self) -> CandleResult<()> {
        self.cache_config.validate()?;
        self.compute_config.validate()?;
        self.metrics_config.validate()?;
        self.performance_config.validate()?;

        for (model, config) in &self.model_configs {
            config.validate().map_err(|e| {
                CandleError::config(
                    &format!("Invalid config for model {}: {}", model, e),
                    "model_config",
                    "valid configuration",
                )
            })?;
        }

        Ok(())
    }
}

/// Model-specific configuration parameters
#[derive(Debug, Clone)]
pub struct ModelSpecificConfig {
    /// Target model
    pub model: CandleModel,
    /// Optimal batch size for this model
    pub optimal_batch_size: usize,
    /// Memory limit for this model in bytes
    pub memory_limit_bytes: u64,
    /// KV cache configuration
    pub kv_cache_size_mb: u32,
    /// Context window size
    pub context_window: u32,
    /// Recommended temperature range
    pub temperature_range: (f32, f32),
    /// Parallel processing configuration
    pub parallel_config: ParallelConfig,
}

impl ModelSpecificConfig {
    /// Create configuration for specific model
    pub fn for_model(model: CandleModel) -> Self {
        match model {
            CandleModel::Llama2_7B | CandleModel::Mistral_7B | CandleModel::CodeLlama_7B => {
                Self {
                    model,
                    optimal_batch_size: 8,
                    memory_limit_bytes: 16 * 1024 * 1024 * 1024, // 16GB
                    kv_cache_size_mb: 2048,                      // 2GB
                    context_window: 4096,
                    temperature_range: (0.1, 2.0),
                    parallel_config: ParallelConfig::medium(),
                }
            }
            CandleModel::Llama2_13B => {
                Self {
                    model,
                    optimal_batch_size: 4,
                    memory_limit_bytes: 32 * 1024 * 1024 * 1024, // 32GB
                    kv_cache_size_mb: 4096,                      // 4GB
                    context_window: 4096,
                    temperature_range: (0.1, 1.5),
                    parallel_config: ParallelConfig::high(),
                }
            }
            CandleModel::Phi3_Mini | CandleModel::Gemma_2B => {
                Self {
                    model,
                    optimal_batch_size: 16,
                    memory_limit_bytes: 8 * 1024 * 1024 * 1024, // 8GB
                    kv_cache_size_mb: 1024,                     // 1GB
                    context_window: 4096,
                    temperature_range: (0.1, 2.0),
                    parallel_config: ParallelConfig::low(),
                }
            }
            CandleModel::Gemma_7B => {
                Self {
                    model,
                    optimal_batch_size: 8,
                    memory_limit_bytes: 20 * 1024 * 1024 * 1024, // 20GB
                    kv_cache_size_mb: 3072,                      // 3GB
                    context_window: 8192,
                    temperature_range: (0.1, 1.8),
                    parallel_config: ParallelConfig::medium(),
                }
            }
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> CandleResult<()> {
        if self.optimal_batch_size == 0 {
            return Err(CandleError::config(
                "Batch size must be positive",
                "optimal_batch_size",
                "> 0",
            ));
        }

        if self.memory_limit_bytes == 0 {
            return Err(CandleError::config(
                "Memory limit must be positive",
                "memory_limit_bytes",
                "> 0",
            ));
        }

        if self.context_window == 0 {
            return Err(CandleError::config(
                "Context window must be positive",
                "context_window",
                "> 0",
            ));
        }

        if self.temperature_range.0 >= self.temperature_range.1 {
            return Err(CandleError::config(
                "Temperature range invalid",
                "temperature_range",
                "min < max",
            ));
        }

        self.parallel_config.validate()?;

        Ok(())
    }
}

/// Parallel processing configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads for model operations
    pub model_threads: usize,
    /// Number of threads for data processing
    pub data_threads: usize,
    /// Enable NUMA-aware scheduling
    pub numa_aware: bool,
    /// Thread affinity settings
    pub thread_affinity: ThreadAffinity,
}

impl ParallelConfig {
    /// Low parallelism configuration
    pub fn low() -> Self {
        Self {
            model_threads: 2,
            data_threads: 2,
            numa_aware: false,
            thread_affinity: ThreadAffinity::None,
        }
    }

    /// Medium parallelism configuration
    pub fn medium() -> Self {
        Self {
            model_threads: 4,
            data_threads: 4,
            numa_aware: true,
            thread_affinity: ThreadAffinity::Core,
        }
    }

    /// High parallelism configuration
    pub fn high() -> Self {
        Self {
            model_threads: 8,
            data_threads: 8,
            numa_aware: true,
            thread_affinity: ThreadAffinity::Socket,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> CandleResult<()> {
        if self.model_threads == 0 {
            return Err(CandleError::config(
                "Model threads must be positive",
                "model_threads",
                "> 0",
            ));
        }

        if self.data_threads == 0 {
            return Err(CandleError::config(
                "Data threads must be positive",
                "data_threads",
                "> 0",
            ));
        }

        Ok(())
    }
}

/// Thread affinity options
#[derive(Debug, Clone, Copy)]
pub enum ThreadAffinity {
    /// No thread affinity
    None,
    /// Bind to specific cores
    Core,
    /// Bind to socket/NUMA node
    Socket,
    /// Custom affinity mask
    Custom(u64),
}

/// Cache configuration for optimal performance
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Model cache size in MB
    pub model_cache_mb: u32,
    /// Tokenizer cache size in MB
    pub tokenizer_cache_mb: u32,
    /// KV cache memory limit in MB
    pub kv_cache_limit_mb: u32,
    /// Enable aggressive caching
    pub aggressive_caching: bool,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Cache warming strategy
    pub warming_strategy: WarmingStrategy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            model_cache_mb: 4096,    // 4GB for models
            tokenizer_cache_mb: 256, // 256MB for tokenizers
            kv_cache_limit_mb: 2048, // 2GB for KV cache
            aggressive_caching: false,
            eviction_policy: EvictionPolicy::LRU,
            warming_strategy: WarmingStrategy::Lazy,
        }
    }
}

impl CacheConfig {
    /// High-performance cache configuration
    pub fn high_performance() -> Self {
        Self {
            model_cache_mb: 8192,    // 8GB
            tokenizer_cache_mb: 512, // 512MB
            kv_cache_limit_mb: 4096, // 4GB
            aggressive_caching: true,
            eviction_policy: EvictionPolicy::LFU,
            warming_strategy: WarmingStrategy::Eager,
        }
    }

    /// Development cache configuration
    pub fn development() -> Self {
        Self {
            model_cache_mb: 1024,    // 1GB
            tokenizer_cache_mb: 128, // 128MB
            kv_cache_limit_mb: 512,  // 512MB
            aggressive_caching: false,
            eviction_policy: EvictionPolicy::LRU,
            warming_strategy: WarmingStrategy::Manual,
        }
    }

    /// Validate cache configuration
    pub fn validate(&self) -> CandleResult<()> {
        if self.model_cache_mb == 0 {
            return Err(CandleError::config(
                "Model cache size must be positive",
                "model_cache_mb",
                "> 0",
            ));
        }

        if self.kv_cache_limit_mb == 0 {
            return Err(CandleError::config(
                "KV cache limit must be positive",
                "kv_cache_limit_mb",
                "> 0",
            ));
        }

        Ok(())
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In, First Out
    FIFO,
    /// Random eviction
    Random,
}

/// Cache warming strategies
#[derive(Debug, Clone, Copy)]
pub enum WarmingStrategy {
    /// Load on first access
    Lazy,
    /// Pre-load common models
    Eager,
    /// Manual warming control
    Manual,
}

/// Compute configuration for optimization
#[derive(Debug, Clone)]
pub struct ComputeConfig {
    /// Enable mixed precision (fp16)
    pub enable_fp16: bool,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Preferred GPU device ID
    pub gpu_device_id: Option<u32>,
    /// Memory optimization level
    pub memory_optimization: MemoryOptimization,
    /// Compute precision
    pub precision: ComputePrecision,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            enable_fp16: false,
            enable_simd: true,
            enable_gpu: true,
            gpu_device_id: None, // Auto-select
            memory_optimization: MemoryOptimization::Balanced,
            precision: ComputePrecision::FP32,
        }
    }
}

impl ComputeConfig {
    /// Optimized compute configuration
    pub fn optimized() -> Self {
        Self {
            enable_fp16: true,
            enable_simd: true,
            enable_gpu: true,
            gpu_device_id: None,
            memory_optimization: MemoryOptimization::Aggressive,
            precision: ComputePrecision::Mixed,
        }
    }

    /// Debug compute configuration
    pub fn debug() -> Self {
        Self {
            enable_fp16: false,
            enable_simd: false,
            enable_gpu: false,
            gpu_device_id: None,
            memory_optimization: MemoryOptimization::Conservative,
            precision: ComputePrecision::FP32,
        }
    }

    /// Validate compute configuration
    pub fn validate(&self) -> CandleResult<()> {
        // All configurations are valid - no constraints to check
        Ok(())
    }
}

/// Memory optimization levels
#[derive(Debug, Clone, Copy)]
pub enum MemoryOptimization {
    /// Conservative memory usage
    Conservative,
    /// Balanced memory/performance
    Balanced,
    /// Aggressive memory optimization
    Aggressive,
}

/// Compute precision options
#[derive(Debug, Clone, Copy)]
pub enum ComputePrecision {
    /// 32-bit floating point
    FP32,
    /// 16-bit floating point
    FP16,
    /// Mixed precision (FP16 + FP32)
    Mixed,
    /// Brain floating point (bfloat16)
    BF16,
}

/// Metrics collection configuration
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Enable performance metrics
    pub enable_performance: bool,
    /// Enable memory metrics
    pub enable_memory: bool,
    /// Enable error metrics
    pub enable_errors: bool,
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Metrics retention period
    pub retention_period: Duration,
    /// Enable real-time metrics
    pub real_time: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enable_performance: true,
            enable_memory: true,
            enable_errors: true,
            collection_interval: Duration::from_secs(1),
            retention_period: Duration::from_secs(3600), // 1 hour
            real_time: false,
        }
    }
}

impl MetricsConfig {
    /// Production metrics configuration
    pub fn production() -> Self {
        Self {
            enable_performance: true,
            enable_memory: true,
            enable_errors: true,
            collection_interval: Duration::from_secs(5),
            retention_period: Duration::from_secs(86400), // 24 hours
            real_time: false,
        }
    }

    /// Verbose metrics for development
    pub fn verbose() -> Self {
        Self {
            enable_performance: true,
            enable_memory: true,
            enable_errors: true,
            collection_interval: Duration::from_millis(100),
            retention_period: Duration::from_secs(1800), // 30 minutes
            real_time: true,
        }
    }

    /// Validate metrics configuration
    pub fn validate(&self) -> CandleResult<()> {
        if self.collection_interval.is_zero() {
            return Err(CandleError::config(
                "Collection interval must be positive",
                "collection_interval",
                "> 0",
            ));
        }

        if self.retention_period.is_zero() {
            return Err(CandleError::config(
                "Retention period must be positive",
                "retention_period",
                "> 0",
            ));
        }

        Ok(())
    }
}

/// Performance settings for optimization
#[derive(Debug, Clone)]
pub struct PerformanceSettings {
    /// Enable all optimizations
    pub enable_all_optimizations: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Profile collection interval
    pub profile_interval: Duration,
    /// Adaptive optimization
    pub adaptive_optimization: bool,
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            enable_all_optimizations: true,
            optimization_level: 2,
            enable_profiling: false,
            profile_interval: Duration::from_secs(10),
            adaptive_optimization: true,
        }
    }
}

impl PerformanceSettings {
    /// Aggressive performance settings
    pub fn aggressive() -> Self {
        Self {
            enable_all_optimizations: true,
            optimization_level: 3,
            enable_profiling: false,
            profile_interval: Duration::from_secs(5),
            adaptive_optimization: true,
        }
    }

    /// Conservative performance settings
    pub fn conservative() -> Self {
        Self {
            enable_all_optimizations: false,
            optimization_level: 1,
            enable_profiling: true,
            profile_interval: Duration::from_secs(30),
            adaptive_optimization: false,
        }
    }

    /// Validate performance settings
    pub fn validate(&self) -> CandleResult<()> {
        if self.optimization_level > 3 {
            return Err(CandleError::config(
                "Optimization level must be 0-3",
                "optimization_level",
                "0 <= level <= 3",
            ));
        }

        if self.profile_interval.is_zero() {
            return Err(CandleError::config(
                "Profile interval must be positive",
                "profile_interval",
                "> 0",
            ));
        }

        Ok(())
    }
}

/// Real-time metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    /// Configuration
    config: MetricsConfig,
    /// Performance metrics
    performance_metrics: PerformanceMetrics,
    /// Memory metrics
    memory_metrics: MemoryMetrics,
    /// Error metrics
    error_metrics: ErrorMetrics,
    /// Collection timestamp
    last_collection: AtomicCell<Instant>,
}

impl MetricsCollector {
    /// Create new metrics collector with simple enable/disable flag
    pub fn new(enabled: bool) -> Self {
        let config = if enabled {
            MetricsConfig::default()
        } else {
            MetricsConfig {
                enable_performance: false,
                enable_memory: false,
                enable_errors: false,
                collection_interval: Duration::from_secs(3600), // Large interval when disabled
                retention_period: Duration::from_secs(60),
                real_time: false,
            }
        };

        Self::with_config(config)
    }

    /// Create new metrics collector with full configuration
    pub fn with_config(config: MetricsConfig) -> Self {
        Self {
            config,
            performance_metrics: PerformanceMetrics::default(),
            memory_metrics: MemoryMetrics::default(),
            error_metrics: ErrorMetrics::default(),
            last_collection: AtomicCell::new(Instant::now()),
        }
    }

    /// Record inference operation
    pub fn record_inference(&self, duration: Duration, tokens_generated: u32) {
        if self.config.enable_performance {
            self.performance_metrics
                .total_inferences
                .store(self.performance_metrics.total_inferences.load() + 1);
            self.performance_metrics.total_inference_time.store(
                self.performance_metrics.total_inference_time.load() + duration.as_nanos() as u64,
            );
            self.performance_metrics.total_tokens_generated.store(
                self.performance_metrics.total_tokens_generated.load() + tokens_generated as u64,
            );
        }
    }

    /// Record memory usage
    pub fn record_memory_usage(&self, bytes_used: u64) {
        if self.config.enable_memory {
            self.memory_metrics.current_memory_usage.store(bytes_used);

            let peak = self.memory_metrics.peak_memory_usage.load();
            if bytes_used > peak {
                self.memory_metrics.peak_memory_usage.store(bytes_used);
            }
        }
    }

    /// Record error occurrence
    pub fn record_error(&self, error_type: &str) {
        if self.config.enable_errors {
            self.error_metrics
                .total_errors
                .store(self.error_metrics.total_errors.load() + 1);
            // In a real implementation, we'd track error types in a HashMap
        }
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.clone()
    }

    /// Get memory metrics
    pub fn memory_metrics(&self) -> MemoryMetrics {
        self.memory_metrics.clone()
    }

    /// Get error metrics
    pub fn error_metrics(&self) -> ErrorMetrics {
        self.error_metrics.clone()
    }

    /// Check if metrics should be collected
    pub fn should_collect(&self) -> bool {
        let now = Instant::now();
        let last = self.last_collection.load();

        if now.duration_since(last) >= self.config.collection_interval {
            self.last_collection.store(now);
            true
        } else {
            false
        }
    }
}

/// Performance metrics tracking
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total inference operations
    total_inferences: AtomicCell<u64>,
    /// Total inference time in nanoseconds
    total_inference_time: AtomicCell<u64>,
    /// Total tokens generated
    total_tokens_generated: AtomicCell<u64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_inferences: AtomicCell::new(0),
            total_inference_time: AtomicCell::new(0),
            total_tokens_generated: AtomicCell::new(0),
        }
    }
}

impl PerformanceMetrics {
    /// Calculate average inference time
    pub fn avg_inference_time(&self) -> Duration {
        let total_time = self.total_inference_time.load();
        let total_inferences = self.total_inferences.load();

        if total_inferences > 0 {
            Duration::from_nanos(total_time / total_inferences)
        } else {
            Duration::ZERO
        }
    }

    /// Calculate tokens per second
    pub fn tokens_per_second(&self) -> f64 {
        let total_time_secs = self.total_inference_time.load() as f64 / 1_000_000_000.0;
        let total_tokens = self.total_tokens_generated.load() as f64;

        if total_time_secs > 0.0 {
            total_tokens / total_time_secs
        } else {
            0.0
        }
    }
}

/// Memory metrics tracking
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Current memory usage in bytes
    current_memory_usage: AtomicCell<u64>,
    /// Peak memory usage in bytes
    peak_memory_usage: AtomicCell<u64>,
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            current_memory_usage: AtomicCell::new(0),
            peak_memory_usage: AtomicCell::new(0),
        }
    }
}

impl MemoryMetrics {
    /// Get current memory usage in MB
    pub fn current_usage_mb(&self) -> f64 {
        self.current_memory_usage.load() as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory usage in MB
    pub fn peak_usage_mb(&self) -> f64 {
        self.peak_memory_usage.load() as f64 / (1024.0 * 1024.0)
    }
}

/// Error metrics tracking
#[derive(Debug, Clone)]
pub struct ErrorMetrics {
    /// Total errors encountered
    total_errors: AtomicCell<u64>,
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self {
            total_errors: AtomicCell::new(0),
        }
    }
}

impl ErrorMetrics {
    /// Get total error count
    pub fn total_errors(&self) -> u64 {
        self.total_errors.load()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_config_creation() {
        let config = CandleGlobalConfig::default();
        assert!(config.validate().is_ok());

        let prod_config = CandleGlobalConfig::production();
        assert!(prod_config.validate().is_ok());
    }

    #[test]
    fn test_model_specific_config() {
        let config = ModelSpecificConfig::for_model(CandleModel::Mistral_7B);
        assert_eq!(config.model, CandleModel::Mistral_7B);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new(true);

        collector.record_inference(Duration::from_millis(100), 50);
        collector.record_memory_usage(1024 * 1024 * 1024); // 1GB
        collector.record_error("test_error");

        let perf_metrics = collector.performance_metrics();
        assert_eq!(perf_metrics.total_inferences.load(), 1);
        assert_eq!(perf_metrics.total_tokens_generated.load(), 50);

        let mem_metrics = collector.memory_metrics();
        assert_eq!(mem_metrics.current_usage_mb(), 1024.0);

        let error_metrics = collector.error_metrics();
        assert_eq!(error_metrics.total_errors(), 1);
    }

    #[test]
    fn test_cache_config() {
        let config = CacheConfig::default();
        assert!(config.validate().is_ok());

        let hp_config = CacheConfig::high_performance();
        assert!(hp_config.validate().is_ok());
        assert!(hp_config.aggressive_caching);
    }
}
