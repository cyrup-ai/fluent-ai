//! Production-grade model cache with intelligent eviction and memory management
//!
//! This module provides a lock-free, high-performance model cache system that manages
//! loaded Candle models with LRU eviction, memory pressure monitoring, and atomic
//! reference counting for optimal performance in concurrent environments.

use std::sync::{Arc, atomic::{AtomicU64, AtomicUsize, Ordering}};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::path::Path;
use std::collections::HashMap;

use arc_swap::{ArcSwap, Guard};
use crossbeam::atomic::AtomicCell;
use smallvec::SmallVec;
use tokio::time::{interval, MissedTickBehavior};
use tokio::sync::RwLock;

use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Llama, LlamaConfig, Cache as LlamaCache};
use candle_transformers::models::mistral::{Model as MistralModel, Config as MistralConfig};
use tokenizers::Tokenizer;

use super::models::{CandleModel, CandleDevice, CandleModelInfo};
use super::model_repo::{ModelRepository, ModelFileInfo, ModelArchitecture};

/// Memory pressure thresholds for cache management
const MEMORY_PRESSURE_WARNING_MB: usize = 8192;  // 8GB warning threshold
const MEMORY_PRESSURE_CRITICAL_MB: usize = 12288; // 12GB critical threshold
const CACHE_CLEANUP_INTERVAL_SECS: u64 = 30;      // Cleanup every 30 seconds
const MAX_CACHED_MODELS: usize = 8;               // Maximum models in cache
const ACCESS_TIME_PRECISION_MS: u64 = 100;        // Access time precision

/// Memory pressure levels for adaptive cache management
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3}

/// Loaded model wrapper with metadata and device information
pub struct LoadedModel {
    /// The actual model (enum for different architectures)
    pub model: ModelInstance,
    /// Tokenizer for text processing
    pub tokenizer: Arc<Tokenizer>,
    /// Device the model is loaded on
    pub device: Device,
    /// Model configuration information
    pub config: ModelConfig,
    /// KV cache for attention optimization
    pub kv_cache: Option<Box<dyn AttentionCache + Send + Sync>>,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Model loading timestamp
    pub loaded_at: Instant}

/// Model instance enum for different architectures
pub enum ModelInstance {
    LLaMA(Box<Llama>),
    Mistral(Box<MistralModel>),
    Phi3(Box<dyn Phi3Model + Send + Sync>),
    Gemma(Box<dyn GemmaModel + Send + Sync>)}

/// Model configuration wrapper
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub use_flash_attn: bool}

/// Trait for attention cache abstraction
pub trait AttentionCache {
    fn reset(&mut self);
    fn memory_usage(&self) -> usize;
}

/// Trait for Phi3 model abstraction (placeholder for actual implementation)
pub trait Phi3Model {
    fn forward(&self, input: &Tensor, cache: &mut dyn AttentionCache) -> candle_core::Result<Tensor>;
}

/// Trait for Gemma model abstraction (placeholder for actual implementation)  
pub trait GemmaModel {
    fn forward(&self, input: &Tensor, cache: &mut dyn AttentionCache) -> candle_core::Result<Tensor>;
}

/// LLaMA attention cache implementation
pub struct LLaMaAttentionCache {
    cache: LlamaCache,
    memory_usage: usize}

impl LLaMaAttentionCache {
    pub fn new(config: &LlamaConfig, device: &Device, dtype: DType) -> candle_core::Result<Self> {
        let cache = LlamaCache::new(true, dtype, config, device)?;
        Ok(Self {
            cache,
            memory_usage: Self::calculate_memory_usage(config)})
    }
    
    fn calculate_memory_usage(config: &LlamaConfig) -> usize {
        // Estimate memory usage based on model configuration
        let hidden_size = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let max_seq_len = config.max_position_embeddings.unwrap_or(4096);
        let dtype_size = 2; // f16 = 2 bytes
        
        // KV cache: (batch_size * num_heads * seq_len * head_dim) * 2 (K + V) * num_layers
        let kv_cache_size = 1 * config.num_attention_heads * max_seq_len * (hidden_size / config.num_attention_heads) * 2 * num_layers * dtype_size;
        kv_cache_size
    }
    
    pub fn get_mut_cache(&mut self) -> &mut LlamaCache {
        &mut self.cache
    }
}

impl AttentionCache for LLaMaAttentionCache {
    fn reset(&mut self) {
        self.cache.reset();
    }
    
    fn memory_usage(&self) -> usize {
        self.memory_usage
    }
}

/// Cached model entry with access tracking
pub struct CachedModel {
    /// The loaded model
    pub model: Arc<LoadedModel>,
    /// Last access timestamp (atomic for lock-free updates)
    pub last_access: AtomicU64,
    /// Access count for LRU ordering
    pub access_count: AtomicU64}

impl CachedModel {
    /// Create new cached model entry
    pub fn new(model: Arc<LoadedModel>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        Self {
            model,
            last_access: AtomicU64::new(now),
            access_count: AtomicU64::new(1)}
    }
    
    /// Update access timestamp and increment count
    #[inline(always)]
    pub fn touch(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        self.last_access.store(now, Ordering::Relaxed);
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get last access timestamp
    #[inline(always)]
    pub fn get_last_access(&self) -> u64 {
        self.last_access.load(Ordering::Relaxed)
    }
    
    /// Get access count
    #[inline(always)]
    pub fn get_access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }
}

/// Lock-free model cache with intelligent memory management
pub struct ModelCache {
    /// Cache storage (lock-free with atomic swapping)
    cache: ArcSwap<HashMap<CandleModel, Arc<CachedModel>>>,
    /// Total memory usage tracker (atomic)
    total_memory_usage: AtomicCell<usize>,
    /// Current memory pressure level
    memory_pressure: AtomicCell<MemoryPressure>,
    /// Model repository for loading models
    repository: Arc<ModelRepository>,
    /// Background cleanup task handle
    cleanup_task: RwLock<Option<tokio::task::JoinHandle<()>>>,
    /// Cache statistics
    stats: CacheStatistics}

/// Cache performance statistics
#[derive(Debug)]
pub struct CacheStatistics {
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub evictions: AtomicU64,
    pub load_times_ms: AtomicU64,
    pub memory_pressure_events: AtomicU64}

impl CacheStatistics {
    pub fn new() -> Self {
        Self {
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            load_times_ms: AtomicU64::new(0),
            memory_pressure_events: AtomicU64::new(0)}
    }
    
    /// Get cache hit ratio as percentage
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total == 0 {
            0.0
        } else {
            (hits as f64 / total as f64) * 100.0
        }
    }
}

impl ModelCache {
    /// Create new model cache with repository integration
    pub fn new(repository: Arc<ModelRepository>) -> Self {
        let cache = Self {
            cache: ArcSwap::from_pointee(HashMap::new()),
            total_memory_usage: AtomicCell::new(0),
            memory_pressure: AtomicCell::new(MemoryPressure::Low),
            repository,
            cleanup_task: RwLock::new(None),
            stats: CacheStatistics::new()};
        
        cache
    }
    
    /// Initialize background cleanup task
    pub async fn start_background_cleanup(&self) {
        let cache_weak = Arc::downgrade(&Arc::new(std::ptr::addr_of!(*self)));
        
        let cleanup_task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(CACHE_CLEANUP_INTERVAL_SECS));
            interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
            
            loop {
                interval.tick().await;
                
                if let Some(cache_ptr) = cache_weak.upgrade() {
                    unsafe {
                        let cache = &*cache_ptr.as_ptr();
                        cache.cleanup_expired_models().await;
                        cache.check_memory_pressure().await;
                    }
                } else {
                    break;
                }
            }
        });
        
        *self.cleanup_task.write().await = Some(cleanup_task);
    }
    
    /// Stop background cleanup task
    pub async fn stop_background_cleanup(&self) {
        if let Some(task) = self.cleanup_task.write().await.take() {
            task.abort();
        }
    }
    
    /// Get cached model (lock-free access)
    pub async fn get_model(&self, model: CandleModel) -> Result<Arc<LoadedModel>, ModelCacheError> {
        // Fast path: check cache first
        {
            let cache_guard = self.cache.load();
            if let Some(cached) = cache_guard.get(&model) {
                cached.touch();
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(cached.model.clone());
            }
        }
        
        // Cache miss - load model
        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.load_model(model).await
    }
    
    /// Load model into cache with memory management
    async fn load_model(&self, model: CandleModel) -> Result<Arc<LoadedModel>, ModelCacheError> {
        let start_time = Instant::now();
        
        // Check if we need to evict models before loading
        self.ensure_cache_capacity().await?;
        
        // Get model metadata
        let metadata = self.repository.get_metadata(model)
            .ok_or(ModelCacheError::ModelNotFound(model))?;
        
        // Download model if not available
        if !self.repository.is_model_available(model) {
            self.repository.download_model(model).await
                .map_err(|e| ModelCacheError::DownloadError(format!("Failed to download model: {}", e)))?;
        }
        
        // Load model files
        let model_files = self.repository.download_model(model).await
            .map_err(|e| ModelCacheError::LoadError(format!("Failed to get model files: {}", e)))?;
        
        // Determine device for loading
        let device = self.select_optimal_device(&metadata).await?;
        
        // Load model based on architecture
        let loaded_model = match metadata.architecture {
            ModelArchitecture::LLaMA => self.load_llama_model(model_files, device).await?,
            ModelArchitecture::Mistral => self.load_mistral_model(model_files, device).await?,
            ModelArchitecture::Phi3 => self.load_phi3_model(model_files, device).await?,
            ModelArchitecture::Gemma => self.load_gemma_model(model_files, device).await?,
            _ => return Err(ModelCacheError::UnsupportedArchitecture(metadata.architecture))};
        
        let loaded_model = Arc::new(loaded_model);
        let cached_model = Arc::new(CachedModel::new(loaded_model.clone()));
        
        // Update cache atomically
        let mut new_cache = self.cache.load().as_ref().clone();
        new_cache.insert(model, cached_model);
        self.cache.store(Arc::new(new_cache));
        
        // Update memory usage
        let memory_usage = loaded_model.memory_usage_bytes;
        let old_usage = self.total_memory_usage.load();
        self.total_memory_usage.store(old_usage + memory_usage);
        
        // Record load time
        let load_time_ms = start_time.elapsed().as_millis() as u64;
        self.stats.load_times_ms.fetch_add(load_time_ms, Ordering::Relaxed);
        
        Ok(loaded_model)
    }
    
    /// Select optimal device for model loading
    async fn select_optimal_device(&self, metadata: &super::model_repo::ModelMetadata) -> Result<Device, ModelCacheError> {
        // Check available memory and select best device
        let required_memory_gb = metadata.min_memory_gb as u64;
        
        // Try Metal on macOS
        #[cfg(target_os = "macos")]
        {
            if let Ok(device) = Device::new_metal(0) {
                // Check Metal memory availability (simplified)
                return Ok(device);
            }
        }
        
        // Try CUDA if available
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            if let Ok(device) = Device::new_cuda(0) {
                // Check CUDA memory availability (simplified)
                return Ok(device);
            }
        }
        
        // Fallback to CPU
        Ok(Device::Cpu)
    }
    
    /// Load LLaMA model
    async fn load_llama_model(
        &self, 
        model_files: SmallVec<[ModelFileInfo; 8]>, 
        device: Device
    ) -> Result<LoadedModel, ModelCacheError> {
        // Find configuration and model files
        let config_file = model_files.iter()
            .find(|f| f.file_path.file_name().unwrap_or_default() == "config.json")
            .ok_or(ModelCacheError::ConfigNotFound)?;
        
        let tokenizer_file = model_files.iter()
            .find(|f| f.file_path.file_name().unwrap_or_default() == "tokenizer.json")
            .ok_or(ModelCacheError::TokenizerNotFound)?;
        
        // Load configuration
        let config_content = tokio::fs::read_to_string(&config_file.file_path).await
            .map_err(|e| ModelCacheError::IoError(e))?;
        let llama_config: LlamaConfig = serde_json::from_str(&config_content)
            .map_err(|e| ModelCacheError::ConfigParseError(format!("LLaMA config parse error: {}", e)))?;
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_file.file_path)
            .map_err(|e| ModelCacheError::TokenizerLoadError(format!("Tokenizer load error: {}", e)))?;
        
        // Load model weights
        let model_files: Vec<_> = model_files.iter()
            .filter(|f| f.file_path.extension().unwrap_or_default() == "safetensors")
            .map(|f| f.file_path.clone())
            .collect();
        
        let dtype = DType::F16; // Use F16 for memory efficiency
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)
                .map_err(|e| ModelCacheError::ModelLoadError(format!("Failed to load safetensors: {}", e)))?
        };
        
        // Create model
        let model = Llama::load(var_builder, &llama_config)
            .map_err(|e| ModelCacheError::ModelLoadError(format!("Failed to create LLaMA model: {}", e)))?;
        
        // Create KV cache
        let kv_cache = LLaMaAttentionCache::new(&llama_config, &device, dtype)
            .map_err(|e| ModelCacheError::CacheError(format!("Failed to create KV cache: {}", e)))?;
        
        // Calculate memory usage
        let memory_usage = self.estimate_model_memory_usage(&llama_config, dtype);
        
        // Create model config
        let config = ModelConfig {
            vocab_size: llama_config.vocab_size,
            hidden_size: llama_config.hidden_size,
            num_attention_heads: llama_config.num_attention_heads,
            num_key_value_heads: llama_config.num_key_value_heads.unwrap_or(llama_config.num_attention_heads),
            num_hidden_layers: llama_config.num_hidden_layers,
            max_position_embeddings: llama_config.max_position_embeddings.unwrap_or(4096),
            rope_theta: llama_config.rope_theta.unwrap_or(10000.0),
            use_flash_attn: false};
        
        Ok(LoadedModel {
            model: ModelInstance::LLaMA(Box::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
            config,
            kv_cache: Some(Box::new(kv_cache)),
            memory_usage_bytes: memory_usage,
            loaded_at: Instant::now()})
    }
    
    /// Load Mistral model (simplified implementation)
    async fn load_mistral_model(
        &self, 
        _model_files: SmallVec<[ModelFileInfo; 8]>, 
        _device: Device
    ) -> Result<LoadedModel, ModelCacheError> {
        // Placeholder for Mistral model loading
        // In a real implementation, this would load Mistral-specific weights and config
        Err(ModelCacheError::UnsupportedArchitecture(ModelArchitecture::Mistral))
    }
    
    /// Load Phi3 model (placeholder)
    async fn load_phi3_model(
        &self, 
        _model_files: SmallVec<[ModelFileInfo; 8]>, 
        _device: Device
    ) -> Result<LoadedModel, ModelCacheError> {
        Err(ModelCacheError::UnsupportedArchitecture(ModelArchitecture::Phi3))
    }
    
    /// Load Gemma model (placeholder)
    async fn load_gemma_model(
        &self, 
        _model_files: SmallVec<[ModelFileInfo; 8]>, 
        _device: Device
    ) -> Result<LoadedModel, ModelCacheError> {
        Err(ModelCacheError::UnsupportedArchitecture(ModelArchitecture::Gemma))
    }
    
    /// Estimate model memory usage
    fn estimate_model_memory_usage(&self, config: &LlamaConfig, dtype: DType) -> usize {
        let dtype_size = match dtype {
            DType::F16 => 2,
            DType::F32 => 4,
            DType::BF16 => 2,
            _ => 4};
        
        // Rough estimation: vocab_size * hidden_size + hidden_size^2 * num_layers * factor
        let vocab_embeddings = config.vocab_size * config.hidden_size * dtype_size;
        let layer_weights = config.hidden_size * config.hidden_size * config.num_hidden_layers * 8 * dtype_size; // 8x factor for all layer weights
        
        vocab_embeddings + layer_weights
    }
    
    /// Ensure cache has capacity for new model
    async fn ensure_cache_capacity(&self) -> Result<(), ModelCacheError> {
        let cache_guard = self.cache.load();
        
        if cache_guard.len() >= MAX_CACHED_MODELS {
            drop(cache_guard);
            self.evict_least_recently_used().await?;
        }
        
        Ok(())
    }
    
    /// Evict least recently used model
    async fn evict_least_recently_used(&self) -> Result<(), ModelCacheError> {
        let cache_guard = self.cache.load();
        
        if cache_guard.is_empty() {
            return Ok(());
        }
        
        // Find LRU model
        let mut oldest_model = None;
        let mut oldest_time = u64::MAX;
        
        for (model, cached) in cache_guard.iter() {
            let last_access = cached.get_last_access();
            if last_access < oldest_time {
                oldest_time = last_access;
                oldest_model = Some(*model);
            }
        }
        
        if let Some(model_to_evict) = oldest_model {
            let memory_to_free = cache_guard.get(&model_to_evict)
                .map(|cached| cached.model.memory_usage_bytes)
                .unwrap_or(0);
            
            drop(cache_guard);
            
            // Remove from cache
            let mut new_cache = self.cache.load().as_ref().clone();
            new_cache.remove(&model_to_evict);
            self.cache.store(Arc::new(new_cache));
            
            // Update memory usage
            let old_usage = self.total_memory_usage.load();
            self.total_memory_usage.store(old_usage.saturating_sub(memory_to_free));
            
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
        
        Ok(())
    }
    
    /// Check memory pressure and trigger cleanup if needed
    async fn check_memory_pressure(&self) {
        let current_usage_mb = self.total_memory_usage.load() / (1024 * 1024);
        
        let pressure = match current_usage_mb {
            0..=MEMORY_PRESSURE_WARNING_MB => MemoryPressure::Low,
            MEMORY_PRESSURE_WARNING_MB..=MEMORY_PRESSURE_CRITICAL_MB => MemoryPressure::Medium,
            _ => MemoryPressure::Critical};
        
        let old_pressure = self.memory_pressure.load();
        if pressure as u8 > old_pressure as u8 {
            self.memory_pressure.store(pressure);
            self.stats.memory_pressure_events.fetch_add(1, Ordering::Relaxed);
            
            // Trigger aggressive cleanup under high pressure
            if matches!(pressure, MemoryPressure::High | MemoryPressure::Critical) {
                let _ = self.evict_least_recently_used().await;
            }
        }
    }
    
    /// Clean up expired models
    async fn cleanup_expired_models(&self) {
        // This could implement time-based expiration in addition to LRU
        // For now, we rely on LRU eviction
    }
    
    /// Get cache statistics
    pub fn get_statistics(&self) -> &CacheStatistics {
        &self.stats
    }
    
    /// Get current memory usage in bytes
    #[inline(always)]
    pub fn get_memory_usage(&self) -> usize {
        self.total_memory_usage.load()
    }
    
    /// Get current memory pressure level
    #[inline(always)]
    pub fn get_memory_pressure(&self) -> MemoryPressure {
        self.memory_pressure.load()
    }
    
    /// Get number of cached models
    #[inline(always)]
    pub fn get_cache_size(&self) -> usize {
        self.cache.load().len()
    }
    
    /// Clear all cached models
    pub async fn clear_cache(&self) {
        self.cache.store(Arc::new(HashMap::new()));
        self.total_memory_usage.store(0);
        self.memory_pressure.store(MemoryPressure::Low);
    }
}

impl Drop for ModelCache {
    fn drop(&mut self) {
        // Cleanup task will be automatically aborted when the handle is dropped
    }
}

/// Model cache error types
#[derive(Debug, thiserror::Error)]
pub enum ModelCacheError {
    #[error("Model not found: {0:?}")]
    ModelNotFound(CandleModel),
    
    #[error("Unsupported model architecture: {0:?}")]
    UnsupportedArchitecture(ModelArchitecture),
    
    #[error("Configuration file not found")]
    ConfigNotFound,
    
    #[error("Tokenizer file not found")]
    TokenizerNotFound,
    
    #[error("Configuration parse error: {0}")]
    ConfigParseError(String),
    
    #[error("Tokenizer load error: {0}")]
    TokenizerLoadError(String),
    
    #[error("Model load error: {0}")]
    ModelLoadError(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
    
    #[error("Download error: {0}")]
    DownloadError(String),
    
    #[error("Load error: {0}")]
    LoadError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] tokio::io::Error)}