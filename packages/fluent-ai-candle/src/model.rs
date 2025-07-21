//! Zero-allocation model loading and management with memory mapping and atomic state

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use arrayvec::ArrayVec;
use candle_core::{Device, Module, Tensor};

use crossbeam_skiplist::SkipMap;
use memmap2::Mmap;

use crate::constants::{DEFAULT_KV_CACHE_SIZE, DEFAULT_TOKEN_BUFFER_SIZE, MAX_MODEL_FILE_SIZE};
use crate::error::{CandleError, CandleResult};
use crate::memory;

/// Supported model types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// LLaMA family models (LLaMA, LLaMA2, Code Llama)
    Llama = 0,
    /// Mistral family models (Mistral 7B, Mixtral)
    Mistral = 1,
    /// Gemma family models
    Gemma = 2,
    /// Phi family models
    Phi = 3,
    /// Qwen family models
    Qwen = 4,
    /// Custom/unknown model
    Custom = 255,
}

impl ModelType {
    /// Get model type from string
    #[inline(always)]
    pub fn from_str(s: &str) -> Self {
        let s = s.to_lowercase();
        if s.contains("llama") {
            Self::Llama
        } else if s.contains("mistral") || s.contains("mixtral") {
            Self::Mistral
        } else if s.contains("gemma") {
            Self::Gemma
        } else if s.contains("phi") {
            Self::Phi
        } else if s.contains("qwen") {
            Self::Qwen
        } else {
            Self::Custom
        }
    }

    /// Get default context length for model type
    #[inline(always)]
    pub fn default_context_length(&self) -> u32 {
        match self {
            Self::Llama => 4096,
            Self::Mistral => 8192,
            Self::Gemma => 8192,
            Self::Phi => 2048,
            Self::Qwen => 8192,
            Self::Custom => 2048,
        }
    }
}

/// Model configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model type
    pub model_type: ModelType,
    /// Context length
    pub context_length: u32,
    /// Vocabulary size
    pub vocab_size: u32,
    /// Number of layers
    pub num_layers: u32,
    /// Hidden dimension
    pub hidden_size: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// RoPE theta parameter
    pub rope_theta: f32,
    /// RoPE frequency base
    pub rope_freq_base: f32,
    /// Use flash attention
    pub use_flash_attn: bool,
    /// Quantization type
    pub quantization: QuantizationType,
}

impl Default for ModelConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            model_type: ModelType::Llama,
            context_length: 4096,
            vocab_size: 32000,
            num_layers: 32,
            hidden_size: 4096,
            num_heads: 32,
            rope_theta: 10000.0,
            rope_freq_base: 1.0,
            use_flash_attn: true,
            quantization: QuantizationType::None,
        }
    }
}

/// Quantization types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// No quantization
    None = 0,
    /// 4-bit quantization (Q4_0)
    Q4_0 = 1,
    /// 4-bit quantization (Q4_1)
    Q4_1 = 2,
    /// 8-bit quantization
    Q8_0 = 3,
}

/// KV cache entry for efficient attention computation
#[repr(C)]
struct KVCacheEntry {
    /// Key tensor
    key: Tensor,
    /// Value tensor
    value: Tensor,
    /// Sequence length
    seq_len: u32,
    /// Last access time (for LRU eviction)
    last_access: AtomicU64,
}

impl KVCacheEntry {
    #[inline(always)]
    fn new(key: Tensor, value: Tensor, seq_len: u32) -> Self {
        Self {
            key,
            value,
            seq_len,
            last_access: AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ),
        }
    }

    #[inline(always)]
    fn touch(&self) {
        self.last_access.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            Ordering::Relaxed,
        );
    }
}

/// Model state for atomic swapping
struct ModelState {
    /// The actual model implementation
    model: Box<dyn Module + Send + Sync>,
    /// Model configuration
    config: ModelConfig,
    /// Model file memory mapping
    _mmap: Option<Mmap>,
}

/// Zero-allocation candle model with lock-free caching
#[repr(C)]
pub struct CandleModel {
    /// Atomic model state for hot-swapping
    model_state: ArcSwap<ModelState>,
    /// Device for computation
    device: Device,
    /// Pre-allocated token buffer
    token_buffer: parking_lot::Mutex<ArrayVec<u32, DEFAULT_TOKEN_BUFFER_SIZE>>,
    /// Lock-free KV cache
    kv_cache: SkipMap<u64, KVCacheEntry>,
    /// Cache size limit
    cache_size_limit: AtomicU32,
    /// Current cache size
    cache_size: AtomicU32,
    /// Model loaded flag
    is_loaded: AtomicBool,
    /// Model loading progress (0-100)
    loading_progress: AtomicU32,
    /// Total memory usage
    memory_usage: AtomicU64,
}

impl CandleModel {
    /// Create a new candle model
    #[inline(always)]
    pub fn new(device: Device) -> Self {
        let initial_state = ModelState {
            model: Box::new(DummyModel),
            config: ModelConfig::default(),
            _mmap: None,
        };

        Self {
            model_state: ArcSwap::new(Arc::new(initial_state)),
            device,
            token_buffer: parking_lot::Mutex::new(ArrayVec::new()),
            kv_cache: SkipMap::new(),
            cache_size_limit: AtomicU32::new(DEFAULT_KV_CACHE_SIZE as u32),
            cache_size: AtomicU32::new(0),
            is_loaded: AtomicBool::new(false),
            loading_progress: AtomicU32::new(0),
            memory_usage: AtomicU64::new(0),
        }
    }

    /// Load model from file path with memory mapping
    #[inline(always)]
    pub async fn load_from_file<P: AsRef<Path>>(&self, path: P) -> CandleResult<()> {
        let path = path.as_ref();

        // Check file size
        let metadata = std::fs::metadata(path)
            .map_err(|e| CandleError::ModelNotFound(format!("File not found: {}", e)))?;

        if metadata.len() > MAX_MODEL_FILE_SIZE as u64 {
            return Err(CandleError::InvalidModelFormat(
                "Model file too large for memory mapping",
            ));
        }

        self.loading_progress.store(10, Ordering::Relaxed);

        // Memory map the file for zero-copy loading
        let file = std::fs::File::open(path)
            .map_err(|e| CandleError::ModelNotFound(format!("Cannot open file: {}", e)))?;

        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| CandleError::ModelNotFound(format!("Cannot memory map file: {}", e)))?;

        self.loading_progress.store(30, Ordering::Relaxed);

        // Detect model type from filename
        let model_type =
            ModelType::from_str(path.file_name().and_then(|n| n.to_str()).unwrap_or(""));

        // Load model based on type
        let (model, config) = match model_type {
            ModelType::Llama => self.load_llama_model(&mmap).await?,
            ModelType::Mistral => self.load_mistral_model(&mmap).await?,
            ModelType::Gemma => self.load_gemma_model(&mmap).await?,
            ModelType::Phi => self.load_phi_model(&mmap).await?,
            ModelType::Qwen => self.load_qwen_model(&mmap).await?,
            ModelType::Custom => {
                return Err(CandleError::InvalidModelFormat("Unsupported model type"))
            }
        };

        self.loading_progress.store(80, Ordering::Relaxed);

        // Create new model state
        let new_state = ModelState {
            model,
            config,
            _mmap: Some(mmap),
        };

        // Atomically swap the model state
        self.model_state.store(Arc::new(new_state));

        // Update memory usage tracking
        let memory_used = metadata.len();
        self.memory_usage.store(memory_used, Ordering::Relaxed);
        memory::track_allocation(memory_used as usize);

        self.loading_progress.store(100, Ordering::Relaxed);
        self.is_loaded.store(true, Ordering::Relaxed);

        Ok(())
    }

    /// Load model from HuggingFace Hub
    #[inline(always)]
    pub async fn load_from_hub(&self, repo_id: &str, filename: &str) -> CandleResult<()> {
        self.loading_progress.store(5, Ordering::Relaxed);

        // Download model file
        let api = hf_hub::api::tokio::Api::new()
            .map_err(|e| CandleError::ModelNotFound(format!("HF Hub API error: {}", e)))?;

        let repo = api.model(repo_id.to_string());

        self.loading_progress.store(20, Ordering::Relaxed);

        let model_path = repo
            .get(filename)
            .await
            .map_err(|e| CandleError::ModelNotFound(format!("Download failed: {}", e)))?;

        self.loading_progress.store(50, Ordering::Relaxed);

        // Load the downloaded file
        self.load_from_file(model_path).await
    }

    /// Forward pass through the model
    #[inline(always)]
    pub fn forward(&self, input_ids: &[u32]) -> CandleResult<Tensor> {
        if !self.is_loaded.load(Ordering::Relaxed) {
            return Err(CandleError::ModelNotFound("Model not loaded".to_string()));
        }

        let state = self.model_state.load();

        // Convert input IDs to tensor
        let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?; // Add batch dimension

        // Forward pass
        let output = state.model.forward(&input_tensor)?;

        Ok(output)
    }

    /// Get model configuration
    #[inline(always)]
    pub fn config(&self) -> ModelConfig {
        self.model_state.load().config.clone()
    }

    /// Check if model is loaded
    #[inline(always)]
    pub fn is_loaded(&self) -> bool {
        self.is_loaded.load(Ordering::Relaxed)
    }

    /// Get loading progress (0-100)
    #[inline(always)]
    pub fn loading_progress(&self) -> u32 {
        self.loading_progress.load(Ordering::Relaxed)
    }

    /// Get memory usage in bytes
    #[inline(always)]
    pub fn memory_usage(&self) -> u64 {
        self.memory_usage.load(Ordering::Relaxed)
    }

    /// Get device information
    #[inline(always)]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Clear KV cache
    #[inline(always)]
    pub fn clear_cache(&self) {
        self.kv_cache.clear();
        self.cache_size.store(0, Ordering::Relaxed);
    }

    /// Get cache statistics
    #[inline(always)]
    pub fn cache_stats(&self) -> (u32, u32) {
        let size = self.cache_size.load(Ordering::Relaxed);
        let limit = self.cache_size_limit.load(Ordering::Relaxed);
        (size, limit)
    }

    // Private helper methods for loading different model types

    async fn load_llama_model(
        &self,
        mmap: &Mmap,
    ) -> CandleResult<(Box<dyn Module + Send + Sync>, ModelConfig)> {
        use candle_transformers::models::llama as llama_models;
        use safetensors::tensor::SafeTensors;

        // Parse safetensors file to get model weights
        let safetensors = SafeTensors::deserialize(mmap).map_err(|e| {
            CandleError::ModelLoadError(format!("Failed to parse safetensors: {}", e))
        })?;

        // Extract model configuration from tensor metadata or use defaults
        let config_json = safetensors
            .metadata()
            .and_then(|m| m.get("model_config"))
            .unwrap_or("{}");

        // Parse LLaMA config or use sensible defaults
        let llama_config =
            if let Ok(parsed_config) = serde_json::from_str::<serde_json::Value>(config_json) {
                llama_models::Config {
                    hidden_size: parsed_config
                        .get("hidden_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4096) as usize,
                    intermediate_size: parsed_config
                        .get("intermediate_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(11008) as usize,
                    vocab_size: parsed_config
                        .get("vocab_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32000) as usize,
                    num_hidden_layers: parsed_config
                        .get("num_hidden_layers")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32) as usize,
                    num_attention_heads: parsed_config
                        .get("num_attention_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32) as usize,
                    num_key_value_heads: parsed_config
                        .get("num_key_value_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32) as usize,
                    rms_norm_eps: parsed_config
                        .get("rms_norm_eps")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1e-6) as f64,
                    rope_theta: parsed_config
                        .get("rope_theta")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(10000.0) as f32,
                    max_position_embeddings: parsed_config
                        .get("max_position_embeddings")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4096) as usize,
                    use_flash_attn: false, // Disable flash attention for compatibility
                }
            } else {
                // Default LLaMA 7B configuration
                llama_models::Config {
                    hidden_size: 4096,
                    intermediate_size: 11008,
                    vocab_size: 32000,
                    num_hidden_layers: 32,
                    num_attention_heads: 32,
                    num_key_value_heads: 32,
                    rms_norm_eps: 1e-6,
                    rope_theta: 10000.0,
                    max_position_embeddings: 4096,
                    use_flash_attn: false,
                }
            };

        // Create variable store from memory-mapped safetensors data 
        let vs = candle_nn::VarBuilder::from_slice_safetensors(mmap, candle_core::DType::F16, &self.device)?;

        // Load LLaMA model
        let llama_model = llama_models::Llama::load(&vs, &llama_config).map_err(|e| {
            CandleError::ModelLoadError(format!("Failed to load LLaMA model: {}", e))
        })?;

        // Create model configuration
        let model_config = ModelConfig {
            model_type: ModelType::Llama,
            context_length: llama_config.max_position_embeddings as u32,
            vocab_size: llama_config.vocab_size as u32,
            hidden_size: llama_config.hidden_size as u32,
            num_layers: llama_config.num_hidden_layers as u32,
            num_attention_heads: llama_config.num_attention_heads as u32,
            ..Default::default()
        };

        Ok((Box::new(llama_model), model_config))
    }

    async fn load_mistral_model(
        &self,
        mmap: &Mmap,
    ) -> CandleResult<(Box<dyn Module + Send + Sync>, ModelConfig)> {
        use candle_transformers::models::mistral as mistral_models;
        use safetensors::tensor::SafeTensors;

        // Parse safetensors file to get model weights
        let safetensors = SafeTensors::deserialize(mmap).map_err(|e| {
            CandleError::ModelLoadError(format!("Failed to parse safetensors: {}", e))
        })?;

        // Extract model configuration from tensor metadata or use defaults
        let config_json = safetensors
            .metadata()
            .and_then(|m| m.get("model_config"))
            .unwrap_or("{}");

        // Parse Mistral config or use sensible defaults
        let mistral_config =
            if let Ok(parsed_config) = serde_json::from_str::<serde_json::Value>(config_json) {
                mistral_models::Config {
                    vocab_size: parsed_config
                        .get("vocab_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32000) as usize,
                    hidden_size: parsed_config
                        .get("hidden_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4096) as usize,
                    intermediate_size: parsed_config
                        .get("intermediate_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(14336) as usize,
                    num_hidden_layers: parsed_config
                        .get("num_hidden_layers")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32) as usize,
                    num_attention_heads: parsed_config
                        .get("num_attention_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32) as usize,
                    num_key_value_heads: parsed_config
                        .get("num_key_value_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(8) as usize,
                    max_position_embeddings: parsed_config
                        .get("max_position_embeddings")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32768) as usize,
                    rms_norm_eps: parsed_config
                        .get("rms_norm_eps")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1e-5) as f64,
                    rope_theta: parsed_config
                        .get("rope_theta")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(10000.0) as f32,
                    sliding_window: parsed_config
                        .get("sliding_window")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize),
                }
            } else {
                // Default Mistral 7B configuration
                mistral_models::Config {
                    vocab_size: 32000,
                    hidden_size: 4096,
                    intermediate_size: 14336,
                    num_hidden_layers: 32,
                    num_attention_heads: 32,
                    num_key_value_heads: 8,
                    max_position_embeddings: 32768,
                    rms_norm_eps: 1e-5,
                    rope_theta: 10000.0,
                    sliding_window: Some(4096),
                }
            };

        // Create variable store from memory-mapped safetensors data 
        let vs = candle_nn::VarBuilder::from_slice_safetensors(mmap, candle_core::DType::F16, &self.device)?;

        // Load Mistral model
        let mistral_model = mistral_models::Model::load(&vs, &mistral_config).map_err(|e| {
            CandleError::ModelLoadError(format!("Failed to load Mistral model: {}", e))
        })?;

        // Create model configuration
        let model_config = ModelConfig {
            model_type: ModelType::Mistral,
            context_length: mistral_config.max_position_embeddings as u32,
            vocab_size: mistral_config.vocab_size as u32,
            hidden_size: mistral_config.hidden_size as u32,
            num_layers: mistral_config.num_hidden_layers as u32,
            num_attention_heads: mistral_config.num_attention_heads as u32,
            ..Default::default()
        };

        Ok((Box::new(mistral_model), model_config))
    }

    async fn load_gemma_model(
        &self,
        mmap: &Mmap,
    ) -> CandleResult<(Box<dyn Module + Send + Sync>, ModelConfig)> {
        use candle_transformers::models::gemma as gemma_models;
        use safetensors::tensor::SafeTensors;

        // Parse safetensors file to get model weights
        let safetensors = SafeTensors::deserialize(mmap).map_err(|e| {
            CandleError::ModelLoadError(format!("Failed to parse safetensors: {}", e))
        })?;

        // Extract model configuration from tensor metadata or use defaults
        let config_json = safetensors
            .metadata()
            .and_then(|m| m.get("model_config"))
            .unwrap_or("{}");

        // Parse Gemma config or use sensible defaults
        let gemma_config =
            if let Ok(parsed_config) = serde_json::from_str::<serde_json::Value>(config_json) {
                gemma_models::Config {
                    vocab_size: parsed_config
                        .get("vocab_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(256000) as usize,
                    hidden_size: parsed_config
                        .get("hidden_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(3072) as usize,
                    intermediate_size: parsed_config
                        .get("intermediate_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(24576) as usize,
                    num_hidden_layers: parsed_config
                        .get("num_hidden_layers")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(28) as usize,
                    num_attention_heads: parsed_config
                        .get("num_attention_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(16) as usize,
                    num_key_value_heads: parsed_config
                        .get("num_key_value_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(16) as usize,
                    head_dim: parsed_config
                        .get("head_dim")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(256) as usize,
                    max_position_embeddings: parsed_config
                        .get("max_position_embeddings")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(8192) as usize,
                    rms_norm_eps: parsed_config
                        .get("rms_norm_eps")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1e-6) as f64,
                    rope_theta: parsed_config
                        .get("rope_theta")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(10000.0) as f32,
                    attention_bias: parsed_config
                        .get("attention_bias")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    mlp_bias: parsed_config
                        .get("mlp_bias")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                }
            } else {
                // Default Gemma 7B configuration
                gemma_models::Config {
                    vocab_size: 256000,
                    hidden_size: 3072,
                    intermediate_size: 24576,
                    num_hidden_layers: 28,
                    num_attention_heads: 16,
                    num_key_value_heads: 16,
                    head_dim: 256,
                    max_position_embeddings: 8192,
                    rms_norm_eps: 1e-6,
                    rope_theta: 10000.0,
                    attention_bias: false,
                    mlp_bias: false,
                }
            };

        // Create variable store from memory-mapped safetensors data 
        let vs = candle_nn::VarBuilder::from_slice_safetensors(mmap, candle_core::DType::F16, &self.device)?;

        // Load Gemma model
        let gemma_model = gemma_models::Model::load(&vs, &gemma_config).map_err(|e| {
            CandleError::ModelLoadError(format!("Failed to load Gemma model: {}", e))
        })?;

        // Create model configuration
        let model_config = ModelConfig {
            model_type: ModelType::Gemma,
            context_length: gemma_config.max_position_embeddings as u32,
            vocab_size: gemma_config.vocab_size as u32,
            hidden_size: gemma_config.hidden_size as u32,
            num_layers: gemma_config.num_hidden_layers as u32,
            num_attention_heads: gemma_config.num_attention_heads as u32,
            ..Default::default()
        };

        Ok((Box::new(gemma_model), model_config))
    }

    async fn load_phi_model(
        &self,
        mmap: &Mmap,
    ) -> CandleResult<(Box<dyn Module + Send + Sync>, ModelConfig)> {
        use candle_transformers::models::phi as phi_models;
        use safetensors::tensor::SafeTensors;

        // Parse safetensors file to get model weights
        let safetensors = SafeTensors::deserialize(mmap).map_err(|e| {
            CandleError::ModelLoadError(format!("Failed to parse safetensors: {}", e))
        })?;

        // Extract model configuration from tensor metadata or use defaults
        let config_json = safetensors
            .metadata()
            .and_then(|m| m.get("model_config"))
            .unwrap_or("{}");

        // Parse Phi config or use sensible defaults
        let phi_config =
            if let Ok(parsed_config) = serde_json::from_str::<serde_json::Value>(config_json) {
                phi_models::Config {
                    vocab_size: parsed_config
                        .get("vocab_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(51200) as usize,
                    hidden_size: parsed_config
                        .get("hidden_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(2560) as usize,
                    intermediate_size: parsed_config
                        .get("intermediate_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(10240) as usize,
                    num_hidden_layers: parsed_config
                        .get("num_hidden_layers")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32) as usize,
                    num_attention_heads: parsed_config
                        .get("num_attention_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32) as usize,
                    num_key_value_heads: parsed_config
                        .get("num_key_value_heads")
                        .and_then(|v| v.as_u64()),
                    max_position_embeddings: parsed_config
                        .get("max_position_embeddings")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(2048) as usize,
                    layer_norm_eps: parsed_config
                        .get("layer_norm_eps")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1e-5) as f64,
                    rope_theta: parsed_config
                        .get("rope_theta")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(10000.0) as f32,
                    partial_rotary_factor: parsed_config
                        .get("partial_rotary_factor")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5) as f64,
                    qk_layernorm: parsed_config
                        .get("qk_layernorm")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                }
            } else {
                // Default Phi-3 Mini configuration
                phi_models::Config {
                    vocab_size: 32064,
                    hidden_size: 3072,
                    intermediate_size: 8192,
                    num_hidden_layers: 32,
                    num_attention_heads: 32,
                    num_key_value_heads: Some(32),
                    max_position_embeddings: 4096,
                    layer_norm_eps: 1e-5,
                    rope_theta: 10000.0,
                    partial_rotary_factor: 0.5,
                    qk_layernorm: false,
                }
            };

        // Create variable store from memory-mapped safetensors data 
        let vs = candle_nn::VarBuilder::from_slice_safetensors(mmap, candle_core::DType::F16, &self.device)?;

        // Load Phi model
        let phi_model = phi_models::Model::load(&vs, &phi_config)
            .map_err(|e| CandleError::ModelLoadError(format!("Failed to load Phi model: {}", e)))?;

        // Create model configuration
        let model_config = ModelConfig {
            model_type: ModelType::Phi,
            context_length: phi_config.max_position_embeddings as u32,
            vocab_size: phi_config.vocab_size as u32,
            hidden_size: phi_config.hidden_size as u32,
            num_layers: phi_config.num_hidden_layers as u32,
            num_attention_heads: phi_config.num_attention_heads as u32,
            ..Default::default()
        };

        Ok((Box::new(phi_model), model_config))
    }

    async fn load_qwen_model(
        &self,
        mmap: &Mmap,
    ) -> CandleResult<(Box<dyn Module + Send + Sync>, ModelConfig)> {
        use candle_transformers::models::qwen2 as qwen_models;
        use safetensors::tensor::SafeTensors;

        // Parse safetensors file to get model weights
        let safetensors = SafeTensors::deserialize(mmap).map_err(|e| {
            CandleError::ModelLoadError(format!("Failed to parse safetensors: {}", e))
        })?;

        // Extract model configuration from tensor metadata or use defaults
        let config_json = safetensors
            .metadata()
            .and_then(|m| m.get("model_config"))
            .unwrap_or("{}");

        // Parse Qwen config or use sensible defaults
        let qwen_config =
            if let Ok(parsed_config) = serde_json::from_str::<serde_json::Value>(config_json) {
                qwen_models::Config {
                    vocab_size: parsed_config
                        .get("vocab_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(151936) as usize,
                    hidden_size: parsed_config
                        .get("hidden_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(3584) as usize,
                    intermediate_size: parsed_config
                        .get("intermediate_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(18944) as usize,
                    num_hidden_layers: parsed_config
                        .get("num_hidden_layers")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(28) as usize,
                    num_attention_heads: parsed_config
                        .get("num_attention_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(28) as usize,
                    num_key_value_heads: parsed_config
                        .get("num_key_value_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4) as usize,
                    max_position_embeddings: parsed_config
                        .get("max_position_embeddings")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32768) as usize,
                    sliding_window: parsed_config
                        .get("sliding_window")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                        .unwrap_or(32768),
                    max_window_layers: parsed_config
                        .get("max_window_layers")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(28) as usize,
                    tie_word_embeddings: parsed_config
                        .get("tie_word_embeddings")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    rope_theta: parsed_config
                        .get("rope_theta")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1000000.0) as f64,
                    rms_norm_eps: parsed_config
                        .get("rms_norm_eps")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1e-6) as f64,
                    use_sliding_window: parsed_config
                        .get("use_sliding_window")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    hidden_act: candle_nn::Activation::Silu,
                }
            } else {
                // Default Qwen2-7B configuration
                qwen_models::Config {
                    vocab_size: 152064,
                    hidden_size: 3584,
                    intermediate_size: 18944,
                    num_hidden_layers: 28,
                    num_attention_heads: 28,
                    num_key_value_heads: 4,
                    max_position_embeddings: 32768,
                    sliding_window: 32768,
                    max_window_layers: 28,
                    tie_word_embeddings: false,
                    rope_theta: 1000000.0,
                    rms_norm_eps: 1e-6,
                    use_sliding_window: false,
                    hidden_act: candle_nn::Activation::Silu,
                }
            };

        // Create variable store from memory-mapped safetensors data 
        let vs = candle_nn::VarBuilder::from_slice_safetensors(mmap, candle_core::DType::F16, &self.device)?;

        // Load Qwen model
        let qwen_model = qwen_models::Model::new(&qwen_config, vs).map_err(|e| {
            CandleError::model_load_error(format!("Failed to load Qwen model: {}", e))
        })?;

        // Create model configuration
        let model_config = ModelConfig {
            model_type: ModelType::Qwen,
            context_length: qwen_config.max_position_embeddings as u32,
            vocab_size: qwen_config.vocab_size as u32,
            hidden_size: qwen_config.hidden_size as u32,
            num_layers: qwen_config.num_hidden_layers as u32,
            num_heads: qwen_config.num_attention_heads as u32,
            ..Default::default()
        };

        Ok((Box::new(qwen_model), model_config))
    }
}

impl Drop for CandleModel {
    fn drop(&mut self) {
        let memory_used = self.memory_usage.load(Ordering::Relaxed);
        if memory_used > 0 {
            memory::track_deallocation(memory_used as usize);
        }
    }
}

/// Dummy model implementation for testing
struct DummyModel;

impl Module for DummyModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Return logits for a small vocabulary (for testing)
        let batch_size = xs.dim(0)?;
        let seq_len = xs.dim(1)?;
        let vocab_size = 1000;

        // Create dummy logits
        Tensor::zeros((batch_size, seq_len, vocab_size), xs.dtype(), xs.device())
    }
}

unsafe impl Send for CandleModel {}
unsafe impl Sync for CandleModel {}
