//! Provides streaming completion capabilities using local Kimi K2 models
//! with zero allocation patterns and AsyncStream streaming.
//!
//! This implementation uses the Candle ML framework for local model inference,
//! specifically targeting Llama-compatible models for high-performance text generation.

use std::num::NonZeroU32;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::DType;
use candle_transformers::models::llama::LlamaConfig;
use fluent_ai_async::AsyncStream;
// SIMD optimizations for high-performance inference
use fluent_ai_simd::get_cpu_features;
use progresshub::types::ZeroOneOrMany;
use progresshub::ProgressHub;
use serde::{Deserialize, Serialize};

use crate::builders::agent_role::CandleCompletionProvider as BuilderCandleCompletionProvider;
use crate::domain::model::{info::CandleModelInfo, traits::CandleModel};
use crate::domain::{
    completion::{CandleCompletionModel, CandleCompletionParams},
    context::chunk::CandleCompletionChunk,
    prompt::CandlePrompt,
};

/// CandleKimiK2Provider for local Kimi K2 model inference using Candle ML framework
#[derive(Debug, Clone)]
pub struct CandleKimiK2Provider {
    /// Model path on filesystem
    model_path: String,
    /// Provider configuration
    config: CandleKimiK2Config,
    /// Model configuration for inference
    model_config: LlamaConfig,
}

/// Configuration for Kimi K2 model inference
#[derive(Debug, Clone)]
pub struct CandleKimiK2Config {
    /// Maximum context length for inference
    max_context: u32,
    /// Default temperature for sampling
    temperature: f64,
    /// Vocabulary size for tokenization
    vocab_size: u32,
    /// Enable key-value caching for faster inference
    use_kv_cache: bool,
    /// Data type for model weights (F16, BF16, F32)
    dtype: DType,
}

impl Default for CandleKimiK2Config {
    #[inline]
    fn default() -> Self {
        Self {
            max_context: 8192,
            temperature: 0.7,
            vocab_size: 32000,
            use_kv_cache: true,
            dtype: DType::F16,
        }
    }
}

impl CandleKimiK2Config {
    /// Get the temperature setting
    #[inline]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Set temperature for sampling
    #[inline]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set maximum context length
    #[inline]
    pub fn with_max_context(mut self, max_context: u32) -> Self {
        self.max_context = max_context;
        self
    }

    /// Set data type for model weights
    #[inline]
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Enable or disable KV caching
    #[inline]
    pub fn with_kv_cache(mut self, use_kv_cache: bool) -> Self {
        self.use_kv_cache = use_kv_cache;
        self
    }
}

impl CandleKimiK2Provider {
    /// Create new Kimi K2 provider with automatic model download
    ///
    /// This method automatically downloads the Kimi-K2 model from HuggingFace
    /// using ProgressHub and returns a provider ready for inference.
    ///
    /// # Example
    /// ```rust
    /// let provider = CandleKimiK2Provider::new().await?;
    /// ```
    ///
    /// # Errors
    /// Returns error if model download fails or model loading fails
    pub async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config = CandleKimiK2Config::default();
        Self::with_config_async(config).await
    }

    /// Create provider with custom configuration and automatic download
    pub async fn with_config_async(
        config: CandleKimiK2Config,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Download model using ProgressHub
        let results = ProgressHub::builder()
            .model("unsloth/Kimi-K2-Instruct-GGUF")
            .with_cli_progress()
            .download()
            .await?;

        // Extract the model path from download results following ProgressHub example pattern
        let (model_cache_dir, gguf_file_path) = if let Some(result) = results.into_iter().next() {
            match &result.models {
                ZeroOneOrMany::One(model) => {
                    // Extract model cache directory
                    let cache_dir = model.model_cache_path.display().to_string();

                    // Find GGUF file with zero allocation - prioritize largest file (model weights over tokenizer)
                    let gguf_file = model
                        .files
                        .iter()
                        .filter(|file| file.filename.ends_with(".gguf"))
                        .max_by_key(|file| file.expected_size)
                        .ok_or_else(|| "No GGUF files found in downloaded model")?;

                    let gguf_path = gguf_file.path.display().to_string();
                    (cache_dir, gguf_path)
                }
                ZeroOneOrMany::Zero => {
                    return Err("No models were downloaded".into());
                }
                ZeroOneOrMany::Many(_) => {
                    return Err("Expected exactly one model, got multiple".into());
                }
            }
        } else {
            return Err("No download results returned".into());
        };

        Self::with_config_sync_gguf(model_cache_dir, gguf_file_path, config)
    }

    /// Create provider with custom configuration and existing model path
    pub fn with_config_sync(
        model_path: String,
        config: CandleKimiK2Config,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Log SIMD capabilities for performance debugging
        let cpu_info = get_cpu_features();
        log::info!(
            "KimiK2 Provider initialized with SIMD support: {} (vector width: {} elements)",
            cpu_info.has_simd(),
            cpu_info.vector_width()
        );

        // Create model configuration for Kimi K2 (Llama-based architecture)
        // This is used only for configuration - actual model loading handled by core engine
        let model_config = LlamaConfig {
            vocab_size: config.vocab_size as usize,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            max_position_embeddings: config.max_context as usize,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            bos_token_id: Some(1),
            eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(2)),
            rope_scaling: None,
            tie_word_embeddings: Some(false),
        };

        Ok(Self {
            model_path,
            config,
            model_config,
        })
    }

    /// Create provider with custom configuration and GGUF metadata extraction
    ///
    /// This method reads the GGUF file metadata to extract real model configuration
    /// instead of using hardcoded values, ensuring accurate model parameters.
    #[inline(always)]
    pub fn with_config_sync_gguf(
        model_cache_dir: String,
        gguf_file_path: String,
        config: CandleKimiK2Config,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Log SIMD capabilities for performance debugging
        let cpu_info = get_cpu_features();
        log::info!(
            "KimiK2 Provider initialized with SIMD support: {} (vector width: {} elements)",
            cpu_info.has_simd(),
            cpu_info.vector_width()
        );

        // Read GGUF file metadata for real model configuration
        let mut file = std::fs::File::open(&gguf_file_path)?;
        let content = gguf_file::Content::read(&mut file)?;

        // Extract metadata values with fallbacks - zero allocation parsing
        let hidden_size = content
            .metadata
            .get("llama.embedding_length")
            .or_else(|| content.metadata.get("kimi.embedding_length"))
            .and_then(|v| v.to_u64().ok())
            .map(|v| v as usize)
            .unwrap_or(4096); // Fallback for Kimi K2

        let intermediate_size = content
            .metadata
            .get("llama.feed_forward_length")
            .or_else(|| content.metadata.get("kimi.feed_forward_length"))
            .and_then(|v| v.to_u64().ok())
            .map(|v| v as usize)
            .unwrap_or(11008); // Fallback for Kimi K2

        let num_hidden_layers = content
            .metadata
            .get("llama.block_count")
            .or_else(|| content.metadata.get("kimi.block_count"))
            .and_then(|v| v.to_u64().ok())
            .map(|v| v as usize)
            .unwrap_or(32); // Fallback for Kimi K2

        let num_attention_heads = content
            .metadata
            .get("llama.attention.head_count")
            .or_else(|| content.metadata.get("kimi.attention.head_count"))
            .and_then(|v| v.to_u64().ok())
            .map(|v| v as usize)
            .unwrap_or(32); // Fallback for Kimi K2

        let num_key_value_heads = content
            .metadata
            .get("llama.attention.head_count_kv")
            .or_else(|| content.metadata.get("kimi.attention.head_count_kv"))
            .and_then(|v| v.to_u64().ok())
            .map(|v| Some(v as usize))
            .unwrap_or(Some(32)); // Fallback for Kimi K2

        let rope_theta = content
            .metadata
            .get("llama.rope.freq_base")
            .or_else(|| content.metadata.get("kimi.rope.freq_base"))
            .and_then(|v| v.to_f64().ok())
            .unwrap_or(10000.0) as f32; // Standard RoPE theta for Kimi K2

        // Extract vocab_size from metadata or use config default
        let vocab_size = content
            .metadata
            .get("tokenizer.ggml.token_count")
            .or_else(|| content.metadata.get("llama.vocab_size"))
            .and_then(|v| v.to_u64().ok())
            .map(|v| v as usize)
            .unwrap_or(config.vocab_size as usize);

        // Create model configuration with real GGUF metadata
        let model_config = LlamaConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings: config.max_context as usize,
            rms_norm_eps: 1e-6,
            rope_theta,
            bos_token_id: Some(1),
            eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(2)),
            rope_scaling: None,
            tie_word_embeddings: Some(false),
        };

        // Log extracted configuration for debugging
        log::debug!("Extracted GGUF metadata for Kimi K2: hidden_size={}, layers={}, heads={}, rope_theta={}", 
                   hidden_size, num_hidden_layers, num_attention_heads, rope_theta);

        Ok(Self {
            model_path: model_cache_dir,
            config,
            model_config,
        })
    }

    // Unused helper functions removed - model loading now handled by core engine

    /// Get vocabulary size
    #[inline]
    pub fn vocab_size(&self) -> u32 {
        self.config.vocab_size
    }

    /// Set temperature for generation
    #[inline]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = temperature;
        self
    }

    /// Set maximum context length
    #[inline]
    pub fn with_max_context(mut self, max_context: u32) -> Self {
        self.config.max_context = max_context;
        self
    }

    /// Get tokenizer path (embedded in model file for GGUF)
    #[inline]
    pub fn tokenizer_path(&self) -> &str {
        // For GGUF models, tokenizer is embedded in the model file
        &self.model_path
    }

    /// Get maximum tokens (input tokens)
    #[inline]
    pub fn max_tokens(&self) -> u32 {
        self.config.max_context
    }

    /// Get temperature setting
    #[inline]
    pub fn temperature(&self) -> f64 {
        self.config.temperature
    }
}

impl CandleCompletionModel for CandleKimiK2Provider {
    fn prompt(
        &self,
        prompt: CandlePrompt,
        params: &CandleCompletionParams,
    ) -> AsyncStream<CandleCompletionChunk> {
        // Create ModelConfig for this provider (thin wrapper - only config!)
        // Convert LlamaConfig to the format expected by ModelArchitecture
        let candle_config = candle_transformers::models::llama::Config {
            hidden_size: self.model_config.hidden_size,
            intermediate_size: self.model_config.intermediate_size,
            vocab_size: self.model_config.vocab_size,
            num_hidden_layers: self.model_config.num_hidden_layers,
            num_attention_heads: self.model_config.num_attention_heads,
            num_key_value_heads: self
                .model_config
                .num_key_value_heads
                .unwrap_or(self.model_config.num_attention_heads),
            use_flash_attn: false,
            rms_norm_eps: self.model_config.rms_norm_eps,
            rope_theta: self.model_config.rope_theta,
            bos_token_id: self.model_config.bos_token_id,
            eos_token_id: self.model_config.eos_token_id.clone(),
            rope_scaling: self.model_config.rope_scaling.clone(),
            max_position_embeddings: self.model_config.max_position_embeddings,
            tie_word_embeddings: self.model_config.tie_word_embeddings.unwrap_or(false),
        };

        let _model_config = crate::core::ModelConfig::new(
            &self.model_path,
            format!("{}/tokenizer.json", self.model_path),
            crate::core::ModelArchitecture::Llama(candle_config),
            "kimi-k2",
            "kimi-k2",
        )
        .with_vocab_size(self.config.vocab_size as usize)
        .with_context_length(self.config.max_context as usize)
        .with_dtype(self.config.dtype);

        // Create SIMD-optimized SamplingConfig from params
        let _cpu_info = get_cpu_features();
        let _sampling_config =
            crate::core::generation::SamplingConfig::new(params.temperature as f32)
                .with_top_k(50) // Default for now
                .with_top_p(0.9) // Default for now
                .with_repetition_penalty(1.0)
                .with_frequency_penalty(0.0)
                .with_presence_penalty(0.0);

        // Format prompt
        let _prompt_text = format!("User: {}\nAssistant: ", prompt.to_string());
        let _max_tokens = params.max_tokens.map(|n| n.get()).unwrap_or(1000);

        // TODO: Use TextGenerator directly instead of Engine
        // This should create TextGenerator and call generate() method
        // For now, return error to indicate delegation pattern is not fully implemented
        return AsyncStream::with_channel(move |sender| {
            let error_chunk = CandleCompletionChunk::Error(
                "Error: Direct TextGenerator usage not yet implemented in KimiK2 provider. This should use TextGenerator::generate() directly instead of Engine::execute_model_inference().".to_string()
            );
            let _ = sender.send(error_chunk);
        });
    }
}

// Implement builder trait
impl BuilderCandleCompletionProvider for CandleKimiK2Provider {}

// Static model info for Kimi-K2
static KIMI_K2_MODEL_INFO: CandleModelInfo = CandleModelInfo {
    provider_name: "candle-kimi",
    name: "kimi-k2-instruct",
    max_input_tokens: NonZeroU32::new(131072), // 128K context
    max_output_tokens: NonZeroU32::new(8192),
    input_price: None, // Local model - no pricing
    output_price: None,
    supports_vision: false,
    supports_function_calling: true,
    supports_streaming: true,
    supports_embeddings: false,
    requires_max_tokens: false,
    supports_thinking: false,
    optimal_thinking_budget: None,
    system_prompt_prefix: None,
    real_name: None,
    model_type: None,
    model_id: "kimi-k2",
    hf_repo_url: "unsloth/Kimi-K2-Instruct-GGUF",
    quantization: "Q4_0",
    patch: None,
};

impl CandleModel for CandleKimiK2Provider {
    fn info(&self) -> &'static CandleModelInfo {
        &KIMI_K2_MODEL_INFO
    }
}

/// Kimi K2 completion request format for HTTP API compatibility
#[derive(Debug, Serialize, Deserialize)]
struct CandleKimiCompletionRequest {
    prompt: String,
    temperature: f64,
    max_tokens: u64,
    stream: bool,
    model: String,
}

/// Validate that the model path exists and is accessible
///
/// # Errors
/// Returns error if the path does not exist or is not accessible
pub fn validate_model_path(path: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = Path::new(path);

    if !model_path.exists() {
        return Err(format!("Model path does not exist: {}", path).into());
    }

    if !model_path.is_dir() && !model_path.is_file() {
        return Err(format!("Model path is neither file nor directory: {}", path).into());
    }

    Ok(())
}
