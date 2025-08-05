//! 
//! Provides streaming completion capabilities using local Kimi K2 models
//! with zero allocation patterns and AsyncStream streaming.
//!
//! This implementation uses the Candle ML framework for local model inference,
//! specifically targeting Llama-compatible models for high-performance text generation.

use std::path::Path;
use serde::{Deserialize, Serialize};
use fluent_ai_async::AsyncStream;
use candle_core::DType;
use candle_transformers::{
    models::llama::LlamaConfig,
};

use crate::domain::{
    completion::{
        CandleCompletionModel, 
        CandleCompletionParams,
    },
    context::chunk::{CandleCompletionChunk, FinishReason},
    prompt::CandlePrompt,
};
use crate::builders::agent_role::{CandleCompletionProvider as BuilderCandleCompletionProvider};

// SIMD optimizations for high-performance inference
use fluent_ai_simd::get_cpu_features;
use crate::core::generation::simd_metrics;

/// CandleKimiK2Provider for local Kimi K2 model inference using Candle ML framework
#[derive(Debug)]
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
    /// Create new Kimi K2 provider with model loading
    ///
    /// # Arguments
    /// * `model_path` - Path to directory containing model files (model.safetensors, tokenizer.json)
    ///
    /// # Example
    /// ```rust
    /// let provider = CandleKimiK2Provider::new("./models/kimi-k2")?;
    /// ```
    ///
    /// # Errors
    /// Returns error if model path doesn't exist or model loading fails
    pub fn new(model_path: impl Into<String>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path_str = model_path.into();
        validate_model_path(&path_str)?;
        
        let config = CandleKimiK2Config::default();
        Self::with_config(path_str, config)
    }
    
    /// Create provider with custom configuration
    pub fn with_config(
        model_path: String, 
        config: CandleKimiK2Config
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Log SIMD capabilities for performance debugging
        let cpu_info = get_cpu_features();
        log::info!("KimiK2 Provider initialized with SIMD support: {} (vector width: {} elements)", 
                   cpu_info.has_simd(), cpu_info.vector_width());
        
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
}

impl CandleCompletionModel for CandleKimiK2Provider {
    fn prompt(&self, prompt: CandlePrompt, params: &CandleCompletionParams) -> AsyncStream<CandleCompletionChunk> {
        // Create ModelConfig for this provider (thin wrapper - only config!)
        // Convert LlamaConfig to the format expected by ModelArchitecture
        let candle_config = candle_transformers::models::llama::Config {
            hidden_size: self.model_config.hidden_size,
            intermediate_size: self.model_config.intermediate_size,
            vocab_size: self.model_config.vocab_size,
            num_hidden_layers: self.model_config.num_hidden_layers,
            num_attention_heads: self.model_config.num_attention_heads,
            num_key_value_heads: self.model_config.num_key_value_heads.unwrap_or(self.model_config.num_attention_heads),
            use_flash_attn: false,
            rms_norm_eps: self.model_config.rms_norm_eps,
            rope_theta: self.model_config.rope_theta,
            bos_token_id: self.model_config.bos_token_id,
            eos_token_id: self.model_config.eos_token_id.clone(),
            rope_scaling: self.model_config.rope_scaling.clone(),
            max_position_embeddings: self.model_config.max_position_embeddings,
            tie_word_embeddings: self.model_config.tie_word_embeddings.unwrap_or(false),
        };
        
        let model_config = crate::core::ModelConfig::new(
            &self.model_path,
            format!("{}/tokenizer.json", self.model_path),
            crate::core::ModelArchitecture::Llama(candle_config),
            "kimi-k2",
            "kimi-k2"
        )
        .with_vocab_size(self.config.vocab_size as usize)
        .with_context_length(self.config.max_context as usize)
        .with_dtype(self.config.dtype);
        
        // Create SIMD-optimized SamplingConfig from params
        let cpu_info = get_cpu_features();
        let sampling_config = crate::core::SamplingConfig {
            temperature: params.temperature as f32,
            top_k: 50, // Default for now
            top_p: 0.9, // Default for now
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            min_prob_threshold: 1e-8,
            seed: None,
            deterministic: false,
            // SIMD optimization based on detected CPU capabilities
            enable_simd: cpu_info.has_simd(),
            simd_threshold: cpu_info.vector_width() * 2, // Optimize threshold for detected SIMD width
        };
        
        // Format prompt
        let prompt_text = format!("User: {}\nAssistant: ", prompt.to_string());
        let max_tokens = params.max_tokens.map(|n| n.get()).unwrap_or(1000);
        
        // Delegate ALL inference to core engine - this is now a thin wrapper!
        let mut engine_stream = crate::core::Engine::execute_model_inference(
            model_config,
            prompt_text,
            max_tokens as u32,
            sampling_config,
        );
        
        // Convert CompletionResponse to CandleCompletionChunk with SIMD performance monitoring
        AsyncStream::with_channel(move |sender| {
            // Reset SIMD metrics at start of inference for accurate measurement
            simd_metrics::reset_simd_metrics();
            
            while let Some(response) = engine_stream.try_next() {
                // Convert engine response to candle chunk
                let chunk = if response.finish_reason.is_some() {
                    // Log SIMD performance metrics on completion
                    if log::log_enabled!(log::Level::Debug) {
                        let report = simd_metrics::get_utilization_report();
                        log::debug!("SIMD Performance Report for KimiK2 inference:\n{}", report);
                    }
                    
                    CandleCompletionChunk::Complete {
                        text: response.text.to_string(),
                        finish_reason: Some(FinishReason::Stop),
                        usage: None,
                    }
                } else {
                    CandleCompletionChunk::Text(response.text.to_string())
                };
                
                if sender.send(chunk).is_err() {
                    break; // Client disconnected
                }
            }
        })
    }
}



// Implement builder trait
impl BuilderCandleCompletionProvider for CandleKimiK2Provider {}

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

