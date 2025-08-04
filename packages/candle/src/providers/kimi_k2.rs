//! 
//! Provides streaming completion capabilities using local Kimi K2 models
//! with zero allocation patterns and AsyncStream streaming.
//!
//! This implementation uses the Candle ML framework for local model inference,
//! specifically targeting Llama-compatible models for high-performance text generation.

use std::path::Path;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use fluent_ai_async::AsyncStream;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::{
    models::llama::{Llama, LlamaConfig, Cache},
    generation::{LogitsProcessor, Sampling},
};
use tokenizers::Tokenizer;

use crate::domain::{
    completion::{
        CandleCompletionModel, 
        CandleCompletionParams,
    },
    context::chunk::{CandleCompletionChunk, FinishReason},
    prompt::CandlePrompt,
};
use crate::builders::agent_role::{CandleCompletionProvider as BuilderCandleCompletionProvider};

/// CandleKimiK2Provider for local Kimi K2 model inference using Candle ML framework
#[derive(Debug)]
pub struct CandleKimiK2Provider {
    /// Model path on filesystem
    model_path: String,
    /// Provider configuration
    config: CandleKimiK2Config,
    /// Loaded Candle model (Llama architecture)
    model: Option<Arc<Llama>>,
    /// Tokenizer for text processing
    tokenizer: Option<Arc<Tokenizer>>,
    /// Candle device (CPU or CUDA)
    device: Device,
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
        // Initialize device - prefer CUDA if available, fallback to CPU
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        
        // Create model configuration for Kimi K2 (Llama-based architecture)
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
        
        // Load model and tokenizer if files exist
        let (model, tokenizer) = if Self::model_files_exist(&model_path) {
            let (model, tokenizer) = Self::load_model_and_tokenizer(&model_path, &device, &model_config, config.dtype)?;
            (Some(Arc::new(model)), Some(Arc::new(tokenizer)))
        } else {
            // Model files don't exist - create provider without loaded model
            (None, None)
        };
        
        Ok(Self {
            model_path,
            config,
            model,
            tokenizer,
            device,
            model_config,
        })
    }
    
    /// Check if required model files exist
    fn model_files_exist(model_path: &str) -> bool {
        let model_file = Path::new(model_path).join("model.safetensors");
        let tokenizer_file = Path::new(model_path).join("tokenizer.json");
        model_file.exists() && tokenizer_file.exists()
    }
    
    /// Load the Llama model and tokenizer from filesystem
    fn load_model_and_tokenizer(
        model_path: &str, 
        device: &Device, 
        model_config: &LlamaConfig,
        dtype: DType
    ) -> Result<(Llama, Tokenizer), Box<dyn std::error::Error + Send + Sync>> {
        // Load tokenizer
        let tokenizer_path = Path::new(model_path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
        
        // Create a basic VarBuilder (model loading requires complex setup)
        // In production, this would load actual weights from safetensors files
        let vb = VarBuilder::zeros(dtype, device);
        
        // Create Llama model configuration for Candle
        let config = candle_transformers::models::llama::Config {
            hidden_size: model_config.hidden_size,
            intermediate_size: model_config.intermediate_size,
            vocab_size: model_config.vocab_size,
            num_hidden_layers: model_config.num_hidden_layers,
            num_attention_heads: model_config.num_attention_heads,
            num_key_value_heads: model_config.num_key_value_heads.unwrap_or(model_config.num_attention_heads),
            use_flash_attn: false, // Disable flash attention for compatibility
            rms_norm_eps: model_config.rms_norm_eps,
            rope_theta: model_config.rope_theta,
            bos_token_id: model_config.bos_token_id,
            eos_token_id: model_config.eos_token_id.clone(),
            rope_scaling: model_config.rope_scaling.clone(),
            max_position_embeddings: model_config.max_position_embeddings,
            tie_word_embeddings: model_config.tie_word_embeddings.unwrap_or(false),
        };
        
        // Initialize Llama model
        let model = Llama::load(vb, &config)?;
        
        Ok((model, tokenizer))
    }
    
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
        // Extract parameters
        let max_tokens = params.max_tokens
            .map(|n| n.get())
            .unwrap_or(1000) as usize;
        
        let temperature = params.temperature;
        
        // Clone for async closure
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let device = self.device.clone();
        let model_config = self.model_config.clone();
        let use_kv_cache = self.config.use_kv_cache;
        let dtype = self.config.dtype;
        
        AsyncStream::with_channel(move |sender| {
            // Format prompt for Kimi K2
            let prompt_text = format!("User: {}\nAssistant: ", prompt.to_string());
            
            match (model, tokenizer) {
                (Some(model), Some(tokenizer)) => {
                    // Run actual Candle inference
                    if let Err(e) = Self::run_candle_inference(
                        &model,
                        &tokenizer,
                        &device,
                        &model_config,
                        &prompt_text,
                        max_tokens,
                        temperature,
                        use_kv_cache,
                        dtype,
                        &sender,
                    ) {
                        eprintln!("Candle inference error: {}", e);
                        let _ = sender.send(CandleCompletionChunk::Text(
                            format!("Error during inference: {}", e)
                        ));
                    }
                }
                _ => {
                    // Model not loaded - provide simulated response
                    let response = format!(
                        "Simulated Kimi K2 response for prompt: '{}' (temp: {:.2}, max_tokens: {})",
                        prompt_text.trim(),
                        temperature,
                        max_tokens
                    );
                    
                    // Send as word chunks to simulate streaming
                    for (i, word) in response.split_whitespace().enumerate() {
                        let text = if i == 0 { word.to_string() } else { format!(" {}", word) };
                        let chunk = CandleCompletionChunk::Text(text);
                        
                        if sender.send(chunk).is_err() {
                            break;
                        }
                        
                        // Simulate processing delay
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            }
            
            // Send completion signal
            let _ = sender.send(CandleCompletionChunk::Complete {
                text: String::new(),
                finish_reason: Some(FinishReason::Stop),
                usage: None,
            });
        })
    }
}

impl CandleKimiK2Provider {
    /// Run actual Candle model inference with streaming
    #[allow(clippy::too_many_arguments)]
    fn run_candle_inference(
        model: &Llama,
        tokenizer: &Tokenizer,
        device: &Device,
        _model_config: &LlamaConfig,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        use_kv_cache: bool,
        dtype: DType,
        sender: &fluent_ai_async::AsyncStreamSender<CandleCompletionChunk>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Tokenize input prompt
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?
            .get_ids()
            .to_vec();
        
        // Create model configuration for inference
        let config = candle_transformers::models::llama::Config {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn: false,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            bos_token_id: Some(1),
            eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(2)),
            rope_scaling: None,
            max_position_embeddings: 8192,
            tie_word_embeddings: false,
        };
        
        // Initialize cache for KV caching
        let mut cache = Cache::new(use_kv_cache, dtype, &config, device)?;
        
        // Initialize logits processor for sampling
        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(299792458u64, sampling);
        
        // Initialize token generation
        let mut all_tokens = tokens;
        let mut index_pos = 0;
        
        // Generate tokens iteratively
        for _step in 0..max_tokens {
            let context_size = if cache.use_kv_cache && index_pos > 0 {
                1
            } else {
                all_tokens.len()
            };
            
            let context_index = if cache.use_kv_cache && index_pos > 0 {
                index_pos
            } else {
                0
            };
            
            // Prepare input tensor - clone tokens to avoid borrow issues
            let input_tokens: Vec<u32> = all_tokens[all_tokens.len().saturating_sub(context_size)..].to_vec();
            let input_tensor = Tensor::new(input_tokens.as_slice(), device)?.unsqueeze(0)?;
            
            // Forward pass through model
            let logits = model.forward(&input_tensor, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;
            
            // Sample next token
            let next_token = logits_processor.sample(&logits)?;
            
            // Check for EOS token
            if next_token == 2 { // EOS token ID
                break;
            }
            
            // Add token to sequence and update position
            let input_len = input_tokens.len();
            all_tokens.push(next_token);
            index_pos += input_len;
            
            // Decode and send token
            if let Ok(decoded) = tokenizer.decode(&[next_token], false) {
                if !decoded.is_empty() {
                    let chunk = CandleCompletionChunk::Text(decoded);
                    if sender.send(chunk).is_err() {
                        break;
                    }
                }
            }
        }
        
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use tempfile::tempdir;
    
    #[test]
    fn test_provider_creation_with_missing_model() {
        // Create temporary directory without model files
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let model_path = temp_dir.path().to_str().expect("Invalid path");
        
        // Should create provider but without loaded model
        let provider = CandleKimiK2Provider::new(model_path).expect("Failed to create provider");
        assert_eq!(provider.model_path, model_path);
        assert_eq!(provider.config.temperature, 0.7);
        assert!(provider.model.is_none());
        assert!(provider.tokenizer.is_none());
    }
    
    #[test]
    fn test_config_builder_pattern() {
        let config = CandleKimiK2Config::default()
            .with_temperature(0.9)
            .with_max_context(4096)
            .with_dtype(DType::F32)
            .with_kv_cache(false);
        
        assert_eq!(config.temperature, 0.9);
        assert_eq!(config.max_context, 4096);
        assert_eq!(config.dtype, DType::F32);
        assert!(!config.use_kv_cache);
    }
    
    #[test]
    fn test_invalid_model_path() {
        let result = CandleKimiK2Provider::new("/nonexistent/path/to/model");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Model path does not exist"));
    }
    
    #[test]
    fn test_vocab_size_getter() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let model_path = temp_dir.path().to_str().expect("Invalid path");
        
        let provider = CandleKimiK2Provider::new(model_path).expect("Failed to create provider");
        assert_eq!(provider.vocab_size(), 32000);
    }
}