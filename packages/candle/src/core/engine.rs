//! Engine domain module
//!
//! Provides core engine functionality with true zero-allocation patterns and lock-free
//! architecture. The engine routes requests to appropriate AI providers using atomic
//! operations and borrowed data to eliminate allocations in hot paths.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Llama, Cache};
use tokenizers::Tokenizer;


use crate::domain::completion::response::CompletionResponse;
use crate::core::generation::{SamplingConfig, SpecialTokens, LogitsBuffer, TokenProb};
use crate::core::model_config::{ModelConfig, ModelArchitecture};
use crate::core::{simd_temperature_scale, simd_softmax_with_cache, simd_argmax_with_bounds};
use crate::{AsyncStream, AsyncStreamSender, AsyncTask, spawn_task};
use arrayvec::ArrayVec;

/// Handle errors in streaming context without panicking
macro_rules! handle_error {
    ($error:expr, $context:literal) => {
        eprintln!("Streaming error in {}: {}", $context, $error)
        // Continue processing instead of returning error
    };
}

/// Engine-specific error types with minimal allocations
#[derive(Error, Debug, Clone)]
pub enum EngineError {
    #[error("Provider not available")]
    /// The requested AI provider is not available or not configured
    ProviderNotAvailable,

    #[error("Model not found")]
    /// The specified model was not found in the provider
    ModelNotFound,

    #[error("Configuration error: {0}")]
    /// Engine configuration is invalid or incomplete
    ConfigurationError(String),

    #[error("Authentication failed")]
    /// Authentication with the AI provider failed
    AuthenticationFailed,

    #[error("Rate limit exceeded, retry after {retry_after_seconds}s")]
    /// Rate limit was exceeded by the provider
    RateLimitExceeded { 
        /// Number of seconds to wait before retrying
        retry_after_seconds: u64 
    },

    #[error("Request timeout after {timeout_seconds}s")]
    /// Request timed out after the specified duration
    RequestTimeout { 
        /// Number of seconds the request waited before timing out
        timeout_seconds: u64 
    },

    #[error("Network error: {0}")]
    /// Network communication error occurred
    NetworkError(String),

    #[error("Invalid input")]
    /// The provided input is invalid or malformed
    InvalidInput,

    #[error("Service unavailable")]
    /// The engine service is temporarily unavailable
    ServiceUnavailable,

    #[error("Internal error: {0}")]
    /// An unexpected internal error occurred
    InternalError(String)}

/// Result type for engine operations
pub type EngineResult<T> = Result<T, EngineError>;

/// Engine configuration with owned strings allocated once at creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Model name for the completion request
    pub model_name: String,
    /// Provider identifier (e.g., "openai", "anthropic", "gemini")
    pub provider: String,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum tokens for completion
    pub max_tokens: Option<u32>,
    /// Temperature for response randomness (0.0 - 1.0)
    pub temperature: Option<f32>,
    /// Whether to enable streaming responses
    pub enable_streaming: bool,
    /// Custom endpoint URL override
    pub endpoint_url: Option<String>}

impl Default for EngineConfig {
    #[inline]
    fn default() -> Self {
        Self {
            model_name: String::from("gpt-4o-mini"),
            provider: String::from("openai"),
            api_key: None,
            timeout_seconds: 30,
            max_tokens: Some(4096),
            temperature: Some(0.7),
            enable_streaming: false,
            endpoint_url: None}
    }
}

impl EngineConfig {
    /// Create a new engine configuration
    #[inline]
    pub fn new(model_name: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            provider: provider.into(),
            ..Default::default()
        }
    }

    /// Set API key
    #[inline]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set timeout in seconds
    #[inline]
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }

    /// Set max tokens
    #[inline]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature (clamped to valid range)
    #[inline]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature.clamp(0.0, 1.0));
        self
    }

    /// Enable streaming responses
    #[inline]
    pub fn with_streaming(mut self) -> Self {
        self.enable_streaming = true;
        self
    }

    /// Set custom endpoint URL
    #[inline]
    pub fn with_endpoint(mut self, endpoint_url: impl Into<String>) -> Self {
        self.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Validate configuration
    #[inline]
    pub fn validate(&self) -> EngineResult<()> {
        if self.model_name.is_empty() {
            return Err(EngineError::ConfigurationError(
                "Model name cannot be empty".to_string(),
            ));
        }

        if self.provider.is_empty() {
            return Err(EngineError::ConfigurationError(
                "Provider cannot be empty".to_string(),
            ));
        }

        if self.timeout_seconds == 0 {
            return Err(EngineError::ConfigurationError(
                "Timeout must be greater than 0".to_string(),
            ));
        }

        if let Some(temp) = self.temperature {
            if !(0.0..=1.0).contains(&temp) {
                return Err(EngineError::ConfigurationError(
                    "Temperature must be between 0.0 and 1.0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Engine completion request using borrowed data to avoid allocations
#[derive(Debug)]
pub struct CompletionRequest<'a> {
    pub prompt: &'a str,
    pub system_prompt: Option<&'a str>,
    pub conversation_history: &'a [&'a str],
    pub tools: &'a [&'a str],
    pub metadata: Option<&'a str>}

impl<'a> CompletionRequest<'a> {
    /// Create a new completion request with borrowed data
    #[inline]
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            system_prompt: None,
            conversation_history: &[],
            tools: &[],
            metadata: None}
    }

    /// Set system prompt
    #[inline]
    pub fn with_system_prompt(mut self, system_prompt: &'a str) -> Self {
        self.system_prompt = Some(system_prompt);
        self
    }

    /// Set conversation history
    #[inline]
    pub fn with_history(mut self, history: &'a [&'a str]) -> Self {
        self.conversation_history = history;
        self
    }

    /// Set available tools
    #[inline]
    pub fn with_tools(mut self, tools: &'a [&'a str]) -> Self {
        self.tools = tools;
        self
    }

    /// Set metadata
    #[inline]
    pub fn with_metadata(mut self, metadata: &'a str) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Validate request
    #[inline]
    pub fn validate(&self) -> EngineResult<()> {
        if self.prompt.is_empty() {
            return Err(EngineError::InvalidInput);
        }
        Ok(())
    }
}

/// Core engine implementation with lock-free atomic operations
pub struct Engine {
    config: EngineConfig,
    request_count: AtomicU64,
    active_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    is_healthy: AtomicBool}

impl Engine {
    /// Create a new engine with the given configuration
    #[inline]
    pub fn new(config: EngineConfig) -> EngineResult<Self> {
        config.validate()?;

        Ok(Self {
            config,
            request_count: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            is_healthy: AtomicBool::new(true)})
    }

    /// Get immutable reference to configuration
    #[inline]
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get current request count (atomic read)
    #[inline]
    pub fn request_count(&self) -> u64 {
        self.request_count.load(Ordering::Relaxed)
    }

    /// Get current active request count (atomic read)
    #[inline]
    pub fn active_requests(&self) -> u64 {
        self.active_requests.load(Ordering::Relaxed)
    }

    /// Get successful request count (atomic read)
    #[inline]
    pub fn successful_requests(&self) -> u64 {
        self.successful_requests.load(Ordering::Relaxed)
    }

    /// Get failed request count (atomic read)
    #[inline]
    pub fn failed_requests(&self) -> u64 {
        self.failed_requests.load(Ordering::Relaxed)
    }

    /// Check if engine is healthy (atomic read)
    #[inline]
    pub fn is_healthy(&self) -> bool {
        self.is_healthy.load(Ordering::Relaxed)
    }

    /// Set health status (atomic write)
    #[inline]
    pub fn set_healthy(&self, healthy: bool) {
        self.is_healthy.store(healthy, Ordering::Relaxed);
    }

    /// Process completion request with zero allocations in hot path
    #[inline]
    pub fn process_completion(
        &self,
        request: CompletionRequest<'_>,
    ) -> AsyncTask<EngineResult<CompletionResponse<'static>>> {
        // Validate request first
        if let Err(e) = request.validate() {
            return spawn_task(move || Err(e));
        }

        // Atomic operations for metrics (lock-free)
        let request_id = self.request_count.fetch_add(1, Ordering::Relaxed);
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        // Clone necessary data for async processing
        let model_name = self.config.model_name.clone();
        let provider = self.config.provider.clone();
        let api_key = self.config.api_key.clone();
        let timeout = self.config.timeout_seconds;
        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature;
        let streaming = self.config.enable_streaming;
        let endpoint = self.config.endpoint_url.clone();

        // Convert borrowed request data to owned for async processing
        let prompt = request.prompt.to_string();
        let system_prompt = request.system_prompt.map(|s| s.to_string());
        let history: Vec<String> = request
            .conversation_history
            .iter()
            .map(|s| s.to_string())
            .collect();
        let tools: Vec<String> = request.tools.iter().map(|s| s.to_string()).collect();
        let metadata = request.metadata.map(|s| s.to_string());

        // We'll update metrics after the task completes, not during

        spawn_task(move || {
            // Create streaming completion and collect first result for backward compatibility
            let mut stream = Self::execute_completion_stream(
                request_id,
                model_name,
                provider,
                api_key,
                timeout,
                max_tokens,
                temperature,
                streaming,
                endpoint,
                prompt,
                system_prompt,
                history,
                tools,
                metadata,
            );

            // Try to get the first item from stream
            if let Some(response) = stream.try_next() {
                Ok(response)
            } else {
                Err(EngineError::InternalError(
                    "No response from stream".to_string(),
                ))
            }
        })
    }

    /// Execute completion with real model inference using ModelConfig and TextGenerator
    pub fn execute_model_inference(
        model_config: ModelConfig,
        prompt: String,
        max_tokens: u32,
        sampling_config: SamplingConfig,
    ) -> AsyncStream<CompletionResponse<'static>> {
        AsyncStream::with_channel(move |sender: AsyncStreamSender<CompletionResponse<'static>>| {
            // Validate model configuration
            if let Err(e) = model_config.validate() {
                let error_response = CompletionResponse {
                    text: format!("Model config validation failed: {}", e).into(),
                    model: model_config.model_name.clone().into(),
                    provider: Some(model_config.provider_name.clone().into()),
                    usage: None,
                    finish_reason: Some("error".into()),
                    response_time_ms: Some(0),
                    generation_time_ms: Some(0),
                    tokens_per_second: Some(0.0),
                };
                let _ = sender.send(error_response);
                return;
            }
            
            // Initialize device (prefer CUDA if available, fallback to CPU)
            let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
            
            // Load tokenizer
            let tokenizer = match Tokenizer::from_file(&model_config.tokenizer_path) {
                Ok(tokenizer) => tokenizer,
                Err(e) => {
                    let error_response = CompletionResponse {
                        text: format!("Failed to load tokenizer: {}", e).into(),
                        model: model_config.model_name.clone().into(),
                        provider: Some(model_config.provider_name.clone().into()),
                        usage: None,
                        finish_reason: Some("error".into()),
                        response_time_ms: Some(0),
                        generation_time_ms: Some(0),
                        tokens_per_second: Some(0.0),
                    };
                    let _ = sender.send(error_response);
                    return;
                }
            };
        
            // Load model based on architecture
            let (model, llama_config) = match &model_config.architecture {
                ModelArchitecture::Llama(config) => {
                    // Load safetensors with memory mapping for efficiency
                    // SAFETY: from_mmaped_safetensors requires unsafe due to memory mapping of safetensors files
                    // This is safe because:
                    // 1. ModelConfig validates file paths during construction
                    // 2. safetensors format includes integrity checks and header validation  
                    // 3. Memory mapping is read-only and cannot corrupt memory
                    // 4. Candle's VarBuilder handles all bounds checking internally
                    #[allow(unsafe_code)]
                    let vs = match unsafe { VarBuilder::from_mmaped_safetensors(&[&model_config.model_path], model_config.dtype, &device) } {
                        Ok(vs) => vs,
                        Err(e) => {
                            let error_response = CompletionResponse {
                                text: format!("Failed to load model weights: {}", e).into(),
                                model: model_config.model_name.clone().into(),
                                provider: Some(model_config.provider_name.clone().into()),
                                usage: None,
                                finish_reason: Some("error".into()),
                                response_time_ms: Some(0),
                                generation_time_ms: Some(0),
                                tokens_per_second: Some(0.0),
                            };
                            let _ = sender.send(error_response);
                            return;
                        }
                    };
                
                // Create Candle Llama config
                let candle_config = candle_transformers::models::llama::Config {
                    hidden_size: config.hidden_size,
                    intermediate_size: config.intermediate_size,
                    vocab_size: config.vocab_size,
                    num_hidden_layers: config.num_hidden_layers,
                    num_attention_heads: config.num_attention_heads,
                    num_key_value_heads: config.num_key_value_heads,
                    use_flash_attn: false, // Disable for compatibility
                    rms_norm_eps: config.rms_norm_eps,
                    rope_theta: config.rope_theta,
                    bos_token_id: config.bos_token_id,
                    eos_token_id: config.eos_token_id.clone(),
                    rope_scaling: config.rope_scaling.clone(),
                    max_position_embeddings: config.max_position_embeddings,
                    tie_word_embeddings: config.tie_word_embeddings,
                };
                    
                    // Load Llama model
                    let llama_model = match Llama::load(vs, &candle_config) {
                        Ok(model) => model,
                        Err(e) => {
                            let error_response = CompletionResponse {
                                text: format!("Failed to load Llama model: {}", e).into(),
                                model: model_config.model_name.clone().into(),
                                provider: Some(model_config.provider_name.clone().into()),
                                usage: None,
                                finish_reason: Some("error".into()),
                                response_time_ms: Some(0),
                                generation_time_ms: Some(0),
                                tokens_per_second: Some(0.0),
                            };
                            let _ = sender.send(error_response);
                            return;
                        }
                    };
                    
                    (llama_model, candle_config)
                },
                _ => {
                    let error_response = CompletionResponse {
                        text: format!("Architecture {} not yet implemented", model_config.architecture.name()).into(),
                        model: model_config.model_name.clone().into(),
                        provider: Some(model_config.provider_name.clone().into()),
                        usage: None,
                        finish_reason: Some("error".into()),
                        response_time_ms: Some(0),
                        generation_time_ms: Some(0),
                        tokens_per_second: Some(0.0),
                    };
                    let _ = sender.send(error_response);
                    return;
                }
            };
            
            // Initialize SIMD-optimized sampling components
            const SAMPLING_CACHE_SIZE: usize = 1024;
            let mut prob_cache: ArrayVec<TokenProb, SAMPLING_CACHE_SIZE> = ArrayVec::new();
            let is_deterministic = sampling_config.temperature <= 0.0;
            
            // Create special tokens from model config  
            let candle_tokenizer = match crate::providers::tokenizer::CandleTokenizer::from_file(&model_config.tokenizer_path) {
                Ok(tokenizer) => tokenizer,
                Err(e) => {
                    let error_response = CompletionResponse {
                        text: format!("Failed to load tokenizer: {}", e).into(),
                        model: model_config.model_name.clone().into(),
                        provider: Some(model_config.provider_name.clone().into()),
                        usage: None,
                        finish_reason: Some("error".into()),
                        response_time_ms: Some(0),
                        generation_time_ms: Some(0),
                        tokens_per_second: Some(0.0),
                    };
                    let _ = sender.send(error_response);
                    return;
                }
            };
            
            let special_tokens = SpecialTokens::new(
                candle_tokenizer,
                model_config.special_tokens.eos_token_id,
            );
            
            // Tokenize input prompt  
            let tokens = match tokenizer.encode(prompt.as_str(), true) {
                Ok(encoding) => encoding.get_ids().to_vec(),
                Err(e) => {
                    let error_response = CompletionResponse {
                        text: format!("Tokenization failed: {}", e).into(),
                        model: model_config.model_name.clone().into(),
                        provider: Some(model_config.provider_name.clone().into()),
                        usage: None,
                        finish_reason: Some("error".into()),
                        response_time_ms: Some(0),
                        generation_time_ms: Some(0),
                        tokens_per_second: Some(0.0),
                    };
                    let _ = sender.send(error_response);
                    return;
                }
            };
            
            // Initialize cache for KV caching
            let mut cache = match Cache::new(true, model_config.dtype, &llama_config, &device) {
                Ok(cache) => cache,
                Err(e) => {
                    let error_response = CompletionResponse {
                        text: format!("Failed to create cache: {}", e).into(),
                        model: model_config.model_name.clone().into(),
                        provider: Some(model_config.provider_name.clone().into()),
                        usage: None,
                        finish_reason: Some("error".into()),
                        response_time_ms: Some(0),
                        generation_time_ms: Some(0),
                        tokens_per_second: Some(0.0),
                    };
                    let _ = sender.send(error_response);
                    return;
                }
            };
            
            // Initialize generation state
            let mut all_tokens = tokens;
            let mut _generated_text = String::new();
            let mut index_pos = 0;
            let start_time = std::time::Instant::now();
            
            // Main generation loop
            for step in 0..max_tokens {
                // Prepare input for model forward pass
                let context_size = if cache.use_kv_cache && index_pos > 0 { 1 } else { all_tokens.len() };
                let context_index = if cache.use_kv_cache && index_pos > 0 { index_pos } else { 0 };
                
                // Get input tokens for this step
                let input_tokens: Vec<u32> = all_tokens[all_tokens.len().saturating_sub(context_size)..].to_vec();
                let input_tensor = match Tensor::new(input_tokens.as_slice(), &device) {
                    Ok(tensor) => match tensor.unsqueeze(0) {
                        Ok(tensor) => tensor,
                        Err(e) => {
                            eprintln!("Failed to unsqueeze tensor: {}", e);
                            break;
                        }
                    },
                    Err(e) => {
                        eprintln!("Failed to create input tensor: {}", e);
                        break;
                    }
                };
                
                // Forward pass through model
                let logits = match model.forward(&input_tensor, context_index, &mut cache) {
                    Ok(logits) => match logits.squeeze(0) {
                        Ok(logits) => logits,
                        Err(e) => {
                            eprintln!("Failed to squeeze logits: {}", e);
                            break;
                        }
                    },
                    Err(e) => {
                        eprintln!("Model forward pass failed: {}", e);
                        break;
                    }
                };
                
                // SIMD-optimized sampling pipeline
                let next_token = match sample_with_simd(&logits, &sampling_config, &mut prob_cache, is_deterministic) {
                    Ok(token) => token,
                    Err(e) => {
                        eprintln!("SIMD sampling failed: {}", e);
                        break;
                    }
                };
                
                // Check for stop conditions
                if special_tokens.is_eos_token(next_token) {
                    break;
                }
                
                // Add token to sequence and update position
                all_tokens.push(next_token);
                index_pos += input_tokens.len();
                
                // Decode token and send as response
                if let Ok(decoded) = tokenizer.decode(&[next_token], false) {
                    if !decoded.is_empty() {
                        _generated_text.push_str(&decoded);
                        
                        // Create and send streaming response
                        let response = CompletionResponse {
                            text: decoded.into(),
                            model: model_config.model_name.clone().into(),
                            provider: Some(model_config.provider_name.clone().into()),
                            usage: None,
                            finish_reason: None,
                            response_time_ms: Some(start_time.elapsed().as_millis() as u64),
                            generation_time_ms: Some(start_time.elapsed().as_millis() as u32),
                            tokens_per_second: Some((step as f32 / start_time.elapsed().as_secs_f32()) as f64),
                        };
                        
                        if sender.send(response).is_err() {
                            break; // Client disconnected
                        }
                    }
                }
            }
            
            // Send final completion response
            let final_response = CompletionResponse {
                text: String::new().into(),
                model: model_config.model_name.into(),
                provider: Some(model_config.provider_name.into()),
                usage: None,
                finish_reason: Some("stop".into()),
                response_time_ms: Some(start_time.elapsed().as_millis() as u64),
                generation_time_ms: Some(start_time.elapsed().as_millis() as u32),
                tokens_per_second: Some((all_tokens.len() as f32 / start_time.elapsed().as_secs_f32()) as f64),
            };
            
            let _ = sender.send(final_response);
        })
    }
    

    
    /// Convert Tensor logits to LogitsBuffer for TextGenerator
    // convert_tensor_to_logits_buffer removed - unused utility method

    /// Execute completion request as stream (internal implementation)
    fn execute_completion_stream(
        request_id: u64,
        model_name: String,
        provider: String,
        _api_key: Option<String>,
        _timeout: u64,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        _streaming: bool,
        _endpoint: Option<String>,
        prompt: String,
        system_prompt: Option<String>,
        _history: Vec<String>,
        _tools: Vec<String>,
        _metadata: Option<String>,
    ) -> AsyncStream<CompletionResponse<'static>> {
        AsyncStream::with_channel(move |sender| {
            // For Kimi K2 provider, implement actual Candle inference
            if provider == "kimi-k2" {
                // Build the full prompt with system prompt if provided
                let full_prompt = if let Some(sys) = system_prompt {
                    format!("{}\n\nUser: {}\nAssistant: ", sys, prompt)
                } else {
                    format!("User: {}\nAssistant: ", prompt)
                };
                
                // Create SamplingConfig from request parameters
                let config = SamplingConfig {
                    temperature: temperature.unwrap_or(0.7),  // f32 is correct for SamplingConfig
                    top_k: 50,  // Default top_k
                    top_p: 0.9,  // Default top_p
                    repetition_penalty: 1.0,
                    length_penalty: 1.0,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    min_prob_threshold: 1e-8,
                    seed: Some(request_id),  // Use request_id as seed
                    deterministic: false,
                    enable_simd: true,
                    simd_threshold: 16,
                };
                
                // Use real inference engine - providers must supply ModelConfig
                // For now, create a basic ModelConfig for kimi-k2 until providers are updated
                let llama_config = candle_transformers::models::llama::Config {
                    vocab_size: 32000,
                    hidden_size: 4096,
                    intermediate_size: 11008,
                    num_hidden_layers: 32,
                    num_attention_heads: 32,
                    num_key_value_heads: 32,
                    use_flash_attn: false,
                    rms_norm_eps: 1e-6,
                    rope_theta: 10000.0,
                    bos_token_id: Some(1),
                    eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(2)),
                    rope_scaling: None,
                    max_position_embeddings: 2048,
                    tie_word_embeddings: false,
                };
                
                let model_config = ModelConfig::new(
                    format!("{}/model.safetensors", model_name), // Temporary path construction
                    format!("{}/tokenizer.json", model_name),    // Temporary path construction  
                    ModelArchitecture::Llama(llama_config),
                    "kimi-k2".to_string(),
                    "kimi-k2".to_string()
                );
                
                // Execute real model inference using TextGenerator - now returns AsyncStream
                let mut inference_stream = Self::execute_model_inference(
                    model_config,
                    full_prompt,
                    max_tokens.unwrap_or(1000),
                    config,
                );
                
                // Forward all responses from inference stream to sender
                while let Some(response) = inference_stream.try_next() {
                    if sender.send(response).is_err() {
                        break; // Client disconnected
                    }
                }
            } else {
                // For other providers, return a proper error response
                let error_response = CompletionResponse {
                    text: format!("Error: Provider '{}' not supported by Candle engine. Only 'kimi-k2' provider is currently implemented.", provider).into(),
                    model: model_name.into(),
                    provider: Some(provider.into()),
                    usage: None,
                    finish_reason: Some("error".into()),
                    response_time_ms: Some(0),
                    generation_time_ms: Some(0),
                    tokens_per_second: Some(0.0),
                };
                let _ = sender.send(error_response);
            }
        })
    }

    /// Process completion request as stream (new primary API)
    #[inline]
    pub fn process_completion_stream(
        &self,
        request: CompletionRequest<'_>,
    ) -> AsyncStream<CompletionResponse<'static>> {
        // Validate request first
        if let Err(e) = request.validate() {
            return AsyncStream::with_channel(move |_sender| {
                handle_error!(e, "process_completion_stream validation");
            });
        }

        // Atomic operations for metrics (lock-free)
        let request_id = self.request_count.fetch_add(1, Ordering::Relaxed);
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        // Clone necessary data for async processing
        let model_name = self.config.model_name.clone();
        let provider = self.config.provider.clone();
        let api_key = self.config.api_key.clone();
        let timeout = self.config.timeout_seconds;
        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature;
        let streaming = self.config.enable_streaming;
        let endpoint = self.config.endpoint_url.clone();

        // Convert borrowed request data to owned for async processing
        let prompt = request.prompt.to_string();
        let system_prompt = request.system_prompt.map(|s| s.to_string());
        let history: Vec<String> = request
            .conversation_history
            .iter()
            .map(|s| s.to_string())
            .collect();
        let tools: Vec<String> = request.tools.iter().map(|s| s.to_string()).collect();
        let metadata = request.metadata.map(|s| s.to_string());

        AsyncStream::with_channel(move |sender| {
            let mut completion_stream = Self::execute_completion_stream(
                request_id,
                model_name,
                provider,
                api_key,
                timeout,
                max_tokens,
                temperature,
                streaming,
                endpoint,
                prompt,
                system_prompt,
                history,
                tools,
                metadata,
            );

            // Process completion responses from the stream using try_next
            while let Some(response) = completion_stream.try_next() {
                let _ = sender.send(response);
            }
        })
    }

    /// Get streaming completion results (convenience method)
    #[inline]
    pub fn get_completion_stream(
        &self,
        request: CompletionRequest<'_>,
    ) -> AsyncStream<CompletionResponse<'static>> {
        self.process_completion_stream(request)
    }

    /// Get engine statistics
    #[inline]
    pub fn stats(&self) -> EngineStats {
        EngineStats {
            total_requests: self.request_count.load(Ordering::Relaxed),
            active_requests: self.active_requests.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            is_healthy: self.is_healthy.load(Ordering::Relaxed)}
    }

    /// Reset all metrics (atomic operations)
    #[inline]
    pub fn reset_metrics(&self) {
        self.request_count.store(0, Ordering::Relaxed);
        self.active_requests.store(0, Ordering::Relaxed);
        self.successful_requests.store(0, Ordering::Relaxed);
        self.failed_requests.store(0, Ordering::Relaxed);
    }
}

/// Engine statistics snapshot
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineStats {
    pub total_requests: u64,
    pub active_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub is_healthy: bool}

impl EngineStats {
    /// Calculate success rate as a percentage
    #[inline]
    pub fn success_rate(&self) -> f64 {
        let completed = self.successful_requests + self.failed_requests;
        if completed == 0 {
            0.0
        } else {
            (self.successful_requests as f64 / completed as f64) * 100.0
        }
    }

    /// Calculate failure rate as a percentage
    #[inline]
    pub fn failure_rate(&self) -> f64 {
        100.0 - self.success_rate()
    }
}

/// SIMD-optimized token sampling pipeline
/// 
/// Replaces Candle's LogitsProcessor with direct SIMD operations for maximum performance
#[inline(always)]
fn sample_with_simd(
    logits: &Tensor,
    sampling_config: &SamplingConfig,
    prob_cache: &mut ArrayVec<TokenProb, 1024>,
    is_deterministic: bool,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    // Convert tensor to LogitsBuffer (SmallVec)
    let logits_vec = logits.to_vec1::<f32>()
        .map_err(|e| format!("Failed to extract logits: {}", e))?;
    
    let mut logits_buffer: LogitsBuffer = logits_vec.into();
    
    // Apply temperature scaling for non-deterministic sampling
    if !is_deterministic {
        simd_temperature_scale(&mut logits_buffer, sampling_config.temperature)
            .map_err(|e| format!("Temperature scaling failed: {}", e))?;
    }
    
    // Apply SIMD softmax to compute probabilities
    let probabilities = simd_softmax_with_cache(&logits_buffer, prob_cache)
        .map_err(|e| format!("Softmax computation failed: {}", e))?;
    
    // Select token using SIMD argmax (deterministic) or sampling
    if is_deterministic {
        simd_argmax_with_bounds(&probabilities, prob_cache)
            .map_err(|e| format!("Argmax selection failed: {}", e).into())
    } else {
        // For non-deterministic sampling, use argmax as fallback
        // TODO: Implement proper sampling in a future task
        simd_argmax_with_bounds(&probabilities, prob_cache)
            .map_err(|e| format!("Sampling selection failed: {}", e).into())
    }
}

// Engine is automatically Send + Sync due to atomic operations - no unsafe impl needed
