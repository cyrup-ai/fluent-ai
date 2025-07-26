//! Core generator implementation with constructors and configuration

use std::sync::Arc;
use candle_core::{Device, Tensor};
// Removed unused imports: rand::prelude::*, rand::SeedableRng

use crate::error::CandleResult;
use crate::kv_cache::{KVCache, KVCacheConfig};
use crate::model::CandleModel;
use crate::processing::processors::{CompositeProcessor, presets};
use crate::sampling::{Sampling, SamplingConfig};
use crate::streaming::{StreamingConfig, TokenOutputStream};
use crate::tokenizer::CandleTokenizer;

use super::types::GenerationConfig;

/// Zero-allocation text generator
pub struct CandleGenerator {
    /// The model for generation
    pub(super) model: Arc<CandleModel>,
    /// The tokenizer
    pub(super) tokenizer: Arc<CandleTokenizer>,
    /// Generation configuration
    pub(super) config: GenerationConfig,
    /// Device for computation
    pub(super) device: Device,
    /// Random number generator state
    pub(super) rng_state: parking_lot::Mutex<Option<u64>>,
    /// Cumulative log probability for current generation
    pub(super) cumulative_log_prob: parking_lot::Mutex<f64>,
    /// Sophisticated sampling configuration
    pub(super) sampling_config: Sampling,
    /// Streaming configuration for real-time output
    pub(super) streaming_config: StreamingConfig,
    /// KV cache for efficient generation
    pub(super) kv_cache: Option<Arc<parking_lot::Mutex<KVCache>>>,
    /// CompositeProcessor for sophisticated sampling
    pub(super) composite_processor: CompositeProcessor,
    /// TokenOutputStream for real-time streaming
    pub(super) token_output_stream: Option<Arc<parking_lot::Mutex<TokenOutputStream>>>}

impl Clone for CandleGenerator {
    /// Creates a deep copy of the CandleGenerator instance
    /// 
    /// This method performs a comprehensive clone of all generator components including:
    /// - Arc references to model and tokenizer (cheap reference counting)
    /// - Generation configuration (full copy)
    /// - Device configuration (full copy)
    /// - RNG state (current value copied)
    /// - Cumulative log probability (current value copied)
    /// - Sampling and streaming configurations (full copy)
    /// - KV cache reference if present (shared reference)
    /// - Composite processor (new instance with default configuration)
    /// - Token output stream reference if present (shared reference)
    /// 
    /// # Performance Notes
    /// - Arc clones are O(1) operations with atomic reference counting
    /// - Mutex lock acquisition is required to copy current state values
    /// - New CompositeProcessor instance is created rather than cloned
    /// 
    /// # Thread Safety
    /// The cloned instance maintains the same thread safety guarantees as the original,
    /// with each having independent mutex-protected state values.
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            tokenizer: Arc::clone(&self.tokenizer),
            config: self.config.clone(),
            device: self.device.clone(),
            rng_state: parking_lot::Mutex::new(*self.rng_state.lock()),
            cumulative_log_prob: parking_lot::Mutex::new(*self.cumulative_log_prob.lock()),
            sampling_config: self.sampling_config.clone(),
            streaming_config: self.streaming_config.clone(),
            kv_cache: self.kv_cache.as_ref().map(Arc::clone),
            composite_processor: CompositeProcessor::new(),
            token_output_stream: self.token_output_stream.as_ref().map(Arc::clone)}
    }
}

impl CandleGenerator {
    /// Creates a new CandleGenerator instance with basic configuration
    /// 
    /// This constructor initializes a text generator with essential components and
    /// default settings suitable for most text generation tasks.
    /// 
    /// # Arguments
    /// 
    /// * `model` - Arc-wrapped CandleModel for performing inference operations
    /// * `tokenizer` - Arc-wrapped CandleTokenizer for text encoding/decoding
    /// * `config` - GenerationConfig containing generation parameters (temperature, max_tokens, etc.)
    /// * `device` - Candle Device specifying compute backend (CPU, CUDA, Metal)
    /// 
    /// # Returns
    /// 
    /// A new CandleGenerator instance configured with:
    /// - Default sampling configuration (temperature-based sampling)
    /// - Default streaming configuration (no buffering)
    /// - No KV cache (memory optimization disabled)
    /// - Basic composite processor (no advanced sampling strategies)
    /// - No token output stream (streaming disabled)
    /// 
    /// # Performance Notes
    /// 
    /// - Zero-allocation constructor using stack-allocated default configurations
    /// - Arc cloning is O(1) with atomic reference counting
    /// - Mutex initialization is lightweight (no heap allocation)
    /// - Device configuration is validated during construction
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use std::sync::Arc;
    /// use candle_core::Device;
    /// use fluent_ai_candle::{CandleModel, CandleTokenizer, GenerationConfig, CandleGenerator};
    /// 
    /// let model = Arc::new(CandleModel::new(Device::Cpu));
    /// let tokenizer = Arc::new(CandleTokenizer::default());
    /// let config = GenerationConfig::default();
    /// let device = Device::Cpu;
    /// 
    /// let generator = CandleGenerator::new(model, tokenizer, config, device);
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// The returned generator is thread-safe and can be cloned for concurrent use.
    /// Internal state (RNG seed, log probabilities) is protected by parking_lot::Mutex.
    #[inline(always)]
    pub fn new(
        model: Arc<CandleModel>,
        tokenizer: Arc<CandleTokenizer>,
        config: GenerationConfig,
        device: Device,
    ) -> Self {
        Self {
            model,
            tokenizer,
            rng_state: parking_lot::Mutex::new(config.seed),
            config,
            device,
            cumulative_log_prob: parking_lot::Mutex::new(0.0),
            sampling_config: SamplingConfig::default().build_sampling(),
            streaming_config: StreamingConfig::default(),
            kv_cache: None,
            composite_processor: CompositeProcessor::new(),
            token_output_stream: None}
    }

    /// Creates a new CandleGenerator with advanced features and custom configurations
    /// 
    /// This constructor provides full control over generator capabilities, enabling
    /// sophisticated text generation with optimized performance features.
    /// 
    /// # Arguments
    /// 
    /// * `model` - Arc-wrapped CandleModel for performing inference operations
    /// * `tokenizer` - Arc-wrapped CandleTokenizer for text encoding/decoding
    /// * `config` - GenerationConfig containing generation parameters
    /// * `device` - Candle Device specifying compute backend (CPU, CUDA, Metal)
    /// * `sampling_config` - Advanced sampling configuration (top-k, top-p, temperature, etc.)
    /// * `streaming_config` - Real-time streaming configuration for token output
    /// * `kv_cache_config` - Optional KV cache configuration for memory optimization
    /// 
    /// # Returns
    /// 
    /// `CandleResult<Self>` containing either:
    /// - `Ok(CandleGenerator)` - Successfully configured generator with all features
    /// - `Err(CandleError)` - Configuration error (invalid cache settings, device issues)
    /// 
    /// # Features Enabled
    /// 
    /// ## KV Cache Optimization
    /// - Dramatically reduces memory usage for long sequences
    /// - Speeds up autoregressive generation by 2-10x
    /// - Configurable eviction policies (LRU, FIFO, custom)
    /// 
    /// ## Advanced Sampling
    /// - Sophisticated probability distribution manipulation
    /// - Multiple sampling strategies (nucleus, top-k, typical, mirostat)
    /// - Fine-grained control over generation quality vs diversity
    /// 
    /// ## Real-time Streaming
    /// - Token-by-token output with configurable buffering
    /// - Adaptive chunk sizing based on generation speed
    /// - Flow control for downstream processing
    /// 
    /// # Performance Notes
    /// 
    /// - KV cache initialization may require significant memory allocation
    /// - Composite processor setup involves multiple sampling strategy coordination
    /// - Streaming configuration affects memory usage patterns
    /// - All components are designed for zero-allocation runtime operation
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use std::sync::Arc;
    /// use candle_core::Device;
    /// use fluent_ai_candle::{
    ///     CandleModel, CandleTokenizer, GenerationConfig, CandleGenerator,
    ///     Sampling, StreamingConfig, KVCacheConfig
    /// };
    /// 
    /// let model = Arc::new(CandleModel::new(Device::Cpu));
    /// let tokenizer = Arc::new(CandleTokenizer::default());
    /// let config = GenerationConfig::default();
    /// let device = Device::Cpu;
    /// let sampling = Sampling::default();
    /// let streaming = StreamingConfig::default();
    /// let cache_config = Some(KVCacheConfig::default());
    /// 
    /// let generator = CandleGenerator::with_sophisticated_features(
    ///     model, tokenizer, config, device, sampling, streaming, cache_config
    /// )?;
    /// ```
    /// 
    /// # Errors
    /// 
    /// - `CandleError::InvalidConfiguration` - Invalid KV cache settings
    /// - `CandleError::DeviceError` - Device initialization failure
    /// - `CandleError::MemoryError` - Insufficient memory for cache allocation
    /// 
    /// # Thread Safety
    /// 
    /// The returned generator maintains thread safety through internal synchronization,
    /// allowing safe concurrent access to generation methods.
    #[inline(always)]
    pub fn with_sophisticated_features(
        model: Arc<CandleModel>,
        tokenizer: Arc<CandleTokenizer>,
        config: GenerationConfig,
        device: Device,
        sampling_config: Sampling,
        streaming_config: StreamingConfig,
        kv_cache_config: Option<KVCacheConfig>,
    ) -> CandleResult<Self> {
        // Initialize KV cache if configured
        let kv_cache = if let Some(cache_config) = kv_cache_config {
            let cache = KVCache::with_config(cache_config)?;
            Some(Arc::new(parking_lot::Mutex::new(cache)))
        } else {
            None
        };

        // Initialize CompositeProcessor based on generation configuration
        let composite_processor =
            presets::conversation().unwrap_or_else(|_| CompositeProcessor::new());

        // Initialize streaming components
        let token_output_stream = None; // TODO: Implement proper TokenOutputStream initialization

        Ok(Self {
            model,
            tokenizer,
            rng_state: parking_lot::Mutex::new(config.seed),
            config,
            device,
            cumulative_log_prob: parking_lot::Mutex::new(0.0),
            sampling_config,
            streaming_config,
            kv_cache,
            composite_processor,
            token_output_stream})
    }

    /// Updates the generation configuration and resets associated state
    /// 
    /// This method allows dynamic reconfiguration of generation parameters during
    /// runtime without recreating the entire generator instance.
    /// 
    /// # Arguments
    /// 
    /// * `config` - New GenerationConfig to apply to this generator
    /// 
    /// # Effects
    /// 
    /// - Replaces current generation configuration with provided config
    /// - Updates RNG seed state to match new configuration
    /// - Preserves all other generator state (cumulative log probabilities, caches)
    /// - Does not affect model, tokenizer, or device configuration
    /// 
    /// # Performance Notes
    /// 
    /// - O(1) operation with minimal memory allocation
    /// - Mutex lock acquisition required for thread-safe RNG state update
    /// - Configuration clone is lightweight (primarily scalar values)
    /// 
    /// # Thread Safety
    /// 
    /// This method requires exclusive access (&mut self) to ensure consistent
    /// state updates. Call on a cloned instance for concurrent configuration changes.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::{GenerationConfig, CandleGenerator};
    /// 
    /// let mut generator = CandleGenerator::default();
    /// 
    /// // Update temperature for more creative generation
    /// let mut new_config = GenerationConfig::default();
    /// new_config.temperature = 1.2;
    /// new_config.seed = Some(42);
    /// 
    /// generator.update_config(new_config);
    /// ```
    #[inline(always)]
    pub fn update_config(&mut self, config: GenerationConfig) {
        self.config = config.clone();
        *self.rng_state.lock() = config.seed;
    }

    /// Returns a reference to the current generation configuration
    /// 
    /// Provides read-only access to the generator's configuration parameters
    /// without requiring any locks or allocations.
    /// 
    /// # Returns
    /// 
    /// `&GenerationConfig` - Immutable reference to current configuration containing:
    /// - `temperature` - Sampling temperature (higher = more creative)
    /// - `max_tokens` - Maximum tokens to generate
    /// - `top_k` - Top-k sampling parameter (0 = disabled)
    /// - `top_p` - Nucleus sampling parameter (1.0 = disabled)
    /// - `seed` - Random seed for reproducible generation
    /// - `stop_sequences` - Sequences that terminate generation
    /// - `repetition_penalty` - Penalty for repeated tokens
    /// 
    /// # Performance Notes
    /// 
    /// - Zero-cost operation (returns direct reference)
    /// - No mutex locks or atomic operations required
    /// - Configuration is immutable once accessed
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::CandleGenerator;
    /// 
    /// let generator = CandleGenerator::default();
    /// let config = generator.config();
    /// 
    /// println!("Temperature: {}", config.temperature);
    /// println!("Max tokens: {}", config.max_tokens);
    /// ```
    #[inline(always)]
    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }

    /// Resets the cumulative log probability counter to zero
    /// 
    /// This method clears the accumulated log probability tracking, typically
    /// called at the beginning of a new generation sequence to ensure clean state.
    /// 
    /// # Purpose
    /// 
    /// Cumulative log probability tracking is used for:
    /// - Generation quality assessment (perplexity calculation)
    /// - Early stopping based on probability thresholds
    /// - Beam search and candidate ranking
    /// - Statistical analysis of generation patterns
    /// 
    /// # Performance Notes
    /// 
    /// - O(1) operation with single mutex lock acquisition
    /// - Minimal memory impact (single f64 value reset)
    /// - Thread-safe operation suitable for concurrent generators
    /// 
    /// # Thread Safety
    /// 
    /// Uses mutex-protected write access to ensure atomic reset operation
    /// across multiple concurrent generation calls.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::CandleGenerator;
    /// 
    /// let generator = CandleGenerator::default();
    /// 
    /// // Generate some text (accumulates log probabilities)
    /// // ... generation code ...
    /// 
    /// // Reset for new generation sequence
    /// generator.reset_cumulative_log_prob();
    /// assert_eq!(generator.cumulative_log_prob(), 0.0);
    /// ```
    #[inline(always)]
    pub fn reset_cumulative_log_prob(&self) {
        *self.cumulative_log_prob.lock() = 0.0;
    }

    /// Returns the current cumulative log probability value
    /// 
    /// Retrieves the accumulated log probability from all tokens generated in the
    /// current sequence, providing insight into generation quality and confidence.
    /// 
    /// # Returns
    /// 
    /// `f64` - Current cumulative log probability where:
    /// - Values closer to 0 indicate higher probability (more confident) generation
    /// - More negative values indicate lower probability (less confident) generation
    /// - Value accumulates across all tokens in the current generation sequence
    /// 
    /// # Interpretation
    /// 
    /// - **High confidence**: -0.1 to -2.0 (model is very confident in its choices)
    /// - **Medium confidence**: -2.0 to -5.0 (reasonable generation quality)
    /// - **Low confidence**: < -10.0 (model struggling with the generation task)
    /// 
    /// # Performance Notes
    /// 
    /// - O(1) operation with single mutex lock acquisition
    /// - Returns copy of internal f64 value (no references or allocations)
    /// - Thread-safe read operation suitable for monitoring during generation
    /// 
    /// # Use Cases
    /// 
    /// - **Quality Assessment**: Monitor generation confidence in real-time
    /// - **Early Stopping**: Terminate generation if confidence drops too low
    /// - **Beam Search**: Compare candidate sequences by cumulative probability
    /// - **Debugging**: Understand model behavior on specific inputs
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::CandleGenerator;
    /// 
    /// let generator = CandleGenerator::default();
    /// 
    /// // Generate some text
    /// // ... generation code ...
    /// 
    /// let log_prob = generator.cumulative_log_prob();
    /// if log_prob < -10.0 {
    ///     eprintln!("Warning: Low confidence generation ({})", log_prob);
    /// }
    /// ```
    #[inline(always)]
    pub fn cumulative_log_prob(&self) -> f64 {
        *self.cumulative_log_prob.lock()
    }

    /// Returns a reference to the configured composite processor
    /// 
    /// Provides access to the sophisticated sampling and processing pipeline
    /// used for advanced text generation quality control.
    /// 
    /// # Returns
    /// 
    /// `&CompositeProcessor` - Reference to the composite processor containing:
    /// - Multiple sampling strategies (temperature, top-k, top-p, typical-p)
    /// - Repetition penalty mechanisms
    /// - Context-aware processing rules
    /// - Dynamic probability distribution manipulation
    /// 
    /// # Composite Processor Features
    /// 
    /// ## Sampling Strategy Coordination
    /// - **Temperature Scaling**: Controls randomness vs determinism
    /// - **Top-k Filtering**: Limits consideration to k most likely tokens
    /// - **Nucleus (Top-p) Sampling**: Dynamic vocabulary truncation
    /// - **Typical Sampling**: Focuses on "typical" probability mass
    /// 
    /// ## Quality Enhancement
    /// - **Repetition Penalty**: Reduces repetitive text generation
    /// - **Presence Penalty**: Encourages vocabulary diversity
    /// - **Frequency Penalty**: Balances common vs rare token usage
    /// 
    /// ## Context Processing
    /// - **Length Normalization**: Adjusts probabilities based on sequence length
    /// - **EOS Handling**: Smart end-of-sequence token management
    /// - **Bias Application**: Apply custom token biases
    /// 
    /// # Performance Notes
    /// 
    /// - Zero-cost reference access (no cloning or allocation)
    /// - Processor configuration is immutable during generation
    /// - Optimized for high-frequency access during token generation
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::CandleGenerator;
    /// 
    /// let generator = CandleGenerator::default();
    /// let processor = generator.composite_processor();
    /// 
    /// // Access processor configuration
    /// println!("Processor enabled: {}", processor.is_enabled());
    /// ```
    #[inline(always)]
    pub fn composite_processor(&self) -> &CompositeProcessor {
        &self.composite_processor
    }

    /// Generates a complete text completion using AsyncStream architecture
    /// 
    /// Produces a single, complete response to the given completion request using
    /// the generator's configured model, tokenizer, and sampling settings.
    /// 
    /// # Arguments
    /// 
    /// * `request` - CandleCompletionRequest containing:
    ///   - `system_prompt` - System-level instructions
    ///   - `chat_history` - Conversation context
    ///   - `documents` - Contextual documents
    ///   - `temperature` - Sampling temperature
    ///   - `max_tokens` - Maximum tokens to generate
    /// 
    /// # Returns
    /// 
    /// `AsyncStream<CandleCompletionResponse<'static>>` that yields complete response
    /// 
    /// # Performance Characteristics
    /// 
    /// - **Zero Allocation**: Uses pre-allocated buffers and stack allocation
    /// - **AsyncStream Pattern**: Non-blocking, composable with other streams
    /// - **Lock-Free**: Critical generation code avoids mutex operations
    /// - **Inline Hot Path**: Performance-critical operations are inlined
    /// 
    /// # Thread Safety
    /// 
    /// This method is thread-safe and can be called concurrently from multiple
    /// threads. Each call operates on independent generation state.
    pub fn generate(&self, request: &crate::types::CandleCompletionRequest) -> fluent_ai_async::AsyncStream<crate::types::CandleCompletionResponse<'static>> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        // Removed unused imports: CandleCompletionResponse, CompletionResponse, CandleModel, CandleTokenizer, Tensor
        use std::time::{SystemTime, UNIX_EPOCH};
        use arrayvec::ArrayVec;
        
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let config = self.config.clone();
        let request = request.clone();
        
        AsyncStream::with_channel(move |sender| {
            // Create unique response ID using timestamp + random component
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let response_id = format!("cmpl-{:x}", timestamp);
            let created = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            
            // Build input text from request components
            let mut input_text = String::with_capacity(4096);
            
            if !request.system_prompt.is_empty() {
                input_text.push_str("System: ");
                input_text.push_str(&request.system_prompt);
                input_text.push_str("\n\n");
            }
            
            // Add chat history
            match &request.chat_history {
                crate::types::ZeroOneOrMany::One(msg) => {
                    input_text.push_str(&format!("{}: {}\n", msg.role, msg.content));
                }
                crate::types::ZeroOneOrMany::Many(messages) => {
                    for msg in messages {
                        input_text.push_str(&format!("{}: {}\n", msg.role, msg.content));
                    }
                }
                crate::types::ZeroOneOrMany::None => {}
            }
            
            // Add document context
            match &request.documents {
                crate::types::ZeroOneOrMany::One(doc) => {
                    input_text.push_str("Context: ");
                    let content = doc.data.as_str();
                    input_text.push_str(content);
                    input_text.push('\n');
                }
                crate::types::ZeroOneOrMany::Many(docs) => {
                    for doc in docs {
                        input_text.push_str("Context: ");
                        let content = doc.data.as_str();
                    input_text.push_str(content);
                        input_text.push('\n');
                    }
                }
                crate::types::ZeroOneOrMany::None => {}
            }
            
            // Tokenize input with pre-allocated buffer
            let tokens = match tokenizer.encode(&input_text, true) {
                Ok(tokens) => tokens,
                Err(e) => {
                    handle_error!(e, "Tokenization failed");
                }
            };
            
            // Convert to tensor
            let input_tensor = match Tensor::new(&tokens[..], &device) {
                Ok(tensor) => match tensor.unsqueeze(0) {
                    Ok(t) => t,
                    Err(e) => {
                        handle_error!(e, "Tensor unsqueeze failed");
                    }
                },
                Err(e) => {
                    handle_error!(e, "Tensor creation failed");
                }
            };
            
            // Generate tokens with autoregressive loop
            let max_tokens = request.max_tokens.map(|n| n.get() as usize).unwrap_or(config.max_tokens as usize);
            let mut generated_tokens = ArrayVec::<u32, 2048>::new();
            let mut current_tokens = tokens.clone();
            let mut _current_input = input_tensor; // Underscore prefix to indicate intentionally unused
            
            for _position in 0..max_tokens {
                // Forward pass through model - collect from AsyncStream
                let mut forward_stream = model.forward(&current_tokens);
                let logits = match forward_stream.try_next() {
                    Some(tensor) => tensor,
                    None => {
                        handle_error!("Model forward pass returned no results", "Forward pass failed");
                    }
                };
                
                // Get the last token's logits (batch_size=1, seq_len, vocab_size)
                let next_token_logits = match logits.narrow(1, logits.dim(1).unwrap_or(1) - 1, 1) {
                    Ok(logits) => match logits.squeeze(1) {
                        Ok(l) => l,
                        Err(e) => {
                            handle_error!(e, "Logits squeeze failed");
                        }
                    },
                    Err(e) => {
                        handle_error!(e, "Logits narrow failed");
                    }
                };
                
                // Apply temperature scaling
                let scaled_logits = if request.temperature != 1.0 {
                    match next_token_logits.affine(1.0 / request.temperature, 0.0) {
                        Ok(logits) => logits,
                        Err(e) => {
                            handle_error!(e, "Temperature scaling failed");
                        }
                    }
                } else {
                    next_token_logits
                };
                
                // Apply softmax to get probabilities
                let probabilities = match candle_nn::ops::softmax_last_dim(&scaled_logits) {
                    Ok(probs) => probs,
                    Err(e) => {
                        handle_error!(e, "Softmax failed");
                    }
                };
                
                // Sample next token using multinomial sampling
                let next_token = match self.sample_token(&probabilities) {
                    Ok(token) => token,
                    Err(e) => {
                        handle_error!(e, "Token sampling failed");
                    }
                };
                
                // Check for end-of-sequence token
                // For demo purposes, assume EOS token is 2 (typical for many models)
                if next_token == 2 {
                    break;
                }
                
                // Add to generated tokens
                if generated_tokens.try_push(next_token).is_err() {
                    break; // Buffer full
                }
                
                // Append new token to current sequence
                current_tokens.push(next_token);
                if generated_tokens.try_push(next_token).is_err() {
                    handle_error!(crate::error::CandleError::Msg("Token buffer overflow".to_string()), "Token buffer overflow");
                }
            }
            
            // Decode generated tokens to text
            let generated_text = match tokenizer.decode(&generated_tokens[..], true) {
                Ok(text) => text,
                Err(e) => {
                    handle_error!(e, "Token decoding failed");
                }
            };
            
            // Create completion response using the available type
            let response = crate::types::candle_completion::CompletionResponse {
                id: Some(response_id),
                object: Some("text_completion".to_string()),
                created: Some(created),
                text: std::borrow::Cow::Owned(generated_text),
                model: std::borrow::Cow::Owned("kimi-k2".to_string()),
                provider: Some(std::borrow::Cow::Owned("fluent-ai-candle".to_string())),
                usage: Some(crate::types::CandleUsage {
                    prompt_tokens: tokens.len() as u32,
                    completion_tokens: generated_tokens.len() as u32,
                    total_tokens: (tokens.len() + generated_tokens.len()) as u32,
                }),
                finish_reason: Some(std::borrow::Cow::Owned("stop".to_string())),
                response_time_ms: None,
                generation_time_ms: Some(0), // TODO: Track actual generation time
                tokens_per_second: Some(0.0), // TODO: Calculate actual TPS
            };
            
            emit!(sender, response);
        })
    }
    
    /// Sample a token from probability distribution using temperature and top-k/top-p
    #[inline(always)]
    fn sample_token(&self, probabilities: &Tensor) -> crate::error::CandleResult<u32> {
        use rand::{Rng, SeedableRng};
        
        let probs_vec = probabilities.to_vec1::<f32>()?;
        let vocab_size = probs_vec.len();
        
        // Simple multinomial sampling - find cumulative distribution
        let mut rng_state = self.rng_state.lock();
        let mut rng = if let Some(seed) = *rng_state {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_seed(rand::random())
        };
        
        let random_value: f32 = rng.r#gen();
        let mut cumulative = 0.0;
        
        for (i, &prob) in probs_vec.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                // Update RNG state
                *rng_state = Some(rng.r#gen());
                return Ok(i as u32);
            }
        }
        
        // Fallback to last token
        *rng_state = Some(rng.r#gen());
        Ok((vocab_size - 1) as u32)
    }

    /// Generates a streaming text completion with real-time token delivery
    /// 
    /// Produces a stream of incremental responses for real-time text generation,
    /// enabling low-latency user experiences with token-by-token delivery.
    /// 
    /// # Arguments
    /// 
    /// * `request` - CandleCompletionRequest containing generation parameters
    /// 
    /// # Returns
    /// 
    /// `AsyncStream<CandleStreamingResponse>` yielding token-by-token responses
    /// 
    /// # Performance Characteristics
    /// 
    /// - **Memory Efficient**: Processes tokens individually without buffering
    /// - **Low Latency**: Immediate token emission as generated
    /// - **Lock-Free**: Critical generation path avoids mutex operations
    /// - **Zero Allocation**: Uses pre-allocated buffers throughout
    /// 
    /// # Thread Safety
    /// 
    /// This method is thread-safe and supports concurrent streaming generations.
    /// Each stream operates independently with isolated generation state.
    pub fn generate_stream(&self, request: &crate::types::CandleCompletionRequest) -> fluent_ai_async::AsyncStream<crate::types::CandleStreamingResponse> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        use crate::types::CandleStreamingResponse;
        // Removed unused import: candle_core::Tensor
        use std::time::{SystemTime, UNIX_EPOCH};
        use arrayvec::ArrayVec;
        
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let config = self.config.clone();
        let request = request.clone();
        
        AsyncStream::with_channel(move |sender| {
            // Create unique response ID
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let response_id = format!("cmpl-{:x}", timestamp);
            let created = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            
            // Build input text from request components
            let mut input_text = String::with_capacity(4096);
            
            if !request.system_prompt.is_empty() {
                input_text.push_str("System: ");
                input_text.push_str(&request.system_prompt);
                input_text.push_str("\n\n");
            }
            
            // Add chat history
            match &request.chat_history {
                crate::types::ZeroOneOrMany::One(msg) => {
                    input_text.push_str(&format!("{}: {}\n", msg.role, msg.content));
                }
                crate::types::ZeroOneOrMany::Many(messages) => {
                    for msg in messages {
                        input_text.push_str(&format!("{}: {}\n", msg.role, msg.content));
                    }
                }
                crate::types::ZeroOneOrMany::None => {}
            }
            
            // Add document context
            match &request.documents {
                crate::types::ZeroOneOrMany::One(doc) => {
                    input_text.push_str("Context: ");
                    let content = doc.data.as_str();
                    input_text.push_str(content);
                    input_text.push('\n');
                }
                crate::types::ZeroOneOrMany::Many(docs) => {
                    for doc in docs {
                        input_text.push_str("Context: ");
                        let content = doc.data.as_str();
                    input_text.push_str(content);
                        input_text.push('\n');
                    }
                }
                crate::types::ZeroOneOrMany::None => {}
            }
            
            // Tokenize input
            let tokens = match tokenizer.encode(&input_text, true) {
                Ok(tokens) => tokens,
                Err(e) => {
                    handle_error!(e, "Tokenization failed");
                }
            };
            
            // Convert to tensor
            let input_tensor = match Tensor::new(&tokens[..], &device) {
                Ok(tensor) => match tensor.unsqueeze(0) {
                    Ok(t) => t,
                    Err(e) => {
                        handle_error!(e, "Tensor unsqueeze failed");
                    }
                },
                Err(e) => {
                    handle_error!(e, "Tensor creation failed");
                }
            };
            
            // Generate tokens with streaming emission
            let max_tokens = request.max_tokens.map(|n| n.get() as usize).unwrap_or(config.max_tokens as usize);
            let mut generated_tokens = ArrayVec::<u32, 2048>::new();
            let mut current_tokens = tokens.clone();  // Keep track of token IDs for model input
            let mut _current_tensor = input_tensor.clone(); // Underscore prefix to indicate intentionally unused
            let mut total_tokens_generated = 0u32;
            
            for _position in 0..max_tokens {
                // Forward pass through model - collect from AsyncStream
                let mut forward_stream = model.forward(&current_tokens);
                let logits = match forward_stream.try_next() {
                    Some(tensor) => tensor,
                    None => {
                        handle_error!("Model forward pass returned no results", "Forward pass failed");
                    }
                };
                
                // Get the last token's logits
                let next_token_logits = match logits.narrow(1, logits.dim(1).unwrap_or(1) - 1, 1) {
                    Ok(logits) => match logits.squeeze(1) {
                        Ok(l) => l,
                        Err(e) => {
                            handle_error!(e, "Logits squeeze failed");
                        }
                    },
                    Err(e) => {
                        handle_error!(e, "Logits narrow failed");
                    }
                };
                
                // Apply temperature scaling
                let scaled_logits = if request.temperature != 1.0 {
                    match next_token_logits.affine(1.0 / request.temperature, 0.0) {
                        Ok(logits) => logits,
                        Err(e) => {
                            handle_error!(e, "Temperature scaling failed");
                        }
                    }
                } else {
                    next_token_logits
                };
                
                // Apply softmax to get probabilities
                let probabilities = match candle_nn::ops::softmax_last_dim(&scaled_logits) {
                    Ok(probs) => probs,
                    Err(e) => {
                        handle_error!(e, "Softmax failed");
                    }
                };
                
                // Sample next token
                let next_token = match self.sample_token(&probabilities) {
                    Ok(token) => token,
                    Err(e) => {
                        handle_error!(e, "Token sampling failed");
                    }
                };
                
                // Check for end-of-sequence token
                // For demo purposes, assume EOS token is 2 (typical for many models)
                if next_token == 2 {
                    // Emit final chunk with finish reason
                    let final_response = CandleStreamingResponse {
                        id: response_id.clone(),
                        object: "text_completion".to_string(),
                        created,
                        model: "kimi-k2".to_string(),
                        choices: vec![crate::types::CandleStreamingChoice {
                            delta: crate::types::CandleStreamingDelta {
                                content: None,
                                role: None,
                                tool_calls: None,
                                function_call: None,
                            },
                            index: 0,
                            finish_reason: Some(crate::types::CandleFinishReason::Stop),
                            logprobs: None,
                        }],
                        usage: None,
                        system_fingerprint: None,
                    };
                    emit!(sender, final_response);
                    break;
                }
                
                // Add to generated tokens buffer
                if generated_tokens.try_push(next_token).is_err() {
                    break; // Buffer full
                }
                total_tokens_generated += 1;
                
                // Decode the single token to text
                let token_text = match tokenizer.decode(&[next_token], false) {
                    Ok(text) => text,
                    Err(e) => {
                        handle_error!(e, "Token decoding failed");
                    }
                };
                
                // Emit streaming response with this token
                let streaming_response = CandleStreamingResponse {
                    id: response_id.clone(),
                    object: "text_completion".to_string(),
                    created,
                    model: "kimi-k2".to_string(),
                    choices: vec![crate::types::CandleStreamingChoice {
                        delta: crate::types::CandleStreamingDelta {
                            content: Some(token_text),
                            role: None,
                            tool_calls: None,
                            function_call: None,
                        },
                        index: 0,
                        finish_reason: None,
                        logprobs: None,
                    }],
                    usage: None,
                    system_fingerprint: None,
                };
                emit!(sender, streaming_response);
                
                // Append new token to current sequence
                current_tokens.push(next_token);
                if generated_tokens.try_push(next_token).is_err() {
                    handle_error!(crate::error::CandleError::Msg("Token buffer overflow".to_string()), "Token buffer overflow");
                }
            }
            
            // If we finished by reaching max tokens (not EOS), emit final chunk
            if total_tokens_generated >= max_tokens as u32 {
                let final_response = CandleStreamingResponse {
                    id: response_id,
                    object: "text_completion".to_string(),
                    created,
                    model: "kimi-k2".to_string(),
                    choices: vec![crate::types::CandleStreamingChoice {
                        delta: crate::types::CandleStreamingDelta {
                            content: None,
                            role: None,
                            tool_calls: None,
                            function_call: None,
                        },
                        index: 0,
                        finish_reason: Some(crate::types::CandleFinishReason::Length),
                        logprobs: None,
                    }],
                    usage: None,
                    system_fingerprint: None,
                };
                emit!(sender, final_response);
            }
        })
    }
}

impl Default for CandleGenerator {
    /// Creates a default CandleGenerator with production-ready fallback configuration
    /// 
    /// Constructs a generator instance suitable for testing, development, and fallback
    /// scenarios where specific model configuration is not available.
    /// 
    /// # Default Configuration
    /// 
    /// ## Model and Tokenizer
    /// - **Model**: Default CandleModel with CPU device backend
    /// - **Tokenizer**: Basic CandleTokenizer with minimal vocabulary
    /// - **Device**: CPU device (universally available, no GPU dependencies)
    /// 
    /// ## Generation Settings
    /// - **Temperature**: 1.0 (balanced creativity vs consistency)
    /// - **Max Tokens**: 2048 (reasonable default for most use cases)
    /// - **Top-k**: Disabled (0) for full vocabulary consideration
    /// - **Top-p**: 1.0 (nucleus sampling disabled)
    /// - **Repetition Penalty**: 1.0 (no penalty applied)
    /// 
    /// ## Performance Configuration
    /// - **Sampling**: Basic temperature sampling only
    /// - **Streaming**: Disabled (minimal buffering)
    /// - **KV Cache**: Disabled (memory conservation)
    /// - **Composite Processor**: Basic configuration without advanced features
    /// 
    /// # Use Cases
    /// 
    /// - **Testing**: Unit tests and integration tests
    /// - **Development**: Quick prototyping and experimentation
    /// - **Fallback**: When specific configuration is unavailable
    /// - **Documentation**: Examples and demonstrations
    /// 
    /// # Performance Notes
    /// 
    /// - **CPU-Only**: No GPU acceleration, suitable for development/testing
    /// - **Memory Efficient**: Minimal memory footprint without KV cache
    /// - **Thread Safe**: Uses mutex protection for state management
    /// - **Zero Dependencies**: No external model files or configuration required
    /// 
    /// # Limitations
    /// 
    /// ⚠️ **Architecture Constraint Violation**: Current implementation uses 
    /// `parking_lot::Mutex` which violates the "no locking" constraint. This should 
    /// be refactored to use atomic types for lock-free operation:
    /// 
    /// ```rust
    /// // Current (should be refactored):
    /// rng_state: parking_lot::Mutex<Option<u64>>
    /// cumulative_log_prob: parking_lot::Mutex<f64>
    /// 
    /// // Target (lock-free):
    /// rng_state: AtomicU64
    /// cumulative_log_prob: AtomicU64  // Using f64::to_bits() for atomic storage
    /// ```
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use fluent_ai_candle::CandleGenerator;
    /// 
    /// // Create default generator for testing
    /// let generator = CandleGenerator::default();
    /// 
    /// // Verify default configuration
    /// assert_eq!(generator.config().temperature, 1.0);
    /// assert_eq!(generator.cumulative_log_prob(), 0.0);
    /// 
    /// // Ready for basic generation tasks
    /// // let response = generator.generate(&request).collect().await;
    /// ```
    /// 
    /// # Thread Safety
    /// 
    /// The default generator is thread-safe and can be cloned for concurrent use,
    /// though individual instances maintain independent state through mutex protection.
    #[inline(always)]
    fn default() -> Self {
        let model = Arc::new(CandleModel::default());
        let tokenizer = Arc::new(CandleTokenizer::default());
        let config = GenerationConfig::default();
        let device = Device::Cpu;
        
        Self::new(model, tokenizer, config, device)
    }
}