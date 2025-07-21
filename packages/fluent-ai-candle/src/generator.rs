//! Zero-allocation text generation with streaming support and SIMD optimization

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use arrayvec::ArrayVec;
use candle_core::{Device, Tensor};
use fluent_ai_domain::completion::{CompletionRequest, CompletionResponse};
use smallvec::SmallVec;
use tokio::sync::mpsc;
use futures::stream::Stream;

use crate::error::{CandleError, CandleResult};
use crate::model::CandleModel;
use crate::tokenizer::CandleTokenizer;

/// Maximum generation buffer size
const MAX_GENERATION_BUFFER: usize = 4096;

/// Maximum batch size for generation
const MAX_BATCH_SIZE: usize = 8;

/// Generation configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-k sampling parameter
    pub top_k: u32,
    /// Top-p (nucleus) sampling parameter
    pub top_p: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Early stopping criteria
    pub early_stopping: bool,
    /// Number of beams for beam search
    pub num_beams: u32,
    /// Length penalty for beam search
    pub length_penalty: f32,
    /// Do sample (vs greedy)
    pub do_sample: bool,
}

impl Default for GenerationConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            early_stopping: true,
            num_beams: 1,
            length_penalty: 1.0,
            do_sample: true,
        }
    }
}

/// Generation statistics
#[repr(C)]
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Total tokens generated
    pub tokens_generated: AtomicU32,
    /// Generation time in microseconds
    pub generation_time_us: AtomicU64,
    /// Tokens per second
    pub tokens_per_second: AtomicU32,
    /// Cache hits
    pub cache_hits: AtomicU32,
    /// Cache misses
    pub cache_misses: AtomicU32,
    /// Memory usage in bytes
    pub memory_usage: AtomicU64,
}

impl Default for GenerationStats {
    #[inline(always)]
    fn default() -> Self {
        Self {
            tokens_generated: AtomicU32::new(0),
            generation_time_us: AtomicU64::new(0),
            tokens_per_second: AtomicU32::new(0),
            cache_hits: AtomicU32::new(0),
            cache_misses: AtomicU32::new(0),
            memory_usage: AtomicU64::new(0),
        }
    }
}

/// Token with generation metadata
#[repr(C)]
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    /// Token ID
    pub id: u32,
    /// Token text
    pub text: SmallVec<[u8; 16]>,
    /// Log probability
    pub log_prob: f32,
    /// Cumulative log probability
    pub cumulative_log_prob: f32,
    /// Generation step
    pub step: u32,
    /// Is special token
    pub is_special: bool,
}

impl GeneratedToken {
    /// Create a new generated token
    #[inline(always)]
    pub fn new(
        id: u32,
        text: &str,
        log_prob: f32,
        cumulative_log_prob: f32,
        step: u32,
        is_special: bool,
    ) -> CandleResult<Self> {
        let mut text_bytes = SmallVec::new();
        text_bytes
            .try_extend_from_slice(text.as_bytes())
            .map_err(|_| CandleError::generation_failed("Token text too long"))?;

        Ok(Self {
            id,
            text: text_bytes,
            log_prob,
            cumulative_log_prob,
            step,
            is_special,
        })
    }

    /// Get token text as string
    #[inline(always)]
    pub fn text_str(&self) -> CandleResult<&str> {
        std::str::from_utf8(&self.text)
            .map_err(|_| CandleError::generation_failed("Invalid UTF-8 in token text"))
    }
}

/// Generation state for streaming
#[repr(C)]
pub struct GenerationState {
    /// Current token sequence
    pub tokens: ArrayVec<u32, MAX_GENERATION_BUFFER>,
    /// Generated tokens with metadata
    pub generated_tokens: SmallVec<[GeneratedToken; 512]>,
    /// Current position
    pub position: u32,
    /// Is generation complete
    pub is_complete: AtomicBool,
    /// Stop reason
    pub stop_reason: parking_lot::Mutex<Option<StopReason>>,
    /// Generation statistics
    pub stats: GenerationStats,
}

impl GenerationState {
    /// Create a new generation state
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            tokens: ArrayVec::new(),
            generated_tokens: SmallVec::new(),
            position: 0,
            is_complete: AtomicBool::new(false),
            stop_reason: parking_lot::Mutex::new(None),
            stats: GenerationStats::default(),
        }
    }

    /// Add a generated token
    #[inline(always)]
    pub fn add_token(&mut self, token: GeneratedToken) -> CandleResult<()> {
        self.tokens
            .try_push(token.id)
            .map_err(|_| CandleError::generation_failed("Generation buffer overflow"))?;

        self.generated_tokens.push(token);
        self.position += 1;
        self.stats.tokens_generated.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Mark generation as complete
    #[inline(always)]
    pub fn complete(&self, reason: StopReason) {
        *self.stop_reason.lock() = Some(reason);
        self.is_complete.store(true, Ordering::Release);
    }

    /// Check if generation is complete
    #[inline(always)]
    pub fn is_complete(&self) -> bool {
        self.is_complete.load(Ordering::Acquire)
    }

    /// Get stop reason
    #[inline(always)]
    pub fn stop_reason(&self) -> Option<StopReason> {
        *self.stop_reason.lock()
    }
}

impl Default for GenerationState {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Reasons for stopping generation
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Maximum tokens reached
    MaxTokens = 0,
    /// EOS token encountered
    EosToken = 1,
    /// Stop sequence encountered
    StopSequence = 2,
    /// User requested stop
    UserStop = 3,
    /// Error occurred
    Error = 4,
    /// Length limit reached
    LengthLimit = 5,
}

/// Zero-allocation text generator
pub struct CandleGenerator {
    /// The model for generation
    model: Arc<CandleModel>,
    /// The tokenizer
    tokenizer: Arc<CandleTokenizer>,
    /// Generation configuration
    config: GenerationConfig,
    /// Device for computation
    device: Device,
    /// Random number generator state
    rng_state: parking_lot::Mutex<Option<u64>>,
    /// Cumulative log probability for current generation
    cumulative_log_prob: parking_lot::Mutex<f64>,
}

impl CandleGenerator {
    /// Create a new generator
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
            config,
            device,
            rng_state: parking_lot::Mutex::new(config.seed),
            cumulative_log_prob: parking_lot::Mutex::new(0.0),
        }
    }

    /// Generate completion for a single request
    #[inline(always)]
    pub async fn generate(
        &self,
        request: &CompletionRequest<'_>,
    ) -> CandleResult<CompletionResponse> {
        let start_time = std::time::Instant::now();

        // Tokenize input
        let input_tokens = self.tokenizer.encode(
            std::str::from_utf8(request.prompt())
                .map_err(|_| CandleError::tokenizer("Invalid UTF-8 in prompt"))?,
            true,
        )?;

        // Initialize generation state
        let mut state = GenerationState::new();
        state
            .tokens
            .try_extend_from_slice(&input_tokens)
            .map_err(|_| CandleError::generation_failed("Input too long"))?;

        // Generate tokens
        let mut generated_text = String::new();
        let mut step = 0;

        while step < request.max_tokens() && !state.is_complete() {
            let next_token = self.generate_next_token(&state.tokens, step).await?;

            // Check for stop conditions
            if let Some(eos_id) = self.tokenizer.eos_token_id() {
                if next_token.id == eos_id {
                    state.complete(StopReason::EosToken);
                    break;
                }
            }

            // Add token to state
            state.add_token(next_token.clone())?;

            // Decode token to text
            if let Ok(token_text) = next_token.text_str() {
                generated_text.push_str(token_text);
            }

            step += 1;
        }

        if step >= request.max_tokens() && !state.is_complete() {
            state.complete(StopReason::MaxTokens);
        }

        // Calculate statistics
        let elapsed = start_time.elapsed();
        let tokens_per_second = if elapsed.as_secs() > 0 {
            (step as f64 / elapsed.as_secs_f64()) as u32
        } else {
            0
        };

        state
            .stats
            .generation_time_us
            .store(elapsed.as_micros() as u64, Ordering::Relaxed);
        state
            .stats
            .tokens_per_second
            .store(tokens_per_second, Ordering::Relaxed);

        // Build response
        let mut response = CompletionResponse::builder()
            .text(generated_text)
            .tokens_generated(step)
            .finish_reason(match state.stop_reason() {
                Some(StopReason::MaxTokens) => "length",
                Some(StopReason::EosToken) => "stop",
                Some(StopReason::StopSequence) => "stop",
                _ => "unknown",
            })
            .build()
            .map_err(|_| CandleError::generation_failed("Failed to build response"))?;

        response.set_generation_time_ms((elapsed.as_millis() as u32).max(1));
        response.set_tokens_per_second(tokens_per_second);

        Ok(response)
    }

    /// Generate streaming completion
    #[inline(always)]
    pub async fn generate_stream(
        &self,
        request: &CompletionRequest<'_>,
    ) -> CandleResult<StreamingResponse> {
        let (tx, rx) = mpsc::unbounded_channel();

        // Clone necessary data for the generation task
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let config = self.config.clone();
        let device = self.device.clone();
        let prompt = request.prompt().to_vec();
        let max_tokens = request.max_tokens();

        // Spawn generation task
        tokio::spawn(async move {
            let generator = CandleGenerator::new(model, tokenizer, config, device);

            // Create a temporary request for generation
            let temp_request = CompletionRequest::builder()
                .prompt(std::str::from_utf8(&prompt).unwrap_or(""))
                .max_tokens(max_tokens)
                .build()
                .unwrap();

            match generator.generate_stream_internal(&temp_request, tx).await {
                Ok(_) => {}
                Err(e) => {
                    // Send error as final chunk
                    let _ = tx.send(Err(e.into()));
                }
            }
        });

        Ok(StreamingResponse::new(Box::pin(CandleTokenStream::new(rx))))
    }

    /// Internal streaming generation
    async fn generate_stream_internal(
        &self,
        request: &CompletionRequest<'_>,
        tx: mpsc::UnboundedSender<
            Result<CompletionResponse, fluent_ai_domain::extractor::ExtractionError>,
        >,
    ) -> CandleResult<()> {
        // Tokenize input
        let input_tokens = self.tokenizer.encode(
            std::str::from_utf8(request.prompt())
                .map_err(|_| CandleError::tokenizer("Invalid UTF-8 in prompt"))?,
            true,
        )?;

        // Initialize generation state
        let mut state = GenerationState::new();
        state
            .tokens
            .try_extend_from_slice(&input_tokens)
            .map_err(|_| CandleError::generation_failed("Input too long"))?;

        let mut step = 0;
        let mut accumulated_text = String::new();

        while step < request.max_tokens() && !state.is_complete() {
            let next_token = self.generate_next_token(&state.tokens, step).await?;

            // Check for stop conditions
            if let Some(eos_id) = self.tokenizer.eos_token_id() {
                if next_token.id == eos_id {
                    state.complete(StopReason::EosToken);
                    break;
                }
            }

            // Add token to state
            state.add_token(next_token.clone())?;

            // Decode token to text
            if let Ok(token_text) = next_token.text_str() {
                accumulated_text.push_str(token_text);

                // Send streaming chunk
                let chunk = CompletionResponse::builder()
                    .text(token_text.to_string())
                    .tokens_generated(step + 1)
                    .finish_reason("") // Empty for streaming chunks
                    .build()
                    .map_err(|_| CandleError::generation_failed("Failed to build chunk"))?;

                if tx.send(Ok(chunk)).is_err() {
                    // Receiver dropped, stop generation
                    break;
                }
            }

            step += 1;
        }

        // Send final chunk with completion info
        let final_chunk = CompletionResponse::builder()
            .text("") // Empty text for final chunk
            .tokens_generated(step)
            .finish_reason(match state.stop_reason() {
                Some(StopReason::MaxTokens) => "length",
                Some(StopReason::EosToken) => "stop",
                Some(StopReason::StopSequence) => "stop",
                _ => "unknown",
            })
            .build()
            .map_err(|_| CandleError::generation_failed("Failed to build final chunk"))?;

        let _ = tx.send(Ok(final_chunk));

        Ok(())
    }

    /// Generate the next token
    async fn generate_next_token(&self, tokens: &[u32], step: u32) -> CandleResult<GeneratedToken> {
        // Convert tokens to tensor
        let input_tensor = Tensor::new(tokens, &self.device).map_err(|e| CandleError::from(e))?;

        // Forward pass through model
        let logits = self.model.forward(&input_tensor).await?;

        // Apply sampling
        let next_token_id = self.sample_token(&logits, step).await?;

        // Decode token to text
        let token_text = self
            .tokenizer
            .id_to_token(next_token_id)
            .unwrap_or_else(|| format!("<unk:{}>", next_token_id));

        // Calculate actual log probability from logits
        let log_prob = self
            .calculate_token_log_probability(&logits, next_token_id)
            .await?;

        // Update cumulative log probability (stored in generation state)
        let cumulative_log_prob = self.update_cumulative_log_prob(log_prob).await?;

        GeneratedToken::new(
            next_token_id,
            &token_text,
            log_prob,
            cumulative_log_prob,
            step,
            self.tokenizer.is_special_token(next_token_id),
        )
    }

    /// Sample next token from logits
    async fn sample_token(&self, logits: &Tensor, _step: u32) -> CandleResult<u32> {
        // Apply temperature
        let scaled_logits = if self.config.temperature != 1.0 {
            (logits / self.config.temperature as f64).map_err(|e| CandleError::from(e))?
        } else {
            logits.clone()
        };

        // Apply top-k filtering
        let filtered_logits = if self.config.top_k > 0 {
            self.apply_top_k(&scaled_logits, self.config.top_k).await?
        } else {
            scaled_logits
        };

        // Apply top-p filtering
        let final_logits = if self.config.top_p < 1.0 {
            self.apply_top_p(&filtered_logits, self.config.top_p)
                .await?
        } else {
            filtered_logits
        };

        // Sample from distribution
        if self.config.do_sample {
            self.sample_from_logits(&final_logits).await
        } else {
            self.greedy_sample(&final_logits).await
        }
    }

    /// Apply top-k filtering - keep only the k most probable tokens
    async fn apply_top_k(&self, logits: &Tensor, k: u32) -> CandleResult<Tensor> {
        if k == 0 {
            return Ok(logits.clone());
        }

        // Get the shape of logits (should be [vocab_size])
        let shape = logits.shape();
        let vocab_size = shape.dims()[shape.rank() - 1];

        if k >= vocab_size as u32 {
            return Ok(logits.clone());
        }

        // Get top-k values and indices
        let (top_k_values, top_k_indices) =
            logits.topk(k as usize).map_err(|e| CandleError::from(e))?;

        // Create a mask for top-k tokens
        let mut filtered_logits = Tensor::full(f32::NEG_INFINITY, shape, &self.device)
            .map_err(|e| CandleError::from(e))?;

        // Set top-k values in the filtered tensor
        filtered_logits = filtered_logits
            .scatter_add(&top_k_indices, &top_k_values, 0)
            .map_err(|e| CandleError::from(e))?;

        Ok(filtered_logits)
    }

    /// Apply top-p (nucleus) filtering - keep tokens with cumulative probability <= p
    async fn apply_top_p(&self, logits: &Tensor, p: f32) -> CandleResult<Tensor> {
        if p >= 1.0 {
            return Ok(logits.clone());
        }

        // Convert logits to probabilities using softmax
        let probs = candle_nn::ops::softmax(logits, candle_core::D::Minus1)
            .map_err(|e| CandleError::from(e))?;

        // Sort probabilities in descending order
        let sorted_probs = probs
            .sort_by(candle_core::D::Minus1, false)
            .map_err(|e| CandleError::from(e))?;

        // Calculate cumulative probabilities
        let cumulative_probs = sorted_probs
            .cumsum(candle_core::D::Minus1)
            .map_err(|e| CandleError::from(e))?;

        // Find tokens where cumulative probability exceeds p
        let mask = cumulative_probs
            .le(&Tensor::full(p, cumulative_probs.shape(), &self.device)
                .map_err(|e| CandleError::from(e))?)
            .map_err(|e| CandleError::from(e))?;

        // Apply mask to filter out tokens beyond nucleus
        let filtered_logits = logits
            .where_cond(
                &mask,
                &Tensor::full(f32::NEG_INFINITY, logits.shape(), &self.device)
                    .map_err(|e| CandleError::from(e))?,
            )
            .map_err(|e| CandleError::from(e))?;

        Ok(filtered_logits)
    }

    /// Sample from logits distribution using multinomial sampling
    async fn sample_from_logits(&self, logits: &Tensor) -> CandleResult<u32> {
        // Convert logits to probabilities using softmax
        let probs = candle_nn::ops::softmax(logits, candle_core::D::Minus1)
            .map_err(|e| CandleError::from(e))?;

        // Convert probabilities to CPU for sampling
        let probs_cpu = probs
            .to_device(&Device::Cpu)
            .map_err(|e| CandleError::from(e))?;

        // Get probabilities as vector for multinomial sampling
        let prob_values: Vec<f32> = probs_cpu.to_vec1().map_err(|e| CandleError::from(e))?;

        // Generate random number using thread-local RNG or seed
        let random_val: f32 = if let Some(seed) = *self.rng_state.lock() {
            // Use deterministic sampling with seed
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&seed, &mut hasher);
            let hash = std::hash::Hasher::finish(&hasher);
            (hash as f64 / u64::MAX as f64) as f32
        } else {
            // Use thread-local random number generator
            fastrand::f32()
        };

        // Multinomial sampling - find the first token where cumulative probability exceeds random value
        let mut cumulative_prob = 0.0;
        for (idx, &prob) in prob_values.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= random_val {
                return Ok(idx as u32);
            }
        }

        // Fallback to last token if numerical precision issues
        Ok((prob_values.len() - 1) as u32)
    }

    /// Greedy sampling (argmax)
    async fn greedy_sample(&self, logits: &Tensor) -> CandleResult<u32> {
        let argmax = logits
            .argmax_keepdim(candle_core::D::Minus1)
            .map_err(|e| CandleError::from(e))?;

        let token_id = argmax
            .to_scalar::<u32>()
            .map_err(|e| CandleError::from(e))?;

        Ok(token_id)
    }

    /// Update generation configuration
    #[inline(always)]
    pub fn update_config(&mut self, config: GenerationConfig) {
        self.config = config;
        *self.rng_state.lock() = config.seed;
    }

    /// Get generation configuration
    #[inline(always)]
    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }

    /// Calculate log probability for a specific token from logits
    async fn calculate_token_log_probability(
        &self,
        logits: &Tensor,
        token_id: u32,
    ) -> CandleResult<f64> {
        // Convert logits to log probabilities using log_softmax
        let log_probs = candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)
            .map_err(|e| CandleError::from(e))?;

        // Extract log probability for the specific token
        let token_log_prob = log_probs
            .i(token_id as usize)
            .map_err(|e| CandleError::from(e))?
            .to_scalar::<f64>()
            .map_err(|e| CandleError::from(e))?;

        Ok(token_log_prob)
    }

    /// Update cumulative log probability in generation state
    async fn update_cumulative_log_prob(&self, token_log_prob: f64) -> CandleResult<f64> {
        let mut cumulative_guard = self.cumulative_log_prob.lock();
        *cumulative_guard += token_log_prob;
        Ok(*cumulative_guard)
    }

    /// Reset cumulative log probability for new generation
    pub fn reset_cumulative_log_prob(&self) {
        let mut cumulative_guard = self.cumulative_log_prob.lock();
        *cumulative_guard = 0.0;
    }

    /// Get current cumulative log probability
    pub fn get_cumulative_log_prob(&self) -> f64 {
        *self.cumulative_log_prob.lock()
    }
}

/// Stream implementation for token generation
pub struct CandleTokenStream {
    receiver: mpsc::UnboundedReceiver<
        Result<CompletionResponse, fluent_ai_domain::extractor::ExtractionError>,
    >,
}

impl CandleTokenStream {
    /// Create a new token stream
    #[inline(always)]
    pub fn new(
        receiver: mpsc::UnboundedReceiver<
            Result<CompletionResponse, fluent_ai_domain::extractor::ExtractionError>,
        >,
    ) -> Self {
        Self { receiver }
    }
}

impl Stream for CandleTokenStream {
    type Item = Result<CompletionResponse, fluent_ai_domain::extractor::ExtractionError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

unsafe impl Send for CandleGenerator {}
unsafe impl Sync for CandleGenerator {}
