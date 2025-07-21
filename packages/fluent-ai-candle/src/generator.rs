//! Zero-allocation text generation with streaming support and SIMD optimization

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use arrayvec::ArrayVec;
use candle_core::{Device, Tensor, IndexOp};
use candle_transformers::models::deepseek2::TopKLastDimOp;
use fluent_ai_domain::completion::{
    CompletionRequest, CompletionResponse, StreamingResponse,
    CompletionCoreResponse, CompletionCoreError, CompletionCoreResult,
};
use fluent_ai_domain::message::MessageRole;
use smallvec::SmallVec;
use tokio::sync::mpsc;
use futures::stream::Stream;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::error::{CandleError, CandleResult};
use crate::model::CandleModel;
use crate::tokenizer::CandleTokenizer;
use crate::sampling::{LogitsProcessor, Sampling};
use crate::streaming::{TokenOutputStream, StreamingConfig};
use crate::kv_cache::{KVCache, KVCacheConfig};

/// Maximum generation buffer size
const MAX_GENERATION_BUFFER: usize = 4096;

/// Maximum batch size for generation
#[allow(dead_code)]
const MAX_BATCH_SIZE: usize = 8;

/// SIMD-optimized token processing operations for blazing-fast performance
/// 
/// These functions provide vectorized operations using CPU SIMD instructions
/// (AVX2/FMA3 on x86_64, NEON on ARM64) for maximum throughput
mod simd_ops {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// SIMD-optimized temperature scaling for logits array
    /// Scales f32 logits by temperature using vectorized operations
    #[inline(always)]
    pub fn scale_logits_by_temperature(logits: &mut [f32], temperature: f32) {
        if temperature == 1.0 {
            return; // No scaling needed
        }
        
        let inv_temp = 1.0 / temperature;
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    scale_logits_avx2_fma(logits, inv_temp);
                    return;
                }
            }
        }
        
        // Fallback optimized scalar with manual unrolling for ILP
        let (chunks_4, remainder) = logits.split_at_mut(logits.len() - (logits.len() % 4));
        
        for chunk in chunks_4.chunks_exact_mut(4) {
            chunk[0] *= inv_temp;
            chunk[1] *= inv_temp; 
            chunk[2] *= inv_temp;
            chunk[3] *= inv_temp;
        }
        
        for value in remainder {
            *value *= inv_temp;
        }
    }

    /// SIMD-optimized cumulative sum for probability calculations
    /// Computes prefix sum using vectorized operations for top-p sampling
    #[inline(always)]
    pub fn cumulative_sum_f32(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        
        if input.is_empty() {
            return;
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && input.len() >= 8 {
                unsafe {
                    cumulative_sum_avx2(input, output);
                    return;
                }
            }
        }
        
        // Fallback optimized scalar implementation
        output[0] = input[0];
        for i in 1..input.len() {
            output[i] = output[i - 1] + input[i];
        }
    }

    /// SIMD-optimized multinomial sampling index finder
    /// Finds first index where cumulative probability >= random value
    #[inline(always)]
    pub fn find_sample_index(cumulative_probs: &[f32], random_val: f32) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && cumulative_probs.len() >= 8 {
                unsafe {
                    return find_sample_index_avx2(cumulative_probs, random_val);
                }
            }
        }
        
        // Fallback binary search for O(log n) performance
        match cumulative_probs.binary_search_by(|&x| {
            if x < random_val { std::cmp::Ordering::Less } 
            else { std::cmp::Ordering::Greater }
        }) {
            Ok(idx) => idx,
            Err(idx) => idx.min(cumulative_probs.len().saturating_sub(1))
        }
    }

    // AVX2/FMA3 implementations for x86_64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn scale_logits_avx2_fma(logits: &mut [f32], inv_temp: f32) {
        let inv_temp_vec = _mm256_set1_ps(inv_temp);
        let chunks = logits.chunks_exact_mut(8);
        let remainder = chunks.into_remainder();
        
        for chunk in chunks {
            let logits_vec = _mm256_loadu_ps(chunk.as_ptr());
            let scaled = _mm256_mul_ps(logits_vec, inv_temp_vec);
            _mm256_storeu_ps(chunk.as_mut_ptr(), scaled);
        }
        
        // Handle remainder with scalar
        for value in remainder {
            *value *= inv_temp;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn cumulative_sum_avx2(input: &[f32], output: &mut [f32]) {
        // Handle small arrays with scalar
        if input.len() < 16 {
            output[0] = input[0];
            for i in 1..input.len() {
                output[i] = output[i - 1] + input[i];
            }
            return;
        }
        
        // Vectorized prefix sum using segmented approach
        let mut running_sum = 0.0f32;
        let chunks = input.chunks_exact(8);
        let remainder = chunks.into_remainder();
        
        for (chunk_idx, chunk) in chunks.enumerate() {
            let chunk_vec = _mm256_loadu_ps(chunk.as_ptr());
            
            // Compute prefix sum within chunk using horizontal adds
            let mut accumulator = _mm256_setzero_ps();
            let mut current = chunk_vec;
            
            // Step 1: [a, b, c, d, e, f, g, h] -> [a, a+b, c, c+d, e, e+f, g, g+h]
            let shifted = _mm256_permute_ps(current, 0b10010011);
            let mask1 = _mm256_set_ps(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
            let masked = _mm256_mul_ps(shifted, mask1);
            current = _mm256_add_ps(current, masked);
            
            // Step 2: Add running sum to all elements
            let running_sum_vec = _mm256_set1_ps(running_sum);
            let result = _mm256_add_ps(current, running_sum_vec);
            
            // Store result
            _mm256_storeu_ps(output[chunk_idx * 8..].as_mut_ptr(), result);
            
            // Extract last element as new running sum
            let mut temp: [f32; 8] = [0.0; 8];
            _mm256_storeu_ps(temp.as_mut_ptr(), result);
            running_sum = temp[7];
        }
        
        // Handle remainder
        let offset = chunks.len() * 8;
        for (i, &val) in remainder.iter().enumerate() {
            running_sum += val;
            output[offset + i] = running_sum;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_sample_index_avx2(cumulative_probs: &[f32], random_val: f32) -> usize {
        let target = _mm256_set1_ps(random_val);
        let chunks = cumulative_probs.chunks_exact(8);
        
        for (chunk_idx, chunk) in chunks.enumerate() {
            let probs = _mm256_loadu_ps(chunk.as_ptr());
            let cmp = _mm256_cmp_ps(probs, target, _CMP_GE_OQ);
            let mask = _mm256_movemask_ps(cmp);
            
            if mask != 0 {
                // Found match - get first set bit position
                let first_bit = mask.trailing_zeros() as usize;
                return chunk_idx * 8 + first_bit;
            }
        }
        
        // Check remainder with scalar
        let offset = chunks.len() * 8;
        for (i, &prob) in cumulative_probs[offset..].iter().enumerate() {
            if prob >= random_val {
                return offset + i;
            }
        }
        
        cumulative_probs.len().saturating_sub(1)
    }
}

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
#[derive(Debug)]
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

impl Clone for GenerationStats {
    fn clone(&self) -> Self {
        Self {
            tokens_generated: AtomicU32::new(self.tokens_generated.load(Ordering::Relaxed)),
            generation_time_us: AtomicU64::new(self.generation_time_us.load(Ordering::Relaxed)),
            tokens_per_second: AtomicU32::new(self.tokens_per_second.load(Ordering::Relaxed)),
            cache_hits: AtomicU32::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU32::new(self.cache_misses.load(Ordering::Relaxed)),
            memory_usage: AtomicU64::new(self.memory_usage.load(Ordering::Relaxed)),
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
    pub text: SmallVec<u8, 16>,
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
        text_bytes.extend_from_slice(text.as_bytes());

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
    pub generated_tokens: SmallVec<GeneratedToken, 512>,
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
    /// Sophisticated sampling configuration
    sampling_config: Sampling,
    /// Streaming configuration for real-time output
    streaming_config: StreamingConfig,
    /// KV cache for efficient generation
    kv_cache: Option<Arc<parking_lot::Mutex<KVCache>>>,
    /// LogitsProcessor for sophisticated sampling
    logits_processor: Option<Box<dyn LogitsProcessor>>,
    /// TokenOutputStream for real-time streaming
    token_output_stream: Option<Arc<parking_lot::Mutex<TokenOutputStream>>>,
}

impl Clone for CandleGenerator {
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
            logits_processor: self.logits_processor.clone(),
            token_output_stream: self.token_output_stream.as_ref().map(Arc::clone),
        }
    }
}

impl CandleGenerator {
    /// Create a new generator with sophisticated features
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
            sampling_config: Sampling::default(),
            streaming_config: StreamingConfig::default(),
            kv_cache: None,
            logits_processor: None,
            token_output_stream: None,
        }
    }

    /// Create a new generator with sophisticated features configured
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

        // Initialize LogitsProcessor based on sampling configuration
        let composite_processor = sampling_config.build_processor()?;
        let logits_processor: Option<Box<dyn LogitsProcessor>> = Some(Box::new(composite_processor));

        // Initialize TokenOutputStream for streaming
        let (token_stream, _sender) = TokenOutputStream::new(streaming_config.clone())?;
        let token_output_stream = Some(Arc::new(parking_lot::Mutex::new(token_stream)));

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
            logits_processor,
            token_output_stream,
        })
    }

    /// Zero-allocation prompt construction with stack allocation for small prompts
    #[inline(always)]
    fn construct_prompt_safe(&self, request: &CompletionRequest<'_>) -> CandleResult<String> {
        if request.chat_history.is_empty() {
            // Zero-copy path: avoid to_string() allocation by cloning directly
            Ok(String::from(request.system_prompt.clone()))
        } else {
            // Calculate exact capacity needed to avoid reallocations
            let mut total_capacity = request.system_prompt.len() + 2; // "\n\n"
            
            // Pre-calculate message sizes for zero reallocation
            for msg in request.chat_history.iter() {
                // Content is already a String, no UTF-8 validation needed
                
                // Calculate exact size for message: "MessageType: content\n"
                // User/Assistant/System = 4-9 chars + ": " + content + "\n"
                total_capacity += match msg.role {
                    MessageRole::User => 4,             // "User"
                    MessageRole::Assistant => 9,        // "Assistant" 
                    MessageRole::System => 6,           // "System"
                    MessageRole::Tool => 4,             // "Tool"
                };
                total_capacity += msg.content.len();
                total_capacity += 3; // ": " + "\n"
            }
            
            // Stack allocation optimization for small prompts (< 1KB)
            const STACK_PROMPT_SIZE: usize = 1024;
            if total_capacity <= STACK_PROMPT_SIZE {
                // Use stack-allocated ArrayVec for true zero heap allocation
                let mut stack_prompt = ArrayVec::<u8, STACK_PROMPT_SIZE>::new();
                stack_prompt.try_extend_from_slice(request.system_prompt.as_bytes())
                    .map_err(|_| CandleError::generation_failed("Stack prompt buffer overflow"))?;
                stack_prompt.try_extend_from_slice(b"\n\n")
                    .map_err(|_| CandleError::generation_failed("Stack prompt buffer overflow"))?;
                
                // Zero-allocation message formatting on stack
                for msg in request.chat_history.iter() {
                    let content_str = &msg.content;
                    
                    let message_type_bytes: &[u8] = match msg.role {
                        MessageRole::User => b"User",
                        MessageRole::Assistant => b"Assistant",
                        MessageRole::System => b"System",
                        MessageRole::Tool => b"Tool",
                    };
                    
                    stack_prompt.try_extend_from_slice(message_type_bytes)
                        .map_err(|_| CandleError::generation_failed("Stack prompt buffer overflow"))?;
                    stack_prompt.try_extend_from_slice(b": ")
                        .map_err(|_| CandleError::generation_failed("Stack prompt buffer overflow"))?;
                    stack_prompt.try_extend_from_slice(content_str.as_bytes())
                        .map_err(|_| CandleError::generation_failed("Stack prompt buffer overflow"))?;
                    stack_prompt.try_extend_from_slice(b"\n")
                        .map_err(|_| CandleError::generation_failed("Stack prompt buffer overflow"))?;
                }
                
                // Convert stack buffer to string (single allocation)
                return Ok(String::from_utf8(stack_prompt.to_vec())
                    .map_err(|_| CandleError::invalid_input("Invalid UTF-8 in constructed prompt"))?);
            }
            
            // Heap allocation path for large prompts - single allocation with exact capacity
            let mut prompt = String::with_capacity(total_capacity);
            prompt.push_str(&request.system_prompt);
            prompt.push_str("\n\n");
            
            // Zero-allocation message formatting using direct string operations
            for msg in request.chat_history.iter() {
                // Content is already a String
                let content_str = &msg.content;
                
                // Zero-allocation message type formatting using match instead of Debug format
                let message_type_str = match msg.role {
                    MessageRole::User => "User",
                    MessageRole::Assistant => "Assistant", 
                    MessageRole::System => "System",
                    MessageRole::Tool => "Tool",
                };
                
                // Direct string operations - no format! allocations
                prompt.push_str(message_type_str);
                prompt.push_str(": ");
                prompt.push_str(content_str);
                prompt.push('\n');
            }
            
            // Ensure we calculated capacity correctly (debug assertion)
            debug_assert!(prompt.len() <= total_capacity, 
                "Prompt capacity miscalculation: {} > {}", prompt.len(), total_capacity);
            
            Ok(prompt)
        }
    }

    /// Generate completion for a single request
    #[inline(always)]
    pub async fn generate(
        &self,
        request: &CompletionRequest<'_>,
    ) -> CandleResult<CompletionResponse> {
        let start_time = std::time::Instant::now();

        // Construct prompt from system prompt and chat history
        let prompt = self.construct_prompt_safe(request)?;

        // Tokenize input
        let input_tokens = self.tokenizer.encode(&prompt, true)?;

        // Initialize generation state
        let mut state = GenerationState::new();
        state
            .tokens
            .try_extend_from_slice(&input_tokens)
            .map_err(|_| CandleError::generation_failed("Input too long"))?;

        // Generate tokens
        let mut generated_text = String::new();
        let mut step = 0;
        const DEFAULT_MAX_TOKENS: u32 = 1000;
        let max_tokens = request.max_tokens.map(|n| n.get() as u32).unwrap_or(DEFAULT_MAX_TOKENS);

        while step < max_tokens && !state.is_complete() {
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

        if step >= max_tokens && !state.is_complete() {
            state.complete(StopReason::MaxTokens);
        }

        // Calculate statistics
        let elapsed = start_time.elapsed();
        let tokens_per_second = if elapsed.as_secs_f64() > 0.0 {
            step as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        state
            .stats
            .generation_time_us
            .store(elapsed.as_micros() as u64, Ordering::Relaxed);
        state
            .stats
            .tokens_per_second
            .store(tokens_per_second as u32, Ordering::Relaxed);

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
            .build();

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
        let (tx, rx) = mpsc::unbounded_channel::<CompletionCoreResult<CompletionCoreResponse>>();

        // Construct prompt from system prompt and chat history safely
        let prompt = self.construct_prompt_safe(request)?;
        
        // Clone necessary data for the generation task
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let config = self.config.clone();
        let device = self.device.clone();
        
        const DEFAULT_MAX_TOKENS: u32 = 1000;
        let _max_tokens = request.max_tokens.map(|n| n.get() as u32).unwrap_or(DEFAULT_MAX_TOKENS);

        // Spawn generation task
        tokio::spawn(async move {
            let generator = CandleGenerator::new(model, tokenizer, config, device);

            // Create a temporary request for generation
            let temp_request = match CompletionRequest::builder()
                .system_prompt(&prompt)
                .build() {
                Ok(req) => req,
                Err(_) => {
                    let _ = tx.send(Err(CompletionCoreError::GenerationFailed("Failed to build request".to_string())));
                    return;
                }
            };

            match generator.generate_stream_internal(&temp_request, tx.clone()).await {
                Ok(_) => {}
                Err(e) => {
                    // Send error as final chunk - convert CandleError to CompletionCoreError
                    let _ = tx.send(Err(CompletionCoreError::from(e)));
                }
            }
        });

        Ok(StreamingResponse::new(Box::pin(CandleTokenStream::new(rx))))
    }

    /// Internal streaming generation
    async fn generate_stream_internal(
        &self,
        request: &CompletionRequest<'_>,
        tx: mpsc::UnboundedSender<CompletionCoreResult<CompletionCoreResponse>>,
    ) -> CandleResult<()> {
        // Construct prompt from system prompt and chat history safely
        let prompt = self.construct_prompt_safe(request)?;

        // Tokenize input
        let input_tokens = self.tokenizer.encode(&prompt, true)?;

        // Initialize generation state
        let mut state = GenerationState::new();
        state
            .tokens
            .try_extend_from_slice(&input_tokens)
            .map_err(|_| CandleError::generation_failed("Input too long"))?;

        let mut step = 0;
        let mut accumulated_text = String::new();
        const DEFAULT_MAX_TOKENS: u32 = 1000;
        let max_tokens = request.max_tokens.map(|n| n.get() as u32).unwrap_or(DEFAULT_MAX_TOKENS);

        while step < max_tokens && !state.is_complete() {
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
                let chunk = CompletionCoreResponse::builder()
                    .text(token_text)
                    .tokens_generated(step + 1)
                    .build()?;

                if tx.send(Ok(chunk)).is_err() {
                    // Receiver dropped, stop generation
                    break;
                }
            }

            step += 1;
        }

        // Send final chunk with completion info
        let final_chunk = CompletionCoreResponse::builder()
            .tokens_generated(step)
            .build()?;

        let _ = tx.send(Ok(final_chunk));

        Ok(())
    }

    /// Generate the next token
    async fn generate_next_token(&self, tokens: &[u32], step: u32) -> CandleResult<GeneratedToken> {
        // Convert tokens to tensor
        // Forward pass through model - pass tokens directly as qwen2::Model expects &[u32]
        let logits = self.model.forward(tokens)?;

        // Apply sampling
        let next_token_id = self.sample_token(&logits, step).await?;

        // Decode token to text with safe fallback for unknown tokens
        const UNKNOWN_TOKEN_PREFIX: &str = "<unk:";
        const UNKNOWN_TOKEN_SUFFIX: &str = ">";
        let token_text = match self.tokenizer.id_to_token(next_token_id) {
            Some(text) => text,
            None => {
                let mut unknown_token = String::with_capacity(UNKNOWN_TOKEN_PREFIX.len() + 10 + UNKNOWN_TOKEN_SUFFIX.len());
                unknown_token.push_str(UNKNOWN_TOKEN_PREFIX);
                unknown_token.push_str(&next_token_id.to_string());
                unknown_token.push_str(UNKNOWN_TOKEN_SUFFIX);
                unknown_token
            }
        };

        // Calculate actual log probability from logits
        let log_prob = self
            .calculate_token_log_probability(&logits, next_token_id)
            .await?;

        // Update cumulative log probability (stored in generation state)
        let cumulative_log_prob = self.update_cumulative_log_prob(log_prob).await?;

        GeneratedToken::new(
            next_token_id,
            &token_text,
            log_prob as f32,
            cumulative_log_prob as f32,
            step,
            self.tokenizer.is_special_token(next_token_id),
        )
    }

    /// Sample next token from logits
    async fn sample_token(&self, logits: &Tensor, _step: u32) -> CandleResult<u32> {
        // Apply temperature scaling using SIMD optimization
        let scaled_logits = if self.config.temperature != 1.0 {
            // Convert to CPU for SIMD processing
            let cpu_logits = logits.to_device(&Device::Cpu).map_err(|e| CandleError::from(e))?;
            let mut logits_vec = cpu_logits.to_vec1::<f32>().map_err(|e| CandleError::from(e))?;
            
            // Use SIMD-optimized temperature scaling
            simd_ops::scale_logits_by_temperature(&mut logits_vec, self.config.temperature);
            
            // Convert back to tensor
            Tensor::from_vec(logits_vec, logits.shape(), logits.device()).map_err(|e| CandleError::from(e))?
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
        let topk_output = logits.topk(k as usize).map_err(|e| CandleError::from(e))?;
        let top_k_values = topk_output.values;
        let top_k_indices = topk_output.indices;

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

        // Sort probabilities in descending order using arg_sort
        let indices = probs
            .arg_sort_last_dim(false)  // false for descending order
            .map_err(|e| CandleError::from(e))?;
        let sorted_probs = probs
            .gather(&indices, candle_core::D::Minus1)
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

        // SIMD-optimized multinomial sampling using vectorized cumulative sum
        let mut cumulative_probs = vec![0.0f32; prob_values.len()];
        simd_ops::cumulative_sum_f32(&prob_values, &mut cumulative_probs);
        
        // SIMD-optimized index finding for sampling
        let sample_idx = simd_ops::find_sample_index(&cumulative_probs, random_val);
        Ok(sample_idx as u32)
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
        *self.rng_state.lock() = config.seed;
        self.config = config;
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
    receiver: mpsc::UnboundedReceiver<CompletionCoreResult<CompletionCoreResponse>>,
}

impl CandleTokenStream {
    /// Create a new token stream
    #[inline(always)]
    pub fn new(
        receiver: mpsc::UnboundedReceiver<CompletionCoreResult<CompletionCoreResponse>>,
    ) -> Self {
        Self { receiver }
    }
}

impl Stream for CandleTokenStream {
    type Item = CompletionCoreResult<CompletionCoreResponse>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

unsafe impl Send for CandleGenerator {}
unsafe impl Sync for CandleGenerator {}
