//! Core types and configuration for text generation

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use arrayvec::ArrayVec;
use smallvec::SmallVec;

use crate::error::{CandleError, CandleResult};

/// Maximum generation buffer size
pub const MAX_GENERATION_BUFFER: usize = 4096;

/// Generation configuration
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
    pub do_sample: bool}

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
            do_sample: true}
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
    pub is_special: bool}

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
            is_special})
    }

    /// Get token text as string
    #[inline(always)]
    pub fn text_str(&self) -> CandleResult<&str> {
        std::str::from_utf8(&self.text)
            .map_err(|_| CandleError::generation_failed("Invalid UTF-8 in token text"))
    }
}

/// Statistics for generation performance tracking
#[repr(C)]
#[derive(Debug)]
pub struct GenerationStats {
    /// Total tokens generated
    pub tokens_generated: AtomicU32,
    /// Generation time in microseconds
    pub generation_time_us: AtomicU64,
    /// Tokens per second throughput
    pub tokens_per_second: AtomicU32,
    /// Cache hit count
    pub cache_hits: AtomicU32,
    /// Cache miss count
    pub cache_misses: AtomicU32,
    /// Memory usage in bytes
    pub memory_usage: AtomicU64}

impl Clone for GenerationStats {
    fn clone(&self) -> Self {
        Self {
            tokens_generated: AtomicU32::new(self.tokens_generated.load(Ordering::Relaxed)),
            generation_time_us: AtomicU64::new(self.generation_time_us.load(Ordering::Relaxed)),
            tokens_per_second: AtomicU32::new(self.tokens_per_second.load(Ordering::Relaxed)),
            cache_hits: AtomicU32::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU32::new(self.cache_misses.load(Ordering::Relaxed)),
            memory_usage: AtomicU64::new(self.memory_usage.load(Ordering::Relaxed))}
    }
}

impl GenerationStats {
    /// Create new generation stats with zero values
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            tokens_generated: AtomicU32::new(0),
            generation_time_us: AtomicU64::new(0),
            tokens_per_second: AtomicU32::new(0),
            cache_hits: AtomicU32::new(0),
            cache_misses: AtomicU32::new(0),
            memory_usage: AtomicU64::new(0)}
    }
}

impl Default for GenerationStats {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
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
    /// Generation statistics for performance tracking
    pub stats: GenerationStats}

impl Clone for GenerationState {
    fn clone(&self) -> Self {
        Self {
            tokens: self.tokens.clone(),
            generated_tokens: self.generated_tokens.clone(),
            position: self.position,
            is_complete: AtomicBool::new(self.is_complete.load(Ordering::Relaxed)),
            stop_reason: parking_lot::Mutex::new(self.stop_reason.lock().clone()),
            stats: self.stats.clone()}
    }
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
            stats: GenerationStats::new()}
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
    LengthLimit = 5}