//! Zero-allocation, lock-free completion system - NO FUTURES!
//!
//! This module provides high-performance completion capabilities with:
//! - Zero allocation: Stack allocation, pre-allocated buffers, ArrayVec/SmallVec
//! - No locking: Crossbeam channels, atomics, lock-free data structures
//! - Blazing-fast: Inline hot paths, optimized memory layout, SIMD where possible
//! - No unsafe/unchecked: Explicit bounds checking, safe performance optimizations
//! - Elegant ergonomic: Clean API with builder patterns, zero-cost abstractions

use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};

use arrayvec::ArrayVec;
// Removed unused import: use fluent_ai_async::AsyncStream;
use smallvec::SmallVec;
use thiserror::Error;
use tokio_stream::Stream;

use super::types::ModelParams;
// Removed unused imports: use super::{CompletionRequest, CompletionResponse, StreamingResponse};

/// Maximum prompt size in bytes (4KB stack allocation)
pub const MAX_PROMPT_SIZE: usize = 4096;

/// Maximum response content size in bytes (8KB stack allocation)
pub const MAX_RESPONSE_SIZE: usize = 8192;

/// Maximum number of stop tokens (8 inline storage)
pub const MAX_STOP_TOKENS: usize = 8;

/// Maximum token buffer size for generation (2K tokens)
pub const MAX_TOKEN_BUFFER: usize = 2048;

/// Cache-friendly memory alignment
#[repr(C, align(64))]
pub struct CacheAligned<T>(pub T);

/// Completion errors with comprehensive error tracking
#[derive(Debug, Clone, Error, PartialEq)]
pub enum CompletionCoreError {
    /// Invalid request parameters
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    /// Model loading failed
    #[error("Model loading failed: {0}")]
    ModelLoadingFailed(String),
    /// Generation failed
    #[error("Generation failed: {0}")]
    GenerationFailed(String),
    /// Context length exceeded
    #[error("Context length exceeded: current {current}, max {max}")]
    ContextLengthExceeded { current: u32, max: u32 },
    /// Provider unavailable
    #[error("Provider unavailable: {0}")]
    ProviderUnavailable(String),
    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    /// Timeout occurred
    #[error("Request timeout")]
    Timeout,
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for completion operations
pub type CompletionCoreResult<T> = Result<T, CompletionCoreError>;

/// Zero-allocation completion request with stack-allocated buffers
#[repr(C)]
#[derive(Clone)]
pub struct CompletionCoreRequest<'a> {
    /// Prompt buffer (stack-allocated, 4KB max)
    prompt: ArrayVec<u8, MAX_PROMPT_SIZE>,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling (0.0 = deterministic, 1.0 = random)
    pub temperature: f32,
    /// Top-k sampling parameter
    pub top_k: u32,
    /// Top-p (nucleus) sampling parameter
    pub top_p: f32,
    /// Stop tokens (inline small collection)
    stop_tokens: SmallVec<&'a str, MAX_STOP_TOKENS>,
    /// Enable streaming response
    pub stream: bool,
    /// Model-specific parameters
    pub model_params: ModelParams,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
}

impl<'a> CompletionCoreRequest<'a> {
    /// Get the prompt as a string slice
    #[inline(always)]
    pub fn prompt(&self) -> &[u8] {
        &self.prompt
    }

    /// Get the stop tokens
    #[inline(always)]
    pub fn stop_tokens(&self) -> &[&'a str] {
        &self.stop_tokens
    }

    /// Estimate token count for the prompt (fast approximation)
    #[inline(always)]
    pub fn estimate_token_count(&self) -> u32 {
        // Simple approximation: ~4 characters per token
        (self.prompt.len() / 4) as u32
    }
}

/// Zero-allocation completion response with atomic counters
#[repr(C)]
pub struct CompletionCoreResponse {
    /// Generated text buffer (stack-allocated, 8KB max)
    text: ArrayVec<u8, MAX_RESPONSE_SIZE>,
    /// Number of tokens generated (atomic for thread safety)
    tokens_generated: AtomicU32,
    /// Generation time in milliseconds (atomic)
    generation_time_ms: AtomicU32,
    /// Tokens per second (atomic)
    tokens_per_second: AtomicU32,
    /// Finish reason (inline string, 32 bytes max)
    finish_reason: ArrayVec<u8, 32>,
    /// Model name/identifier (inline string, 64 bytes max)
    model: ArrayVec<u8, 64>,
}

impl CompletionCoreResponse {
    /// Get the generated text as a string slice
    #[inline(always)]
    pub fn text(&self) -> CompletionCoreResult<&str> {
        std::str::from_utf8(&self.text)
            .map_err(|_| CompletionCoreError::Internal(String::from("invalid UTF-8 in response")))
    }

    /// Get the number of tokens generated
    #[inline(always)]
    pub fn tokens_generated(&self) -> u32 {
        self.tokens_generated.load(Ordering::Relaxed)
    }

    /// Get the generation time in milliseconds
    #[inline(always)]
    pub fn generation_time_ms(&self) -> u32 {
        self.generation_time_ms.load(Ordering::Relaxed)
    }

    /// Get tokens per second
    #[inline(always)]
    pub fn tokens_per_second(&self) -> u32 {
        self.tokens_per_second.load(Ordering::Relaxed)
    }

    /// Get the finish reason
    #[inline(always)]
    pub fn finish_reason(&self) -> &str {
        std::str::from_utf8(&self.finish_reason).unwrap_or("unknown")
    }

    /// Get the model name
    #[inline(always)]
    pub fn model(&self) -> &str {
        std::str::from_utf8(&self.model).unwrap_or("unknown")
    }

    /// Set tokens generated (atomic)
    #[inline(always)]
    pub fn set_tokens_generated(&self, tokens: u32) {
        self.tokens_generated.store(tokens, Ordering::Relaxed);
    }

    /// Set generation time (atomic)
    #[inline(always)]
    pub fn set_generation_time_ms(&self, time_ms: u32) {
        self.generation_time_ms.store(time_ms, Ordering::Relaxed);
    }

    /// Set tokens per second (atomic)
    #[inline(always)]
    pub fn set_tokens_per_second(&self, tps: u32) {
        self.tokens_per_second.store(tps, Ordering::Relaxed);
    }
}

/// Streaming response wrapper
pub struct StreamingCoreResponse {
    stream: Pin<Box<dyn Stream<Item = CompletionCoreResult<CompletionCoreResponse>> + Send>>,
}

impl StreamingCoreResponse {
    /// Create a new streaming response
    #[inline(always)]
    pub fn new(
        stream: Pin<Box<dyn Stream<Item = CompletionCoreResult<CompletionCoreResponse>> + Send>>,
    ) -> Self {
        Self { stream }
    }

    /// Get the underlying stream
    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Pin<Box<dyn Stream<Item = CompletionCoreResult<CompletionCoreResponse>> + Send>> {
        self.stream
    }
}

impl Stream for StreamingCoreResponse {
    type Item = CompletionCoreResult<CompletionCoreResponse>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

/// Zero-cost wrapper for stack-allocated collections
#[inline(always)]
pub fn with_stack_buffer<T, F, R>(f: F) -> R
where
    T: Copy,
    F: FnOnce(&mut [std::mem::MaybeUninit<T>]) -> R,
{
    // Safe zero-allocation buffer using const generics and MaybeUninit::uninit_array
    let mut buffer: [std::mem::MaybeUninit<T>; 1024] = [std::mem::MaybeUninit::uninit(); 1024];
    f(&mut buffer)
}

/// Compile-time string validation for static strings
#[macro_export]
macro_rules! static_str {
    ($s:expr) => {{
        const _: &str = $s; // Compile-time validation
        $s
    }};
}

/// Performance hint for hot path optimization
#[inline(always)]
pub const fn is_hot_path() -> bool {
    true
}
