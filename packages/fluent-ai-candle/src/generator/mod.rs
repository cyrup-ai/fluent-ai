//! Zero-allocation text generation with streaming support and SIMD optimization

pub mod core;
pub mod simd;
pub mod types;

// Re-export public types and functions
pub use core::CandleGenerator;
pub use simd::{scale_logits_by_temperature, cumulative_sum_f32, find_sample_index};
pub use types::{
    GenerationConfig, GeneratedToken, GenerationStats, GenerationState, StopReason,
    MAX_GENERATION_BUFFER};

// Type aliases for compatibility
use crate::types::{
    CandleCompletionRequest, CandleCompletionResponse, CandleStreamingResponse};

pub type CompletionRequest = CandleCompletionRequest;
pub type CompletionResponse<'a> = CandleCompletionResponse<'a>;
pub type StreamingResponse = CandleStreamingResponse;