//! Production-ready HuggingFace API provider implementation
//!
//! This module provides a complete, battle-tested implementation for HuggingFace's API
//! with zero-allocation patterns, robust error handling, and comprehensive feature support.
//!
//! ## Features
//! - Full Llama/Gemma/Qwen model support with proper token handling
//! - Complete message/content type system with function calling
//! - Vision support for image analysis (Qwen-VL models)
//! - Tool/function calling with HuggingFace function definitions
//! - Proper error handling with HuggingFace API response envelope
//! - JSON payload building with zero-alloc merge_inplace
//! - Streaming support with SSE decoding
//! - Usage token tracking (prompt/completion tokens)
//! - Model configuration through ModelInfo defaults

pub mod client;
pub mod completion;

#[cfg(feature = "image")]
pub mod image_generation;
pub mod streaming;
pub mod transcription;

// Explicit re-exports to avoid ambiguity
pub use client::HuggingFaceProvider as SubProvider;
pub use client::{HuggingFaceClient, HuggingFaceClient as Client, HuggingFaceProvider};
pub use completion::HuggingFaceCompletionBuilder as HuggingfaceCompletionBuilder;
pub use completion::{
    GEMMA_2, HuggingFaceChoice, HuggingFaceCompletionBuilder, HuggingFaceCompletionRequest,
    HuggingFaceDelta, HuggingFaceFunction, HuggingFaceFunctionDelta, HuggingFaceMessage,
    HuggingFaceStreamChunk, HuggingFaceToolCall, HuggingFaceToolCallDelta, HuggingFaceUsage,
    META_LLAMA_3_1, PHI_4, QWEN_QVQ_PREVIEW, QWEN2_5, QWEN2_5_CODER, QWEN2_VL,
    SMALLTHINKER_PREVIEW, available_models, completion_builder, get_model_config,
};
#[cfg(feature = "image")]
pub use image_generation::{FLUX_1, KOLORS, STABLE_DIFFUSION_3};
pub use transcription::{WHISPER_LARGE_V3, WHISPER_LARGE_V3_TURBO, WHISPER_SMALL};
