//! Production-ready DeepSeek API provider implementation
//!
//! This module provides a complete, battle-tested implementation for DeepSeek's API
//! with zero-allocation patterns, robust error handling, and comprehensive feature support.
//!
//! ## Features
//! - Full DeepSeek model support (chat, reasoner, v3, r1) with proper token handling
//! - Complete message/content type system with function calling
//! - Tool/function calling with DeepSeek function definitions
//! - Proper error handling with DeepSeek API response envelope
//! - JSON payload building with zero-alloc merge_inplace
//! - Streaming support with SSE decoding
//! - Usage token tracking (prompt/completion tokens)
//! - Model configuration through ModelInfo defaults

pub mod client;
pub mod completion;
pub mod streaming;

// Explicit re-exports to avoid ambiguity
pub use client::{DeepSeekClient, DeepSeekProvider};
pub use completion::{
    DEEPSEEK_CHAT, DEEPSEEK_R1, DEEPSEEK_REASONER, DEEPSEEK_V3, DeepSeekChoice,
    DeepSeekCompletionBuilder, DeepSeekCompletionRequest, DeepSeekDelta, DeepSeekFunction,
    DeepSeekFunctionDelta, DeepSeekMessage, DeepSeekStreamChunk, DeepSeekToolCall,
    DeepSeekToolCallDelta, DeepSeekUsage, available_models, completion_builder, get_model_config,
};
