//! Processing context integration for sophisticated sampling strategies
//!
//! This module provides context management for logits processing with:
//! - Integration with fluent-ai-simd ProcessingContext
//! - Extension for candle-specific processing needs
//! - Token history tracking with bounded memory usage
//! - Zero allocation patterns with stack-based storage
//! - Production-ready error handling

pub mod context_core;
pub mod context_builder;
pub mod context_stats;
pub mod context_utils;
pub mod context_analysis;

// Re-export core types for convenience
pub use context_core::{
    ProcessingContext, BaseProcessingContext, 
    MAX_VOCAB_SIZE, MAX_CONTEXT_SIZE, DEFAULT_CONTEXT_SIZE
};
pub use context_builder::ContextBuilder;
pub use context_stats::{RepetitionPenaltyType, SequenceStats};

// Re-export utility functions
pub use context_utils::{
    text_generation_context, code_generation_context, conversation_context,
    optimal_context_size, validate_context_config
};
