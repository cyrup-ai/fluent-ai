//! Utility functions for context management
//!
//! Provides convenience functions for creating contexts optimized for
//! specific use cases and memory constraints.

use super::context_builder::ContextBuilder;
use super::context_core::{ProcessingContext, MAX_CONTEXT_SIZE, MAX_VOCAB_SIZE};
use crate::processing::error::ProcessingError;
use crate::processing::traits::ProcessingResult;

/// Create context for text generation
#[inline(always)]
pub fn text_generation_context(
    vocab_size: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> ProcessingResult<ProcessingContext> {
    ContextBuilder::new(vocab_size)
        .context_size(2048)
        .temperature(temperature)
        .top_k(top_k)
        .top_p(top_p)
        .build()
}

/// Create context for code generation
#[inline(always)]
pub fn code_generation_context(vocab_size: usize) -> ProcessingResult<ProcessingContext> {
    ContextBuilder::new(vocab_size)
        .context_size(4096) // Longer context for code
        .temperature(0.2) // Low temperature for precision
        .top_k(Some(20)) // Focused vocabulary
        .top_p(Some(0.95)) // High nucleus threshold
        .build()
}

/// Create context for conversation
#[inline(always)]
pub fn conversation_context(vocab_size: usize) -> ProcessingResult<ProcessingContext> {
    ContextBuilder::new(vocab_size)
        .context_size(1024) // Moderate context
        .temperature(0.7) // Balanced temperature
        .top_k(Some(40)) // Balanced vocabulary
        .top_p(Some(0.9)) // Standard nucleus sampling
        .build()
}

/// Calculate optimal context size for given constraints
#[inline(always)]
pub fn optimal_context_size(vocab_size: usize, target_memory_mb: usize) -> usize {
    // Estimate memory usage per token in context
    let bytes_per_token = std::mem::size_of::<u32>(); // Token ID
    let bytes_per_freq = std::mem::size_of::<u32>(); // Frequency counter
    let overhead = vocab_size * bytes_per_freq; // Frequency array

    let available_bytes = (target_memory_mb * 1024 * 1024).saturating_sub(overhead);
    let max_tokens = available_bytes / bytes_per_token;

    max_tokens.min(MAX_CONTEXT_SIZE)
}

/// Validate context configuration
pub fn validate_context_config(vocab_size: usize, context_size: usize) -> ProcessingResult<()> {
    if vocab_size == 0 || vocab_size > MAX_VOCAB_SIZE {
        return Err(ProcessingError::configuration(format!(
            "Invalid vocabulary size: {}",
            vocab_size
        )));
    }

    if context_size == 0 || context_size > MAX_CONTEXT_SIZE {
        return Err(ProcessingError::configuration(format!(
            "Invalid context size: {}",
            context_size
        )));
    }

    Ok(())
}