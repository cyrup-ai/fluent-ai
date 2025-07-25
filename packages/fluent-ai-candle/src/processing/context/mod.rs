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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let context = ProcessingContext::new(1000, 512);
        assert!(context.is_ok());

        let ctx = context.unwrap();
        assert_eq!(ctx.vocab_size(), 1000);
        assert_eq!(ctx.context_size(), 512);
        assert_eq!(ctx.position(), 0);
        assert_eq!(ctx.sequence_length(), 0);
    }

    #[test]
    fn test_context_builder() {
        let context = ContextBuilder::new(1000)
            .context_size(256)
            .temperature(0.8)
            .top_k(Some(50))
            .top_p(Some(0.9))
            .build();

        assert!(context.is_ok());
        let ctx = context.unwrap();
        assert_eq!(ctx.vocab_size(), 1000);
        assert_eq!(ctx.context_size(), 256);
    }

    #[test]
    fn test_token_operations() {
        let mut context = ProcessingContext::new(100, 50).unwrap();
        
        // Test adding tokens
        assert!(context.add_token(10).is_ok());
        assert!(context.add_token(20).is_ok());
        assert!(context.add_token(10).is_ok()); // Repeat token
        
        assert_eq!(context.sequence_length(), 3);
        assert_eq!(context.position(), 3);
        assert_eq!(context.token_frequency(10), 2);
        assert_eq!(context.token_frequency(20), 1);
        assert!(context.has_token(10));
        assert!(context.has_token(20));
        assert!(!context.has_token(30));
    }

    #[test]
    fn test_utility_functions() {
        let context = text_generation_context(1000, 0.7, Some(40), Some(0.9));
        assert!(context.is_ok());

        let context = code_generation_context(1000);
        assert!(context.is_ok());

        let context = conversation_context(1000);
        assert!(context.is_ok());

        let optimal_size = optimal_context_size(1000, 10);
        assert!(optimal_size > 0);

        assert!(validate_context_config(1000, 512).is_ok());
        assert!(validate_context_config(0, 512).is_err());
        assert!(validate_context_config(1000, 0).is_err());
    }

    #[test]
    fn test_repetition_patterns() {
        let mut context = ProcessingContext::new(100, 50).unwrap();
        
        // Add pattern: 1, 2, 3, 1, 2, 3
        let tokens = vec![1, 2, 3, 1, 2, 3];
        assert!(context.add_tokens(&tokens).is_ok());
        
        assert!(context.has_repetition_pattern(3));
        assert!(!context.has_repetition_pattern(2));
        assert!(!context.has_repetition_pattern(4));
    }

    #[test]
    fn test_diversity_metrics() {
        let mut context = ProcessingContext::new(100, 50).unwrap();
        
        // Add diverse tokens
        let tokens = vec![1, 2, 3, 4, 5];
        assert!(context.add_tokens(&tokens).is_ok());
        
        assert_eq!(context.unique_token_count(), 5);
        assert!((context.diversity_score() - 1.0).abs() < 0.01);
        
        // Add repeated tokens
        let more_tokens = vec![1, 1, 1, 1, 1];
        assert!(context.add_tokens(&more_tokens).is_ok());
        
        assert_eq!(context.unique_token_count(), 5);
        assert!(context.diversity_score() < 1.0);
    }
}