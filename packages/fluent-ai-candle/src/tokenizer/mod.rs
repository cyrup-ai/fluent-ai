//! Tokenizer Module - DECOMPOSED FROM 650 LINES
//!
//! This module provides HuggingFace tokenizer integration for Candle ML Framework
//! with high-performance tokenization, Hub integration, and zero-allocation patterns.
//!
//! The original 650-line tokenizer.rs has been decomposed into focused modules:
//! - config: Configuration types and builders (120 lines)
//! - core: Core tokenizer implementation (170 lines)
//! - special_tokens: Special token management (100 lines)
//! - encoding: Text encoding operations (80 lines)
//! - decoding: Token decoding operations (50 lines)
//! - utils: Utility functions and helpers (70 lines)

pub mod config;
pub mod core;
pub mod special_tokens;
pub mod encoding;
pub mod decoding;
pub mod utils;

// Re-export main types for ergonomic API
pub use config::{
    TokenizerConfig, TokenizerConfigBuilder, PaddingConfig, 
    TruncationConfig, TruncationStrategy
};

pub use core::{CandleTokenizer, MAX_TOKEN_BUFFER};

pub use utils::{
    load_popular_tokenizer, config_for_model_type, validate_tokenizer
};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tokenizer_config_builder() {
        let config = TokenizerConfigBuilder::new()
            .add_bos_token(true)
            .add_eos_token(false)
            .max_length(Some(1024))
            .build();

        assert!(config.add_bos_token);
        assert!(!config.add_eos_token);
        assert_eq!(config.max_length, Some(1024));
    }

    #[test]
    fn test_chat_message() {
        let msg = crate::types::CandleMessage::system("You are a helpful assistant");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are a helpful assistant");

        let user_msg = crate::types::CandleMessage::user("Hello world");
        assert_eq!(user_msg.role, "user");
        assert_eq!(user_msg.content, "Hello world");
    }

    #[test]
    fn test_utils() {
        let config = utils::config_for_model_type("llama");
        assert!(config.add_bos_token);
        assert!(!config.add_eos_token);

        let config = utils::config_for_model_type("phi");
        assert!(!config.add_bos_token);
        assert!(config.add_eos_token);
    }
}