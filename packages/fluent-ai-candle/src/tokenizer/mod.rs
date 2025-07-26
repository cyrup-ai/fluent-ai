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
