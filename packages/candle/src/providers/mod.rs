//! Candle Model Providers
//!
//! This module provides model providers for the Candle ML framework integration.
//! All providers implement the CandleCompletionModel trait for consistent streaming inference.

pub mod kimi_k2;
pub mod qwen3_coder;
pub mod tokenizer; // tokenizers dependency is available in Cargo.toml

// Re-export primary provider types
pub use kimi_k2::{CandleKimiK2Config, CandleKimiK2Provider};
pub use qwen3_coder::{CandleQwen3CoderConfig, CandleQwen3CoderProvider};
pub use tokenizer::{CandleTokenizer, CandleTokenizerConfig};
