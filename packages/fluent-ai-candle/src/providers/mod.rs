//! Candle Model Providers
//!
//! This module provides model providers for the Candle ML framework integration.
//! All providers implement the CandleCompletionModel trait for consistent streaming inference.

pub mod kimi_k2;
pub mod tokenizer;

// Re-export primary provider types
pub use kimi_k2::{CandleKimiK2Provider, CandleKimiK2Config};
pub use tokenizer::{CandleTokenizer, CandleTokenizerConfig};