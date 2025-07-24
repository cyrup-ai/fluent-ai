//! Fluent AI model implementations
//!
//! This module contains the production-ready model implementations
//! for the fluent-ai-candle framework. Currently supports:
//! - Kimi K2: 1T parameter MoE model with 32B activated parameters

pub mod kimi_k2;

// Re-export the main components
pub use kimi_k2::{
    KimiK2Config as Config, QuantFormat, kimi_k2_fp8, kimi_k2_fp16,
    loader::{LoaderEvent, ModelShard, load_model},
    model::{KimiK2Config, KimiK2Model},
    tokenizer::KimiK2Tokenizer,
};
