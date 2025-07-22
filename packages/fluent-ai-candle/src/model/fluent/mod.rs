//! Fluent AI model implementations
//!
//! This module contains the production-ready model implementations
//! for the fluent-ai-candle framework. Currently supports:
//! - Kimi K2: 1T parameter MoE model with 32B activated parameters

pub mod kimi_k2;

// Re-export the main components
pub use kimi_k2::{
    example::KimiK2Example,
    loader::{load_model, LoaderEvent, ModelShard},
    model::{KimiK2Config, KimiK2Model},
    tokenizer::{ChatMessage, KimiK2Tokenizer},
    KimiK2Config as Config, QuantFormat, KIMI_K2_FP16, KIMI_K2_FP8,
};