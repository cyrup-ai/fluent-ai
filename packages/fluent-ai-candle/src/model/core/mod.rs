//! Core model implementation with atomic state management
//!
//! This module provides:
//! - Zero-allocation CandleModel with hot-swapping
//! - Atomic state management for model loading/unloading
//! - Integration with KV cache manager
//! - Generation statistics and memory tracking

pub mod candle_model;
pub mod dummy_model;
pub mod model_creators;
pub mod model_loader;
pub mod model_state;
pub mod model_stats;

// Re-export the main CandleModel for ergonomic access
pub use candle_model::CandleModel;