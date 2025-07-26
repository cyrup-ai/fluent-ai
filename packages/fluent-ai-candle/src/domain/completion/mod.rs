//! Candle completion module - Consolidated completion functionality
//!
//! This module consolidates Candle completion-related functionality from completion.rs and candle_completion.rs
//! into a clean, unified module structure with zero-allocation patterns and production-ready functionality.
//!
//! ## Architecture
//! - `core.rs` - Core Candle completion traits and domain types
//! - `request.rs` - Candle completion request types and builders
//! - `response.rs` - Candle completion response types and builders
//! - `candle.rs` - Zero-allocation, lock-free Candle completion system
//! - `types.rs` - Shared Candle types and constants

pub mod candle;
pub mod core;
pub mod request;
pub mod response;
pub mod types;

// Re-export commonly used Candle types for convenience
pub use core::{CandleCompletionBackend, CandleCompletionModel};

pub use candle::{
    CandleCompletionCoreError, CandleCompletionCoreRequest, CandleCompletionCoreResponse, CandleCompletionCoreResult,
    CandleStreamingCoreResponse};

// Type aliases for convenience
pub type CandleCompletionResult<T> = CandleCompletionCoreResult<T>;
pub type CandleStreamingResponse = CandleStreamingCoreResponse;
pub use request::{CandleCompletionRequest, CandleCompletionRequestError};
pub use response::{CandleCompactCompletionResponse, CandleCompletionResponse};
pub use types::{CandleCompletionChunk, CandleCompletionParams, CandleModelParams, CandleToolDefinition};
