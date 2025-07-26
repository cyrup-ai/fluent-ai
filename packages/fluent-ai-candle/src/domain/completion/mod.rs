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
pub mod chunk;
pub mod core;
pub mod model;
pub mod request;
pub mod response;
pub mod types;

// Re-export commonly used Candle types for convenience
pub use core::{CandleCompletionBackend, CandleCompletionModel};

pub use candle::{
    CompletionCoreError, CompletionCoreRequest, CompletionCoreResponse, CompletionCoreResult,
    StreamingCoreResponse};

// Type aliases for convenience
pub type CandleCompletionResult<T> = CompletionCoreResult<T>;
pub type CandleStreamingResponse = StreamingCoreResponse;
pub use request::{CompletionRequest as CandleCompletionRequest, CompletionRequestError as CandleCompletionRequestError};
pub use response::{CompactCompletionResponse as CandleCompactCompletionResponse, CompletionResponse as CandleCompletionResponse};
pub use types::{CandleCompletionChunk, CandleCompletionParams, CandleModelParams, CandleToolDefinition};
