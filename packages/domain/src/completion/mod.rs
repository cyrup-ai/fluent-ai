//! Completion module - Consolidated completion functionality
//!
//! This module consolidates completion-related functionality from completion.rs and candle_completion.rs
//! into a clean, unified module structure with zero-allocation patterns and production-ready functionality.
//!
//! ## Architecture
//! - `core.rs` - Core completion traits and domain types
//! - `request.rs` - Completion request types and builders
//! - `response.rs` - Completion response types and builders
//! - `candle.rs` - Zero-allocation, lock-free completion system
//! - `types.rs` - Shared types and constants

pub mod candle;
pub mod core;
pub mod request;
pub mod response;
pub mod types;

// Re-export commonly used types for convenience
pub use core::{CompletionBackend, CompletionModel};

pub use candle::{
    CompletionCoreError, CompletionCoreRequest, CompletionCoreResponse, CompletionCoreResult,
    StreamingCoreResponse};

// Type aliases for convenience
pub type CompletionResult<T> = CompletionCoreResult<T>;
pub type StreamingResponse = StreamingCoreResponse;
pub use request::{CompletionRequest, CompletionRequestError};
pub use response::{CompactCompletionResponse, CompletionResponse};
pub use types::{CompletionParams, ModelParams, ToolDefinition};
