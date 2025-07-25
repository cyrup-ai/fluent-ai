//! CandleCompletionClient module with decomposed architecture
//!
//! This module provides a production-ready completion client for the Candle ML framework
//! with zero-allocation patterns and lock-free design. The client has been decomposed
//! into focused modules for better maintainability while preserving the original API.

pub mod builder;
pub mod completion;
pub mod config;

// Re-export core types for public API compatibility
pub use builder::{CandleClientBuilder, HasPrompt, NeedsPrompt};
pub use completion::{CandleCompletionClient, CandleMetrics, CANDLE_METRICS};
pub use config::{
    CandleClientConfig, DeviceType, ModelArchitecture, ModelConfig, QuantizationType,
    MAX_DOCUMENTS, MAX_MESSAGES, MAX_TOOLS,
};

// Type aliases for backward compatibility
pub type Message = crate::types::CandleMessage;
pub type Document = crate::types::CandleDocument;
pub type CompletionRequest = crate::types::CandleCompletionRequest;
pub type CompletionResponse<'a> = crate::types::CandleCompletionResponse<'a>;
pub type StreamingResponse = crate::types::CandleStreamingResponse;