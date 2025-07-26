//! Completion-related builders extracted from fluent_ai_domain

pub mod candle_completion_builder;
pub mod completion_request_builder;
pub mod completion_response_builder;

// Re-export for convenience
pub use candle_completion_builder::{CandleCompletionCoreRequestBuilder, CandleCompletionCoreResponseBuilder};
pub use completion_request_builder::{CandleCompletionRequestBuilder, CandleCompletionRequestError};
pub use completion_response_builder::{CandleCompletionResponseBuilder, CandleCompactCompletionResponseBuilder};