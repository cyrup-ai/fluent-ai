pub mod streaming;

// Re-export streaming types
pub use streaming::*;

// Common streaming types that are used throughout the codebase
pub type StreamingResultDyn =
    Box<dyn std::future::Future<Output = Result<String, Box<dyn std::error::Error>>> + Send>;
pub type AsyncStreamDyn =
    Box<dyn futures::Stream<Item = Result<String, Box<dyn std::error::Error>>> + Send + Unpin>;

// Export streaming completion response types
pub use crate::completion::StreamingCompletionResponse;
