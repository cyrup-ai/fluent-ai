// ============================================================================
// File: src/providers/deepseek/mod.rs
// ----------------------------------------------------------------------------
// DeepSeek provider implementation with typestate builder pattern
// ============================================================================

mod client;
mod completion;
mod streaming;

pub use client::{Client, DeepSeekCompletionBuilder};
pub use completion::DeepSeekCompletionModel;

// Re-export model constants
pub use completion::{DEEPSEEK_CHAT, DEEPSEEK_REASONER};
