// ============================================================================
// File: src/providers/perplexity/streaming.rs
// ----------------------------------------------------------------------------
// Perplexity streaming types (uses OpenAI-compatible streaming)
// ============================================================================

// Perplexity uses OpenAI-compatible streaming, so we re-export that
pub use crate::providers::openai::StreamingCompletionResponse;
