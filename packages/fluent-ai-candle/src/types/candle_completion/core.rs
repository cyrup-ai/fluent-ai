//! Core completion traits and domain types
//!
//! Contains the fundamental traits and interfaces for completion functionality.

use fluent_ai_async::AsyncStream;

use super::completion_response::CompletionResponse;
use super::request::CandleCompletionRequest;
use crate::types::{CandleCompletionChunk, CandleCompletionParams};

/// Core trait for completion models
pub trait CandleCompletionModel: Send + Sync + 'static {
    /// Generate completion from prompt
    ///
    /// # Arguments
    /// * `prompt` - The input prompt for generation
    /// * `params` - Generation parameters
    ///
    /// # Returns
    /// Stream of completion chunks
    fn prompt(
        &self,
        prompt: &str,
        params: &CandleCompletionParams,
    ) -> AsyncStream<CandleCompletionChunk>;
}

/// Backend for completion processing
pub trait CandleCompletionBackend: Send + Sync + 'static {
    /// Submit a completion request
    ///
    /// # Arguments
    /// * `request` - The completion request
    ///
    /// # Returns
    /// Async task that resolves to the completion result
    fn submit_completion<'a>(
        &'a self,
        request: CandleCompletionRequest,
    ) -> fluent_ai_async::AsyncTask<CompletionResponse<'a>>;
}
