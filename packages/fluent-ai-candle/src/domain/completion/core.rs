//! Core Candle completion traits and domain types
//!
//! Contains the fundamental traits and interfaces for Candle completion functionality.

use fluent_ai_async::AsyncStream;

use super::request::CompletionRequest;
use super::response::CompletionResponse;
use super::types::CandleCompletionParams;
use super::types::CandleCompletionChunk;
use crate::prompt::CandlePrompt;

/// Core trait for Candle completion models
pub trait CandleCompletionModel: Send + Sync + 'static {
    /// Generate completion from prompt
    ///
    /// # Arguments
    /// * `prompt` - The input prompt for generation
    /// * `params` - Generation parameters
    ///
    /// # Returns
    /// Stream of completion chunks
    fn prompt(&self, prompt: CandlePrompt, params: &CandleCompletionParams) -> AsyncStream<CandleCompletionChunk>;
}

/// Backend for Candle completion processing
pub trait CandleCompletionBackend: Send + Sync + 'static {
    /// Submit a Candle completion request
    ///
    /// # Arguments
    /// * `request` - The Candle completion request
    ///
    /// # Returns
    /// Async task that resolves to the Candle completion result
    fn submit_completion<'a>(
        &'a self,
        request: CandleCompletionRequest,
    ) -> fluent_ai_async::AsyncTask<CandleCompletionResponse<'a>>;
}
