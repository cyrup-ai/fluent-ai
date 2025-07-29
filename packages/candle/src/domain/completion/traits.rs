//! Core completion traits - EXACT REPLICA of domain with Candle prefixes
//!
//! Contains CandleCompletionModel and CandleCompletionBackend traits that exactly match
//! domain/src/completion/core.rs with zero over-engineering.

use fluent_ai_async::AsyncStream;
use super::{
    request::CompletionRequest,
    response::CompletionResponse,
    types::CandleCompletionParams,
};
use crate::domain::completion::{CandleCompletionRequest, CandleCompletionResponse};
use crate::domain::{
    context::chunk::CompletionChunk,
    prompt::CandlePrompt,
};
use crate::domain::completion::CandleCompletionChunk;

/// Core trait for completion models - EXACT REPLICA of domain CompletionModel
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

/// Backend for completion processing - EXACT REPLICA of domain CompletionBackend
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
    ) -> fluent_ai_async::AsyncTask<CandleCompletionResponse<'a>>;
}

// Backward compatibility trait alias for existing code
pub trait CandleCompletionProvider: CandleCompletionModel {}

// Blanket implementation
impl<T: CandleCompletionModel> CandleCompletionProvider for T {}