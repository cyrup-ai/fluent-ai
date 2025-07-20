//! Core completion traits and domain types
//!
//! Contains the fundamental traits and interfaces for completion functionality.

use crate::async_task::AsyncStream;
use crate::chunk::CompletionChunk;
use crate::prompt::Prompt;

use super::types::CompletionParams;
use super::request::CompletionRequest;
use super::response::CompletionResponse;

/// Core trait for completion models
pub trait CompletionModel: Send + Sync + 'static {
    /// Generate completion from prompt
    ///
    /// # Arguments
    /// * `prompt` - The input prompt for generation
    /// * `params` - Generation parameters
    ///
    /// # Returns
    /// Stream of completion chunks
    fn prompt<'a>(
        &'a self,
        prompt: Prompt<'a>,
        params: &'a CompletionParams,
    ) -> AsyncStream<CompletionChunk<'a>>;
}

/// Backend for completion processing
pub trait CompletionBackend: Send + Sync + 'static {
    /// Submit a completion request
    ///
    /// # Arguments
    /// * `request` - The completion request
    ///
    /// # Returns
    /// Async task that resolves to the completion result
    fn submit_completion<'a>(
        &'a self,
        request: CompletionRequest<'a>,
    ) -> crate::async_task::AsyncTask<CompletionResponse<'a>>;
}
