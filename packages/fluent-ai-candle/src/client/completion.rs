//! Completion methods and streaming functionality
//!
//! This module contains the completion-related methods for CandleCompletionClient,
//! including both single-shot and streaming completion implementations.

use std::sync::atomic::Ordering;

use fluent_ai_async::{AsyncStream, handle_error};

use super::core::CandleCompletionClient;
use crate::types::{
    CandleCompletionError, CandleCompletionRequest, CandleCompletionResponse,
    CandleStreamingResponse,
};

// Type aliases for local use
type CompletionRequest = CandleCompletionRequest;
type CompletionResponse<'a> = CandleCompletionResponse<'a>;

impl CandleCompletionClient {
    /// Generate completion with zero allocation - SYNCHRONOUS ONLY
    #[inline(always)]
    pub fn complete(
        &self,
        _request: CompletionRequest,
    ) -> AsyncStream<CompletionResponse<'static>> {
        let client = self.clone();

        AsyncStream::with_channel(move |sender| {
            // Check if client is initialized
            if !client.is_initialized() {
                let error = CandleCompletionError::InvalidRequest {
                    message: "Client not initialized".to_string(),
                };
                handle_error!(error, "Client not initialized");
            }

            // Update concurrent request counter - ATOMIC OPERATION
            client
                .metrics
                .concurrent_requests
                .fetch_add(1, Ordering::Relaxed);

            // TODO: Implement synchronous generation method
            // For now, handle error since sync generation is not implemented
            client.record_request_stats(false, 0, false);

            // Decrement concurrent counter
            client
                .metrics
                .concurrent_requests
                .fetch_sub(1, Ordering::Relaxed);

            // Handle error instead of sending Result
            handle_error!(
                CandleCompletionError::GenerationFailed {
                    reason: "Synchronous generation not yet implemented".to_string(),
                },
                "Synchronous generation not implemented"
            );
        })
    }

    /// Generate streaming completion with prompt - RETURNS UNWRAPPED AsyncStream
    #[inline(always)]
    pub fn prompt(&self, prompt_text: &str) -> AsyncStream<CandleStreamingResponse> {
        let client = self.clone();
        let prompt = prompt_text.to_string();

        AsyncStream::with_channel(move |sender| {
            // Check if client is initialized
            if !client.is_initialized() {
                log::error!(
                    "Stream error in {}: Client not initialized. Details: {}",
                    file!(),
                    "Client not initialized"
                );
                return;
            }

            // Build request
            let _request = match CompletionRequest::builder().system_prompt(prompt).build() {
                Ok(req) => req,
                Err(e) => {
                    log::error!(
                        "Stream error in {}: Request building failed. Details: {}",
                        file!(),
                        format!("Failed to build request: {}", e)
                    );
                    return;
                }
            };

            // Update concurrent request counter - ATOMIC OPERATION
            client
                .metrics
                .concurrent_requests
                .fetch_add(1, Ordering::Relaxed);

            // TODO: Generate streaming response synchronously
            // For now, create a placeholder streaming response since async calls are not allowed
            let streaming_response = CandleStreamingResponse::new(
                "placeholder-streaming".to_string(),
                "candle-placeholder".to_string(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            );

            let _ = sender.send(streaming_response);

            // Decrement concurrent counter
            client
                .metrics
                .concurrent_requests
                .fetch_sub(1, Ordering::Relaxed);
        })
    }

    /// Get the model name/identifier for this client
    pub fn model_name(&self) -> &'static str {
        "candle-model"
    }

    /// Record request statistics (internal method)
    pub(super) fn record_request_stats(&self, success: bool, tokens_generated: u32, is_streaming: bool) {
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);

        if success {
            self.metrics
                .successful_requests
                .fetch_add(1, Ordering::Relaxed);
            self.metrics
                .total_tokens_generated
                .fetch_add(tokens_generated as usize, Ordering::Relaxed);

            if is_streaming {
                self.metrics
                    .streaming_requests
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                self.metrics
                    .batch_requests
                    .fetch_add(1, Ordering::Relaxed);
            }
        } else {
            self.metrics
                .failed_requests
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

unsafe impl Send for CandleCompletionClient {}
unsafe impl Sync for CandleCompletionClient {}