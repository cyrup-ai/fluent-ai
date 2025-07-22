// ============================================================================
// File: src/providers/azure_new/streaming.rs
// ----------------------------------------------------------------------------
// Azure OpenAI streaming implementation with AsyncTask pattern
// ============================================================================

#![allow(clippy::type_complexity)]

use fluent_ai_domain::completion::StreamingCoreResponse as RigStreaming;
use fluent_ai_domain::completion::{CompletionCoreError, CompletionRequest};
use serde_json; // Note: merge function needs to be implemented if needed
use serde_json::json;
use tokio::{self as rt, task::spawn as AsyncTask};

use super::completion::CompletionModel;
use crate::clients::openai::{self, send_compatible_streaming_request};

// ─────────────────────────── streaming response type ───────────────────

#[derive(Clone)]
pub struct StreamingCompletionResponse {
    // Azure uses OpenAI compatibility, so we can just re-export
    pub inner: openai::StreamingCompletionResponse,
}

impl From<openai::StreamingCompletionResponse> for StreamingCompletionResponse {
    fn from(inner: openai::StreamingCompletionResponse) -> Self {
        Self { inner }
    }
}

// ───────────────────────── CompletionModel::stream ──────────────────────

impl CompletionModel {
    /// Public sync façade: returns **one** AsyncTask that resolves to a
    /// `StreamingCompletionResponse`, hiding *all* async machinery.
    #[inline]
    pub fn stream(
        &self,
        req: CompletionRequest,
    ) -> AsyncTask<Result<RigStreaming<StreamingCompletionResponse>, CompletionError>> {
        rt::spawn_async(self.clone().drive_stream(req))
    }

    // ---------------- internal async driver (NOT public) ----------------

    async fn drive_stream(
        self,
        completion_request: CompletionRequest,
    ) -> Result<RigStreaming<StreamingCompletionResponse>, CompletionError> {
        // Azure uses OpenAI-compatible streaming API
        let mut request = self.create_completion_request(completion_request)?;

        request = merge(
            request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let builder = self
            .client
            .post_chat_completion(self.model.as_str())
            .json(&request);

        // Use OpenAI's compatible streaming implementation
        let openai_response = send_compatible_streaming_request(builder).await?;

        // Convert to our Azure-specific response type
        Ok(RigStreaming::stream(Box::pin(
            openai_response.into_stream().map(|result| {
                result.map(|choice| choice.map_response(StreamingCompletionResponse::from))
            }),
        )))
    }
}
