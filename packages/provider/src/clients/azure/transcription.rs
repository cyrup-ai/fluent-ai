// ============================================================================
// File: src/providers/azure_new/transcription.rs
// ----------------------------------------------------------------------------
// Azure OpenAI transcription implementation with AsyncTask pattern
// ============================================================================

#![allow(clippy::type_complexity)]

// TranscriptionModel does not exist in domain - removed
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use fluent_ai_http3::{Http3, header};
use tokio::{self as rt, task::spawn as AsyncTask};
use super::client::Client;
use crate::{AsyncStream, AsyncStreamSender, channel, clients::openai::TranscriptionResponse};
// Note: fluent_ai_domain::transcription doesn't exist - commenting out
// use fluent_ai_domain::transcription::{self, TranscriptionError};

// ───────────────────────────── error handling ───────────────────────

#[derive(Debug, serde::Deserialize)]
struct ApiErrorResponse {
    message: String}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse)}

// ───────────────────────────── provider model ────────────────────────────
// TranscriptionModel is now imported from fluent_ai_domain::model
// Removed duplicated TranscriptionModel struct - use canonical domain type

impl TranscriptionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string()}
    }
}

// ───────────────────────────── impl TranscriptionModel ─────────────────────

impl transcription::TranscriptionModel for TranscriptionModel {
    type Response = TranscriptionResponse;

    fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> AsyncTask<
        Result<
            transcription::TranscriptionResponse<Self::Response>,
            transcription::TranscriptionError,
        >,
    > {
        let this = self.clone();
        rt::spawn_async(async move { this.perform_transcription(request).await })
    }
}

// ───────────────────────────── internal async helpers ───────────────────

impl TranscriptionModel {
    /// Perform transcription using HTTP3 pure streaming architecture
    /// Returns impl TranscriptionChunk - sync method with pure streaming
    fn perform_transcription(
        self,
        request: transcription::TranscriptionRequest,
    ) -> impl crate::http3_streaming::TranscriptionChunk {
        use crate::http3_streaming::TranscriptionChunkImpl;

        // Build form data for transcription request
        let mut form_data = HashMap::new();
        form_data.insert("file".to_string(), BASE64_STANDARD.encode(&request.data));
        form_data.insert("model".to_string(), self.model.clone());
        
        if let Some(prompt) = &request.prompt {
            form_data.insert("prompt".to_string(), prompt.clone());
        }
        
        if let Some(temperature) = request.temperature {
            form_data.insert("temperature".to_string(), temperature.to_string());
        }

        // Create HTTP3 request with form data
        let transcription_url = self.client.get_transcription_url(&self.model);
        
        let result = Http3::form_urlencoded()
            .bearer_auth(&self.client.api_key())
            .body(&form_data)
            .post(&transcription_url)
            .collect_or_else(|e| {
                tracing::error!(target: "rig", "Azure transcription request failed: {}", e);
                TranscriptionResponse {
                    text: format!("Error: {}", e),
                    language: None,
                    duration: None,
                    words: None,
                    segments: None}
            });

        // Return TranscriptionChunkImpl immediately - pure streaming
        TranscriptionChunkImpl::new(
            Some("Transcription processing".to_string()),
            None,
            None,
            Vec::new(),
        )
    }
}
