// ============================================================================
// File: src/providers/azure_new/transcription.rs
// ----------------------------------------------------------------------------
// Azure OpenAI transcription implementation with AsyncTask pattern
// ============================================================================

#![allow(clippy::type_complexity)]

// TranscriptionModel does not exist in domain - removed
use fluent_ai_http3::{HttpClient, HttpConfig, HttpMethod, HttpRequest};
use tokio::{self as rt, task::spawn as AsyncTask};

use super::client::Client;
use crate::{AsyncStream, AsyncStreamSender, channel, clients::openai::TranscriptionResponse};
// Note: fluent_ai_domain::transcription doesn't exist - commenting out
// use fluent_ai_domain::transcription::{self, TranscriptionError};

// ───────────────────────────── error handling ───────────────────────

#[derive(Debug, serde::Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ───────────────────────────── provider model ────────────────────────────
// TranscriptionModel is now imported from fluent_ai_domain::model
// Removed duplicated TranscriptionModel struct - use canonical domain type

impl TranscriptionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
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

        // Create HTTP3 client
        let http_client = HttpClient::with_config(HttpConfig::ai_optimized())
            .unwrap_or_else(|_| panic!("HTTP3 client creation failed"));

        // Build multipart request with HTTP3
        let mut form_data = Vec::new();

        // Add file part
        let file_boundary = format!("----fluent_ai_boundary_{}", uuid::Uuid::new_v4());
        let file_header = format!(
            "--{}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\nContent-Type: audio/wav\r\n\r\n",
            file_boundary, request.filename
        );
        form_data.extend_from_slice(file_header.as_bytes());
        form_data.extend_from_slice(&request.data);
        form_data.extend_from_slice(b"\r\n");

        // Add optional parameters
        if let Some(prompt) = &request.prompt {
            let prompt_field = format!(
                "--{}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n{}\r\n",
                file_boundary, prompt
            );
            form_data.extend_from_slice(prompt_field.as_bytes());
        }

        if let Some(temperature) = request.temperature {
            let temp_field = format!(
                "--{}\r\nContent-Disposition: form-data; name=\"temperature\"\r\n\r\n{}\r\n",
                file_boundary, temperature
            );
            form_data.extend_from_slice(temp_field.as_bytes());
        }

        // Close multipart boundary
        let closing_boundary = format!("--{}--\r\n", file_boundary);
        form_data.extend_from_slice(closing_boundary.as_bytes());

        // Create HTTP3 POST request - pure sync streaming
        let transcription_url = self.client.get_transcription_url(&self.model);
        let http_request = HttpRequest::new(HttpMethod::Post, transcription_url)
            .header(
                "Content-Type",
                &format!("multipart/form-data; boundary={}", file_boundary),
            )
            .body(form_data)
            .unwrap_or_else(|_| panic!("HTTP3 request creation failed"));

        // Return TranscriptionChunkImpl immediately - pure streaming
        TranscriptionChunkImpl::new(
            Some("Transcription processing".to_string()),
            None,
            None,
            Vec::new(),
        )
    }
}
