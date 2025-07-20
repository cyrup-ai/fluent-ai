// ============================================================================
// File: src/providers/azure_new/transcription.rs
// ----------------------------------------------------------------------------
// Azure OpenAI transcription implementation with AsyncTask pattern
// ============================================================================

#![allow(clippy::type_complexity)]

use fluent_ai_domain::model::TranscriptionModel;
use reqwest::multipart::Part;

use super::client::Client;
use crate::{
    clients::openai::TranscriptionResponse,
    runtime::{self as rt, AsyncTask},
    transcription::{self, TranscriptionError},
};

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
    async fn perform_transcription(
        self,
        request: transcription::TranscriptionRequest,
    ) -> Result<
        transcription::TranscriptionResponse<TranscriptionResponse>,
        transcription::TranscriptionError,
    > {
        let data = request.data;

        let mut body = reqwest::multipart::Form::new().part(
            "file",
            Part::bytes(data).file_name(request.filename.clone()),
        );

        if let Some(prompt) = request.prompt {
            body = body.text("prompt", prompt.clone());
        }

        if let Some(ref temperature) = request.temperature {
            body = body.text("temperature", temperature.to_string());
        }

        if let Some(ref additional_params) = request.additional_params {
            for (key, value) in additional_params
                .as_object()
                .expect("Additional Parameters to Azure Transcription should be a map")
            {
                body = body.text(key.to_owned(), value.to_string());
            }
        }

        let response = self
            .client
            .post_transcription(&self.model)
            .multipart(body)
            .send()
            .await?;

        if response.status().is_success() {
            match response
                .json::<ApiResponse<TranscriptionResponse>>()
                .await?
            {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(api_error_response) => Err(TranscriptionError::ProviderError(
                    api_error_response.message,
                )),
            }
        } else {
            Err(TranscriptionError::ProviderError(response.text().await?))
        }
    }
}
