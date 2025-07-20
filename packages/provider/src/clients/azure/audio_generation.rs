// ============================================================================
// File: src/providers/azure_new/audio_generation.rs
// ----------------------------------------------------------------------------
// Azure OpenAI audio generation implementation with AsyncTask pattern
// ============================================================================

#![allow(clippy::type_complexity)]

use bytes::Bytes;
use serde_json::json;

use super::client::Client;
use crate::{
    audio_generation::{
        self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
    },
    client::AudioGenerationClient,
    runtime::{self, AsyncTask},
};

// ───────────────────────────── provider model ────────────────────────────

#[derive(Clone)]
pub struct AudioGenerationModel {
    client: Client,
    pub model: String,
}

impl AudioGenerationModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

// ───────────────────────────── impl AudioGenerationModel ─────────────────────

impl audio_generation::AudioGenerationModel for AudioGenerationModel {
    type Response = Bytes;

    fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> AsyncTask<Result<AudioGenerationResponse<Self::Response>, AudioGenerationError>> {
        let this = self.clone();
        runtime::spawn_async(async move { this.perform_audio_generation(request).await })
    }
}

// ───────────────────────────── internal async helpers ───────────────────

impl AudioGenerationModel {
    async fn perform_audio_generation(
        self,
        request: AudioGenerationRequest,
    ) -> Result<AudioGenerationResponse<Bytes>, AudioGenerationError> {
        let request = json!({
            "model": self.model,
            "input": request.text,
            "voice": request.voice,
            "speed": request.speed,
        });

        let response = self
            .client
            .post_audio_generation(&self.model)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(AudioGenerationError::ProviderError(format!(
                "{}: {}",
                response.status(),
                response.text().await?
            )));
        }

        let bytes = response.bytes().await?;

        Ok(AudioGenerationResponse {
            audio: bytes.to_vec(),
            response: bytes,
        })
    }
}

// ───────────────────────────── client implementation ───────────────────

impl AudioGenerationClient for Client {
    type AudioGenerationModel = AudioGenerationModel;

    fn audio_generation_model(&self, model: &str) -> Self::AudioGenerationModel {
        AudioGenerationModel::new(self.clone(), model)
    }
}
