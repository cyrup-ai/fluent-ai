use base64::Engine;
use base64::prelude::BASE64_STANDARD;
// TranscriptionModel does not exist in domain - removed
use serde::Deserialize;
use serde_json::json;
use super::client::HuggingFaceClient as Client;
use super::completion::ApiResponse;
use crate::transcription;
use crate::transcription::TranscriptionError;
use fluent_ai_http3::{Http3, header};

pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "openai/whisper-large-v3-turbo";
pub const WHISPER_SMALL: &str = "openai/whisper-small";

#[derive(Debug, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String}

impl TryFrom<TranscriptionResponse>
    for transcription::TranscriptionResponse<TranscriptionResponse>
{
    type Error = TranscriptionError;

    fn try_from(value: TranscriptionResponse) -> Result<Self, Self::Error> {
        Ok(transcription::TranscriptionResponse {
            text: value.text.clone(),
            response: value})
    }
}

// TranscriptionModel is now imported from fluent_ai_domain::model
// Removed duplicated TranscriptionModel struct - use canonical domain type

/// HuggingFace transcription model implementation
#[derive(Debug, Clone)]
pub struct HuggingFaceTranscriptionModel {
    client: Client,
    model: String}

impl TranscriptionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string()}
    }
}
impl transcription::TranscriptionModel for TranscriptionModel {
    type Response = TranscriptionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<transcription::TranscriptionResponse<Self::Response>, TranscriptionError> {
        let data = request.data;
        let data = BASE64_STANDARD.encode(data);

        let request = json!({
            "inputs": data
        });

        let route = self
            .client
            .sub_provider
            .transcription_endpoint(&self.model)?;
        let url = format!("{}{}", self.client.base_url(), route);
        
        let result = Http3::json()
            .bearer_auth(&self.client.api_key())
            .body(&request)
            .post(&url)
            .collect_or_else(|e| {
                tracing::error!(target: "rig", "HuggingFace transcription request failed: {}", e);
                Err(TranscriptionError::ProviderError(e.to_string()))
            })
            .and_then(|response: TranscriptionResponse| {
                tracing::debug!(target: "rig", "HuggingFace transcription response: {:?}", response);
                response.try_into()
            });
            
        result
    }
}
