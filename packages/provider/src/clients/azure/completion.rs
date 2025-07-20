// ============================================================================
// File: src/providers/azure_new/completion.rs
// ----------------------------------------------------------------------------
// Azure OpenAI completion implementation with AsyncTask pattern
// ============================================================================

#![allow(clippy::type_complexity)]

use std::{convert::Infallible, str::FromStr};

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::client::Client;
use crate::{
    OneOrMany,
    clients::openai::{self, TranscriptionResponse, send_compatible_streaming_request},
    completion::CompletionRequest,
    completion::{self, CompletionError, StreamingCompletionResponse as RigStreaming},
    json_util::{self, merge},
    message::{self, MessageError},
    runtime::{self as rt, AsyncTask},
    streaming::StreamingCompletionResponse,
};

// ───────────────────────────── public constants ──────────────────────────

/// `o1` completion model
pub const O1: &str = "o1";
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-mini` completion model
pub const O1_MINI: &str = "o1-mini";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini` completion model
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4o-realtime-preview` completion model
pub const GPT_4O_REALTIME_PREVIEW: &str = "gpt-4o-realtime-preview";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";
/// `gpt-3.5-turbo-16k` completion model
pub const GPT_35_TURBO_16K: &str = "gpt-3.5-turbo-16k";

// ───────────────────────────── response helpers ───────────────────────

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ───────────────────────────── provider model ────────────────────────────

pub use crate::Models as CompletionModel;

impl CompletionModel {
    #[inline(always)]
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_owned(),
        }
    }
}

// ───────────────────────────── impl CompletionModel ─────────────────────

impl completion::CompletionModel for CompletionModel {
    type Response = openai::CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> AsyncTask<Result<completion::CompletionResponse<Self::Response>, CompletionError>> {
        let this = self.clone();
        rt::spawn_async(async move { this.perform_completion(req).await })
    }

    fn stream(
        &self,
        req: CompletionRequest,
    ) -> AsyncTask<Result<RigStreaming<Self::StreamingResponse>, CompletionError>> {
        let this = self.clone();
        rt::spawn_async(async move { this.perform_stream(req).await })
    }
}

// ───────────────────────────── internal async helpers ───────────────────

impl CompletionModel {
    // Actual POST call – lifted into an internal method so the public trait stays sync.
    async fn perform_completion(
        self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let response = self
            .client
            .post_chat_completion(&self.model)
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let t = response.text().await?;
            tracing::debug!(target: "rig", "Azure completion response: {}", t);

            match serde_json::from_str::<ApiResponse<openai::CompletionResponse>>(&t)? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "Azure completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                    );
                    response.try_into()
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }

    // streaming variant – re-uses OpenAI compatibility layer
    async fn perform_stream(
        self,
        completion_request: CompletionRequest,
    ) -> Result<RigStreaming<openai::StreamingCompletionResponse>, CompletionError> {
        let mut request = self.create_completion_request(completion_request)?;

        request = merge(
            request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let builder = self
            .client
            .post_chat_completion(self.model.as_str())
            .json(&request);

        send_compatible_streaming_request(builder).await
    }

    // helper: create the outbound JSON without sending – reused by both paths
    fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<serde_json::Value, CompletionError> {
        let mut full_history: Vec<openai::Message> = match &completion_request.preamble {
            Some(preamble) => vec![openai::Message::system(preamble)],
            None => vec![],
        };

        if let Some(docs) = completion_request.normalized_documents() {
            let docs: Vec<openai::Message> = docs.try_into()?;
            full_history.extend(docs);
        }

        let chat_history: Vec<openai::Message> = completion_request
            .chat_history
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<openai::Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        let request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": completion_request.tools.into_iter().map(openai::ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        let request = if let Some(params) = completion_request.additional_params {
            json_util::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }
}
