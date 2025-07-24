// ============================================================================
// File: src/providers/perplexity/completion.rs
// ----------------------------------------------------------------------------
// Perplexity completion types and implementation
// ============================================================================

// Import centralized HTTP structs - no more local definitions!
use fluent_ai_http_structs::{
    builders::{ChatBuilder, Http3Builders, HttpRequestBuilder},
    common::{AuthMethod, ContentTypes, HttpHeaders, HttpUtils, Provider},
    errors::{HttpStructError, HttpStructResult},
    perplexity::{
        PerplexityChatRequest, PerplexityChatResponse, PerplexityChoice, PerplexityContent,
        PerplexityMessage, PerplexityResponseMessage, PerplexityStreamingChunk, PerplexityUsage,
    },
    validation::{ValidateRequest, ValidationResult},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::client::Client;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionModel, CompletionRequest},
    json_util, message,
};

// ================================================================
// Perplexity Completion API
// ================================================================
/// `sonar-pro` completion model - default for Perplexity
pub const SONAR_PRO: &str = "sonar-pro";
/// `sonar` completion model
pub const SONAR: &str = "sonar";

/// Legacy alias for centralized Perplexity response type
pub type CompletionResponse = fluent_ai_http_structs::perplexity::PerplexityChatResponse;

/// Legacy alias for centralized message type
pub type Message = fluent_ai_http_structs::perplexity::PerplexityMessage;

/// Legacy alias for centralized role type  
pub type Role = fluent_ai_http_structs::perplexity::PerplexityRole;

/// Legacy alias for centralized delta type
pub type Delta = fluent_ai_http_structs::perplexity::PerplexityDelta;

/// Legacy alias for centralized choice type
pub type Choice = fluent_ai_http_structs::perplexity::PerplexityChoice;

/// Legacy alias for centralized usage type
pub type Usage = fluent_ai_http_structs::perplexity::PerplexityUsage;

// Display implementation is provided by centralized PerplexityUsage type

impl TryFrom<CompletionResponse> for completion::CompletionResponse {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        match &choice.message {
            message
                if message.role
                    == fluent_ai_http_structs::perplexity::PerplexityRole::Assistant =>
            {
                Ok(completion::CompletionResponse {
                    choice: OneOrMany::one(message.content.clone().into()),
                    raw_response: response,
                })
            }
            _ => Err(CompletionError::ResponseError(
                "Response contained no assistant message".to_owned(),
            )),
        }
    }
}

/// Perplexity completion model implementation
#[derive(Debug, Clone)]
pub struct PerplexityCompletionModel {
    client: Client,
    model: String,
}

impl PerplexityCompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<PerplexityChatRequest<'_>, CompletionError> {
        // Use the centralized builder with validation
        let builder = Http3Builders::perplexity();
        let mut chat_builder = builder.chat(&self.model);

        // Add preamble as system message if present
        if let Some(preamble) = &completion_request.preamble {
            chat_builder = chat_builder.add_text_message("system", preamble);
        }

        // Add documents as context
        if let Some(docs) = completion_request.normalized_documents() {
            for doc in docs {
                let content = format!("Document: {}", doc.content());
                chat_builder = chat_builder.add_text_message("user", &content);
            }
        }

        // Add chat history
        for msg in completion_request.chat_history {
            match msg.role() {
                fluent_ai_domain::message::MessageRole::User => {
                    if let Some(text) = msg.content().text() {
                        chat_builder = chat_builder.add_text_message("user", text);
                    }
                }
                fluent_ai_domain::message::MessageRole::Assistant => {
                    if let Some(text) = msg.content().text() {
                        chat_builder = chat_builder.add_text_message("assistant", text);
                    }
                }
                fluent_ai_domain::message::MessageRole::System => {
                    if let Some(text) = msg.content().text() {
                        chat_builder = chat_builder.add_text_message("system", text);
                    }
                }
            }
        }

        // Set parameters with validation using centralized utilities
        if let Some(temp) = completion_request.temperature {
            chat_builder = chat_builder.temperature(
                HttpUtils::validate_temperature(temp as f32, Provider::Perplexity).map_err(|e| {
                    CompletionError::InvalidRequest(format!("Invalid temperature: {}", e))
                })? as f64,
            );
        }

        if let Some(max_tokens) = completion_request.max_tokens {
            chat_builder = chat_builder.max_tokens(
                HttpUtils::validate_max_tokens(max_tokens, Provider::Perplexity).map_err(|e| {
                    CompletionError::InvalidRequest(format!("Invalid max_tokens: {}", e))
                })?,
            );
        }

        // Build and validate the request
        match chat_builder.build() {
            Ok(request) => Ok(request),
            Err(e) => Err(CompletionError::InvalidRequest(format!(
                "Request building failed: {}",
                e
            ))),
        }
    }
}

impl TryFrom<message::Message> for Message {
    type Error = fluent_ai_domain::message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        Ok(match message {
            message::Message::User { content } => {
                let collapsed_content = content
                    .into_iter()
                    .map(|content| match content {
                        message::UserContent::Text(message::Text { text }) => Ok(text),
                        _ => Err(fluent_ai_domain::message::MessageError::ConversionError(
                            "Only text content is supported by Perplexity".to_owned(),
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join("\n");

                fluent_ai_http_structs::perplexity::PerplexityMessage {
                    role: fluent_ai_http_structs::perplexity::PerplexityRole::User,
                    content: collapsed_content,
                }
            }

            message::Message::Assistant { content } => {
                let collapsed_content = content
                    .into_iter()
                    .map(|content| {
                        Ok(match content {
                            message::AssistantContent::Text(message::Text { text }) => text,
                            _ => return Err(fluent_ai_domain::message::MessageError::ConversionError(
                                "Only text assistant message content is supported by Perplexity"
                                    .to_owned(),
                            )),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join("\n");

                fluent_ai_http_structs::perplexity::PerplexityMessage {
                    role: fluent_ai_http_structs::perplexity::PerplexityRole::Assistant,
                    content: collapsed_content,
                }
            }
        })
    }
}

impl From<Message> for message::Message {
    fn from(message: Message) -> Self {
        match message.role {
            fluent_ai_http_structs::perplexity::PerplexityRole::User => {
                message::Message::user(message.content)
            }
            fluent_ai_http_structs::perplexity::PerplexityRole::Assistant => {
                message::Message::assistant(message.content)
            }
            // System messages get coerced into user messages for ease of error handling.
            // They should be handled on the outside of `Message` conversions via the preamble.
            fluent_ai_http_structs::perplexity::PerplexityRole::System => {
                message::Message::user(message.content)
            }
        }
    }
}

impl completion::CompletionModel for PerplexityCompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = crate::providers::openai::StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let request_body = self.create_completion_request(completion_request)?;

        // Use centralized serialization
        let body_bytes = match serde_json::to_vec(&request_body) {
            Ok(bytes) => bytes,
            Err(e) => {
                return Err(CompletionError::Internal(format!(
                    "Serialization error: {}",
                    e
                )));
            }
        };

        let response = self
            .client
            .post("/chat/completions")
            .body(body_bytes)
            .header("content-type", "application/json")
            .send()
            .await?;

        if response.status().is_success() {
            match response
                .json::<super::client::ApiResponse<CompletionResponse>>()
                .await?
            {
                super::client::ApiResponse::Ok(completion) => {
                    tracing::info!(target: "rig",
                        "Perplexity completion token usage: {}",
                        completion.usage
                    );
                    Ok(completion.try_into()?)
                }
                super::client::ApiResponse::Err(error) => {
                    Err(CompletionError::ProviderError(error.message))
                }
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        let mut request_body = self.create_completion_request(completion_request)?;

        // Enable streaming in the centralized request
        request_body.stream = Some(true);

        // Use centralized serialization
        let body_bytes = match serde_json::to_vec(&request_body) {
            Ok(bytes) => bytes,
            Err(e) => {
                return Err(CompletionError::Internal(format!(
                    "Serialization error: {}",
                    e
                )));
            }
        };

        let builder = self
            .client
            .post("/chat/completions")
            .body(body_bytes)
            .header("content-type", "application/json");

        crate::providers::openai::send_compatible_streaming_request(builder).await
    }
}
