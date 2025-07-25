// ================================================================
//! xAI Completion Integration
//! From [xAI Reference](https://docs.x.ai/docs/api-reference#chat-completions)
// ================================================================

// Import centralized HTTP structs - no more local definitions!
// Re-export the domain CompletionModel trait
pub use fluent_ai_domain::completion::CompletionModel;
use fluent_ai_domain::completion::{self, CompletionRequest};
use super::types::{
    XaiChatRequest, XaiChatResponse, XaiChoice, XaiContent, XaiFunction, XaiMessage,
    XaiResponseMessage, XaiStreamingChunk, XaiTool, XaiUsage};
use fluent_ai_http3::{Http3, HttpResult};
use crate::utils::{HttpUtils, Provider};
use serde_json::{Value, json};

use super::client::Client;
use crate::completion_provider::{CompletionError, CompletionResponse as DomainCompletionResponse};
use crate::json_util;
use crate::streaming::StreamingCompletionResponse;

/// xAI completion models as of 2025-06-04
pub const GROK_2_1212: &str = "grok-2-1212";
pub const GROK_2_VISION_1212: &str = "grok-2-vision-1212";
pub const GROK_3: &str = "grok-3";
pub const GROK_3_FAST: &str = "grok-3-fast";
pub const GROK_3_MINI: &str = "grok-3-mini";
pub const GROK_3_MINI_FAST: &str = "grok-3-mini-fast";
pub const GROK_2_IMAGE_1212: &str = "grok-2-image-1212";

// =================================================================
// Rig Implementation Types
// =================================================================

/// xAI completion model implementation
#[derive(Clone)]
pub struct XaiCompletionModel {
    client: Client,
    model: String}

impl XaiCompletionModel {
    pub(crate) fn create_completion_request(
        &self,
        completion_request: fluent_ai_domain::completion::CompletionRequest,
    ) -> Result<XaiChatRequest<'_>, CompletionError> {
        // Use the centralized builder with validation
        let builder = Http3Builders::xai();
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
                HttpUtils::validate_temperature(temp as f32, Provider::XAI).map_err(|e| {
                    CompletionError::RequestError(format!("Invalid temperature: {}", e))
                })? as f64,
            );
        }

        if let Some(max_tokens) = completion_request.max_tokens {
            chat_builder = chat_builder.max_tokens(
                HttpUtils::validate_max_tokens(max_tokens, Provider::XAI).map_err(|e| {
                    CompletionError::RequestError(format!("Invalid max_tokens: {}", e))
                })?,
            );
        }

        // Add tools if present
        if !completion_request.tools.is_empty() {
            let mut xai_tools = arrayvec::ArrayVec::new();
            for tool in completion_request.tools.into_iter() {
                if xai_tools.len() < super::types::MAX_TOOLS {
                    let xai_tool = XaiTool {
                        tool_type: "function",
                        function: XaiFunction {
                            name: tool.name(),
                            description: tool.description(),
                            parameters: tool.parameters().clone()}};
                    let _ = xai_tools.push(xai_tool);
                }
            }
            chat_builder = chat_builder.with_tools(xai_tools);
        }

        // Build and validate the request
        match chat_builder.build() {
            Ok(request) => Ok(request),
            Err(e) => Err(CompletionError::RequestError(format!(
                "Request building failed: {}",
                e
            )))}
    }
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string()}
    }
}

impl completion::CompletionModel for XaiCompletionModel {
    type Response = XaiChatResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: fluent_ai_domain::completion::CompletionRequest,
    ) -> Result<DomainCompletionResponse<XaiChatResponse>, CompletionError> {
        let request_body = self.create_completion_request(completion_request)?;

        // Use centralized serialization
        let body_bytes = match serde_json::to_vec(&request_body) {
            Ok(bytes) => bytes,
            Err(e) => {
                return Err(CompletionError::RequestError(format!(
                    "Serialization error: {}",
                    e
                )));
            }
        };

        let response = self
            .client
            .make_request("v1/chat/completions", body_bytes)
            .await
            .map_err(|e| CompletionError::HttpError(e.to_string()))?;

        if response.status().is_success() {
            let body = response.body();
            let completion_response: XaiChatResponse =
                serde_json::from_slice(body).map_err(|e| {
                    CompletionError::ResponseError(format!("Deserialization error: {}", e))
                })?;

            tracing::info!(target: "rig",
                "XAI completion token usage: {:?}",
                completion_response.usage.clone().map(|usage| format!("{usage}")).unwrap_or_else(|| "N/A".to_string())
            );

            Ok(DomainCompletionResponse {
                raw_response: completion_response.clone(),
                content: completion_response.try_into()?,
                token_usage: None, // Token usage is in the raw_response
                metadata: Default::default()})
        } else {
            let error_body = String::from_utf8_lossy(response.body());
            Err(CompletionError::ProviderError(error_body.to_string()))
        }
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingResponse<Self::StreamingResponse>, CompletionError> {
        let mut request_body = self.create_completion_request(request)?;

        // Enable streaming in the centralized request
        request_body.stream = Some(true);

        // Use centralized serialization
        let body_bytes = match serde_json::to_vec(&request_body) {
            Ok(bytes) => bytes,
            Err(e) => {
                return Err(CompletionError::RequestError(format!(
                    "Serialization error: {}",
                    e
                )));
            }
        };

        let response = self
            .client
            .make_request("v1/chat/completions", body_bytes)
            .await
            .map_err(|e| CompletionError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let error_body = String::from_utf8_lossy(response.body());
            return Err(CompletionError::ProviderError(error_body.to_string()));
        }

        // Create streaming response using the streaming module
        let sse_stream = response.sse();
        Ok(crate::streaming::StreamingResponse::from_sse_stream(
            sse_stream,
        ))
    }
}

impl TryFrom<XaiChatResponse> for crate::completion_provider::ZeroOneOrMany<String> {
    type Error = CompletionError;

    fn try_from(response: XaiChatResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        // Extract text content from the response
        if let Some(content) = &choice.message.content {
            Ok(crate::completion_provider::ZeroOneOrMany::One(
                content.clone(),
            ))
        } else {
            Err(CompletionError::ResponseError(
                "Response did not contain valid text content".to_string(),
            ))
        }
    }
}

// TODO: Implement local XAI types to replace unauthorized fluent_ai_http_structs
// All type aliases referencing fluent_ai_http_structs have been removed
