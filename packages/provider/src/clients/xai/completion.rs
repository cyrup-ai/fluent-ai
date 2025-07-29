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
use fluent_ai_http3::{Http3, HttpError};
use serde_json::{Value, json};
use arrayvec::ArrayVec;

// CRITICAL: Import model information from model-info package (single source of truth)
use model_info::{Provider, XaiProvider, ModelInfo as ModelInfoFromPackage};

use super::client::Client;
use crate::completion_provider::{CompletionError, CompletionResponse as DomainCompletionResponse};
use crate::json_util;
use crate::streaming::StreamingCompletionResponse;

// Model constants removed - use model-info package exclusively
// All xAI model information is provided by ./packages/model-info

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
        let mut messages = ArrayVec::new();

        // Add preamble as system message if present
        if let Some(preamble) = &completion_request.preamble {
            messages
                .try_push(XaiMessage {
                    role: "system",
                    content: XaiContent::Text(preamble)})
                .map_err(|_| CompletionError::RequestError("Request too large".to_string()))?;
        }

        // Add documents as context
        if let Some(docs) = completion_request.normalized_documents() {
            for doc in docs {
                let content = format!("Document: {}", doc.content());
                messages
                    .try_push(XaiMessage {
                        role: "user",
                        content: XaiContent::Text(Box::leak(content.into_boxed_str()))})
                    .map_err(|_| CompletionError::RequestError("Request too large".to_string()))?;
            }
        }

        // Add chat history
        for msg in completion_request.chat_history {
            match msg.role() {
                fluent_ai_domain::message::MessageRole::User => {
                    if let Some(text) = msg.content().text() {
                        messages
                            .try_push(XaiMessage {
                                role: "user",
                                content: XaiContent::Text(text)})
                            .map_err(|_| CompletionError::RequestError("Request too large".to_string()))?;
                    }
                }
                fluent_ai_domain::message::MessageRole::Assistant => {
                    if let Some(text) = msg.content().text() {
                        messages
                            .try_push(XaiMessage {
                                role: "assistant",
                                content: XaiContent::Text(text)})
                            .map_err(|_| CompletionError::RequestError("Request too large".to_string()))?;
                    }
                }
                fluent_ai_domain::message::MessageRole::System => {
                    if let Some(text) = msg.content().text() {
                        messages
                            .try_push(XaiMessage {
                                role: "system",
                                content: XaiContent::Text(text)})
                            .map_err(|_| CompletionError::RequestError("Request too large".to_string()))?;
                    }
                }
            }
        }

        // Set parameters with direct validation - zero allocation
        let temperature = completion_request.temperature.map(|temp| temp.clamp(0.0, 2.0));
        let max_tokens = completion_request.max_tokens.map(|tokens| tokens.clamp(1, 131072));

        // Add tools if present
        let tools = if !completion_request.tools.is_empty() {
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
            Some(xai_tools)
        } else {
            None
        };

        Ok(XaiChatRequest {
            model: &self.model,
            messages,
            temperature,
            max_tokens,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            tools,
            stream: Some(false),
        })
    }
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string()}
    }

    /// Load model information from model-info package (single source of truth)
    pub fn load_model_info(&self) -> fluent_ai_async::AsyncStream<ModelInfoFromPackage> {
        let provider = Provider::Xai(XaiProvider);
        provider.get_model_info(&self.model)
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
        // Use Http3::json() directly instead of client abstraction
        let completion_response: XaiChatResponse = Http3::json()
            .api_key(&self.client.api_key())
            .body(&request_body)
            .post(&format!("{}/v1/chat/completions", self.client.base_url()))
            .collect()
            .await
            .map_err(|e| CompletionError::HttpError(e.to_string()))?;

        tracing::info!(target: "rig",
            "XAI completion token usage: {:?}",
            completion_response.usage.clone().map(|usage| format!("{usage}")).unwrap_or_else(|| "N/A".to_string())
        );

        Ok(DomainCompletionResponse {
            raw_response: completion_response.clone(),
            content: completion_response.try_into()?,
            token_usage: None, // Token usage is in the raw_response
            metadata: Default::default(),
        })
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingResponse<Self::StreamingResponse>, CompletionError> {
        let mut request_body = self.create_completion_request(request)?;

        // Enable streaming in the centralized request
        request_body.stream = Some(true);

        // Use Http3::json() for streaming with direct SSE handling
        let mut response_stream = Http3::json()
            .api_key(&self.client.api_key())
            .body(&request_body)
            .post(&format!("{}/v1/chat/completions", self.client.base_url()));

        if !response_stream.is_success() {
            return Err(CompletionError::HttpError("Request failed".to_string()));
        }

        // Create streaming response using the streaming module
        let sse_stream = response_stream.sse();
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

// =============================================================================
// Model Constants (for compatibility with existing imports)
// =============================================================================

/// Grok-3 model
pub const GROK_3: &str = "grok-3";
