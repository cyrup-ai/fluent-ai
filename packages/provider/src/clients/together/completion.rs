// ================================================================
//! Together AI Completion Integration
//! From [Together AI Reference](https://docs.together.ai/docs/chat-overview)
// ================================================================

// ================================================================
// Together Completion Models
// ================================================================

// Import centralized HTTP structs - no more local definitions!
// Re-export the domain CompletionModel trait
pub use fluent_ai_domain::CompletionModel;
use fluent_ai_domain::completion::{CompletionCoreError as CompletionError, CompletionRequest};
use super::types::{
    TogetherChatRequest, TogetherChatResponse, TogetherChoice, TogetherContent,
    TogetherFunction, TogetherMessage, TogetherResponseMessage, TogetherStreamingChunk,
    TogetherTool, TogetherUsage};
use serde_json::json;
use arrayvec::ArrayVec;
use fluent_ai_http3::Http3;

// CRITICAL: Import model information from model-info package (single source of truth)
use model_info::{Provider, TogetherProvider, ModelInfo as ModelInfoFromPackage};

use super::client::{Client, together_ai_api_types::ApiResponse};
use crate::streaming::StreamingCompletionResponse;
use crate::{clients::openai, json_util};

// Model constants removed - use model-info package exclusively
// All Together AI model information is provided by ./packages/model-info

// =================================================================
// Rig Implementation Types
// =================================================================

/// Together AI completion model implementation
#[derive(Clone)]
pub struct TogetherCompletionModel {
    client: Client,
    model: String}

impl TogetherCompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string()}
    }

    /// Load model information from model-info package (single source of truth)
    pub fn load_model_info(&self) -> fluent_ai_async::AsyncStream<ModelInfoFromPackage> {
        let provider = Provider::Together(TogetherProvider);
        provider.get_model_info(&self.model)
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<TogetherChatRequest<'_>, CompletionError> {
        let mut messages = ArrayVec::new();

        // Add preamble as system message if present
        if let Some(preamble) = &completion_request.preamble {
            messages
                .try_push(TogetherMessage {
                    role: "system",
                    content: TogetherContent::Text(preamble)})
                .map_err(|_| CompletionError::InvalidRequest("Request too large".to_string()))?;
        }

        // Add documents as context
        if let Some(docs) = completion_request.normalized_documents() {
            for doc in docs {
                let content = format!("Document: {}", doc.content());
                messages
                    .try_push(TogetherMessage {
                        role: "user",
                        content: TogetherContent::Text(Box::leak(content.into_boxed_str()))})
                    .map_err(|_| CompletionError::InvalidRequest("Request too large".to_string()))?;
            }
        }

        // Add chat history
        for msg in completion_request.chat_history {
            match msg.role() {
                fluent_ai_domain::message::MessageRole::User => {
                    if let Some(text) = msg.content().text() {
                        messages
                            .try_push(TogetherMessage {
                                role: "user",
                                content: TogetherContent::Text(text)})
                            .map_err(|_| CompletionError::InvalidRequest("Request too large".to_string()))?;
                    }
                }
                fluent_ai_domain::message::MessageRole::Assistant => {
                    if let Some(text) = msg.content().text() {
                        messages
                            .try_push(TogetherMessage {
                                role: "assistant",
                                content: TogetherContent::Text(text)})
                            .map_err(|_| CompletionError::InvalidRequest("Request too large".to_string()))?;
                    }
                }
                fluent_ai_domain::message::MessageRole::System => {
                    if let Some(text) = msg.content().text() {
                        messages
                            .try_push(TogetherMessage {
                                role: "system",
                                content: TogetherContent::Text(text)})
                            .map_err(|_| CompletionError::InvalidRequest("Request too large".to_string()))?;
                    }
                }
            }
        }

        // Set parameters with direct validation - zero allocation
        let temperature = completion_request.temperature.map(|temp| temp.clamp(0.0, 2.0));
        let max_tokens = completion_request.max_tokens.map(|tokens| tokens.clamp(1, 8192));

        // Add tools if present
        let tools = if !completion_request.tools.is_empty() {
            let mut together_tools = arrayvec::ArrayVec::new();
            for tool in completion_request.tools.into_iter() {
                if together_tools.len() < super::types::MAX_TOOLS {
                    let together_tool = TogetherTool {
                        tool_type: "function",
                        function: TogetherFunction {
                            name: tool.name(),
                            description: tool.description(),
                            parameters: tool.parameters().clone()}};
                    let _ = together_tools.push(together_tool);
                }
            }
            Some(together_tools)
        } else {
            None
        };

        Ok(TogetherChatRequest {
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
}

impl completion::CompletionModel for TogetherCompletionModel {
    type Response = TogetherChatResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> fluent_ai_async::AsyncStream<completion::CompletionResponse<TogetherChatResponse>> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let api_key = self.client.api_key().clone();
        let base_url = self.client.base_url().clone();
        let request_body = match self.create_completion_request(completion_request) {
            Ok(body) => body,
            Err(e) => {
                return AsyncStream::with_channel(|sender| {
                    handle_error!(e, "Request creation failed");
                });
            }
        };

        AsyncStream::with_channel(move |sender| {
            // Use Http3::json() directly without await - NO FUTURES
            let response = Http3::json()
                .api_key(&api_key)
                .body(&request_body)
                .post(&format!("{}/v1/chat/completions", base_url))
                .collect();

            match response {
                Ok(ApiResponse::Ok(completion_response)) => {
                    tracing::info!(target: "rig",
                        "Together completion token usage: {:?}",
                        completion_response.usage.clone().map(|usage| format!("{usage}")).unwrap_or_else(|| "N/A".to_string())
                    );
                    match completion_response.try_into() {
                        Ok(response) => emit!(sender, response),
                        Err(e) => handle_error!(e, "Response conversion failed"),
                    }
                }
                Ok(ApiResponse::Error(err)) => {
                    handle_error!(CompletionError::ProviderError(err.error), "API error");
                }
                Err(e) => {
                    handle_error!(CompletionError::ProviderError(format!("Request failed: {}", e)), "HTTP request failed");
                }
            }
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> fluent_ai_async::AsyncStream<Self::StreamingResponse> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        
        let api_key = self.client.api_key().clone();
        let base_url = self.client.base_url().clone();
        let mut request_body = match self.create_completion_request(request) {
            Ok(body) => body,
            Err(e) => {
                return AsyncStream::with_channel(|sender| {
                    handle_error!(e, "Request creation failed");
                });
            }
        };

        // Enable streaming in the request
        request_body.stream = Some(true);

        AsyncStream::with_channel(move |sender| {
            // Use Http3::json() for streaming with SSE handling - NO FUTURES
            let mut response_stream = Http3::json()
                .api_key(&api_key)
                .body(&request_body)
                .post(&format!("{}/v1/chat/completions", base_url));

            // Process SSE stream using AsyncStream patterns
            response_stream.on_chunk(|chunk| {
                match chunk {
                    Ok(sse_data) => {
                        // Parse SSE data and emit streaming responses
                        if let Ok(streaming_response) = parse_sse_chunk(&sse_data) {
                            emit!(sender, streaming_response);
                        }
                    }
                    Err(e) => handle_error!(e, "SSE chunk processing failed"),
                }
            });
        })
    }
}

// =============================================================================
// Model Constants (for compatibility with existing imports)
// =============================================================================

/// Llama 3.2 11B Vision Instruct Turbo model
pub const LLAMA_3_2_11B_VISION_INSTRUCT_TURBO: &str = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo";
