// ================================================================
//! xAI Completion Integration
//! From [xAI Reference](https://docs.x.ai/docs/api-reference#chat-completions)
// ================================================================

// Import centralized HTTP structs - no more local definitions!
// Re-export the domain CompletionModel trait
use arrayvec::ArrayVec;
use cyrup_sugars::ZeroOneOrMany;
pub use fluent_ai_domain::completion::CompletionModel;
use fluent_ai_domain::completion::{self, CompletionRequest};
// Model constants removed - use model-info package exclusively
// All xAI model information is provided by ./packages/model-info

// Type aliases for domain types used in client builders
pub use fluent_ai_domain::context::document::Document;
use fluent_ai_domain::util::json_util;
use fluent_ai_http3::{Http3, HttpError};
// CRITICAL: Import model information from model-info package (single source of truth)
use model_info::{ModelInfo as ModelInfoFromPackage, discovery::Provider};
use serde_json::{Value, json};

use super::client::Client;
use super::types::{
    XaiChatRequest, XaiChatResponse, XaiChoice, XaiContent, XaiFunction, XaiMessage,
    XaiResponseMessage, XaiStreamingChunk, XaiTool, XaiUsage,
};
use crate::completion_provider::{CompletionError, CompletionResponse as DomainCompletionResponse};
use crate::streaming_completion_provider::{ConnectionStatus, StreamingCompletionProvider};

// =================================================================
// Rig Implementation Types
// =================================================================

/// xAI completion model implementation
#[derive(Clone)]
pub struct XaiCompletionModel {
    client: Client,
    model: String,
}

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
                    content: XaiContent::Text(preamble),
                })
                .map_err(|_| CompletionError::RequestError("Request too large".to_string()))?;
        }

        // Add documents as context
        if let Some(docs) = completion_request.normalized_documents() {
            for doc in docs {
                let content = format!("Document: {}", doc.content());
                messages
                    .try_push(XaiMessage {
                        role: "user",
                        content: XaiContent::Text(Box::leak(content.into_boxed_str())),
                    })
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
                                content: XaiContent::Text(text),
                            })
                            .map_err(|_| {
                                CompletionError::RequestError("Request too large".to_string())
                            })?;
                    }
                }
                fluent_ai_domain::message::MessageRole::Assistant => {
                    if let Some(text) = msg.content().text() {
                        messages
                            .try_push(XaiMessage {
                                role: "assistant",
                                content: XaiContent::Text(text),
                            })
                            .map_err(|_| {
                                CompletionError::RequestError("Request too large".to_string())
                            })?;
                    }
                }
                fluent_ai_domain::message::MessageRole::System => {
                    if let Some(text) = msg.content().text() {
                        messages
                            .try_push(XaiMessage {
                                role: "system",
                                content: XaiContent::Text(text),
                            })
                            .map_err(|_| {
                                CompletionError::RequestError("Request too large".to_string())
                            })?;
                    }
                }
            }
        }

        // Set parameters with direct validation - zero allocation
        let temperature = completion_request
            .temperature
            .map(|temp| temp.clamp(0.0, 2.0));
        let max_tokens = completion_request
            .max_tokens
            .map(|tokens| tokens.clamp(1, 131072));

        // Add tools if present
        let tools = if !completion_request.tools.is_empty() {
            let mut xai_tools = arrayvec::ArrayVec::new();
            for tool in completion_request.tools.into_iter() {
                if xai_tools.len() < crate::MAX_TOOLS {
                    let xai_tool = XaiTool {
                        tool_type: "function",
                        function: XaiFunction {
                            name: tool.name(),
                            description: tool.description(),
                            parameters: tool.parameters().clone(),
                        },
                    };
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
            model: model.to_string(),
        }
    }

    /// Load model information from model-info package (single source of truth)
    pub fn load_model_info(&self) -> fluent_ai_async::AsyncStream<ModelInfoFromPackage> {
        let provider = Provider::XAI;
        provider.get_model_info(&self.model)
    }
}

impl completion::CompletionModel for XaiCompletionModel {
    fn prompt(
        &self,
        prompt: fluent_ai_domain::Prompt,
        params: &fluent_ai_domain::completion::types::CompletionParams,
    ) -> fluent_ai_async::AsyncStream<fluent_ai_domain::context::CompletionChunk> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        use fluent_ai_http3::Http3;
        use serde_json::json;

        let client = self.client.clone();
        let model = self.model.clone();

        // Streaming-only completion using proper architecture - NO FUTURES
        AsyncStream::with_channel(move |sender| {
            // Convert domain prompt to XAI-compatible chat messages format
            let messages = match prompt {
                fluent_ai_domain::Prompt::Text(text) => {
                    vec![json!({"role": "user", "content": text})]
                }
                fluent_ai_domain::Prompt::Messages(msgs) => msgs
                    .into_iter()
                    .map(|msg| {
                        json!({
                            "role": msg.role,
                            "content": msg.content
                        })
                    })
                    .collect(),
                fluent_ai_domain::Prompt::SystemUserAssistant {
                    system,
                    user,
                    assistant,
                } => {
                    let mut messages = Vec::with_capacity(3);
                    if let Some(sys) = system {
                        messages.push(json!({"role": "system", "content": sys}));
                    }
                    messages.push(json!({"role": "user", "content": user}));
                    if let Some(asst) = assistant {
                        messages.push(json!({"role": "assistant", "content": asst}));
                    }
                    messages
                }
            };

            // Build optimized XAI request payload
            let request_body = json!({
                "model": model,
                "messages": messages,
                "stream": true,
                "temperature": params.temperature.unwrap_or(0.7),
                "max_tokens": params.max_tokens.unwrap_or(4096),
                "top_p": params.top_p.unwrap_or(1.0)
            });

            // Use Http3 streaming patterns from examples - NO async/await
            let response_stream = Http3::json()
                .api_key(&client.api_key)
                .body(&request_body)
                .on_chunk(|result: Result<fluent_ai_http3::HttpChunk, fluent_ai_http3::HttpError>| -> fluent_ai_http3::HttpChunk {
                    match result {
                        Ok(chunk) => chunk,
                        Err(error) => {
                            tracing::error!("XAI streaming HTTP error: {:?}", error);
                            fluent_ai_http3::HttpChunk::bad_chunk(format!("XAI streaming HTTP error: {:?}", error))
                        }
                    }
                })
                .post("https://api.x.ai/v1/chat/completions");

            // Process chunks using streaming-only patterns with ChunkHandler
            response_stream.on_chunk(|chunk| {
                match chunk {
                    Ok(chunk_bytes) => {
                        // Parse Server-Sent Events format with zero allocation
                        if let Some(completion_chunk) = Self::parse_xai_sse_chunk(&chunk_bytes) {
                            emit!(sender, completion_chunk);
                        }
                    }
                    Err(e) => {
                        handle_error!(e, "XAI streaming HTTP error");
                    }
                }
            });
        })
    }

    /// Parse XAI Server-Sent Events chunk with zero-allocation optimization
    ///
    /// # Performance Features
    /// - Minimal heap allocations
    /// - Fast byte-level parsing
    /// - Early exit optimizations
    #[inline]
    fn parse_xai_sse_chunk(
        chunk_bytes: &[u8],
    ) -> Option<fluent_ai_domain::context::CompletionChunk> {
        let chunk_str = std::str::from_utf8(chunk_bytes).ok()?;

        // Process SSE format with zero allocation
        for line in chunk_str.lines() {
            if line.len() > 6 && line.starts_with("data: ") {
                let json_data = &line[6..];

                // Handle termination marker
                if json_data == "[DONE]" {
                    return Some(fluent_ai_domain::context::CompletionChunk::Complete {
                        text: String::new(),
                        finish_reason: Some(fluent_ai_domain::context::FinishReason::Stop),
                        usage: None,
                    });
                }

                // Fast JSON parsing for XAI format
                if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(json_data) {
                    // Extract content from XAI streaming response format
                    if let Some(choices) = json_value.get("choices").and_then(|c| c.as_array()) {
                        if let Some(first_choice) = choices.first() {
                            if let Some(delta) = first_choice.get("delta") {
                                if let Some(content) = delta.get("content").and_then(|c| c.as_str())
                                {
                                    if !content.is_empty() {
                                        return Some(
                                            fluent_ai_domain::context::CompletionChunk::Text(
                                                content.to_string(),
                                            ),
                                        );
                                    }
                                }
                            }

                            // Check for completion finish
                            if let Some(finish_reason) =
                                first_choice.get("finish_reason").and_then(|fr| fr.as_str())
                            {
                                let reason = match finish_reason.as_str() {
                                    "stop" => fluent_ai_domain::context::FinishReason::Stop,
                                    "length" => fluent_ai_domain::context::FinishReason::Length,
                                    "tool_calls" => {
                                        fluent_ai_domain::context::FinishReason::ToolCalls
                                    }
                                    _ => fluent_ai_domain::context::FinishReason::Stop,
                                };
                                return Some(
                                    fluent_ai_domain::context::CompletionChunk::Complete {
                                        text: String::new(),
                                        finish_reason: Some(reason),
                                        usage: None,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }

        None
    }
}

// =============================================================================
// StreamingCompletionProvider Implementation (Enforces Architecture)
// =============================================================================

impl StreamingCompletionProvider for XaiCompletionModel {
    fn stream_completion(
        &self,
        request: CompletionRequest,
    ) -> fluent_ai_async::AsyncStream<fluent_ai_domain::context::CompletionChunk> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        use fluent_ai_http3::Http3;

        let client = self.client.clone();
        let model = self.model.clone();

        AsyncStream::with_channel(move |sender| {
            // Create XAI request from domain request
            let xai_request = match XaiCompletionModel::build_xai_request(&request, &model) {
                Ok(req) => req,
                Err(e) => {
                    handle_error!(e, "Failed to build XAI request");
                    return;
                }
            };

            // Stream completion using Http3 ChunkHandler patterns
            let response_stream = Http3::json()
                .api_key(&client.api_key)
                .body(&xai_request)
                .on_chunk(|result: Result<fluent_ai_http3::HttpChunk, fluent_ai_http3::HttpError>| -> fluent_ai_http3::HttpChunk {
                    match result {
                        Ok(chunk) => chunk,
                        Err(error) => {
                            tracing::error!("XAI HTTP streaming error: {:?}", error);
                            fluent_ai_http3::HttpChunk::bad_chunk(format!("XAI HTTP streaming error: {:?}", error))
                        }
                    }
                })
                .post("https://api.x.ai/v1/chat/completions");

            // Process streaming response chunks
            response_stream.on_chunk(|chunk| match chunk {
                Ok(chunk_bytes) => {
                    if let Some(completion_chunk) =
                        crate::streaming_completion_provider::utils::parse_sse_chunk(&chunk_bytes)
                    {
                        emit!(sender, completion_chunk);
                    }
                }
                Err(e) => {
                    handle_error!(e, "XAI HTTP streaming error");
                }
            });
        })
    }

    fn provider_name(&self) -> &'static str {
        "xai"
    }

    fn test_connection(&self) -> fluent_ai_async::AsyncStream<ConnectionStatus> {
        use fluent_ai_async::{AsyncStream, emit, handle_error};
        use fluent_ai_http3::Http3;

        let client = self.client.clone();

        AsyncStream::with_channel(move |sender| {
            emit!(sender, ConnectionStatus::Testing);

            // Test connection with simple model list request using ChunkHandler
            let response_stream = Http3::json()
                .api_key(&client.api_key)
                .on_chunk(|result: Result<fluent_ai_http3::HttpChunk, fluent_ai_http3::HttpError>| -> fluent_ai_http3::HttpChunk {
                    match result {
                        Ok(chunk) => chunk,
                        Err(error) => {
                            tracing::error!("XAI connection test error: {:?}", error);
                            fluent_ai_http3::HttpChunk::bad_chunk(format!("XAI connection test error: {:?}", error))
                        }
                    }
                })
                .get("https://api.x.ai/v1/models");

            response_stream.on_chunk(|chunk| match chunk {
                Ok(_) => {
                    emit!(sender, ConnectionStatus::Connected);
                }
                Err(e) => {
                    emit!(
                        sender,
                        ConnectionStatus::Failed(format!("Connection test failed: {}", e))
                    );
                }
            });
        })
    }
}

// Helper methods for XAI request building
impl XaiCompletionModel {
    /// Build XAI-compatible request from domain CompletionRequest
    fn build_xai_request(
        request: &CompletionRequest,
        model: &str,
    ) -> Result<serde_json::Value, CompletionError> {
        use serde_json::json;

        // Convert domain messages to XAI format
        let mut messages = Vec::new();

        // Add preamble as system message if present
        if let Some(preamble) = &request.preamble {
            messages.push(json!({"role": "system", "content": preamble}));
        }

        // Add chat history
        for msg in &request.chat_history {
            let role = match msg.role() {
                fluent_ai_domain::message::MessageRole::User => "user",
                fluent_ai_domain::message::MessageRole::Assistant => "assistant",
                fluent_ai_domain::message::MessageRole::System => "system",
            };

            if let Some(text) = msg.content().text() {
                messages.push(json!({"role": role, "content": text}));
            }
        }

        // Add main prompt message
        if let Some(text) = request.prompt.content().text() {
            let role = match request.prompt.role() {
                fluent_ai_domain::message::MessageRole::User => "user",
                fluent_ai_domain::message::MessageRole::Assistant => "assistant",
                fluent_ai_domain::message::MessageRole::System => "system",
            };
            messages.push(json!({"role": role, "content": text}));
        }

        // Build XAI request payload
        Ok(json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "temperature": request.temperature.unwrap_or(0.7),
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "top_p": request.top_p.unwrap_or(1.0)
        }))
    }
}

impl TryFrom<XaiChatResponse> for ZeroOneOrMany<String> {
    type Error = CompletionError;

    fn try_from(response: XaiChatResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        // Extract text content from the response
        if let Some(content) = &choice.message.content {
            Ok(ZeroOneOrMany::One(content.clone()))
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
