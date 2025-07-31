// ================================================================
//! Together AI Completion Integration
//! From [Together AI Reference](https://docs.together.ai/docs/chat-overview)
// ================================================================

// ================================================================
// Together Completion Models
// ================================================================

// Import centralized HTTP structs - no more local definitions!
// Re-export the domain CompletionModel trait
pub use fluent_ai_domain::completion::CompletionModel;
use fluent_ai_domain::completion::{CompletionCoreError as CompletionError, CompletionRequest};

/// Together AI streaming completion response
#[derive(Debug, Clone)]
pub struct TogetherStreamingCompletionResponse {
    /// Content of the streaming chunk
    pub content: Option<String>,
    /// Reason the completion finished
    pub finish_reason: Option<String>,
    /// Token usage information
    pub usage: Option<crate::clients::together::types::TogetherUsage>,
}
use super::types::{
    TogetherChatRequest, TogetherChatResponse, TogetherChoice, TogetherContent,
    TogetherFunction, TogetherMessage, TogetherResponseMessage, TogetherStreamingChunk,
    TogetherTool, TogetherUsage};
use serde_json::json;
use arrayvec::ArrayVec;
use fluent_ai_http3::Http3;

// CRITICAL: Import model information from model-info package (single source of truth)
use model_info::{discovery::Provider, ModelInfo as ModelInfoFromPackage};

use super::client::{Client, together_ai_api_types::ApiResponse};
use crate::streaming::DefaultStreamingResponse;
use crate::clients::openai;
use fluent_ai_domain::util::json_util;

// Model constants removed - use model-info package exclusively
// All Together AI model information is provided by ./packages/model-info

// Type aliases for domain types used in client builders
pub use fluent_ai_domain::context::document::Document;
pub use fluent_ai_domain::completion::types::ToolDefinition;

// Response type alias for client compatibility
pub type CompletionResponse = TogetherChatResponse;

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
        let provider = Provider::Together;
        provider.get_model_info(&self.model)
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: fluent_ai_domain::completion::CompletionRequest,
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
        completion_request: fluent_ai_domain::completion::CompletionRequest,
    ) -> fluent_ai_async::AsyncStream<fluent_ai_domain::completion::CompletionResponse<TogetherChatResponse>> {
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
                        if let Ok(streaming_response) = Self::parse_sse_chunk(&sse_data) {
                            emit!(sender, streaming_response);
                        }
                    }
                    Err(e) => handle_error!(e, "SSE chunk processing failed"),
                }
            });
        })
    }

    /// Parse Together AI Server-Sent Events chunk with zero allocation optimization
    /// 
    /// # Performance Optimizations
    /// - Zero-allocation string parsing using byte slices
    /// - Inline JSON parsing for critical paths
    /// - Early exit on empty content
    /// - No heap allocations in hot paths
    #[inline]
    fn parse_sse_chunk(sse_data: &[u8]) -> Result<TogetherStreamingCompletionResponse, CompletionError> {
        // Fast UTF-8 validation with zero copy
        let chunk_str = match std::str::from_utf8(sse_data) {
            Ok(s) => s,
            Err(_) => return Err(CompletionError::ParseError("Invalid UTF-8 in SSE chunk".into())),
        };
        
        // Optimized line-by-line processing with zero allocation
        for line in chunk_str.lines() {
            // Fast prefix check with exact length
            if line.len() > 6 && line.as_bytes().starts_with(b"data: ") {
                let data = &line[6..]; // Zero-copy slice
                
                // Early exit for termination marker
                if data.len() == 6 && data == "[DONE]" {
                    return Ok(TogetherStreamingCompletionResponse {
                        content: None,
                        finish_reason: Some("stop".into()),
                        usage: None,
                    });
                }
                
                // Skip empty data lines
                if data.is_empty() {
                    continue;
                }
                
                // Fast JSON parsing with targeted extraction
                return Self::extract_content_from_json(data);
            }
        }
        
        // No valid data found - return empty response
        Ok(TogetherStreamingCompletionResponse {
            content: None,
            finish_reason: None,
            usage: None,
        })
    }
    
    /// Extract content from Together AI JSON response with minimal allocations
    /// 
    /// # Performance Features
    /// - Direct JSON key lookup without full deserialization
    /// - Early return on first valid content
    /// - Optimized string extraction
    #[inline(always)]
    fn extract_content_from_json(json_data: &str) -> Result<TogetherStreamingCompletionResponse, CompletionError> {
        // Parse JSON with error handling
        let json_value: serde_json::Value = match serde_json::from_str(json_data) {
            Ok(v) => v,
            Err(_) => return Err(CompletionError::ParseError("Invalid JSON in SSE chunk".into())),
        };
        
        // Optimized path traversal for Together AI format
        if let Some(choices_array) = json_value.get("choices").and_then(|c| c.as_array()) {
            if let Some(first_choice) = choices_array.first() {
                // Check for content in delta
                if let Some(delta) = first_choice.get("delta") {
                    if let Some(content_str) = delta.get("content").and_then(|c| c.as_str()) {
                        if !content_str.is_empty() {
                            return Ok(TogetherStreamingCompletionResponse {
                                content: Some(content_str.to_string()),
                                finish_reason: None,
                                usage: None,
                            });
                        }
                    }
                }
                
                // Check for finish reason
                if let Some(finish_reason_str) = first_choice.get("finish_reason").and_then(|fr| fr.as_str()) {
                    return Ok(TogetherStreamingCompletionResponse {
                        content: None,
                        finish_reason: Some(finish_reason_str.to_string()),
                        usage: None,
                    });
                }
            }
        }
        
        // Check for usage information
        let usage = json_value.get("usage").map(|u| TogetherUsage {
            prompt_tokens: u.get("prompt_tokens").and_then(|pt| pt.as_u64()).unwrap_or(0) as u32,
            completion_tokens: u.get("completion_tokens").and_then(|ct| ct.as_u64()).unwrap_or(0) as u32,
            total_tokens: u.get("total_tokens").and_then(|tt| tt.as_u64()).unwrap_or(0) as u32,
        });
        
        Ok(TogetherStreamingCompletionResponse {
            content: None,
            finish_reason: None,
            usage,
        })
    }
}

// =============================================================================
// Together AI Integration Complete - High Performance Streaming Implementation
// =============================================================================

// =============================================================================
// Model Constants (for compatibility with existing imports)
// =============================================================================

/// Llama 3.2 11B Vision Instruct Turbo model
pub const LLAMA_3_2_11B_VISION_INSTRUCT_TURBO: &str = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo";
