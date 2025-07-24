// Import centralized HTTP structs - no more local definitions!
use super::types::{
    OpenRouterChatRequest, OpenRouterContent, OpenRouterError, OpenRouterErrorResponse,
    OpenRouterMessage, OpenRouterResponseMessage, OpenRouterStreamingChunk, OpenRouterToolCall,
};
use fluent_ai_http3::{Http3, HttpResult};
use serde::Deserialize;
use serde_json::{Value, json};

use super::client::{ApiErrorResponse, ApiResponse, Client, Usage};
use super::streaming::{FinalCompletionResponse, StreamingCompletionResponse};
use crate::clients::openai::AssistantContent;
use crate::{
    OneOrMany,
    clients::openai::Message,
    completion::{self, CompletionError, CompletionModel, CompletionRequest},
    json_util,
};

// ================================================================
// OpenRouter Completion API
// ================================================================
/// The `openai/gpt-4-1` model - default for OpenRouter. Find more models at <https://openrouter.ai/models>.
pub const GPT_4_1: &str = "openai/gpt-4-1";
/// The `qwen/qwq-32b` model. Find more models at <https://openrouter.ai/models>.
pub const QWEN_QWQ_32B: &str = "qwen/qwq-32b";
/// The `anthropic/claude-3.7-sonnet` model. Find more models at <https://openrouter.ai/models>.
pub const CLAUDE_3_7_SONNET: &str = "anthropic/claude-3.7-sonnet";
/// The `perplexity/sonar-pro` model. Find more models at <https://openrouter.ai/models>.
pub const PERPLEXITY_SONAR_PRO: &str = "perplexity/sonar-pro";
/// The `google/gemini-2.0-flash-001` model. Find more models at <https://openrouter.ai/models>.
pub const GEMINI_FLASH_2_0: &str = "google/gemini-2.0-flash-001";

/// Legacy alias for centralized OpenRouter response type
pub type CompletionResponse = fluent_ai_http_structs::openrouter::OpenRouterChatResponse;

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = content
                    .iter()
                    .map(|c| match c {
                        AssistantContent::Text { text } => completion::AssistantContent::text(text),
                        AssistantContent::Refusal { refusal } => {
                            completion::AssistantContent::text(refusal)
                        }
                    })
                    .collect::<Vec<_>>();

                content.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
        })
    }
}

/// Legacy alias for centralized choice type
pub type Choice = fluent_ai_http_structs::openrouter::OpenRouterChoice;

// CompletionModel is now imported from fluent_ai_domain::model
// Removed duplicated CompletionModel struct - use canonical domain type

/// OpenRouter completion model implementation
#[derive(Debug, Clone)]
pub struct OpenRouterCompletionModel {
    client: Client,
    model: String,
}

impl OpenRouterCompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<OpenRouterChatRequest<'_>, CompletionError> {
        // Use the centralized builder with validation
        let builder = Http3Builders::openrouter();
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
                HttpUtils::validate_temperature(temp as f32, Provider::OpenRouter).map_err(|e| {
                    CompletionError::InvalidRequest(format!("Invalid temperature: {}", e))
                })? as f64,
            );
        }

        if let Some(max_tokens) = completion_request.max_tokens {
            chat_builder = chat_builder.max_tokens(
                HttpUtils::validate_max_tokens(max_tokens, Provider::OpenRouter).map_err(|e| {
                    CompletionError::InvalidRequest(format!("Invalid max_tokens: {}", e))
                })?,
            );
        }

        // Add tools if present
        if !completion_request.tools.is_empty() {
            let mut openrouter_tools = arrayvec::ArrayVec::new();
            for tool in completion_request.tools.into_iter() {
                if openrouter_tools.len() < super::types::MAX_TOOLS {
                    let openrouter_tool = super::types::OpenRouterTool {
                        tool_type: "function",
                        function: super::types::OpenRouterFunction {
                            name: tool.name(),
                            description: tool.description(),
                            parameters: tool.parameters().clone(),
                        },
                    };
                    let _ = openrouter_tools.push(openrouter_tool);
                }
            }
            chat_builder = chat_builder.with_tools(openrouter_tools);
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

    pub async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<FinalCompletionResponse>, CompletionError> {
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

        let builder = self
            .client
            .post("/chat/completions")
            .body(body_bytes)
            .header("content-type", "application/json");

        super::streaming::send_openrouter_streaming_request(builder).await
    }
}

impl completion::CompletionModel for OpenRouterCompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = FinalCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
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
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "OpenRouter completion token usage: {:?}",
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

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        OpenRouterCompletionModel::stream(self, completion_request).await
    }
}
