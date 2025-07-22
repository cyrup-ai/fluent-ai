// ================================================================
//! xAI Completion Integration
//! From [xAI Reference](https://docs.x.ai/docs/api-reference#chat-completions)
// ================================================================

// Re-export the domain CompletionModel trait
pub use fluent_ai_domain::completion::CompletionModel;
use fluent_ai_domain::completion::{self, CompletionRequest};
use serde_json::{Value, json};

use self::xai_api_types::{CompletionResponse, ToolDefinition};
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
    model: String,
}

impl XaiCompletionModel {
    pub(crate) fn create_completion_request(
        &self,
        completion_request: fluent_ai_domain::completion::CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Convert documents into user messages
        let docs: Option<Vec<serde_json::Value>> =
            completion_request.normalized_documents().map(|docs| {
                docs.into_iter()
                    .map(|doc| serde_json::json!({"role": "user", "content": doc.content}))
                    .collect()
            });

        // Convert existing chat history to JSON values
        let chat_history: Vec<serde_json::Value> = completion_request
            .chat_history
            .into_iter()
            .map(|message| {
                // Convert domain Message to xAI format
                match message {
                    fluent_ai_domain::message::Message::User(content) => {
                        serde_json::json!({"role": "user", "content": content})
                    }
                    fluent_ai_domain::message::Message::Assistant(content) => {
                        serde_json::json!({"role": "assistant", "content": content})
                    }
                    fluent_ai_domain::message::Message::System(content) => {
                        serde_json::json!({"role": "system", "content": content})
                    }
                    fluent_ai_domain::message::Message::ToolCall {
                        name, arguments, ..
                    } => {
                        serde_json::json!({
                            "role": "assistant",
                            "tool_calls": [{
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments
                                }
                            }]
                        })
                    }
                    fluent_ai_domain::message::Message::ToolResult { name, result, .. } => {
                        serde_json::json!({
                            "role": "tool",
                            "tool_call_id": name,
                            "content": result
                        })
                    }
                }
            })
            .collect();

        // Init full history with preamble (or empty if non-existent)
        let mut full_history: Vec<serde_json::Value> = match &completion_request.preamble {
            Some(preamble) => vec![serde_json::json!({"role": "system", "content": preamble})],
            None => vec![],
        };

        // Docs appear right after preamble, if they exist
        if let Some(docs) = docs {
            full_history.extend(docs)
        }

        // Chat history and prompt appear in the order they were provided
        full_history.extend(chat_history);

        let mut request = if completion_request.tools.is_empty() {
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
                "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        request = if let Some(params) = completion_request.additional_params {
            json_util::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for XaiCompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: fluent_ai_domain::completion::CompletionRequest,
    ) -> Result<DomainCompletionResponse<CompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;
        let request_body = serde_json::to_vec(&request)
            .map_err(|e| CompletionError::RequestError(format!("Serialization error: {}", e)))?;

        let response = self
            .client
            .make_request("v1/chat/completions", request_body)
            .await
            .map_err(|e| CompletionError::HttpError(e.to_string()))?;

        if response.status().is_success() {
            // Use pure HTTP3 streaming - delegate to domain layer
            // Provider uses domain, domain uses HTTP3 directly
            let completion: CompletionResponse =
                todo!("Delegate to domain layer for HTTP3 streaming");

            Ok(DomainCompletionResponse {
                raw_response: completion.clone(),
                content: completion.try_into()?,
                token_usage: None, // TODO: Extract token usage from response
                metadata: Default::default(),
            })
        } else {
            // Use pure HTTP3 streaming for error responses - delegate to domain
            Err(CompletionError::ProviderError(
                "HTTP error - delegate to domain layer".to_string(),
            ))
        }
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingResponse<Self::StreamingResponse>, CompletionError> {
        let mut request_json = self.create_completion_request(request)?;
        request_json["stream"] = serde_json::json!(true);

        let request_body = serde_json::to_vec(&request_json)
            .map_err(|e| CompletionError::RequestError(format!("Serialization error: {}", e)))?;

        let response = self
            .client
            .make_request("v1/chat/completions", request_body)
            .await
            .map_err(|e| CompletionError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            // Use pure HTTP3 streaming for error responses - delegate to domain
            return Err(CompletionError::ProviderError(
                "HTTP error - delegate to domain layer".to_string(),
            ));
        }

        // Create streaming response using the streaming module
        let sse_stream = response.sse();
        Ok(crate::streaming::StreamingResponse::from_sse_stream(
            sse_stream,
        ))
    }
}

impl TryFrom<xai_api_types::CompletionResponse>
    for crate::completion_provider::ZeroOneOrMany<String>
{
    type Error = CompletionError;

    fn try_from(response: xai_api_types::CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        // Extract text content from the response
        if let Some(content) = choice.message.get("content").and_then(|c| c.as_str()) {
            Ok(crate::completion_provider::ZeroOneOrMany::One(
                content.to_string(),
            ))
        } else {
            Err(CompletionError::ResponseError(
                "Response did not contain valid text content".to_string(),
            ))
        }
    }
}

pub mod xai_api_types {
    use serde::{Deserialize, Serialize};

    impl From<completion::ToolDefinition> for ToolDefinition {
        fn from(tool: completion::ToolDefinition) -> Self {
            Self {
                r#type: "function".into(),
                function: tool,
            }
        }
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ToolDefinition {
        pub r#type: String,
        pub function: completion::ToolDefinition,
    }

    #[derive(Debug, Deserialize)]
    pub struct Function {
        pub name: String,
        pub arguments: String,
    }

    #[derive(Debug, Deserialize)]
    pub struct CompletionResponse {
        pub id: String,
        pub model: String,
        pub choices: Vec<Choice>,
        pub created: i64,
        pub object: String,
        pub system_fingerprint: String,
        pub usage: Usage,
    }

    #[derive(Debug, Deserialize)]
    pub struct Choice {
        pub finish_reason: String,
        pub index: i32,
        pub message: serde_json::Value,
    }

    #[derive(Debug, Deserialize)]
    pub struct Usage {
        pub completion_tokens: i32,
        pub prompt_tokens: i32,
        pub total_tokens: i32,
    }
}
