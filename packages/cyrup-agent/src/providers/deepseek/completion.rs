// ============================================================================
// File: src/providers/deepseek/completion.rs
// ----------------------------------------------------------------------------
// DeepSeek completion model implementation
// ============================================================================

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_util, message,
    runtime::{self, AsyncTask},
    streaming::StreamingCompletionResponse,
    OneOrMany,
};

use super::client::Client;
use super::streaming;

// ============================================================================
// Model Constants
// ============================================================================
/// `deepseek-chat` completion model
pub const DEEPSEEK_CHAT: &str = "deepseek-chat";
/// `deepseek-reasoner` completion model
pub const DEEPSEEK_REASONER: &str = "deepseek-reasoner";

// ============================================================================
// API Response Types
// ============================================================================
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

// ============================================================================
// Message types (DeepSeek uses custom format)
// ============================================================================
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum DeepSeekMessage {
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_util::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

impl DeepSeekMessage {
    pub fn system(content: &str) -> Self {
        DeepSeekMessage::System {
            content: content.to_owned(),
            name: None,
        }
    }
}

// ============================================================================
// Tool types
// ============================================================================
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    pub function: Function,
    pub index: u64,
    #[serde(rename = "type")]
    pub r#type: ToolType,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Function,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            },
        }
    }
}

// ============================================================================
// Response types (reuse OpenAI format mostly)
// ============================================================================
pub use crate::providers::openai::CompletionResponse;

// ============================================================================
// Message conversion implementations
// ============================================================================
impl TryFrom<message::Message> for Vec<DeepSeekMessage> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => {
                let mut messages = Vec::new();
                for content_item in content.iter() {
                    match content_item {
                        message::UserContent::Text(text) => {
                            messages.push(DeepSeekMessage::User {
                                content: text.text.clone(),
                                name: None,
                            });
                        }
                        _ => {
                            return Err(message::MessageError::ConversionError(
                                "Only text content is supported".to_string(),
                            ))
                        }
                    }
                }
                Ok(messages)
            }
            message::Message::Assistant { content } => {
                let mut text_content = String::new();
                let mut tool_calls = Vec::new();

                for content_item in content.iter() {
                    match content_item {
                        message::AssistantContent::Text(text) => {
                            if !text_content.is_empty() {
                                text_content.push('\n');
                            }
                            text_content.push_str(&text.text);
                        }
                        message::AssistantContent::ToolCall(tool_call) => {
                            tool_calls.push(ToolCall {
                                id: tool_call.id.clone(),
                                function: Function {
                                    name: tool_call.function.name.clone(),
                                    arguments: tool_call.function.arguments.clone(),
                                },
                                index: 0,
                                r#type: ToolType::Function,
                            });
                        }
                    }
                }

                Ok(vec![DeepSeekMessage::Assistant {
                    content: text_content,
                    name: None,
                    tool_calls,
                }])
            }
        }
    }
}

// ============================================================================
// Completion Model
// ============================================================================
#[derive(Clone)]
pub struct DeepSeekCompletionModel {
    pub client: Client,
    pub model: String,
}

impl DeepSeekCompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Build up the order of messages (context, chat_history, prompt)
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(completion_request.chat_history);

        // Initialize full history with preamble (or empty if non-existent)
        let mut full_history: Vec<DeepSeekMessage> = completion_request
            .preamble
            .map_or_else(Vec::new, |preamble| {
                vec![DeepSeekMessage::system(&preamble)]
            });

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<DeepSeekMessage>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

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
                "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
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

impl completion::CompletionModel for DeepSeekCompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = streaming::StreamingCompletionResponse;

    fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> AsyncTask<Result<completion::CompletionResponse<CompletionResponse>, CompletionError>>
    {
        let (tx, task) = runtime::channel();
        let client = self.client.clone();

        match self.create_completion_request(completion_request) {
            Ok(request) => {
                runtime::spawn_async(async move {
                    let response = client.post("/chat/completions").json(&request).send().await;

                    let result = match response {
                        Ok(response) => {
                            if response.status().is_success() {
                                match response.json::<ApiResponse<CompletionResponse>>().await {
                                    Ok(ApiResponse::Ok(response)) => {
                                        tracing::info!(target: "rig",
                                            "deepseek completion token usage: {:?}",
                                            response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                                        );
                                        response.try_into()
                                    }
                                    Ok(ApiResponse::Err(err)) => {
                                        Err(CompletionError::ProviderError(err.message))
                                    }
                                    Err(e) => {
                                        Err(CompletionError::DeserializationError(e.to_string()))
                                    }
                                }
                            } else {
                                match response.text().await {
                                    Ok(text) => Err(CompletionError::ProviderError(text)),
                                    Err(e) => Err(CompletionError::RequestError(e.to_string())),
                                }
                            }
                        }
                        Err(e) => Err(CompletionError::RequestError(e.to_string())),
                    };

                    tx.finish(result);
                });
            }
            Err(e) => {
                tx.finish(Err(e));
            }
        }

        task
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> AsyncTask<Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>>
    {
        let (tx, task) = runtime::channel();
        let client = self.client.clone();

        match self.create_completion_request(request) {
            Ok(mut request) => {
                request = json_util::merge(
                    request,
                    json!({"stream": true, "stream_options": {"include_usage": true}}),
                );

                runtime::spawn_async(async move {
                    let builder = client.post("/chat/completions").json(&request);
                    let result = streaming::send_deepseek_streaming_request(builder).await;
                    tx.finish(result);
                });
            }
            Err(e) => {
                tx.finish(Err(e));
            }
        }

        task
    }
}
