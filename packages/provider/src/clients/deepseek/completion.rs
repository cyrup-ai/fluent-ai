// ============================================================================
// File: src/providers/deepseek/completion.rs
// ----------------------------------------------------------------------------
// DeepSeek completion model implementation
// ============================================================================

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use fluent_ai_domain::{
    completion::{CompletionRequest, ToolDefinition as DomainToolDefinition}, 
    message::{self, Message, AssistantContent, MessageError},
    AsyncTask, spawn_async, OneOrMany,
};
use crate::{json_util, streaming::StreamingCompletionResponse};
use crate::Model;

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
pub use crate::clients::openai::CompletionResponse;

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
    pub model: crate::Models,
}

impl DeepSeekCompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        let model_enum = match model {
            "deepseek-chat" => crate::Models::DeepseekChat,
            "deepseek-reasoner" => crate::Models::DeepseekReasoner,
            "deepseek-v3" => crate::Models::DeepseekV3,
            "deepseek-r1" => crate::Models::DeepseekR10528,
            _ => crate::Models::DeepseekChat, // Default fallback
        };
        Self {
            client,
            model: model_enum,
        }
    }

    fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Create a simple user message from the prompt
        let message = DeepSeekMessage::User {
            content: completion_request.prompt,
            name: None,
        };

        let request = json!({
            "model": completion_request.model,
            "messages": vec![message],
            "temperature": completion_request.temperature,
        });

        let request = if let Some(max_tokens) = completion_request.max_tokens {
            json_util::merge(request, json!({"max_tokens": max_tokens}))
        } else {
            request
        };

        Ok(request)
    }
}

impl crate::client::completion::CompletionModel for DeepSeekCompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = streaming::StreamingCompletionResponse;

    fn completion(
        &self,
        req: crate::client::completion::CompletionRequest,
    ) -> AsyncTask<Result<crate::client::completion::CompletionResponse<Self::Response>, crate::client::completion::CompletionError>> {
        let (tx, task) = runtime::channel();
        let client = self.client.clone();

        // Convert the client completion request to our format
        let prompt = if let Some(msg) = req.chat_history.first() {
            match msg {
                crate::completion::message::Message::User { content } => {
                    content.iter().find_map(|c| match c {
                        crate::completion::message::UserContent::Text(text) => Some(text.0.clone()),
                        _ => None,
                    }).unwrap_or_default()
                }
                crate::completion::message::Message::Assistant { content } => {
                    content.iter().find_map(|c| match c {
                        crate::completion::message::AssistantContent::Text(text) => Some(text.0.clone()),
                        _ => None,
                    }).unwrap_or_default()
                }
            }
        } else {
            "".to_string()
        };
        
        let completion_request = CompletionRequest {
            prompt,
            model: self.model.info().name.clone(),
            temperature: req.temperature.unwrap_or(0.7),
            max_tokens: req.max_tokens.map(|t| t as u32),
        };

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
                                        // Convert to the expected format
                                        let choice = if let Some(choice) = response.choices.first() {
                                            crate::OneOrMany::one(crate::completion::message::AssistantContent::Text(
                                                crate::completion::message::Text(choice.message.content.clone())
                                            ))
                                        } else {
                                            return Err(crate::client::completion::CompletionError::Response("No choices in response".to_string()));
                                        };

                                        Ok(crate::client::completion::CompletionResponse {
                                            choice,
                                            raw_response: response,
                                        })
                                    }
                                    Ok(ApiResponse::Err(err)) => {
                                        Err(crate::client::completion::CompletionError::Response(err.message))
                                    }
                                    Err(e) => {
                                        Err(crate::client::completion::CompletionError::Response(e.to_string()))
                                    }
                                }
                            } else {
                                match response.text().await {
                                    Ok(text) => Err(crate::client::completion::CompletionError::Response(text)),
                                    Err(e) => Err(crate::client::completion::CompletionError::Response(e.to_string())),
                                }
                            }
                        }
                        Err(e) => Err(crate::client::completion::CompletionError::Http(e)),
                    };

                    tx.finish(result);
                });
            }
            Err(e) => {
                tx.finish(Err(crate::client::completion::CompletionError::Response(e.to_string())));
            }
        }

        task
    }

    fn stream(
        &self,
        req: crate::client::completion::CompletionRequest,
    ) -> AsyncTask<Result<crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>, crate::client::completion::CompletionError>> {
        let (tx, task) = runtime::channel();
        let client = self.client.clone();

        // Convert the client completion request to our format
        let prompt = if let Some(msg) = req.chat_history.first() {
            match msg {
                crate::completion::message::Message::User { content } => {
                    content.iter().find_map(|c| match c {
                        crate::completion::message::UserContent::Text(text) => Some(text.0.clone()),
                        _ => None,
                    }).unwrap_or_default()
                }
                crate::completion::message::Message::Assistant { content } => {
                    content.iter().find_map(|c| match c {
                        crate::completion::message::AssistantContent::Text(text) => Some(text.0.clone()),
                        _ => None,
                    }).unwrap_or_default()
                }
            }
        } else {
            "".to_string()
        };
        
        let completion_request = CompletionRequest {
            prompt,
            model: self.model.info().name.clone(),
            temperature: req.temperature.unwrap_or(0.7),
            max_tokens: req.max_tokens.map(|t| t as u32),
        };

        match self.create_completion_request(completion_request) {
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
                tx.finish(Err(crate::client::completion::CompletionError::Response(e.to_string())));
            }
        }

        task
    }
}
