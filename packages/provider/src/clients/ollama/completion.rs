// ============================================================================
// File: src/providers/ollama/completion.rs
// ----------------------------------------------------------------------------
// Ollama completion model implementation
// ============================================================================

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::client::Client;
use super::streaming;
use crate::{
    OneOrMany,
    completion::{
        self, AssistantContent, CompletionError, CompletionRequest, Message as CompletionMessage,
    },
    embeddings::{self, Embedding, EmbeddingError, EmbeddingModel as EmbeddingModelTrait},
    json_util,
    message::{self, Message, MessageError, Text, ToolResultContent, UserContent},
    runtime::{self, AsyncTask},
    streaming::StreamingCompletionResponse,
};

// ============================================================================
// Model Constants
// ============================================================================
pub const LLAMA3_2: &str = "llama3.2";
pub const LLAVA: &str = "llava";
pub const MISTRAL: &str = "mistral";
pub const MISTRAL_MAGISTRAR_SMALL: &str = "mistral-magistrar-small";

// Embedding model constants
pub const ALL_MINILM: &str = "all-minilm";
pub const NOMIC_EMBED_TEXT: &str = "nomic-embed-text";

// ============================================================================
// API Error and Response Structures
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
// Completion Response
// ============================================================================
#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub model: String,
    pub created_at: String,
    pub message: ProviderMessage,
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse {
    type Error = CompletionError;

    fn try_from(resp: CompletionResponse) -> Result<Self, Self::Error> {
        match resp.message {
            // Process only if an assistant message is present.
            ProviderMessage::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut assistant_contents = Vec::new();
                // Add the assistant's text content if any.
                if !content.is_empty() {
                    assistant_contents.push(AssistantContent::text(&content));
                }
                // Process tool_calls following Ollama's chat response definition.
                // Each ToolCall has an id, a type, and a function field.
                for tc in tool_calls.iter() {
                    assistant_contents.push(AssistantContent::tool_call(
                        tc.function.name.clone(),
                        tc.function.name.clone(),
                        tc.function.arguments.clone(),
                    ));
                }
                let choice = OneOrMany::many(assistant_contents).map_err(|_| {
                    CompletionError::ResponseError("No content provided".to_owned())
                })?;
                let raw_response = CompletionResponse {
                    model: resp.model,
                    created_at: resp.created_at,
                    done: resp.done,
                    done_reason: resp.done_reason,
                    total_duration: resp.total_duration,
                    load_duration: resp.load_duration,
                    prompt_eval_count: resp.prompt_eval_count,
                    prompt_eval_duration: resp.prompt_eval_duration,
                    eval_count: resp.eval_count,
                    eval_duration: resp.eval_duration,
                    message: ProviderMessage::Assistant {
                        content,
                        images: None,
                        name: None,
                        tool_calls,
                    },
                };
                Ok(completion::CompletionResponse {
                    choice,
                    raw_response,
                })
            }
            _ => Err(CompletionError::ResponseError(
                "Chat response does not include an assistant message".into(),
            )),
        }
    }
}

// ============================================================================
// CompletionModel is now imported from fluent_ai_domain::model
// Removed duplicated CompletionModel struct - use canonical domain type

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_owned(),
        }
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Build up the order of messages (context, chat_history)
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(completion_request.chat_history);

        // Initialize full history with preamble (or empty if non-existent)
        let mut full_history: Vec<ProviderMessage> = completion_request
            .preamble
            .map_or_else(Vec::new, |preamble| {
                vec![ProviderMessage::system(&preamble)]
            });

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(|msg| msg.try_into())
                .collect::<Result<Vec<ProviderMessage>, _>>()?,
        );

        // Convert internal prompt into a provider Message
        let options = if let Some(extra) = completion_request.additional_params {
            json_util::merge(
                json!({ "temperature": completion_request.temperature }),
                extra,
            )
        } else {
            json!({ "temperature": completion_request.temperature })
        };

        let mut request_payload = json!({
            "model": self.model,
            "messages": full_history,
            "options": options,
            "stream": false,
        });
        if !completion_request.tools.is_empty() {
            request_payload["tools"] = json!(
                completion_request
                    .tools
                    .into_iter()
                    .map(|tool| tool.into())
                    .collect::<Vec<ToolDefinition>>()
            );
        }

        tracing::debug!(target: "rig", "Chat mode payload: {}", request_payload);

        Ok(request_payload)
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = streaming::StreamingCompletionResponse<CompletionResponse>;

    fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> AsyncTask<Result<completion::CompletionResponse<Self::Response>, CompletionError>> {
        let (tx, task) = runtime::channel();
        let client = self.client.clone();
        let model = self.model.clone();

        match self.create_completion_request(completion_request) {
            Ok(request_payload) => {
                runtime::spawn_async(async move {
                    let response = client.post("api/chat").json(&request_payload).send().await;

                    let result = match response {
                        Ok(response) => {
                            if response.status().is_success() {
                                let text = match response.text().await {
                                    Ok(text) => text,
                                    Err(e) => {
                                        tx.finish(Err(CompletionError::ProviderError(
                                            e.to_string(),
                                        )));
                                        return;
                                    }
                                };
                                tracing::debug!(target: "rig", "Ollama chat response: {}", text);
                                match serde_json::from_str::<CompletionResponse>(&text) {
                                    Ok(chat_resp) => chat_resp.try_into(),
                                    Err(e) => Err(CompletionError::ProviderError(e.to_string())),
                                }
                            } else {
                                let err_text = match response.text().await {
                                    Ok(text) => text,
                                    Err(e) => e.to_string(),
                                };
                                Err(CompletionError::ProviderError(err_text))
                            }
                        }
                        Err(e) => Err(CompletionError::ProviderError(e.to_string())),
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
            Ok(mut request_payload) => {
                json_util::merge_inplace(&mut request_payload, json!({"stream": true}));

                runtime::spawn_async(async move {
                    let builder = client.post("api/chat").json(&request_payload);
                    let result = streaming::send_ollama_streaming_request(builder).await;
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

// Embedding Model
// EmbeddingModel is now imported from fluent_ai_domain::model
// Removed duplicated EmbeddingModel struct - use canonical domain type

/// Ollama embedding model implementation
#[derive(Debug, Clone)]
pub struct OllamaEmbeddingModel {
    client: Client,
    model: String,
    ndims: usize,
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_owned(),
            ndims,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f64>>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message)
    }
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

impl EmbeddingModelTrait for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    fn ndims(&self) -> usize {
        self.ndims
    }

    fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> AsyncTask<Result<Vec<Embedding>, EmbeddingError>> {
        let (tx, task) = runtime::channel();
        let client = self.client.clone();
        let model = self.model.clone();
        let docs: Vec<String> = documents.into_iter().collect();

        runtime::spawn_async(async move {
            let payload = json!({
                "model": model,
                "input": docs.clone(),
            });

            let response = client.post("api/embed").json(&payload).send().await;

            let result = match response {
                Ok(response) => {
                    if response.status().is_success() {
                        match response.json::<EmbeddingResponse>().await {
                            Ok(api_resp) => {
                                if api_resp.embeddings.len() != docs.len() {
                                    Err(EmbeddingError::ResponseError(
                                        "Number of returned embeddings does not match input".into(),
                                    ))
                                } else {
                                    Ok(api_resp
                                        .embeddings
                                        .into_iter()
                                        .zip(docs.into_iter())
                                        .map(|(vec, document)| Embedding { document, vec })
                                        .collect())
                                }
                            }
                            Err(e) => Err(EmbeddingError::ProviderError(e.to_string())),
                        }
                    } else {
                        match response.text().await {
                            Ok(text) => Err(EmbeddingError::ProviderError(text)),
                            Err(e) => Err(EmbeddingError::ProviderError(e.to_string())),
                        }
                    }
                }
                Err(e) => Err(EmbeddingError::ProviderError(e.to_string())),
            };

            tx.finish(result);
        });

        task
    }
}

// ============================================================================
// Tool Definition Conversion
// ============================================================================

/// Ollama-required tool definition format.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub type_field: String, // Fixed as "function"
    pub function: completion::ToolDefinition,
}

/// Convert internal ToolDefinition (from the completion module) into Ollama's tool definition.
impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        ToolDefinition {
            type_field: "function".to_owned(),
            function: completion::ToolDefinition {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    // pub id: String,
    #[serde(default, rename = "type")]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: Value,
}

// ============================================================================
// Provider Message Definition
// ============================================================================
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ProviderMessage {
    User {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(default)]
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, deserialize_with = "json_util::null_or_vec")]
        tool_calls: Vec<ToolCall>,
    },
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "tool")]
    ToolResult { name: String, content: String },
}

/// Conversion from an internal Rig message (crate::message::Message) to a provider Message.
/// (Only User and Assistant variants are supported.)
impl TryFrom<Message> for ProviderMessage {
    type Error = MessageError;

    fn try_from(internal_msg: Message) -> Result<Self, Self::Error> {
        match internal_msg {
            Message::User { content, .. } => {
                let mut texts = Vec::new();
                let mut images = Vec::new();
                for uc in content.into_iter() {
                    match uc {
                        UserContent::Text(t) => texts.push(t.text),
                        UserContent::Image(img) => images.push(img.data),
                        UserContent::ToolResult(result) => {
                            let content = result
                                .content
                                .into_iter()
                                .map(OllamaToolResultContent::try_from)
                                .collect::<Result<Vec<OllamaToolResultContent>, MessageError>>()?;

                            let content = OneOrMany::many(content).map_err(|x| {
                                MessageError::ConversionError(format!(
                                    "Couldn't make a OneOrMany from a list of tool results: {x}"
                                ))
                            })?;

                            return Ok(ProviderMessage::ToolResult {
                                name: result.id,
                                content: content.first().text,
                            });
                        }
                        _ => {} // Audio variant removed since Ollama API does not support it.
                    }
                }
                let content_str = texts.join(" ");
                let images_opt = if images.is_empty() {
                    None
                } else {
                    Some(images)
                };
                Ok(ProviderMessage::User {
                    content: content_str,
                    images: images_opt,
                    name: None,
                })
            }
            Message::Assistant { content, .. } => {
                let mut texts = Vec::new();
                let mut tool_calls = Vec::new();
                for ac in content.into_iter() {
                    match ac {
                        message::AssistantContent::Text(t) => texts.push(t.text),
                        message::AssistantContent::ToolCall(tc) => {
                            tool_calls.push(ToolCall {
                                r#type: ToolType::Function, // Assuming internal tool call provides these fields
                                function: Function {
                                    name: tc.function.name,
                                    arguments: tc.function.arguments,
                                },
                            });
                        }
                    }
                }
                let content_str = texts.join(" ");
                Ok(ProviderMessage::Assistant {
                    content: content_str,
                    images: None,
                    name: None,
                    tool_calls,
                })
            }
        }
    }
}

/// Conversion from provider Message to a completion message.
/// This is needed so that responses can be converted back into chat history.
impl From<ProviderMessage> for CompletionMessage {
    fn from(msg: ProviderMessage) -> Self {
        match msg {
            ProviderMessage::User { content, .. } => CompletionMessage::User {
                content: OneOrMany::one(message::UserContent::Text(Text { text: content })),
            },
            ProviderMessage::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut assistant_contents =
                    vec![message::AssistantContent::Text(Text { text: content })];
                for tc in tool_calls {
                    assistant_contents.push(message::AssistantContent::tool_call(
                        tc.function.name.clone(),
                        tc.function.name,
                        tc.function.arguments,
                    ));
                }
                CompletionMessage::Assistant {
                    content: OneOrMany::many(assistant_contents).map_err(|e| {
                        CompletionError::ServerError(format!(
                            "Failed to create assistant content: {}",
                            e
                        ))
                    })?,
                }
            }
            // System and ToolResult are converted to User message as needed.
            ProviderMessage::System { content, .. } => CompletionMessage::User {
                content: OneOrMany::one(message::UserContent::Text(Text { text: content })),
            },
            ProviderMessage::ToolResult { name, content } => CompletionMessage::User {
                content: OneOrMany::one(message::UserContent::tool_result(
                    name,
                    OneOrMany::one(message::ToolResultContent::Text(Text { text: content })),
                )),
            },
        }
    }
}

impl ProviderMessage {
    /// Constructs a system message.
    pub fn system(content: &str) -> Self {
        ProviderMessage::System {
            content: content.to_owned(),
            images: None,
            name: None,
        }
    }
}

// ============================================================================
// Additional Message Types
// ============================================================================
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct OllamaToolResultContent {
    text: String,
}

impl TryFrom<ToolResultContent> for OllamaToolResultContent {
    type Error = MessageError;

    fn try_from(value: ToolResultContent) -> Result<Self, Self::Error> {
        let ToolResultContent::Text(Text { text }) = value else {
            return Err(MessageError::ConversionError(
                "Non-text tool results not supported".into(),
            ));
        };

        Ok(Self { text })
    }
}

impl From<String> for OllamaToolResultContent {
    fn from(s: String) -> Self {
        OllamaToolResultContent { text: s }
    }
}
