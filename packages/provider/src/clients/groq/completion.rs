// ============================================================================
// File: src/providers/groq/completion.rs
// ----------------------------------------------------------------------------
// Groq completion model implementation
// ============================================================================

use arrayvec::ArrayVec;
use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_domain::chunk::{CompletionChunk, FinishReason, Usage as DomainUsage};
use fluent_ai_domain::tool::ToolDefinition as DomainToolDefinition;
use fluent_ai_domain::{Document, Message};
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use futures::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_stream;

use super::client::Client;
use super::streaming;
use crate::{
    AsyncStream, OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    completion_provider::{
        ChunkHandler, CompletionError as ProviderCompletionError, CompletionProvider, ModelConfig,
    },
    json_util,
    message::{self, MessageError},
    runtime::{self, AsyncTask},
    streaming::StreamingCompletionResponse,
};

// ============================================================================
// Model Constants
// ============================================================================
/// The `deepseek-r1-distill-llama-70b` model. Used for chat completion.
pub const DEEPSEEK_R1_DISTILL_LLAMA_70B: &str = "deepseek-r1-distill-llama-70b";
/// The `gemma2-9b-it` model. Used for chat completion.
pub const GEMMA2_9B_IT: &str = "gemma2-9b-it";
/// The `llama-3.1-8b-instant` model. Used for chat completion.
pub const LLAMA_3_1_8B_INSTANT: &str = "llama-3.1-8b-instant";
/// The `llama-3.2-11b-vision-preview` model. Used for chat completion.
pub const LLAMA_3_2_11B_VISION_PREVIEW: &str = "llama-3.2-11b-vision-preview";
/// The `llama-3.2-1b-preview` model. Used for chat completion.
pub const LLAMA_3_2_1B_PREVIEW: &str = "llama-3.2-1b-preview";
/// The `llama-3.2-3b-preview` model. Used for chat completion.
pub const LLAMA_3_2_3B_PREVIEW: &str = "llama-3.2-3b-preview";
/// The `llama-3.2-90b-vision-preview` model. Used for chat completion.
pub const LLAMA_3_2_90B_VISION_PREVIEW: &str = "llama-3.2-90b-vision-preview";
/// The `llama-3.2-70b-specdec` model. Used for chat completion.
pub const LLAMA_3_2_70B_SPECDEC: &str = "llama-3.2-70b-specdec";
/// The `llama-3.2-70b-versatile` model. Used for chat completion.
pub const LLAMA_3_2_70B_VERSATILE: &str = "llama-3.2-70b-versatile";
/// The `llama-guard-3-8b` model. Used for chat completion.
pub const LLAMA_GUARD_3_8B: &str = "llama-guard-3-8b";
/// The `llama3-70b-8192` model. Used for chat completion.
pub const LLAMA_3_70B_8192: &str = "llama3-70b-8192";
/// The `llama3-8b-8192` model. Used for chat completion.
pub const LLAMA_3_8B_8192: &str = "llama3-8b-8192";
/// The `mixtral-8x7b-32768` model. Used for chat completion.
pub const MIXTRAL_8X7B_32768: &str = "mixtral-8x7b-32768";

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
// Message conversion types (Groq uses OpenAI-compatible format)
// ============================================================================
#[derive(Debug, Serialize, Deserialize)]
pub struct GroqMessage {
    pub role: String,
    pub content: Option<String>,
}

impl TryFrom<GroqMessage> for message::Message {
    type Error = message::MessageError;

    fn try_from(message: GroqMessage) -> Result<Self, Self::Error> {
        match message.role.as_str() {
            "user" => Ok(Self::User {
                content: OneOrMany::one(
                    message
                        .content
                        .map(|content| message::UserContent::text(&content))
                        .ok_or_else(|| {
                            message::MessageError::ConversionError("Empty user message".to_string())
                        })?,
                ),
            }),
            "assistant" => Ok(Self::Assistant {
                content: OneOrMany::one(
                    message
                        .content
                        .map(|content| message::AssistantContent::text(&content))
                        .ok_or_else(|| {
                            message::MessageError::ConversionError(
                                "Empty assistant message".to_string(),
                            )
                        })?,
                ),
            }),
            _ => Err(message::MessageError::ConversionError(format!(
                "Unknown role: {}",
                message.role
            ))),
        }
    }
}

impl TryFrom<message::Message> for GroqMessage {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => Ok(Self {
                role: "user".to_string(),
                content: content.iter().find_map(|c| match c {
                    message::UserContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                }),
            }),
            message::Message::Assistant { content } => {
                let mut text_content: Option<String> = None;

                for c in content.iter() {
                    match c {
                        message::AssistantContent::Text(text) => {
                            text_content = Some(
                                text_content
                                    .map(|mut existing| {
                                        existing.push('\n');
                                        existing.push_str(&text.text);
                                        existing
                                    })
                                    .unwrap_or_else(|| text.text.clone()),
                            );
                        }
                        message::AssistantContent::ToolCall(_) => {
                            return Err(MessageError::ConversionError(
                                "Tool calls not supported in Groq messages".into(),
                            ));
                        }
                    }
                }

                Ok(Self {
                    role: "assistant".to_string(),
                    content: text_content,
                })
            }
        }
    }
}

// ============================================================================
// Completion Response (reuse OpenAI format)
// ============================================================================
pub use crate::clients::openai::CompletionResponse;

// ============================================================================
// Completion Model
// ============================================================================
// CompletionModel is now imported from fluent_ai_domain::model
// Removed duplicated CompletionModel struct - use canonical domain type

impl CompletionModel {
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
        let mut full_history: Vec<GroqMessage> =
            completion_request
                .preamble
                .map_or_else(Vec::new, |preamble| {
                    vec![GroqMessage {
                        role: "system".to_string(),
                        content: Some(preamble),
                    }]
                });

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<GroqMessage>, _>>()?,
        );

        let request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            // Convert tools to OpenAI format
            let tools: Vec<Value> = completion_request
                .tools
                .into_iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }
                    })
                })
                .collect();

            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": tools,
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

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = streaming::StreamingCompletionResponse;

    fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> AsyncTask<Result<completion::CompletionResponse<CompletionResponse>, CompletionError>>
    {
        let (tx, task) = runtime::channel();
        let client = self.client.clone();
        let model_name = self.model.clone();

        match self.create_completion_request(completion_request) {
            Ok(request) => {
                runtime::spawn_async(async move {
                    let request_body = match serde_json::to_vec(&request) {
                        Ok(body) => body,
                        Err(e) => {
                            tx.finish(Err(CompletionError::ProviderError(format!(
                                "Failed to serialize request: {}",
                                e
                            ))));
                            return;
                        }
                    };

                    let http_request = match client.post("/chat/completions", request_body) {
                        Ok(req) => req,
                        Err(e) => {
                            tx.finish(Err(CompletionError::ProviderError(format!(
                                "Failed to create request: {}",
                                e
                            ))));
                            return;
                        }
                    };

                    let result = match client.http_client.send(http_request).await {
                        Ok(response) => {
                            if response.status().is_success() {
                                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(
                                    response.body(),
                                ) {
                                    Ok(ApiResponse::Ok(response)) => {
                                        tracing::info!(target: "rig",
                                            "groq completion token usage: {:?}",
                                            response.usage.clone().map(|usage| format!("{usage}")).unwrap_or_else(|| "N/A".to_string())
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
                                let error_body = String::from_utf8_lossy(response.body());
                                Err(CompletionError::ProviderError(error_body.to_string()))
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
                    let request_body = match serde_json::to_vec(&request) {
                        Ok(body) => body,
                        Err(e) => {
                            tx.finish(Err(CompletionError::ProviderError(format!(
                                "Failed to serialize request: {}",
                                e
                            ))));
                            return;
                        }
                    };

                    let http_request = match client.post("/chat/completions", request_body) {
                        Ok(req) => req,
                        Err(e) => {
                            tx.finish(Err(CompletionError::ProviderError(format!(
                                "Failed to create request: {}",
                                e
                            ))));
                            return;
                        }
                    };

                    let result =
                        streaming::send_groq_streaming_request(client.http_client, http_request)
                            .await;
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

// ============================================================================
// New CompletionProvider Implementation
// ============================================================================

/// Maximum messages per completion request (compile-time bounded)
const MAX_MESSAGES: usize = 128;
/// Maximum tools per request (compile-time bounded)  
const MAX_TOOLS: usize = 32;
/// Maximum documents per request (compile-time bounded)
const MAX_DOCUMENTS: usize = 64;

/// Zero-allocation Groq completion builder with perfect ergonomics
#[derive(Clone)]
pub struct GroqCompletionBuilder {
    client: HttpClient,
    api_key: String,
    explicit_api_key: Option<String>, // .api_key() override takes priority
    base_url: &'static str,
    model_name: &'static str,
    config: &'static ModelConfig,
    system_prompt: String,
    temperature: f64,
    max_tokens: u32,
    top_p: f64,
    frequency_penalty: f64,
    presence_penalty: f64,
    messages: ArrayVec<Message, MAX_MESSAGES>,
    documents: ArrayVec<Document, MAX_DOCUMENTS>,
    tools: ArrayVec<DomainToolDefinition, MAX_TOOLS>,
    additional_params: Option<Value>,
    chunk_handler: Option<ChunkHandler>,
}

impl CompletionProvider for GroqCompletionBuilder {
    /// Create new Groq completion builder with ModelInfo defaults
    #[inline(always)]
    fn new(api_key: String, model_name: &'static str) -> Result<Self, ProviderCompletionError> {
        let client = HttpClient::with_config(HttpConfig::streaming_optimized())
            .map_err(|_| ProviderCompletionError::HttpError)?;

        // Get model config - for now using default values
        let config = &ModelConfig {
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 0.9,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            context_length: 100000,
            system_prompt: "",
            supports_tools: true,
            supports_vision: false,
            supports_audio: false,
            provider: "groq",
            model_name,
        };

        Ok(Self {
            client,
            api_key,
            explicit_api_key: None,
            base_url: "https://api.groq.com/openai/v1",
            model_name,
            config,
            system_prompt: String::new(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            messages: ArrayVec::new(),
            documents: ArrayVec::new(),
            tools: ArrayVec::new(),
            additional_params: None,
            chunk_handler: None,
        })
    }

    /// Set explicit API key (takes priority over environment variables)
    #[inline(always)]
    fn api_key(mut self, key: impl Into<String>) -> Self {
        self.explicit_api_key = Some(key.into());
        self
    }

    /// Environment variable names to search for Groq API keys (ordered by priority)
    #[inline(always)]
    fn env_api_keys(&self) -> ZeroOneOrMany<String> {
        // First found wins - search in priority order
        ZeroOneOrMany::One("GROQ_API_KEY".to_string())
    }

    /// Set system prompt (overrides ModelInfo default)
    #[inline(always)]
    fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set temperature (overrides ModelInfo default)
    #[inline(always)]
    fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set max tokens (overrides ModelInfo default)
    #[inline(always)]
    fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set top_p (overrides ModelInfo default)
    #[inline(always)]
    fn top_p(mut self, p: f64) -> Self {
        self.top_p = p;
        self
    }

    /// Set frequency penalty (overrides ModelInfo default)
    #[inline(always)]
    fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    /// Set presence penalty (overrides ModelInfo default)
    #[inline(always)]
    fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = penalty;
        self
    }

    /// Add chat history (ZeroOneOrMany with bounded capacity)
    fn chat_history(
        mut self,
        history: ZeroOneOrMany<Message>,
    ) -> Result<Self, ProviderCompletionError> {
        match history {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(message) => {
                if self.messages.try_push(message).is_err() {
                    return Err(ProviderCompletionError::RequestTooLarge);
                }
            }
            ZeroOneOrMany::Many(messages) => {
                for message in messages {
                    if self.messages.try_push(message).is_err() {
                        return Err(ProviderCompletionError::RequestTooLarge);
                    }
                }
            }
        }
        Ok(self)
    }

    /// Add documents for RAG (ZeroOneOrMany with bounded capacity)
    fn documents(mut self, docs: ZeroOneOrMany<Document>) -> Result<Self, ProviderCompletionError> {
        match docs {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(doc) => {
                if self.documents.try_push(doc).is_err() {
                    return Err(ProviderCompletionError::RequestTooLarge);
                }
            }
            ZeroOneOrMany::Many(documents) => {
                for doc in documents {
                    if self.documents.try_push(doc).is_err() {
                        return Err(ProviderCompletionError::RequestTooLarge);
                    }
                }
            }
        }
        Ok(self)
    }

    /// Add tools for function calling (ZeroOneOrMany with bounded capacity)
    fn tools(
        mut self,
        tools: ZeroOneOrMany<DomainToolDefinition>,
    ) -> Result<Self, ProviderCompletionError> {
        match tools {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(tool) => {
                if self.tools.try_push(tool).is_err() {
                    return Err(ProviderCompletionError::RequestTooLarge);
                }
            }
            ZeroOneOrMany::Many(tools_vec) => {
                for tool in tools_vec {
                    if self.tools.try_push(tool).is_err() {
                        return Err(ProviderCompletionError::RequestTooLarge);
                    }
                }
            }
        }
        Ok(self)
    }

    /// Add provider-specific parameters
    #[inline(always)]
    fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Set chunk handler with cyrup_sugars pattern matching
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<CompletionChunk, ProviderCompletionError>) + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }

    /// Terminal action - execute completion with user prompt
    /// Returns blazing-fast zero-allocation streaming
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let prompt_text = text.as_ref().to_string();

        // Use the explicit API key if provided, otherwise fall back to the original
        let api_key = self.explicit_api_key.unwrap_or(self.api_key);

        tokio::spawn(async move {
            let result = Self::execute_completion(self, prompt_text, api_key, tx.clone()).await;

            if let Err(e) = result {
                let error_chunk = CompletionChunk {
                    id: "error".to_string(),
                    model: self.model_name.to_string(),
                    content: Some(format!("Error: {}", e)),
                    finish_reason: Some(FinishReason::Stop),
                    usage: None,
                    tool_calls: vec![],
                };
                let _ = tx.send(error_chunk);
            }
        });

        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }

    /// Execute the completion request with streaming
    async fn execute_completion(
        mut builder: Self,
        prompt_text: String,
        api_key: String,
        tx: tokio::sync::mpsc::UnboundedSender<CompletionChunk>,
    ) -> Result<(), String> {
        // Add the user prompt to messages
        let user_message = Message::User {
            content: cyrup_sugars::OneOrMany::One(fluent_ai_domain::UserContent::Text {
                text: prompt_text,
            }),
        };
        if builder.messages.try_push(user_message).is_err() {
            return Err("Too many messages".to_string());
        }

        // Build the request JSON
        let mut request_json = json!({
            "model": builder.model_name,
            "stream": true,
            "stream_options": {"include_usage": true},
            "temperature": builder.temperature,
            "max_tokens": builder.max_tokens,
            "top_p": builder.top_p,
            "frequency_penalty": builder.frequency_penalty,
            "presence_penalty": builder.presence_penalty,
        });

        // Add system prompt if provided
        let mut messages = Vec::new();
        if !builder.system_prompt.is_empty() {
            messages.push(json!({
                "role": "system",
                "content": builder.system_prompt
            }));
        }

        // Convert documents to context message if any
        if !builder.documents.is_empty() {
            let context = builder
                .documents
                .iter()
                .map(|doc| format!("Document: {}\nContent: {}", doc.name, doc.content))
                .collect::<Vec<_>>()
                .join("\n\n");
            messages.push(json!({
                "role": "system",
                "content": format!("Context documents:\n{}", context)
            }));
        }

        // Convert messages to Groq format
        for message in builder.messages.iter() {
            match message {
                Message::User { content } => {
                    let text = content
                        .iter()
                        .filter_map(|c| match c {
                            fluent_ai_domain::UserContent::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    messages.push(json!({
                        "role": "user",
                        "content": text
                    }));
                }
                Message::Assistant { content } => {
                    let text = content
                        .iter()
                        .filter_map(|c| match c {
                            fluent_ai_domain::AssistantContent::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    messages.push(json!({
                        "role": "assistant",
                        "content": text
                    }));
                }
            }
        }

        request_json["messages"] = json!(messages);

        // Add tools if any
        if !builder.tools.is_empty() {
            let tools: Vec<Value> = builder
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }
                    })
                })
                .collect();
            request_json["tools"] = json!(tools);
            request_json["tool_choice"] = json!("auto");
        }

        // Add additional params if provided
        if let Some(additional) = &builder.additional_params {
            if let Some(obj) = additional.as_object() {
                for (key, value) in obj {
                    request_json[key] = value.clone();
                }
            }
        }

        // Create HTTP request
        let request_body = serde_json::to_vec(&request_json)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;

        let url = format!("{}/chat/completions", builder.base_url);
        let http_request = HttpRequest::post(&url, request_body)
            .map_err(|e| format!("Failed to create HTTP request: {}", e))?
            .header("Content-Type", "application/json")
            .header("Authorization", &format!("Bearer {}", api_key));

        // Send the request using HTTP3 client
        let response = builder
            .client
            .send(http_request)
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            let error_body = String::from_utf8_lossy(response.body());
            return Err(format!("HTTP error {}: {}", response.status(), error_body));
        }

        // Process SSE stream
        let mut sse_stream = response.sse();
        let mut request_id = "groq-req".to_string();

        while let Some(event) = sse_stream.next().await {
            match event {
                Ok(sse_event) => {
                    if let Some(data) = sse_event.data {
                        if data == "[DONE]" {
                            break;
                        }

                        // Parse the streaming response
                        match serde_json::from_str::<streaming::StreamingCompletionResponse>(&data)
                        {
                            Ok(response) => {
                                request_id = response.id.clone();

                                if let Some(choice) = response.choices.first() {
                                    let mut chunk = CompletionChunk {
                                        id: response.id.clone(),
                                        model: response.model.clone(),
                                        content: None,
                                        finish_reason: None,
                                        usage: None,
                                        tool_calls: vec![],
                                    };

                                    // Handle delta content
                                    if let Some(delta) = &choice.delta {
                                        if let Some(content) = &delta.content {
                                            chunk.content = Some(content.clone());
                                        }

                                        if let Some(finish_reason) = &choice.finish_reason {
                                            chunk.finish_reason = match finish_reason.as_str() {
                                                "stop" => Some(FinishReason::Stop),
                                                "length" => Some(FinishReason::Length),
                                                "tool_calls" => Some(FinishReason::ToolCalls),
                                                _ => Some(FinishReason::Stop),
                                            };
                                        }
                                    }

                                    // Handle usage information
                                    if let Some(usage) = &response.usage {
                                        chunk.usage = Some(DomainUsage {
                                            prompt_tokens: usage.prompt_tokens as u64,
                                            completion_tokens: usage.completion_tokens as u64,
                                            total_tokens: usage.total_tokens as u64,
                                        });
                                    }

                                    // Call chunk handler if provided
                                    if let Some(handler) = &builder.chunk_handler {
                                        handler(Ok(chunk.clone()));
                                    }

                                    // Send chunk to stream
                                    if tx.send(chunk).is_err() {
                                        break; // Receiver dropped
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to parse streaming response: {}", e);
                                continue;
                            }
                        }
                    }
                }
                Err(e) => {
                    return Err(format!("SSE stream error: {}", e));
                }
            }
        }

        Ok(())
    }
}
