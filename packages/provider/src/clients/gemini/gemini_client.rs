//! Core Gemini completion client implementation with zero-allocation patterns
//!
//! This module provides the main CompletionProvider implementation with blazing-fast
//! performance, elegant ergonomics, and production-ready error handling.

use std::convert::TryFrom;
use std::sync::Arc;

use arrayvec::ArrayVec;
use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_domain::chunk::{CompletionChunk, FinishReason, Usage};
use fluent_ai_domain::tool::ToolDefinition;
use fluent_ai_domain::{AsyncTask, Document, Message, spawn_async};
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use serde_json::{Map, Value};
use tracing::{debug, error, info, warn};

use super::Client;
use super::gemini_error::{GeminiError, GeminiResult};
use super::gemini_streaming::{GeminiStreamProcessor, StreamingResponse};
use super::gemini_types::*;
use fluent_ai_domain::completion::{self, CompletionCoreError as CompletionError, CompletionRequest};
use crate::{
    AsyncStream, OneOrMany,
    completion_provider::{ChunkHandler, CompletionProvider, ModelConfig, ModelInfo},
    streaming::StreamingCompletionResponse,
};

/// Maximum messages per completion request (compile-time bounded)
const MAX_MESSAGES: usize = 128;
/// Maximum tools per request (compile-time bounded)  
const MAX_TOOLS: usize = 32;
/// Maximum documents per request (compile-time bounded)
const MAX_DOCUMENTS: usize = 64;

// =================================================================
// Rig Implementation (Compatibility)
// =================================================================

// CompletionModel is now imported from fluent_ai_domain::model
// Removed duplicated CompletionModel struct - use canonical domain type

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = GenerateContentResponse;
    type StreamingResponse = StreamingCompletionResponse;

    fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> crate::runtime::AsyncTask<
        Result<completion::CompletionResponse<GenerateContentResponse>, CompletionError>,
    > {
        let (tx, task) = crate::runtime::channel();
        let client = self.client.clone();
        let model = self.model.clone();

        match create_request_body(completion_request) {
            Ok(request) => {
                crate::runtime::spawn_async(async move {
                    debug!(
                        "Sending completion request to Gemini API {}",
                        serde_json::to_string_pretty(&request)
                            .unwrap_or_else(|_| "Failed to serialize".to_string())
                    );

                    let response = client
                        .post(&format!("/v1beta/models/{}:generateContent", model))
                        .json(&request)
                        .send()
                        .await;

                    let result = match response {
                        Ok(response) => {
                            if response.status().is_success() {
                                match response.json::<GenerateContentResponse>().await {
                                    Ok(response) => {
                                        match response.usage_metadata {
                                            Some(ref usage) => info!(target: "rig",
                                                "Gemini completion token usage: {}",
                                                usage
                                            ),
                                            None => info!(target: "rig",
                                                "Gemini completion token usage: n/a",
                                            ),
                                        }
                                        debug!("Received response");
                                        completion::CompletionResponse::try_from(response)
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
    ) -> crate::runtime::AsyncTask<
        Result<
            crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
            CompletionError,
        >,
    > {
        let (tx, task) = crate::runtime::channel();
        let client = self.client.clone();
        let model = self.model.clone();

        crate::runtime::spawn_async(async move {
            let result = CompletionModel { client, model }
                .stream_internal(request)
                .await;
            tx.finish(result);
        });

        task
    }
}

impl CompletionModel {
    async fn stream_internal(
        &self,
        request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>,
        CompletionError,
    > {
        // This would be implemented for backward compatibility
        // For now, return a basic error to encourage migration to new API
        Err(CompletionError::ProviderError(
            "Streaming not implemented - use GeminiCompletionBuilder instead".to_string(),
        ))
    }
}

// =================================================================
// Modern CompletionProvider Implementation
// =================================================================

/// Zero-allocation Gemini completion builder with perfect ergonomics
#[derive(Clone)]
pub struct GeminiCompletionBuilder {
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
    top_k: u32,
    frequency_penalty: f64,
    presence_penalty: f64,
    chat_history: ArrayVec<Message, MAX_MESSAGES>,
    documents: ArrayVec<Document, MAX_DOCUMENTS>,
    tools: ArrayVec<ToolDefinition, MAX_TOOLS>,
    additional_params: Option<Value>,
    chunk_handler: Option<ChunkHandler>,
    streaming_processor: Arc<GeminiStreamProcessor>,
}

impl CompletionProvider for GeminiCompletionBuilder {
    /// Create new Gemini completion builder with ModelInfo defaults
    #[inline(always)]
    fn new(
        api_key: String,
        model_name: &'static str,
    ) -> Result<Self, crate::completion_provider::CompletionError> {
        let client = HttpClient::with_config(HttpConfig::streaming_optimized())
            .map_err(|_| crate::completion_provider::CompletionError::HttpError)?;

        let config = crate::clients::gemini::model_info::get_model_config(model_name);
        let streaming_processor = Arc::new(GeminiStreamProcessor::new(client.clone()));

        Ok(Self {
            client,
            api_key,
            explicit_api_key: None,
            base_url: "https://generativelanguage.googleapis.com",
            model_name,
            config,
            system_prompt: config.system_prompt.to_string(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            top_k: 40, // Gemini default
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            chat_history: ArrayVec::new(),
            documents: ArrayVec::new(),
            tools: ArrayVec::new(),
            additional_params: None,
            chunk_handler: None,
            streaming_processor,
        })
    }

    /// Set explicit API key (takes priority over environment variables)
    #[inline(always)]
    fn api_key(mut self, key: impl Into<String>) -> Self {
        self.explicit_api_key = Some(key.into());
        self
    }

    /// Environment variable names to search for Gemini API keys (ordered by priority)
    #[inline(always)]
    fn env_api_keys(&self) -> ZeroOneOrMany<String> {
        // First found wins - search in priority order
        ZeroOneOrMany::Many(vec![
            "GEMINI_API_KEY".to_string(),        // Primary Gemini key
            "GOOGLE_GEMINI_API_KEY".to_string(), // Google-specific Gemini key
            "GOOGLE_AI_API_KEY".to_string(),     // Google AI Platform key
            "GOOGLE_API_KEY".to_string(),        // General Google API key
        ])
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
    #[inline(always)]
    fn chat_history(
        mut self,
        history: ZeroOneOrMany<Message>,
    ) -> Result<Self, crate::completion_provider::CompletionError> {
        match history {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(msg) => {
                self.chat_history
                    .try_push(msg)
                    .map_err(|_| crate::completion_provider::CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(msgs) => {
                for msg in msgs {
                    self.chat_history.try_push(msg).map_err(|_| {
                        crate::completion_provider::CompletionError::RequestTooLarge
                    })?;
                }
            }
        }
        Ok(self)
    }

    /// Add documents for RAG (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn documents(
        mut self,
        docs: ZeroOneOrMany<Document>,
    ) -> Result<Self, crate::completion_provider::CompletionError> {
        match docs {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(doc) => {
                self.documents
                    .try_push(doc)
                    .map_err(|_| crate::completion_provider::CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(documents) => {
                for doc in documents {
                    self.documents.try_push(doc).map_err(|_| {
                        crate::completion_provider::CompletionError::RequestTooLarge
                    })?;
                }
            }
        }
        Ok(self)
    }

    /// Add tools for function calling (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn tools(
        mut self,
        tools: ZeroOneOrMany<ToolDefinition>,
    ) -> Result<Self, crate::completion_provider::CompletionError> {
        match tools {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(tool) => {
                self.tools
                    .try_push(tool)
                    .map_err(|_| crate::completion_provider::CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(tool_list) => {
                for tool in tool_list {
                    self.tools.try_push(tool).map_err(|_| {
                        crate::completion_provider::CompletionError::RequestTooLarge
                    })?;
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

    /// Set chunk handler with cyrup_sugars pattern matching syntax
    #[inline(always)]
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<CompletionChunk, crate::completion_provider::CompletionError>)
            + Send
            + Sync
            + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }

    /// Terminal action - execute completion with user prompt (blazing-fast streaming)
    #[inline(always)]
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk> {
        let (sender, receiver) = crate::async_stream_channel();
        let prompt_text = text.as_ref().to_string();

        spawn_async(async move {
            match self.execute_streaming_completion(prompt_text).await {
                Ok(stream) => {
                    use futures_util::StreamExt;
                    let mut stream = Box::pin(stream);

                    while let Some(chunk_result) = stream.next().await {
                        // Apply cyrup_sugars pattern matching if handler provided
                        if let Some(ref handler) = self.chunk_handler {
                            handler(chunk_result.clone());
                        } else {
                            // Default env_logger behavior (zero allocation)
                            match &chunk_result {
                                Ok(chunk) => debug!("Chunk: {:?}", chunk),
                                Err(e) => error!("Chunk error: {}", e),
                            }
                        }

                        match chunk_result {
                            Ok(chunk) => {
                                if sender.send(chunk).is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                if sender.send(CompletionChunk::error(e.message())).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to start completion: {}", e);
                    let _ = sender.send(CompletionChunk::error(e.message()));
                }
            }
        });

        receiver
    }
}

impl GeminiCompletionBuilder {
    /// Execute streaming completion with zero-allocation HTTP3 (blazing-fast)
    #[inline(always)]
    async fn execute_streaming_completion(
        &self,
        prompt: String,
    ) -> GeminiResult<
        crate::AsyncStream<Result<CompletionChunk, crate::completion_provider::CompletionError>>,
    > {
        let request_body = self.build_gemini_request(&prompt)?;

        // Use explicit API key if set, otherwise use discovered key
        let auth_key = self.explicit_api_key.as_ref().unwrap_or(&self.api_key);

        // Use the optimized streaming processor
        let stream = self
            .streaming_processor
            .execute_streaming_completion(request_body, self.model_name, auth_key)
            .await?;

        Ok(stream)
    }

    /// Build Gemini request with zero allocation where possible
    #[inline(always)]
    fn build_gemini_request(&self, prompt: &str) -> GeminiResult<GenerateContentRequest> {
        let mut contents = Vec::new();

        // Add documents as context
        for doc in &self.documents {
            contents.push(Content {
                parts: OneOrMany::one(format!("Document: {}", doc.content()).into()),
                role: Some(Role::User),
            });
        }

        // Add chat history
        for msg in &self.chat_history {
            contents.push(self.convert_domain_message_to_gemini(msg)?);
        }

        // Add user prompt
        contents.push(Content {
            parts: OneOrMany::one(prompt.to_string().into()),
            role: Some(Role::User),
        });

        let system_instruction = if !self.system_prompt.is_empty() {
            Some(Content {
                parts: OneOrMany::one(self.system_prompt.clone().into()),
                role: Some(Role::Model),
            })
        } else {
            None
        };

        let mut generation_config = GenerationConfig::default();
        generation_config.temperature = Some(self.temperature);
        generation_config.max_output_tokens = Some(self.max_tokens as u64);
        generation_config.top_p = Some(self.top_p);
        generation_config.top_k = Some(self.top_k as i32);

        let tools = if self.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools_to_gemini()?)
        };

        Ok(GenerateContentRequest {
            contents,
            generation_config: Some(generation_config),
            safety_settings: None,
            tools,
            tool_config: None,
            system_instruction,
        })
    }

    /// Convert domain Message to Gemini format
    #[inline(always)]
    fn convert_domain_message_to_gemini(&self, msg: &Message) -> GeminiResult<Content> {
        match msg.role() {
            fluent_ai_domain::message::MessageRole::User => {
                let content = msg
                    .content()
                    .text()
                    .ok_or_else(|| GeminiError::parse_error("Message content must be text"))?;
                Ok(Content {
                    parts: OneOrMany::one(content.to_string().into()),
                    role: Some(Role::User),
                })
            }
            fluent_ai_domain::message::MessageRole::Assistant => {
                let mut parts = Vec::new();

                if let Some(text) = msg.content().text() {
                    parts.push(Part::Text(text.to_string()));
                }

                for tool_call in msg.tool_calls() {
                    parts.push(Part::FunctionCall(FunctionCall {
                        name: tool_call.function().name().to_string(),
                        args: tool_call.function().arguments().clone(),
                    }));
                }

                Ok(Content {
                    parts: OneOrMany::many(parts).map_err(|_| {
                        GeminiError::parse_error("Failed to convert assistant message parts")
                    })?,
                    role: Some(Role::Model),
                })
            }
            fluent_ai_domain::message::MessageRole::System => {
                let content = msg.content().text().ok_or_else(|| {
                    GeminiError::parse_error("System message content must be text")
                })?;
                Ok(Content {
                    parts: OneOrMany::one(content.to_string().into()),
                    role: Some(Role::Model),
                })
            }
        }
    }

    /// Convert domain ToolDefinition to Gemini format
    #[inline(always)]
    fn convert_tools_to_gemini(&self) -> GeminiResult<Vec<Tool>> {
        let mut tools = Vec::new();

        for tool in &self.tools {
            let parameters: Option<Schema> =
                if tool.parameters() == &serde_json::json!({"type": "object", "properties": {}}) {
                    None
                } else {
                    Some(tool.parameters().clone().try_into()?)
                };

            tools.push(Tool {
                function_declarations: FunctionDeclaration {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters,
                },
                code_execution: None,
            });
        }

        Ok(tools)
    }
}

// =================================================================
// Helper Functions (Compatibility)
// =================================================================

pub(crate) fn create_request_body(
    completion_request: CompletionRequest,
) -> Result<GenerateContentRequest, CompletionError> {
    let mut full_history = Vec::new();
    full_history.extend(completion_request.chat_history);

    let additional_params = completion_request
        .additional_params
        .unwrap_or_else(|| Value::Object(Map::new()));

    let mut generation_config = serde_json::from_value::<GenerationConfig>(additional_params)?;

    if let Some(temp) = completion_request.temperature {
        generation_config.temperature = Some(temp);
    }

    if let Some(max_tokens) = completion_request.max_tokens {
        generation_config.max_output_tokens = Some(max_tokens);
    }

    let system_instruction = completion_request.preamble.clone().map(|preamble| Content {
        parts: OneOrMany::one(preamble.into()),
        role: Some(Role::Model),
    });

    let request = GenerateContentRequest {
        contents: full_history
            .into_iter()
            .map(|msg| {
                msg.try_into()
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))
            })
            .collect::<Result<Vec<_>, _>>()?,
        generation_config: Some(generation_config),
        safety_settings: None,
        tools: Some(
            completion_request
                .tools
                .into_iter()
                .map(Tool::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        tool_config: None,
        system_instruction,
    };

    Ok(request)
}

/// Public constructor for Gemini completion builder
#[inline(always)]
pub fn completion_builder(
    api_key: String,
    model_name: &'static str,
) -> Result<GeminiCompletionBuilder, crate::completion_provider::CompletionError> {
    GeminiCompletionBuilder::new(api_key, model_name)
}

#[cfg(test)]
mod tests {
    use fluent_ai_http3::HttpConfig;

    use super::*;

    #[tokio::test]
    async fn test_completion_builder_creation() {
        let builder = completion_builder("test-key".to_string(), GEMINI_1_5_FLASH);
        assert!(builder.is_ok());

        let builder = builder.unwrap();
        assert_eq!(builder.model_name, GEMINI_1_5_FLASH);
        assert_eq!(builder.api_key, "test-key");
    }

    #[test]
    fn test_completion_model_creation() {
        let client = Client::new("test-url", "test-key");
        let model = CompletionModel::new(client, GEMINI_1_5_PRO);
        assert_eq!(model.model, GEMINI_1_5_PRO);
    }

    #[test]
    fn test_env_api_keys() {
        let builder = completion_builder("test-key".to_string(), GEMINI_1_5_FLASH).unwrap();
        let env_keys = builder.env_api_keys();

        match env_keys {
            ZeroOneOrMany::Many(keys) => {
                assert!(keys.contains(&"GEMINI_API_KEY".to_string()));
                assert!(keys.contains(&"GOOGLE_AI_API_KEY".to_string()));
            }
            _ => panic!("Expected many env keys"),
        }
    }

    #[tokio::test]
    async fn test_builder_configuration() {
        let builder = completion_builder("test-key".to_string(), GEMINI_1_5_FLASH)
            .unwrap()
            .system_prompt("Test system prompt")
            .temperature(0.7)
            .max_tokens(1000)
            .top_p(0.9);

        assert_eq!(builder.system_prompt, "Test system prompt");
        assert_eq!(builder.temperature, 0.7);
        assert_eq!(builder.max_tokens, 1000);
        assert_eq!(builder.top_p, 0.9);
    }
}
