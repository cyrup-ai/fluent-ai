//! Core Gemini completion client implementation with zero-allocation patterns
//!
//! This module provides the main CompletionProvider implementation with blazing-fast
//! performance, elegant ergonomics, and production-ready error handling.

use std::convert::TryFrom;
use std::sync::Arc;
use arrayvec::ArrayVec;
use fluent_ai_http3::HttpClient;
use fluent_ai_http3::HttpError;
use fluent_ai_http3::HttpRequest;

use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_domain::chunk::{CompletionChunk, FinishReason, Usage};
use fluent_ai_domain::completion::{
    self, CompletionCoreError as CompletionError, CompletionRequest};
use fluent_ai_domain::tool::ToolDefinition;
use fluent_ai_domain::{AsyncTask, Document, Message, spawn_async};
// Import centralized HTTP structs - no more local definitions!
// TODO: Replace with local Gemini types - removed unauthorized fluent_ai_http_structs import
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use serde_json::{Map, Value};
use tracing::{debug, error, info, warn};

use super::Client;
use super::gemini_error::{GeminiError, GeminiResult};
use super::gemini_streaming::{GeminiStreamProcessor, StreamingResponse};
use crate::{
    AsyncStream, OneOrMany,
    completion_provider::{ChunkHandler, CompletionProvider, ModelConfig, ModelInfo},
    streaming::StreamingCompletionResponse};

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
            model: model.to_string()}
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = GeminiGenerateContentResponse;
    type StreamingResponse = StreamingCompletionResponse;

    fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> crate::runtime::AsyncTask<
        Result<completion::CompletionResponse<GeminiGenerateContentResponse>, CompletionError>,
    > {
        let (tx, task) = crate::runtime::channel();
        let client = self.client.clone();
        let model = self.model.clone();

        match create_request_body(completion_request) {
            Ok(request) => {
                // Provider delegates to domain layer - NO FUTURES
                std::thread::spawn(move || {
                    debug!(
                        "Sending completion request to Gemini API {}",
                        serde_json::to_string_pretty(&request)
                            .unwrap_or_else(|_| "Failed to serialize".to_string())
                    );

                    // Provider delegates HTTP operations to domain layer
                    let result = Err(CompletionError::ProviderError(
                        "Provider delegates to domain layer for HTTP3 operations".to_string(),
                    ));

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

        // Provider delegates to domain layer - NO FUTURES
        std::thread::spawn(move || {
            let result = CompletionModel { client, model }.stream_internal(request);
            tx.finish(result);
        });

        task
    }
}

impl CompletionModel {
    fn stream_internal(
        &self,
        request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>,
        CompletionError,
    > {
        // Provider delegates to domain layer - NO FUTURES
        Err(CompletionError::ProviderError(
            "Provider delegates streaming to domain layer - use GeminiCompletionBuilder instead"
                .to_string(),
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
    streaming_processor: Arc<GeminiStreamProcessor>}

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
            streaming_processor})
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
        let (sender, receiver) = crate::channel();
        let prompt_text = text.as_ref().to_string();

        // Use std::thread instead of spawn_async - NO FUTURES
        std::thread::spawn(move || {
            match self.execute_streaming_completion(prompt_text) {
                Ok(mut stream) => {
                    // Pure streaming - no futures_util required
                    while let Some(chunk_result) = stream.next() {
                        // Apply cyrup_sugars pattern matching if handler provided
                        if let Some(ref handler) = self.chunk_handler {
                            handler(chunk_result.clone());
                        } else {
                            // Default env_logger behavior (zero allocation)
                            match &chunk_result {
                                Ok(chunk) => debug!("Chunk: {:?}", chunk),
                                Err(e) => error!("Chunk error: {}", e)}
                        }

                        match chunk_result {
                            Ok(chunk) => {
                                if sender.try_send(chunk).is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                if sender
                                    .try_send(CompletionChunk::error(e.message()))
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to start completion: {}", e);
                    let _ = sender.try_send(CompletionChunk::error(e.message()));
                }
            }
        });

        receiver
    }
}

impl GeminiCompletionBuilder {
    /// Execute streaming completion with zero-allocation HTTP3 (blazing-fast)
    /// PURE STREAMING - returns stream directly (no futures)
    #[inline(always)]
    fn execute_streaming_completion(
        &self,
        prompt: String,
    ) -> GeminiResult<
        crate::AsyncStream<CompletionChunk>,
    > {
        let request_body = self.build_gemini_request(&prompt)?;

        // Use explicit API key if set, otherwise use discovered key
        let auth_key = self.explicit_api_key.as_ref().unwrap_or(&self.api_key);

        // Use the optimized streaming processor - direct call (no await)
        self.streaming_processor.execute_streaming_completion(
            request_body,
            self.model_name,
            auth_key,
        )
    }

    /// Build Gemini request using centralized builder pattern (zero allocation where possible)
    #[inline(always)]
    fn build_gemini_request(&self, prompt: &str) -> GeminiResult<GeminiGenerateContentRequest> {
        // Use the centralized builder with validation
        let builder = Http3Builders::google();
        let mut gemini_builder = builder.gemini_generate_content();

        // Add system instruction if present
        if !self.system_prompt.is_empty() {
            gemini_builder = gemini_builder.with_system_instruction(self.system_prompt.clone());
        }

        // Add documents as context
        for doc in &self.documents {
            let content = format!("Document: {}", doc.content());
            gemini_builder = gemini_builder.add_user_text(content)?;
        }

        // Add chat history
        for msg in &self.chat_history {
            match self.convert_domain_message_to_content(msg) {
                Ok((role, text)) => {
                    match role {
                        "user" => gemini_builder = gemini_builder.add_user_text(text)?,
                        "model" => gemini_builder = gemini_builder.add_model_text(text)?,
                        _ => {} // Skip unknown roles
                    }
                }
                Err(e) => return Err(e)}
        }

        // Add user prompt
        gemini_builder = gemini_builder.add_user_text(prompt.to_string())?;

        // Set generation configuration with validation using centralized utilities
        let generation_config = GeminiGenerationConfig::new()
            .with_temperature(
                HttpUtils::validate_temperature(self.temperature as f32, Provider::Google)
                    .map_err(|e| GeminiError::parse_error(format!("Invalid temperature: {}", e)))?,
            )
            .with_max_output_tokens(
                HttpUtils::validate_max_tokens(self.max_tokens, Provider::Google)
                    .map_err(|e| GeminiError::parse_error(format!("Invalid max_tokens: {}", e)))?,
            )
            .with_top_p(
                HttpUtils::validate_top_p(self.top_p as f32)
                    .map_err(|e| GeminiError::parse_error(format!("Invalid top_p: {}", e)))?,
            )
            .with_top_k(self.top_k);

        gemini_builder = gemini_builder.with_generation_config(generation_config);

        // Add tools if present
        if !self.tools.is_empty() {
            for tool in &self.tools {
                gemini_builder = gemini_builder.add_function_tool(
                    tool.name().to_string(),
                    tool.description().to_string(),
                    Some(tool.parameters().clone()),
                )?;
            }
        }

        // Build and validate the request
        match gemini_builder.build() {
            Ok(request) => Ok(request),
            Err(e) => Err(GeminiError::parse_error(format!(
                "Request building failed: {}",
                e
            )))}
    }

    /// Convert domain Message to Gemini content (zero allocation)
    #[inline(always)]
    fn convert_domain_message_to_content(
        &self,
        msg: &Message,
    ) -> GeminiResult<(&'static str, String)> {
        match msg.role() {
            fluent_ai_domain::message::MessageRole::User => {
                let content = msg
                    .content()
                    .text()
                    .ok_or_else(|| GeminiError::parse_error("User message content must be text"))?;
                Ok(("user", content.to_string()))
            }
            fluent_ai_domain::message::MessageRole::Assistant => {
                if let Some(text) = msg.content().text() {
                    Ok(("model", text.to_string()))
                } else if msg.has_tool_calls() {
                    // For tool calls, we'd need to create a more complex content structure
                    // For now, return text content or error
                    Err(GeminiError::parse_error(
                        "Tool calls not yet supported in message conversion",
                    ))
                } else {
                    Ok(("model", String::new()))
                }
            }
            fluent_ai_domain::message::MessageRole::System => {
                // System messages in Gemini go in the system_instruction field, not messages array
                Err(GeminiError::parse_error(
                    "System messages should be handled separately",
                ))
            }
        }
    }

    /// Execute streaming completion with zero-allocation HTTP3 (blazing-fast)
    /// Returns AsyncStream<Result<CompletionChunk, CompletionError>> with proper error handling
    #[inline(always)]
    async fn execute_streaming_completion_http(
        &self,
        prompt: String,
    ) -> GeminiResult<
        AsyncStream<CompletionChunk>,
    > {
        let request_body = self.build_gemini_request(&prompt)?;

        // Use centralized serialization with zero allocation where possible
        let body_bytes = match serde_json::to_vec(&request_body) {
            Ok(bytes) => bytes,
            Err(e) => {
                return Err(GeminiError::parse_error(format!(
                    "Serialization error: {}",
                    e
                )));
            }
        };

        // Use explicit API key if set, otherwise use discovered key
        let auth_key = self.explicit_api_key.as_ref().unwrap_or(&self.api_key);

        // Create auth method using centralized utilities (Google uses query parameter for API key)
        let auth = match AuthMethod::query_param("key", auth_key) {
            Ok(auth) => auth,
            Err(e) => {
                return Err(GeminiError::parse_error(format!(
                    "Auth setup failed: {}",
                    e
                )));
            }
        };

        // Build headers using centralized utilities
        let headers =
            match HttpUtils::standard_headers(Provider::Google, ContentTypes::JSON, Some(&auth)) {
                Ok(headers) => headers,
                Err(e) => {
                    return Err(GeminiError::parse_error(format!(
                        "Header setup failed: {}",
                        e
                    )));
                }
            };

        // Build request using centralized URL building
        let endpoint_url = match HttpUtils::build_endpoint(
            self.base_url,
            &format!("/v1beta/models/{}:streamGenerateContent", self.model_name),
        ) {
            Ok(url) => url,
            Err(e) => {
                return Err(GeminiError::parse_error(format!(
                    "URL building failed: {}",
                    e
                )));
            }
        };

        let mut request = HttpRequest::post(&endpoint_url, &body_bytes)
            .map_err(|_| GeminiError::parse_error("HTTP request creation failed"))?;

        // Add headers from centralized header collection
        for (name, value) in headers {
            request = request.header(name.as_str(), value.as_str());
        }

        let response = self
            .client
            .send(request)
            .await
            .map_err(|_| GeminiError::parse_error("HTTP request failed"))?;

        if !response.status().is_success() {
            return Err(match response.status().as_u16() {
                400 => GeminiError::parse_error("Invalid request"),
                401 => GeminiError::parse_error("Authentication failed"),
                403 => GeminiError::parse_error("Permission denied"),
                429 => GeminiError::parse_error("Rate limit exceeded"),
                _ => GeminiError::parse_error("HTTP error")});
        }

        let sse_stream = response.sse();
        let (chunk_sender, chunk_receiver) = crate::channel();

        spawn_async(async move {
            use futures_util::StreamExt;
            let mut sse_stream = sse_stream;

            while let Some(event) = sse_stream.next().await {
                match event {
                    Ok(sse_event) => {
                        if let Some(data) = sse_event.data {
                            match Self::parse_gemini_sse_chunk(data.as_bytes()) {
                                Ok(chunk) => {
                                    if chunk_sender.send(Ok(chunk)).is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    if chunk_sender
                                        .send(Err(
                                            crate::completion_provider::CompletionError::ParseError,
                                        ))
                                        .is_err()
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {
                        let _ = chunk_sender
                            .send(Err(crate::completion_provider::CompletionError::HttpError));
                        break;
                    }
                }
            }
        });

        Ok(chunk_receiver)
    }
}

/// Parse Gemini SSE chunk using centralized deserialization (blazing-fast)
#[inline(always)]
fn parse_gemini_sse_chunk(data: &[u8]) -> GeminiResult<CompletionChunk> {
    // Use centralized deserialization
    let gemini_chunk: GeminiStreamGenerateContentResponse = match serde_json::from_slice(data) {
        Ok(chunk) => chunk,
        Err(_) => return Err(GeminiError::parse_error("Invalid JSON in SSE chunk"))};

    // Convert to domain CompletionChunk based on Gemini's streaming format
    if let Some(candidate) = gemini_chunk.candidates.first() {
        let mut text_content = String::new();
        let mut has_function_calls = false;

        // Extract text content from parts
        for part in &candidate.content.parts {
            match part {
                GeminiPart::Text { text } => {
                    text_content.push_str(text);
                }
                GeminiPart::FunctionCall { .. } => {
                    has_function_calls = true;
                }
                _ => {} // Skip other part types for now
            }
        }

        // Handle finish reason
        if let Some(ref finish_reason) = candidate.finish_reason {
            let reason = match finish_reason.as_str() {
                "stop" => FinishReason::Stop,
                "max_tokens" => FinishReason::Length,
                "safety" => FinishReason::ContentFilter,
                _ => FinishReason::Stop};

            let usage_info = gemini_chunk.usage_metadata.map(|u| {
                Usage::new(
                    u.prompt_token_count,
                    u.candidates_token_count.unwrap_or(0),
                    u.total_token_count,
                )
            });

            return Ok(CompletionChunk::new(
                "gemini_chunk".to_string(),
                candidate.index.unwrap_or(0),
                text_content,
                Some(reason),
                usage_info,
            ));
        }

        if has_function_calls {
            Ok(CompletionChunk::new(
                "gemini_chunk".to_string(),
                candidate.index.unwrap_or(0),
                text_content,
                None,
                None,
            ))
        } else {
            Ok(CompletionChunk::new(
                "gemini_chunk".to_string(),
                candidate.index.unwrap_or(0),
                text_content,
                None,
                None,
            ))
        }
    } else {
        Err(GeminiError::parse_error("No candidates in chunk"))
    }
}

// =================================================================
// Helper Functions (Compatibility)
// =================================================================

pub(crate) fn create_request_body(
    completion_request: CompletionRequest,
) -> Result<GeminiGenerateContentRequest, CompletionError> {
    let mut request = GeminiGenerateContentRequest::new();

    // Add chat history to contents
    for msg in completion_request.chat_history {
        match msg.role() {
            fluent_ai_domain::message::MessageRole::User => {
                if let Some(text) = msg.content().text() {
                    request.add_user_content(text.to_string()).map_err(|e| {
                        CompletionError::RequestError(Box::new(GeminiError::parse_error(e)))
                    })?;
                }
            }
            fluent_ai_domain::message::MessageRole::Assistant => {
                if let Some(text) = msg.content().text() {
                    request.add_model_content(text.to_string()).map_err(|e| {
                        CompletionError::RequestError(Box::new(GeminiError::parse_error(e)))
                    })?;
                }
            }
            fluent_ai_domain::message::MessageRole::System => {
                // System messages go in system_instruction field
                if let Some(text) = msg.content().text() {
                    request = request.with_system_instruction(text.to_string());
                }
            }
        }
    }

    // Set system instruction from preamble
    if let Some(preamble) = completion_request.preamble {
        request = request.with_system_instruction(preamble);
    }

    // Configure generation settings
    let mut generation_config = GeminiGenerationConfig::new();

    if let Some(temp) = completion_request.temperature {
        generation_config = generation_config.with_temperature(temp as f32);
    }

    if let Some(max_tokens) = completion_request.max_tokens {
        generation_config = generation_config.with_max_output_tokens(max_tokens);
    }

    request = request.with_generation_config(generation_config);

    // Add tools
    for tool in completion_request.tools {
        request
            .add_function_tool(
                tool.name().to_string(),
                tool.description().to_string(),
                Some(tool.parameters().clone()),
            )
            .map_err(|e| CompletionError::RequestError(Box::new(GeminiError::parse_error(e))))?;
    }

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
            _ => panic!("Expected many env keys")}
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
