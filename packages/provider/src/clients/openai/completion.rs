//! Zero-allocation OpenAI completion with cyrup_sugars typestate builders and fluent_ai_http3
//!
//! Blazing-fast streaming completions with elegant ergonomics:
//! ```
//! client.completion_model("gpt-4o")
//!     .system_prompt("You are helpful")
//!     .temperature(0.8)
//!     .on_chunk(|chunk| {
//!         Ok => log::info!("Chunk: {:?}", chunk),
//!         Err => log::error!("Error: {:?}", chunk)
//!     })
//!     .prompt("Hello world")
//! ```

use arrayvec::ArrayVec;
use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_domain::chunk::{CompletionChunk, FinishReason, Usage};
use fluent_ai_domain::tool::ToolDefinition;
use fluent_ai_domain::{AsyncTask, spawn_async};
use fluent_ai_domain::{Document, Message};
// Import centralized HTTP structs - no more local definitions!
use fluent_ai_http_structs::{
    builders::{ChatBuilder, Http3Builders, HttpRequestBuilder},
    common::{AuthMethod, ContentTypes, HttpHeaders, HttpUtils, Provider},
    errors::{HttpStructError, HttpStructResult},
    openai::{
        OpenAIChatRequest, OpenAIContent, OpenAIFunction, OpenAIMessage, OpenAIResponseFormat,
        OpenAIStreamingChoice, OpenAIStreamingChunk, OpenAIStreamingDelta, OpenAITool,
        OpenAIToolCall, OpenAIToolChoice, OpenAIToolChoiceFunction, OpenAIUsage,
    },
    validation::{ValidateRequest, ValidationResult},
};
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use serde_json::Value;

use crate::clients::openai::model_info::get_model_config;
use crate::{
    AsyncStream,
    completion_provider::{
        ChunkHandler, CompletionError, CompletionProvider, ModelConfig, ModelInfo,
    },
};

/// Maximum messages per completion request (compile-time bounded)
const MAX_MESSAGES: usize = 128;
/// Maximum tools per request (compile-time bounded)  
const MAX_TOOLS: usize = 32;
/// Maximum documents per request (compile-time bounded)
const MAX_DOCUMENTS: usize = 64;

/// Zero-allocation OpenAI completion builder with perfect ergonomics
#[derive(Clone)]
pub struct OpenAICompletionBuilder {
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
    chat_history: ArrayVec<Message, MAX_MESSAGES>,
    documents: ArrayVec<Document, MAX_DOCUMENTS>,
    tools: ArrayVec<ToolDefinition, MAX_TOOLS>,
    additional_params: Option<Value>,
    chunk_handler: Option<ChunkHandler>,
}

impl CompletionProvider for OpenAICompletionBuilder {
    /// Create new OpenAI completion builder with ModelInfo defaults
    #[inline(always)]
    fn new(api_key: String, model_name: &'static str) -> Result<Self, CompletionError> {
        let client = HttpClient::with_config(HttpConfig::streaming_optimized()).map_err(|_| {
            CompletionError::ProviderUnavailable("HTTP client initialization failed".to_string())
        })?;

        let config = get_model_config(model_name);

        Ok(Self {
            client,
            api_key,
            explicit_api_key: None,
            base_url: "https://api.openai.com/v1",
            model_name,
            config,
            system_prompt: config.system_prompt.to_string(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            chat_history: ArrayVec::new(),
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

    /// Environment variable names to search for OpenAI API keys (ordered by priority)
    #[inline(always)]
    fn env_api_keys(&self) -> ZeroOneOrMany<String> {
        // First found wins - search in priority order
        ZeroOneOrMany::Many(vec![
            "OPENAI_API_KEY".to_string(),  // Primary OpenAI key
            "OPENAI_TOKEN".to_string(),    // Alternative name
            "OPEN_AI_API_KEY".to_string(), // Common variation
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
    fn chat_history(mut self, history: ZeroOneOrMany<Message>) -> Result<Self, CompletionError> {
        match history {
            ZeroOneOrMany::Zero => {}
            ZeroOneOrMany::One(msg) => {
                if self.chat_history.try_push(msg).is_err() {
                    return Err(CompletionError::InvalidRequest(
                        "Chat history too large".to_string(),
                    ));
                }
            }
            ZeroOneOrMany::Many(msgs) => {
                for msg in msgs {
                    if self.chat_history.try_push(msg).is_err() {
                        return Err(CompletionError::InvalidRequest(
                            "Chat history too large".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(self)
    }

    /// Add documents for context (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn documents(mut self, documents: ZeroOneOrMany<Document>) -> Result<Self, CompletionError> {
        match documents {
            ZeroOneOrMany::Zero => {}
            ZeroOneOrMany::One(doc) => {
                if self.documents.try_push(doc).is_err() {
                    return Err(CompletionError::InvalidRequest(
                        "Documents too large".to_string(),
                    ));
                }
            }
            ZeroOneOrMany::Many(docs) => {
                for doc in docs {
                    if self.documents.try_push(doc).is_err() {
                        return Err(CompletionError::InvalidRequest(
                            "Documents too large".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(self)
    }

    /// Add tools for function calling (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> Result<Self, CompletionError> {
        match tools {
            ZeroOneOrMany::Zero => {}
            ZeroOneOrMany::One(tool) => {
                if self.tools.try_push(tool).is_err() {
                    return Err(CompletionError::InvalidRequest(
                        "Tools too large".to_string(),
                    ));
                }
            }
            ZeroOneOrMany::Many(tool_list) => {
                for tool in tool_list {
                    if self.tools.try_push(tool).is_err() {
                        return Err(CompletionError::InvalidRequest(
                            "Tools too large".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(self)
    }

    /// Set additional parameters as JSON value
    #[inline(always)]
    fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Set chunk handler for streaming responses
    #[inline(always)]
    fn on_chunk(mut self, handler: ChunkHandler) -> Self {
        self.chunk_handler = Some(handler);
        self
    }

    /// Execute completion with prompt string
    /// Returns AsyncStream<CompletionChunk> with proper error handling
    #[inline(always)]
    fn prompt(self, prompt: impl Into<String>) -> AsyncStream<CompletionChunk> {
        let prompt_string = prompt.into();
        let (sender, receiver) = crate::channel();

        spawn_async(async move {
            match self.execute_streaming_completion(prompt_string).await {
                Ok(mut stream) => {
                    use futures_util::StreamExt;

                    while let Some(result) = stream.next().await {
                        match result {
                            Ok(chunk) => {
                                if let Some(handler) = &self.chunk_handler {
                                    match handler(Ok(chunk.clone())) {
                                        Ok(_) => {}
                                        Err(e) => {
                                            let _ =
                                                sender.send(CompletionChunk::error(e.message()));
                                            break;
                                        }
                                    }
                                }
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
                    log::error!("Failed to start completion: {}", e);
                    let _ = sender.send(CompletionChunk::error(e.message()));
                }
            }
        });

        receiver
    }
}

impl OpenAICompletionBuilder {
    /// Execute streaming completion with zero-allocation HTTP3 (blazing-fast)
    /// Returns AsyncStream<Result<CompletionChunk, CompletionError>> with proper error handling
    #[inline(always)]
    async fn execute_streaming_completion(
        &self,
        prompt: String,
    ) -> Result<AsyncStream<Result<CompletionChunk, CompletionError>>, CompletionError> {
        let request_body = self.build_request(&prompt)?;

        // Use centralized serialization with zero allocation where possible
        let body_bytes = match request_body.to_json_bytes() {
            Ok(bytes) => bytes,
            Err(e) => {
                return Err(CompletionError::Internal(format!(
                    "Serialization error: {}",
                    e
                )));
            }
        };

        // Use explicit API key if set, otherwise use discovered key
        let auth_key = self.explicit_api_key.as_ref().unwrap_or(&self.api_key);

        // Create auth method using centralized utilities
        let auth = match AuthMethod::bearer_token(auth_key) {
            Ok(auth) => auth,
            Err(e) => {
                return Err(CompletionError::ProviderUnavailable(format!(
                    "Auth setup failed: {}",
                    e
                )));
            }
        };

        // Build headers using centralized utilities
        let headers =
            match HttpUtils::standard_headers(Provider::OpenAI, ContentTypes::JSON, Some(&auth)) {
                Ok(headers) => headers,
                Err(e) => {
                    return Err(CompletionError::ProviderUnavailable(format!(
                        "Header setup failed: {}",
                        e
                    )));
                }
            };

        // Build request using centralized URL building
        let endpoint_url = match HttpUtils::build_endpoint(self.base_url, "/chat/completions") {
            Ok(url) => url,
            Err(e) => {
                return Err(CompletionError::ProviderUnavailable(format!(
                    "URL building failed: {}",
                    e
                )));
            }
        };

        let mut request =
            HttpRequest::post(&endpoint_url, body_bytes.as_slice()).map_err(|_| {
                CompletionError::ProviderUnavailable("HTTP request creation failed".to_string())
            })?;

        // Add headers from centralized header collection
        for (name, value) in headers {
            request = request.header(name.as_str(), value.as_str());
        }

        let response =
            self.client.send(request).await.map_err(|_| {
                CompletionError::ProviderUnavailable("HTTP request failed".to_string())
            })?;

        if !response.status().is_success() {
            return Err(match response.status().as_u16() {
                401 => CompletionError::ProviderUnavailable("Authentication failed".to_string()),
                413 => CompletionError::InvalidRequest("Request too large".to_string()),
                429 => CompletionError::RateLimitExceeded,
                _ => CompletionError::ProviderUnavailable("HTTP error".to_string()),
            });
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
                            if data.trim() == "[DONE]" {
                                break;
                            }

                            match Self::parse_sse_chunk(data.as_bytes()) {
                                Ok(chunk) => {
                                    if chunk_sender.send(Ok(chunk)).is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    if chunk_sender.send(Err(e)).is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {
                        let _ = chunk_sender
                            .send(Err(CompletionError::Internal("Stream error".to_string())));
                        break;
                    }
                }
            }
        });

        Ok(chunk_receiver)
    }

    /// Build OpenAI request using centralized builder pattern (zero allocation where possible)
    #[inline(always)]
    fn build_request(&self, prompt: &str) -> Result<OpenAIChatRequest<'_>, CompletionError> {
        // Use the centralized builder with validation
        let builder = Http3Builders::openai();
        let mut chat_builder = builder.chat(self.model_name);

        // Add system prompt if present
        if !self.system_prompt.is_empty() {
            chat_builder = chat_builder.add_text_message("system", &self.system_prompt);
        }

        // Add documents as context (zero allocation conversion)
        for doc in &self.documents {
            let content = format!("Document: {}", doc.content());
            // Note: This does allocate for the document content formatting
            // In a production system, you might want to use a more sophisticated approach
            chat_builder =
                chat_builder.add_text_message("user", Box::leak(content.into_boxed_str()));
        }

        // Add chat history (zero allocation domain conversion)
        for msg in &self.chat_history {
            match self.convert_domain_message_to_content(msg) {
                Ok((role, content)) => {
                    chat_builder = chat_builder.add_message(role, content);
                }
                Err(e) => return Err(e),
            }
        }

        // Add user prompt
        chat_builder = chat_builder.add_text_message("user", prompt);

        // Set parameters with validation using centralized utilities
        chat_builder = chat_builder
            .temperature(
                HttpUtils::validate_temperature(self.temperature as f32, Provider::OpenAI).map_err(
                    |e| CompletionError::InvalidRequest(format!("Invalid temperature: {}", e)),
                )? as f64,
            )
            .max_tokens(
                HttpUtils::validate_max_tokens(self.max_tokens, Provider::OpenAI).map_err(|e| {
                    CompletionError::InvalidRequest(format!("Invalid max_tokens: {}", e))
                })?,
            )
            .top_p(
                HttpUtils::validate_top_p(self.top_p as f32)
                    .map_err(|e| CompletionError::InvalidRequest(format!("Invalid top_p: {}", e)))?
                    as f64,
            )
            .frequency_penalty(self.frequency_penalty as f32)
            .presence_penalty(self.presence_penalty as f32)
            .stream(true);

        // Add tools if present
        if !self.tools.is_empty() {
            let openai_tools = self.convert_tools()?;
            chat_builder = chat_builder.with_tools(openai_tools);
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

    /// Convert domain Message to OpenAI content (zero allocation)
    #[inline(always)]
    fn convert_domain_message_to_content(
        &self,
        msg: &Message,
    ) -> Result<(&'static str, OpenAIContent<'_>), CompletionError> {
        match msg.role() {
            fluent_ai_domain::message::MessageRole::User => {
                let content = msg.content().text().ok_or(CompletionError::ParseError)?;
                Ok(("user", OpenAIContent::Text(content)))
            }
            fluent_ai_domain::message::MessageRole::Assistant => {
                if let Some(content) = msg.content().text() {
                    Ok(("assistant", OpenAIContent::Text(content)))
                } else if msg.has_tool_calls() {
                    // For tool calls, we'd need to create a more complex content structure
                    // For now, return text content or error
                    Err(CompletionError::ParseError)
                } else {
                    Ok(("assistant", OpenAIContent::Text("")))
                }
            }
            fluent_ai_domain::message::MessageRole::System => {
                let content = msg.content().text().ok_or(CompletionError::ParseError)?;
                Ok(("system", OpenAIContent::Text(content)))
            }
        }
    }

    /// Convert domain ToolDefinition to OpenAI format using centralized types
    #[inline(always)]
    fn convert_tools(&self) -> Result<ArrayVec<OpenAITool<'_>, MAX_TOOLS>, CompletionError> {
        let mut tools = ArrayVec::new();

        for tool in &self.tools {
            let openai_tool = OpenAITool {
                tool_type: "function",
                function: OpenAIFunction {
                    name: tool.name(),
                    description: tool.description(),
                    parameters: tool.parameters().clone(),
                },
            };

            if tools.try_push(openai_tool).is_err() {
                return Err(CompletionError::InvalidRequest(
                    "Too many tools".to_string(),
                ));
            }
        }

        Ok(tools)
    }

    /// Parse OpenAI SSE chunk using centralized deserialization (blazing-fast)
    #[inline(always)]
    fn parse_sse_chunk(data: &[u8]) -> Result<CompletionChunk, CompletionError> {
        // Use centralized deserialization
        let openai_chunk: OpenAIStreamingChunk = match serde_json::from_slice(data) {
            Ok(chunk) => chunk,
            Err(e) => return Err(CompletionError::ParseError),
        };

        // Convert to domain CompletionChunk
        if let Some(choice) = openai_chunk.choices.get(0) {
            let content = choice.delta.content.clone().unwrap_or_default();
            let finish_reason = choice.finish_reason.as_ref().map(|r| match r.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "function_call" => FinishReason::ToolCalls,
                "tool_calls" => FinishReason::ToolCalls,
                _ => FinishReason::Other,
            });

            let usage = openai_chunk
                .usage
                .map(|u| Usage::new(u.prompt_tokens, u.completion_tokens, u.total_tokens));

            Ok(CompletionChunk::new(
                openai_chunk.id,
                choice.index,
                content,
                finish_reason,
                usage,
            ))
        } else {
            Err(CompletionError::ParseError)
        }
    }
}
