//! Zero-allocation Anthropic completion with cyrup_sugars typestate builders and fluent_ai_http3
//!
//! Blazing-fast streaming completions with elegant ergonomics:
//! ```
//! client.completion_model("claude-3-5-sonnet-20241022")
//!     .system_prompt("You are helpful")
//!     .temperature(0.8)
//!     .on_chunk(|chunk| {
//!         Ok => log::info!("Chunk: {:?}", chunk),
//!         Err => log::error!("Error: {:?}", chunk)
//!     })
//!     .prompt("Hello world")
//! ```

use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_domain::chunk::{CompletionChunk, FinishReason, Usage};
use fluent_ai_domain::spawn_async;
use fluent_ai_domain::tool::ToolDefinition;
use fluent_ai_domain::{Document, Message};
// Use local Anthropic request/response types
use super::types::{
    AnthropicCacheControl, AnthropicChatRequest, AnthropicContent, AnthropicContentBlock,
    AnthropicMessage, AnthropicStreamingChoice, AnthropicStreamingChunk,
    AnthropicStreamingDelta, AnthropicSystemMessage, AnthropicThinkingConfig, AnthropicTool,
    AnthropicToolResult, AnthropicToolUse, AnthropicUsage
};
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest};
use log;
use serde_json::Value;

use super::messages::ContentBlock;
use crate::{
    AsyncStream,
    completion_provider::{
        ChunkHandler, CompletionError, CompletionProvider, ModelConfig, ModelInfo}};

/// Maximum messages per completion request (compile-time bounded)
const MAX_MESSAGES: usize = 128;
/// Maximum tools per request (compile-time bounded)  
const MAX_TOOLS: usize = 32;
/// Maximum documents per request (compile-time bounded)
const MAX_DOCUMENTS: usize = 64;
/// Maximum search results per request (compile-time bounded)
const MAX_SEARCH_RESULTS: usize = 16;

/// Get ModelInfo configuration for Anthropic models
#[inline(always)]
pub fn get_model_config(model_name: &'static str) -> &'static ModelConfig {
    crate::model_info::get_model_config(model_name)
}

/// Search result data for citation support (using centralized types)
#[derive(Debug, Clone)]
pub struct SearchResultData {
    pub source: String,
    pub title: String,
    pub content: Vec<ContentBlock>}

/// Zero-allocation Anthropic completion builder with perfect ergonomics
#[derive(Clone)]
pub struct AnthropicCompletionBuilder {
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
    // Prompt caching configuration
    prompt_caching_enabled: bool,
    auto_cache_large_content: bool,
    // Extended thinking configuration
    thinking_enabled: bool,
    thinking_config: Option<AnthropicThinkingConfig>,
    // Search results for citation support
    search_results: ArrayVec<SearchResultData, MAX_SEARCH_RESULTS>}

/// Anthropic-specific builder extensions available only for Anthropic provider
pub trait AnthropicExtensions {
    /// Add search results for citation support
    ///
    /// Search results enable Claude to cite sources properly and provide
    /// natural citations in responses. Each search result includes:
    /// - source: URL or identifier for the source
    /// - title: Descriptive title of the content
    /// - content: Array of content blocks (typically text)
    fn with_search_results(self, search_results: Vec<SearchResultData>) -> Self;

    /// Add a single search result for citation support
    fn with_search_result(
        self,
        source: impl Into<String>,
        title: impl Into<String>,
        content: Vec<ContentBlock>,
    ) -> Self;
}

impl CompletionProvider for AnthropicCompletionBuilder {
    /// Create new Anthropic completion builder with ModelInfo defaults
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
            base_url: "https://api.anthropic.com/v1",
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
            prompt_caching_enabled: false,
            auto_cache_large_content: false,
            thinking_enabled: false,
            thinking_config: None,
            search_results: ArrayVec::new()})
    }

    /// Set explicit API key (takes priority over environment variables)
    #[inline(always)]
    fn api_key(mut self, key: impl Into<String>) -> Self {
        self.explicit_api_key = Some(key.into());
        self
    }

    /// Environment variable names to search for Anthropic API keys (ordered by priority)
    #[inline(always)]
    fn env_api_keys(&self) -> ZeroOneOrMany<String> {
        ZeroOneOrMany::Many(vec![
            "ANTHROPIC_API_KEY".to_string(),
            "CLAUDE_API_KEY".to_string(),
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

impl AnthropicExtensions for AnthropicCompletionBuilder {
    /// Add search results for citation support
    fn with_search_results(mut self, search_results: Vec<SearchResultData>) -> Self {
        for result in search_results {
            if self.search_results.try_push(result).is_err() {
                break; // Silently ignore if too many results
            }
        }
        self
    }

    /// Add a single search result for citation support
    fn with_search_result(
        mut self,
        source: impl Into<String>,
        title: impl Into<String>,
        content: Vec<ContentBlock>,
    ) -> Self {
        let result = SearchResultData {
            source: source.into(),
            title: title.into(),
            content};
        let _ = self.search_results.try_push(result);
        self
    }
}

impl AnthropicCompletionBuilder {
    /// Enable prompt caching for large context efficiency
    #[inline(always)]
    pub fn enable_prompt_caching(mut self) -> Self {
        self.prompt_caching_enabled = true;
        self
    }

    /// Auto-cache large content blocks (>1024 characters)
    #[inline(always)]
    pub fn auto_cache_large_content(mut self) -> Self {
        self.auto_cache_large_content = true;
        self
    }

    /// Enable extended thinking mode with custom configuration
    #[inline(always)]
    pub fn enable_thinking(mut self, config: Option<AnthropicThinkingConfig>) -> Self {
        self.thinking_enabled = true;
        self.thinking_config = config;
        self
    }

    /// Execute streaming completion with zero-allocation HTTP3 (blazing-fast)
    /// Returns AsyncStream<Result<CompletionChunk, CompletionError>> with proper error handling
    #[inline(always)]
    async fn execute_streaming_completion(
        &self,
        prompt: String,
    ) -> Result<AsyncStream<CompletionChunk>, CompletionError> {
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

        // Create auth method using centralized utilities (Anthropic uses x-api-key header)
        let auth = match AuthMethod::api_key(auth_key) {
            Ok(auth) => auth,
            Err(e) => {
                return Err(CompletionError::ProviderUnavailable(format!(
                    "Auth setup failed: {}",
                    e
                )));
            }
        };

        // Build headers using centralized utilities
        let mut headers =
            match HttpUtils::standard_headers(Provider::Anthropic, ContentTypes::JSON, Some(&auth))
            {
                Ok(headers) => headers,
                Err(e) => {
                    return Err(CompletionError::ProviderUnavailable(format!(
                        "Header setup failed: {}",
                        e
                    )));
                }
            };

        // Add Anthropic-specific headers
        let anthropic_version_header = (
            ArrayString::from("anthropic-version").map_err(|_| {
                CompletionError::ProviderUnavailable("Header name too long".to_string())
            })?,
            ArrayString::from("2023-06-01").map_err(|_| {
                CompletionError::ProviderUnavailable("Header value too long".to_string())
            })?,
        );
        if headers.try_push(anthropic_version_header).is_err() {
            return Err(CompletionError::ProviderUnavailable(
                "Too many headers".to_string(),
            ));
        }

        // Build request using centralized URL building
        let endpoint_url = match HttpUtils::build_endpoint(self.base_url, "/messages") {
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
                _ => CompletionError::ProviderUnavailable("HTTP error".to_string())});
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

    /// Build Anthropic request using centralized builder pattern (zero allocation where possible)
    #[inline(always)]
    fn build_request(&self, prompt: &str) -> Result<AnthropicChatRequest<'_>, CompletionError> {
        // Use the centralized builder with validation
        let builder = Http3Builders::anthropic();
        let mut chat_builder = builder.chat(self.model_name);

        // Set system prompt if present
        if !self.system_prompt.is_empty() {
            chat_builder = chat_builder.system_prompt(&self.system_prompt);
        }

        // Add documents as context
        for doc in &self.documents {
            let content = format!("Document: {}", doc.content());
            chat_builder =
                chat_builder.add_text_message("user", Box::leak(content.into_boxed_str()));
        }

        // Add chat history
        for msg in &self.chat_history {
            match self.convert_domain_message_to_content(msg) {
                Ok((role, content)) => {
                    chat_builder = chat_builder.add_message(role, content);
                }
                Err(e) => return Err(e)}
        }

        // Add user prompt
        chat_builder = chat_builder.add_text_message("user", prompt);

        // Set parameters with validation using centralized utilities
        chat_builder = chat_builder
            .temperature(
                HttpUtils::validate_temperature(self.temperature as f32, Provider::Anthropic)
                    .map_err(|e| {
                        CompletionError::InvalidRequest(format!("Invalid temperature: {}", e))
                    })? as f64,
            )
            .max_tokens(
                HttpUtils::validate_max_tokens(self.max_tokens, Provider::Anthropic).map_err(
                    |e| CompletionError::InvalidRequest(format!("Invalid max_tokens: {}", e)),
                )?,
            )
            .top_p(
                HttpUtils::validate_top_p(self.top_p as f32)
                    .map_err(|e| CompletionError::InvalidRequest(format!("Invalid top_p: {}", e)))?
                    as f64,
            )
            .stream(true);

        // Add tools if present
        if !self.tools.is_empty() {
            let anthropic_tools = self.convert_tools()?;
            chat_builder = chat_builder.with_tools(anthropic_tools);
        }

        // Enable thinking mode if configured
        if self.thinking_enabled {
            if let Some(config) = &self.thinking_config {
                chat_builder = chat_builder.enable_thinking_with_config(config.clone());
            } else {
                chat_builder = chat_builder.enable_thinking();
            }
        }

        // Enable prompt caching if configured
        if self.prompt_caching_enabled {
            chat_builder = chat_builder.enable_prompt_caching();
        }

        // Build and validate the request
        match chat_builder.build() {
            Ok(request) => Ok(request),
            Err(e) => Err(CompletionError::InvalidRequest(format!(
                "Request building failed: {}",
                e
            )))}
    }

    /// Convert domain Message to Anthropic content (zero allocation)
    #[inline(always)]
    fn convert_domain_message_to_content(
        &self,
        msg: &Message,
    ) -> Result<(&'static str, AnthropicContent<'_>), CompletionError> {
        match msg.role() {
            fluent_ai_domain::message::MessageRole::User => {
                let content = msg.content().text().ok_or(CompletionError::ParseError)?;
                Ok(("user", AnthropicContent::Text(content)))
            }
            fluent_ai_domain::message::MessageRole::Assistant => {
                if let Some(content) = msg.content().text() {
                    Ok(("assistant", AnthropicContent::Text(content)))
                } else if msg.has_tool_calls() {
                    // For tool calls, we'd need to create a more complex content structure
                    // For now, return text content or error
                    Err(CompletionError::ParseError)
                } else {
                    Ok(("assistant", AnthropicContent::Text("")))
                }
            }
            fluent_ai_domain::message::MessageRole::System => {
                // System messages in Anthropic go in the system field, not messages array
                Err(CompletionError::ParseError)
            }
        }
    }

    /// Convert domain ToolDefinition to Anthropic format using centralized types
    #[inline(always)]
    fn convert_tools(&self) -> Result<ArrayVec<AnthropicTool<'_>, MAX_TOOLS>, CompletionError> {
        let mut tools = ArrayVec::new();

        for tool in &self.tools {
            let anthropic_tool = AnthropicTool {
                name: tool.name(),
                description: tool.description(),
                input_schema: tool.parameters().clone()};

            if tools.try_push(anthropic_tool).is_err() {
                return Err(CompletionError::InvalidRequest(
                    "Too many tools".to_string(),
                ));
            }
        }

        Ok(tools)
    }

    /// Parse Anthropic SSE chunk using centralized deserialization (blazing-fast)
    #[inline(always)]
    fn parse_sse_chunk(data: &[u8]) -> Result<CompletionChunk, CompletionError> {
        // Use centralized deserialization
        let anthropic_chunk: AnthropicStreamingChunk = match serde_json::from_slice(data) {
            Ok(chunk) => chunk,
            Err(_) => return Err(CompletionError::ParseError)};

        // Convert to domain CompletionChunk based on Anthropic's streaming format
        match anthropic_chunk.chunk_type.as_str() {
            "content_block_delta" => {
                if let Some(delta) = anthropic_chunk.delta {
                    let content = delta.text.unwrap_or_default();
                    Ok(CompletionChunk::new(
                        "anthropic_chunk".to_string(),
                        anthropic_chunk.index.unwrap_or(0),
                        content,
                        None,
                        None,
                    ))
                } else {
                    Err(CompletionError::ParseError)
                }
            }
            "content_block_stop" => Ok(CompletionChunk::new(
                "anthropic_chunk".to_string(),
                anthropic_chunk.index.unwrap_or(0),
                String::new(),
                Some(FinishReason::Stop),
                None,
            )),
            "message_delta" => {
                let usage = anthropic_chunk.usage.map(|u| {
                    Usage::new(
                        u.input_tokens,
                        u.output_tokens,
                        u.input_tokens + u.output_tokens,
                    )
                });
                let finish_reason =
                    anthropic_chunk
                        .delta
                        .and_then(|d| d.stop_reason)
                        .map(|r| match r.as_str() {
                            "end_turn" => FinishReason::Stop,
                            "max_tokens" => FinishReason::Length,
                            "tool_use" => FinishReason::ToolCalls,
                            _ => FinishReason::Other});

                Ok(CompletionChunk::new(
                    "anthropic_chunk".to_string(),
                    0,
                    String::new(),
                    finish_reason,
                    usage,
                ))
            }
            _ => {
                // Skip other chunk types (message_start, content_block_start, etc.)
                Ok(CompletionChunk::new(
                    "anthropic_chunk".to_string(),
                    0,
                    String::new(),
                    None,
                    None,
                ))
            }
        }
    }
}

/// Public constructor for Anthropic completion builder
#[inline(always)]
pub fn completion_builder(
    api_key: String,
    model_name: &'static str,
) -> Result<AnthropicCompletionBuilder, CompletionError> {
    AnthropicCompletionBuilder::new(api_key, model_name)
}
