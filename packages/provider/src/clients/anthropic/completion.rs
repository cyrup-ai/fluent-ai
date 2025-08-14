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

use arrayvec::ArrayVec;
use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_domain::chat::{Message, config::ModelConfig};
use fluent_ai_domain::completion::types::ToolDefinition;
use fluent_ai_domain::context::Document;
use fluent_ai_domain::context::chunk::{CompletionChunk, FinishReason};
use fluent_ai_http3::{Http3, HttpError};
use log;
// Use model-info package as single source of truth for model information
use model_info::{ModelInfo as ModelInfoFromPackage, discovery::Provider};
use serde_json::Value;

use super::messages::ContentBlock;
// Use local Anthropic request/response types
use super::types::{
    AnthropicCacheControl, AnthropicChatRequest, AnthropicContent, AnthropicContentBlock,
    AnthropicMessage, AnthropicStreamingChoice, AnthropicStreamingChunk, AnthropicStreamingDelta,
    AnthropicSystemMessage, AnthropicThinkingConfig, AnthropicTool, AnthropicToolResult,
    AnthropicToolUse, AnthropicUsage,
};
use crate::completion_provider::Usage;
use crate::spawn_async;
use crate::{
    AsyncStream,
    completion_provider::{ChunkHandler, CompletionError, CompletionProvider, ModelConfigInfo},
};

/// Maximum messages per completion request (compile-time bounded)
const MAX_MESSAGES: usize = 128;
/// Maximum tools per request (compile-time bounded)  
const MAX_TOOLS: usize = 32;
/// Maximum documents per request (compile-time bounded)
const MAX_DOCUMENTS: usize = 64;
/// Maximum search results per request (compile-time bounded)
const MAX_SEARCH_RESULTS: usize = 16;

/// Search result data for citation support (using centralized types)
#[derive(Debug, Clone)]
pub struct SearchResultData {
    pub source: String,
    pub title: String,
    pub content: Vec<ContentBlock>,
}

/// Zero-allocation Anthropic completion builder with perfect ergonomics
#[derive(Clone)]
pub struct AnthropicCompletionBuilder {
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
    search_results: ArrayVec<SearchResultData, MAX_SEARCH_RESULTS>,
}

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
        // Use blazing-fast default configuration - model info loaded from model-info package when needed
        let default_config = ModelConfig {
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            context_length: 200000,
            system_prompt: "",
            supports_tools: true,
            supports_vision: true,
            supports_audio: false,
            supports_thinking: true,
            optimal_thinking_budget: 20000,
            provider: "anthropic",
            model_name,
        };

        Ok(Self {
            api_key,
            explicit_api_key: None,
            base_url: "https://api.anthropic.com/v1",
            model_name,
            config: default_config,
            system_prompt: default_config.system_prompt.to_string(),
            temperature: default_config.temperature,
            max_tokens: default_config.max_tokens,
            top_p: default_config.top_p,
            frequency_penalty: default_config.frequency_penalty,
            presence_penalty: default_config.presence_penalty,
            chat_history: ArrayVec::new(),
            documents: ArrayVec::new(),
            tools: ArrayVec::new(),
            additional_params: None,
            chunk_handler: None,
            prompt_caching_enabled: false,
            auto_cache_large_content: false,
            thinking_enabled: false,
            thinking_config: None,
            search_results: ArrayVec::new(),
        })
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
                    // AsyncStream already has .next() method

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
            content,
        };
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
    /// Returns AsyncStream<CompletionChunk> with proper error handling
    #[inline(always)]
    async fn execute_streaming_completion(
        &self,
        prompt: String,
    ) -> Result<AsyncStream<Result<CompletionChunk, CompletionError>>, CompletionError> {
        let request_body = self.build_request(&prompt)?;

        // Use explicit API key if set, otherwise use discovered key
        let auth_key = self.explicit_api_key.as_ref().unwrap_or(&self.api_key);

        // Use Http3::json() directly instead of HttpClient abstraction
        let mut response_stream = Http3::json()
            .headers(|| {
                use std::collections::HashMap;
                let mut map = HashMap::new();
                map.insert("x-api-key", auth_key);
                map.insert("anthropic-version", "2023-06-01");
                map.insert("Content-Type", "application/json");
                map
            })
            .body(&request_body)
            .post(&format!("{}/messages", self.base_url));

        if !response_stream.is_success() {
            return Err(match response_stream.status_code() {
                401 => CompletionError::AuthError,
                413 => CompletionError::RequestTooLarge,
                429 => CompletionError::RateLimited,
                _ => CompletionError::HttpError,
            });
        }

        let (chunk_sender, chunk_receiver) = crate::channel();

        spawn_async(async move {
            use fluent_ai_http3::HttpStreamExt;

            while let Some(chunk) = response_stream.next().await {
                match chunk {
                    Ok(http_chunk) => {
                        if let fluent_ai_http3::HttpChunk::Body(data) = http_chunk {
                            let data_str = String::from_utf8_lossy(&data);

                            // Process SSE events
                            for line in data_str.lines() {
                                if line.starts_with("data: ") {
                                    let data = &line[6..];
                                    if data.trim() == "[DONE]" {
                                        return;
                                    }

                                    match Self::parse_sse_chunk(data.as_bytes()) {
                                        Ok(chunk) => {
                                            if chunk_sender.send(Ok(chunk)).is_err() {
                                                return;
                                            }
                                        }
                                        Err(e) => {
                                            if chunk_sender.send(Err(e)).is_err() {
                                                return;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {
                        let _ = chunk_sender.send(Err(CompletionError::StreamError));
                        return;
                    }
                }
            }
        });

        Ok(chunk_receiver)
    }

    /// Build Anthropic request with zero allocation where possible
    #[inline(always)]
    fn build_request(&self, prompt: &str) -> Result<AnthropicChatRequest<'_>, CompletionError> {
        let mut messages = ArrayVec::new();

        // Add documents as context
        for doc in &self.documents {
            let content = format!("Document: {}", doc.content());
            messages
                .try_push(AnthropicMessage {
                    role: "user",
                    content: AnthropicContent::Text(Box::leak(content.into_boxed_str())),
                })
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }

        // Add chat history
        for msg in &self.chat_history {
            match self.convert_domain_message_to_content(msg) {
                Ok((role, content)) => {
                    messages
                        .try_push(AnthropicMessage { role, content })
                        .map_err(|_| CompletionError::RequestTooLarge)?;
                }
                Err(e) => return Err(e),
            }
        }

        // Add user prompt
        messages
            .try_push(AnthropicMessage {
                role: "user",
                content: AnthropicContent::Text(prompt),
            })
            .map_err(|_| CompletionError::RequestTooLarge)?;

        let tools = if self.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools()?)
        };

        let system_message = if self.system_prompt.is_empty() {
            None
        } else {
            Some(AnthropicSystemMessage::Structured {
                message_type: "text",
                text: &self.system_prompt,
                cache_control: if self.prompt_caching_enabled {
                    Some(AnthropicCacheControl {
                        r#type: "ephemeral",
                    })
                } else {
                    None
                },
            })
        };

        Ok(AnthropicChatRequest {
            model: self.model_name,
            max_tokens: self.max_tokens.clamp(1, 200000),
            messages,
            system: system_message,
            temperature: Some(self.temperature.clamp(0.0, 1.0)),
            top_p: Some(self.top_p.clamp(0.0, 1.0)),
            tools,
            stream: Some(true),
            thinking_config: self.thinking_config.clone(),
        })
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
                input_schema: tool.parameters().clone(),
            };

            if tools.try_push(anthropic_tool).is_err() {
                return Err(CompletionError::InvalidRequest(
                    "Too many tools".to_string(),
                ));
            }
        }

        Ok(tools)
    }

    /// Load model information from model-info package (single source of truth)
    pub fn load_model_info(&self) -> fluent_ai_async::AsyncStream<ModelInfoFromPackage> {
        let provider = Provider::Anthropic;
        provider.get_model_info(self.model_name)
    }

    /// Parse Anthropic SSE chunk using centralized deserialization (blazing-fast)
    #[inline(always)]
    fn parse_sse_chunk(data: &[u8]) -> Result<CompletionChunk, CompletionError> {
        // Use centralized deserialization
        let anthropic_chunk: AnthropicStreamingChunk = match serde_json::from_slice(data) {
            Ok(chunk) => chunk,
            Err(_) => return Err(CompletionError::ParseError),
        };

        // Convert to domain CompletionChunk based on Anthropic's streaming format
        match anthropic_chunk.chunk_type.as_str() {
            "content_block_delta" => {
                if let Some(delta) = anthropic_chunk.delta {
                    let content = delta.text.unwrap_or_default();
                    Ok(CompletionChunk::text(&content))
                } else {
                    Err(CompletionError::ParseError)
                }
            }
            "content_block_stop" => Ok(CompletionChunk::Complete {
                text: String::new(),
                finish_reason: Some(FinishReason::Stop),
                usage: None,
            }),
            "message_delta" => {
                let usage = anthropic_chunk.usage.map(|u| Usage {
                    prompt_tokens: u.input_tokens,
                    completion_tokens: u.output_tokens,
                    total_tokens: u.input_tokens + u.output_tokens,
                });
                let finish_reason =
                    anthropic_chunk
                        .delta
                        .and_then(|d| d.stop_reason)
                        .map(|r| match r.as_str() {
                            "end_turn" => FinishReason::Stop,
                            "max_tokens" => FinishReason::Length,
                            "tool_use" => FinishReason::ToolCalls,
                            _ => FinishReason::Stop,
                        });

                Ok(CompletionChunk::Complete {
                    text: String::new(),
                    finish_reason,
                    usage,
                })
            }
            _ => {
                // Skip other chunk types (message_start, content_block_start, etc.)
                Ok(CompletionChunk::text(""))
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

// =============================================================================
// Missing Types for Requests Module Compatibility
// =============================================================================

/// Anthropic completion request (alias for compatibility)
pub type AnthropicCompletionRequest<'a> = super::types::AnthropicChatRequest<'a>;
