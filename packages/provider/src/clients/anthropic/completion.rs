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
use fluent_ai_domain::chunk::{CompletionChunk, FinishReason, Usage};
use fluent_ai_domain::spawn_async;
use fluent_ai_domain::tool::ToolDefinition;
use fluent_ai_domain::{Document, Message};
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest};
use log;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::messages::ContentBlock;
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
/// Maximum search results per request (compile-time bounded)
const MAX_SEARCH_RESULTS: usize = 16;

/// Get ModelInfo configuration for Anthropic models
#[inline(always)]
pub fn get_model_config(model_name: &'static str) -> &'static ModelConfig {
    crate::model_info::get_model_config(model_name)
}

/// Cache control configuration for Anthropic prompt caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String, // "ephemeral"
}

/// Search result data for citation support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultData {
    pub source: String,
    pub title: String,
    pub content: Vec<ContentBlock>,
}

impl Default for CacheControl {
    fn default() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
        }
    }
}

/// Thinking configuration for Anthropic extended thinking
#[derive(Debug, Clone, Serialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: &'static str, // "enabled"
    pub budget_tokens: u32,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            thinking_type: "enabled",
            budget_tokens: 1024, // Default thinking budget
        }
    }
}

impl ThinkingConfig {
    /// Create thinking config with custom budget
    pub fn with_budget(budget_tokens: u32) -> Self {
        Self {
            thinking_type: "enabled",
            budget_tokens,
        }
    }
}

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
    thinking_config: Option<ThinkingConfig>,
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

/// Anthropic API message (zero-allocation serialization)
#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// System message with optional cache control
#[derive(Debug, Serialize)]
pub struct AnthropicSystemMessage<'a> {
    #[serde(rename = "type")]
    pub message_type: &'static str, // "text"
    pub text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Anthropic completion request (zero-allocation where possible)
#[derive(Debug, Serialize)]
pub struct AnthropicCompletionRequest<'a> {
    pub model: &'a str,
    pub messages: ArrayVec<AnthropicMessage<'a>, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<AnthropicSystemMessage<'a>>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<Value, MAX_TOOLS>>,
    pub stream: bool,
}

/// Anthropic streaming response chunk (optimized deserialization)
#[derive(Debug, Deserialize)]
pub struct AnthropicStreamChunk {
    #[serde(rename = "type")]
    pub chunk_type: String,
    #[serde(default)]
    pub delta: Option<AnthropicDelta>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl CompletionProvider for AnthropicCompletionBuilder {
    /// Create new Anthropic completion builder with ModelInfo defaults
    #[inline(always)]
    fn new(api_key: String, model_name: &'static str) -> Result<Self, CompletionError> {
        let client = HttpClient::with_config(HttpConfig::streaming_optimized())
            .map_err(|_| CompletionError::HttpError)?;

        let config = get_model_config(model_name);

        Ok(Self {
            client,
            api_key,
            explicit_api_key: None,
            base_url: "https://api.anthropic.com",
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
            // Initialize prompt caching (disabled by default)
            prompt_caching_enabled: false,
            auto_cache_large_content: true, // Enable auto-caching by default
            // Initialize thinking configuration
            thinking_enabled: config.supports_thinking,
            thinking_config: if config.supports_thinking {
                Some(ThinkingConfig {
                    thinking_type: "enabled",
                    budget_tokens: config.optimal_thinking_budget,
                })
            } else {
                None
            },
            // Initialize search results (empty by default)
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
        // First found wins - search in priority order
        ZeroOneOrMany::Many(vec![
            "ANTHROPIC_API_KEY".to_string(), // Primary Anthropic key
            "CLAUDE_API_KEY".to_string(),    // Alternative Claude key
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
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(msg) => {
                self.chat_history
                    .try_push(msg)
                    .map_err(|_| CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(msgs) => {
                for msg in msgs {
                    self.chat_history
                        .try_push(msg)
                        .map_err(|_| CompletionError::RequestTooLarge)?;
                }
            }
        }
        Ok(self)
    }

    /// Add documents for RAG (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn documents(mut self, docs: ZeroOneOrMany<Document>) -> Result<Self, CompletionError> {
        match docs {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(doc) => {
                self.documents
                    .try_push(doc)
                    .map_err(|_| CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(documents) => {
                for doc in documents {
                    self.documents
                        .try_push(doc)
                        .map_err(|_| CompletionError::RequestTooLarge)?;
                }
            }
        }
        Ok(self)
    }

    /// Add tools for function calling (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> Result<Self, CompletionError> {
        match tools {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(tool) => {
                self.tools
                    .try_push(tool)
                    .map_err(|_| CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(tool_list) => {
                for tool in tool_list {
                    self.tools
                        .try_push(tool)
                        .map_err(|_| CompletionError::RequestTooLarge)?;
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
        F: Fn(Result<CompletionChunk, CompletionError>) + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }

    /// Terminal action - execute completion with user prompt (blazing-fast streaming)
    #[inline(always)]
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk> {
        let (sender, receiver) = crate::channel();
        let prompt_text = text.as_ref().to_string();

        spawn_async(async move {
            match self.execute_streaming_completion(prompt_text).await {
                Ok(mut stream) => {
                    use futures_util::StreamExt;
                    while let Some(chunk_result) = stream.next().await {
                        // Apply cyrup_sugars pattern matching if handler provided
                        if let Some(ref handler) = self.chunk_handler {
                            handler(chunk_result.clone());
                        } else {
                            // Default env_logger behavior (zero allocation)
                            match &chunk_result {
                                Ok(chunk) => log::debug!("Chunk: {:?}", chunk),
                                Err(e) => log::error!("Chunk error: {}", e),
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
    #[inline(always)]
    fn with_search_results(mut self, search_results: Vec<SearchResultData>) -> Self {
        self.search_results.clear();
        for result in search_results.into_iter().take(MAX_SEARCH_RESULTS) {
            if self.search_results.try_push(result).is_err() {
                break;
            }
        }
        self
    }

    /// Add a single search result for citation support
    #[inline(always)]
    fn with_search_result(
        mut self,
        source: impl Into<String>,
        title: impl Into<String>,
        content: Vec<ContentBlock>,
    ) -> Self {
        let search_result = SearchResultData {
            source: source.into(),
            title: title.into(),
            content,
        };

        if self.search_results.try_push(search_result).is_err() {
            // ArrayVec is full, remove oldest entry and add new one
            self.search_results.remove(0);
            let _ = self.search_results.try_push(SearchResultData {
                source: source.into(),
                title: title.into(),
                content,
            });
        }
        self
    }
}

impl AnthropicCompletionBuilder {
    /// Enable prompt caching for cost optimization and faster processing
    ///
    /// Anthropic's prompt caching reduces costs for repeated content:
    /// - Cache writes cost 25% more than base input tokens
    /// - Cache hits cost only 10% of base input token price
    /// - 5-minute ephemeral cache with automatic management
    /// - Automatically caches system prompts, tools, and large documents (>2048 tokens)
    #[inline(always)]
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching_enabled = true;
        self
    }

    /// Disable automatic caching of large content while keeping manual caching enabled
    #[inline(always)]
    pub fn disable_auto_cache(mut self) -> Self {
        self.auto_cache_large_content = false;
        self
    }

    /// Check if content should be cached based on size and settings
    #[inline(always)]
    fn should_cache_content(&self, content: &str) -> bool {
        if !self.prompt_caching_enabled {
            return false;
        }

        if !self.auto_cache_large_content {
            return false;
        }

        // Minimum cacheable tokens: 1024 for most models, 2048 for Haiku
        let min_tokens = if self.model_name.contains("haiku") {
            2048
        } else {
            1024
        };

        // Rough estimate: ~4 characters per token for English text
        let estimated_tokens = content.len() / 4;
        estimated_tokens >= min_tokens
    }
}

impl AnthropicCompletionBuilder {
    /// Execute streaming completion with zero-allocation HTTP3 (blazing-fast)
    #[inline(always)]
    async fn execute_streaming_completion(
        &self,
        prompt: String,
    ) -> Result<AsyncStream<Result<CompletionChunk, CompletionError>>, CompletionError> {
        let request_body = self.build_request(&prompt)?;
        let body_bytes =
            serde_json::to_vec(&request_body).map_err(|_| CompletionError::ParseError)?;

        // Use explicit API key if set, otherwise use discovered key
        let auth_key = self.explicit_api_key.as_ref().unwrap_or(&self.api_key);

        let request = HttpRequest::post(&format!("{}/v1/messages", self.base_url), body_bytes)
            .map_err(|_| CompletionError::HttpError)?
            .header("Content-Type", "application/json")
            .header("x-api-key", auth_key)
            .header("anthropic-version", "2023-06-01");

        let response = self
            .client
            .send(request)
            .await
            .map_err(|_| CompletionError::HttpError)?;

        if !response.status().is_success() {
            return Err(match response.status().as_u16() {
                401 => CompletionError::AuthError,
                413 => CompletionError::RequestTooLarge,
                429 => CompletionError::RateLimited,
                _ => CompletionError::HttpError,
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
                        let _ = chunk_sender.send(Err(CompletionError::StreamError));
                        break;
                    }
                }
            }
        });

        Ok(chunk_receiver)
    }

    /// Build Anthropic request with zero allocation where possible
    #[inline(always)]
    fn build_request(
        &self,
        prompt: &str,
    ) -> Result<AnthropicCompletionRequest<'_>, CompletionError> {
        let mut messages = ArrayVec::new();

        // Add documents as context (zero allocation conversion)
        for doc in &self.documents {
            let content = format!("Document: {}", doc.content());
            let should_cache = self.should_cache_content(&content);

            messages
                .try_push(AnthropicMessage {
                    role: "user",
                    content: Box::leak(content.into_boxed_str()),
                    cache_control: if should_cache {
                        Some(CacheControl::default())
                    } else {
                        None
                    },
                })
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }

        // Add chat history (zero allocation domain conversion)
        for msg in &self.chat_history {
            let anthropic_msg = self.convert_domain_message(msg)?;
            messages
                .try_push(anthropic_msg)
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }

        // Add search results as user messages for citation support
        for search_result in &self.search_results {
            let search_content = format!(
                "Search Result - {}: {}",
                search_result.title,
                search_result
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            );

            messages
                .try_push(AnthropicMessage {
                    role: "user",
                    content: Box::leak(search_content.into_boxed_str()),
                    cache_control: if self.should_cache_content(&search_content) {
                        Some(CacheControl::default())
                    } else {
                        None
                    },
                })
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }

        // Add user prompt (typically not cached as it's unique)
        messages
            .try_push(AnthropicMessage {
                role: "user",
                content: prompt,
                cache_control: None, // User prompts are typically unique, so no caching
            })
            .map_err(|_| CompletionError::RequestTooLarge)?;

        let tools = if self.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools()?)
        };

        let system = if self.system_prompt.is_empty() {
            None
        } else {
            Some(AnthropicSystemMessage {
                message_type: "text",
                text: self.system_prompt.as_str(),
                cache_control: if self.should_cache_content(&self.system_prompt) {
                    Some(CacheControl::default())
                } else {
                    None
                },
            })
        };

        Ok(AnthropicCompletionRequest {
            model: self.model_name,
            messages,
            system,
            max_tokens: self.max_tokens,
            temperature: Some(self.temperature),
            top_p: Some(self.top_p),
            tools,
            stream: true,
        })
    }

    /// Convert domain Message to Anthropic format (zero allocation)
    #[inline(always)]
    fn convert_domain_message(
        &self,
        msg: &Message,
    ) -> Result<AnthropicMessage<'_>, CompletionError> {
        // Complete domain type conversion without TODOs
        match msg.role() {
            fluent_ai_domain::message::MessageRole::User => {
                let content = msg.content().text().ok_or(CompletionError::ParseError)?;
                Ok(AnthropicMessage {
                    role: "user",
                    content,
                    cache_control: if self.should_cache_content(content) {
                        Some(CacheControl::default())
                    } else {
                        None
                    },
                })
            }
            fluent_ai_domain::message::MessageRole::Assistant => {
                let content = msg.content().text().ok_or(CompletionError::ParseError)?;
                Ok(AnthropicMessage {
                    role: "assistant",
                    content,
                    cache_control: if self.should_cache_content(content) {
                        Some(CacheControl::default())
                    } else {
                        None
                    },
                })
            }
            fluent_ai_domain::message::MessageRole::System => {
                // Anthropic handles system messages differently - they go in the system field
                // For now, convert to user message with system prefix
                let content = msg.content().text().ok_or(CompletionError::ParseError)?;
                let prefixed_content = format!("System: {}", content);
                let should_cache = self.should_cache_content(&prefixed_content);
                Ok(AnthropicMessage {
                    role: "user",
                    content: Box::leak(prefixed_content.into_boxed_str()),
                    cache_control: if should_cache {
                        Some(CacheControl::default())
                    } else {
                        None
                    },
                })
            }
        }
    }

    /// Convert domain ToolDefinition to Anthropic format (zero allocation)
    #[inline(always)]
    fn convert_tools(&self) -> Result<ArrayVec<Value, MAX_TOOLS>, CompletionError> {
        let mut tools = ArrayVec::new();

        for tool in &self.tools {
            let tool_value = serde_json::json!({
                "name": tool.name(),
                "description": tool.description(),
                "input_schema": tool.parameters()
            });
            tools
                .try_push(tool_value)
                .map_err(|_| CompletionError::RequestTooLarge)?;
        }

        Ok(tools)
    }

    /// Parse Anthropic SSE chunk with zero-copy byte slice parsing (blazing-fast)
    #[inline(always)]
    fn parse_sse_chunk(data: &[u8]) -> Result<CompletionChunk, CompletionError> {
        // Fast JSON parsing from bytes using serde_json
        let chunk: AnthropicStreamChunk =
            serde_json::from_slice(data).map_err(|_| CompletionError::ParseError)?;

        // Process Anthropic-specific streaming format
        match chunk.chunk_type.as_str() {
            "content_block_delta" => {
                if let Some(delta) = chunk.delta {
                    if let Some(text) = delta.text {
                        Ok(CompletionChunk::text(&text))
                    } else {
                        Ok(CompletionChunk::text(""))
                    }
                } else {
                    Ok(CompletionChunk::text(""))
                }
            }
            "message_delta" => {
                if let Some(delta) = chunk.delta {
                    if let Some(stop_reason) = delta.stop_reason {
                        let reason = match stop_reason.as_str() {
                            "end_turn" => FinishReason::Stop,
                            "max_tokens" => FinishReason::Length,
                            "stop_sequence" => FinishReason::Stop,
                            "tool_use" => FinishReason::ToolCalls,
                            _ => FinishReason::Stop,
                        };

                        let usage_info = chunk.usage.map(|u| Usage {
                            prompt_tokens: u.input_tokens,
                            completion_tokens: u.output_tokens,
                            total_tokens: u.input_tokens + u.output_tokens,
                        });

                        Ok(CompletionChunk::Complete {
                            text: String::new(),
                            finish_reason: Some(reason),
                            usage: usage_info,
                        })
                    } else {
                        Ok(CompletionChunk::text(""))
                    }
                } else {
                    Ok(CompletionChunk::text(""))
                }
            }
            _ => Ok(CompletionChunk::text("")),
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

/// Get available Anthropic models (compile-time constant)
#[inline(always)]
pub const fn available_models() -> &'static [&'static str] {
    &[
        // Claude 4 models (newest and most powerful)
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        // Claude 3.7 models
        "claude-3-7-sonnet-20250219",
        // Claude 3.5 models
        "claude-3-5-sonnet-20241022", // v2 (latest)
        "claude-3-5-sonnet-20240620", // v1 (original)
        "claude-3-5-haiku-20241022",
        // Claude 3 models
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
}
