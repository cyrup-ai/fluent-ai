//! Universal CompletionProvider trait enabling `Model::X.prompt()` syntax
//!
//! Zero-allocation streaming completions with ZeroOneOrMany defaults:
//! ```
//! Model::OpenaiGpt4o.prompt("Hello world")
//! Model::MistralMagistral.completion()?.temperature(0.8).prompt("Hello")
//! ```

use std::env;

use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_domain::chunk::CompletionChunk;
use fluent_ai_domain::completion::CompletionCoreError;
use fluent_ai_domain::tool::ToolDefinition;
use fluent_ai_domain::{Document, Message};
use serde_json::Value;

use crate::AsyncStream;

/// Typestate: Builder needs prompt to complete
#[derive(Debug, Clone, Copy)]
pub struct NeedsPrompt;

/// Typestate: Builder ready to execute  
#[derive(Debug, Clone, Copy)]
pub struct Ready;

/// Re-export domain completion error for provider use
pub use CompletionCoreError as CompletionError;

/// Chunk handler with cyrup_sugars pattern matching
pub type ChunkHandler = Box<dyn Fn(Result<CompletionChunk, CompletionError>) + Send + Sync>;

/// Automatic API key discovery using provider's env_api_keys() method
#[inline(always)]
pub fn discover_api_key_for_provider<T: CompletionProvider>(provider: &T) -> Option<String> {
    let env_keys = provider.env_api_keys();

    match env_keys {
        ZeroOneOrMany::None => {
            log::error!("No environment variable patterns defined for provider");
            None
        }
        ZeroOneOrMany::One(key) => {
            if let Ok(value) = env::var(key) {
                if !value.is_empty() {
                    log::debug!("Found API key using {}", key);
                    return Some(value);
                }
            }
            log::error!("No API key found using {}", key);
            None
        }
        ZeroOneOrMany::Many(keys) => {
            // Search in order - first found wins (ZeroOneOrMany is ordered)
            for key in &keys {
                if let Ok(value) = env::var(key) {
                    if !value.is_empty() {
                        log::debug!("Found API key using {} (first found in ordered list)", key);
                        return Some(value);
                    }
                }
            }
            log::error!("No API key found. Tried in order: {:?}", keys);
            None
        }
    }
}

/// Universal completion provider trait
///
/// All parameters use ZeroOneOrMany with ModelInfo defaults
/// Enables blazing-fast zero-allocation streaming completions
pub trait CompletionProvider: Clone + Send + Sync + 'static {
    /// Create new builder with ModelInfo defaults loaded at compile time
    fn new(api_key: String, model_name: &'static str) -> Result<Self, CompletionError>;

    /// Set explicit API key (takes priority over environment variables)
    fn api_key(self, key: impl Into<String>) -> Self;

    /// Environment variable names to search for API keys (provider-specific)
    /// Defaults to all well-known patterns for this provider
    /// Returns ZeroOneOrMany of environment variable names to search in order
    fn env_api_keys(&self) -> ZeroOneOrMany<String> {
        // Default implementation returns None - each provider should override
        ZeroOneOrMany::None
    }

    /// Set system prompt (overrides ModelInfo default)
    fn system_prompt(self, prompt: impl Into<String>) -> Self;

    /// Set temperature (overrides ModelInfo default)
    fn temperature(self, temp: f64) -> Self;

    /// Set max tokens (overrides ModelInfo default)
    fn max_tokens(self, tokens: u32) -> Self;

    /// Set top_p (overrides ModelInfo default)
    fn top_p(self, p: f64) -> Self;

    /// Set frequency penalty (overrides ModelInfo default)
    fn frequency_penalty(self, penalty: f64) -> Self;

    /// Set presence penalty (overrides ModelInfo default)
    fn presence_penalty(self, penalty: f64) -> Self;

    /// Add chat history (ZeroOneOrMany with bounded capacity)
    fn chat_history(self, history: ZeroOneOrMany<Message>) -> Result<Self, CompletionError>;

    /// Add documents for RAG (ZeroOneOrMany with bounded capacity)
    fn documents(self, docs: ZeroOneOrMany<Document>) -> Result<Self, CompletionError>;

    /// Add tools for function calling (ZeroOneOrMany with bounded capacity)
    fn tools(self, tools: ZeroOneOrMany<ToolDefinition>) -> Result<Self, CompletionError>;

    /// Add provider-specific parameters
    fn additional_params(self, params: Value) -> Self;

    /// Set chunk handler with cyrup_sugars pattern matching
    ///
    /// ```
    /// .on_chunk(|chunk| {
    ///     Ok => log::info!("Chunk: {:?}", chunk),
    ///     Err => log::error!("Error: {:?}", chunk)
    /// })
    /// ```
    fn on_chunk<F>(self, handler: F) -> Self
    where
        F: Fn(Result<CompletionChunk, CompletionError>) + Send + Sync + 'static;

    /// Terminal action - execute completion with user prompt
    /// Returns blazing-fast zero-allocation streaming
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk>;
}

/// Model trait for direct prompting syntax
///
/// Enables: `Model::MistralMagistral.prompt("What time is it in Paris?")`
/// Each model variant knows its provider and configuration at compile time
pub trait ModelPrompt: ModelInfo {
    /// Associated provider type for this model
    type Provider: CompletionProvider;

    /// Direct prompt execution with ModelInfo defaults
    /// Zero allocation, blazing-fast streaming
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk>
    where
        Self: Sized + ModelInfo,
    {
        // Create temporary provider to get env_api_keys list
        let temp_provider = match Self::Provider::new("temp".to_string(), self.name()) {
            Ok(provider) => provider,
            Err(e) => {
                // Return error stream
                let (sender, receiver) = crate::channel();
                let _ = sender.send(CompletionChunk::error(&e.to_string()));
                return receiver;
            }
        };

        // Discover API key using provider's env_api_keys() method
        let api_key = match discover_api_key_for_provider(&temp_provider) {
            Some(key) => key,
            None => {
                // Return error stream - discovery function already logged the error
                let (sender, receiver) = crate::channel();
                let _ = sender.send(CompletionChunk::error("Missing API key"));
                return receiver;
            }
        };

        // Create provider with discovered API key and execute
        match Self::Provider::new(api_key, self.name()) {
            Ok(provider) => provider.prompt(text),
            Err(e) => {
                // Return error stream
                let (sender, receiver) = crate::channel();
                let _ = sender.send(CompletionChunk::error(&e.to_string()));
                receiver
            }
        }
    }

    /// Create completion builder for advanced configuration
    /// All parameters default from ModelInfo - zero allocation setup
    fn completion(self) -> Result<Self::Provider, CompletionError>
    where
        Self: Sized + ModelInfo,
    {
        // Create temporary provider to get env_api_keys list
        let temp_provider = Self::Provider::new("temp".to_string(), self.name())?;

        // Discover API key using provider's env_api_keys() method
        let api_key = discover_api_key_for_provider(&temp_provider).ok_or(
            CompletionError::ProviderUnavailable("Missing API key".to_string()),
        )?;

        // Create provider with discovered API key and ModelInfo defaults
        Self::Provider::new(api_key, self.name())
    }
}

/// Compile-time model configuration (zero allocation constants)
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub max_tokens: u32,
    pub temperature: f64,
    pub top_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub context_length: u32,
    pub system_prompt: &'static str,
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub supports_thinking: bool,
    pub optimal_thinking_budget: u32,
    pub provider: &'static str,
    pub model_name: &'static str,
}

/// Universal model info trait for compile-time defaults
///
/// All model configurations are const and known at compile time
/// for blazing-fast zero-allocation initialization
pub trait ModelInfo {
    /// Compile-time model configuration (zero allocation)
    const CONFIG: ModelConfig;

    /// Get model name (zero allocation string literal)
    #[inline(always)]
    fn name(&self) -> &'static str {
        Self::CONFIG.model_name
    }

    /// Get provider name (zero allocation string literal)
    #[inline(always)]
    fn provider(&self) -> &'static str {
        Self::CONFIG.provider
    }

    /// Check tool support at compile time
    #[inline(always)]
    fn supports_tools() -> bool {
        Self::CONFIG.supports_tools
    }

    /// Check vision support at compile time
    #[inline(always)]
    fn supports_vision() -> bool {
        Self::CONFIG.supports_vision
    }

    /// Check audio support at compile time
    #[inline(always)]
    fn supports_audio() -> bool {
        Self::CONFIG.supports_audio
    }

    /// Get context length at compile time
    #[inline(always)]
    fn context_length() -> u32 {
        Self::CONFIG.context_length
    }

    /// Get max output tokens at compile time
    #[inline(always)]
    fn max_output_tokens() -> u32 {
        Self::CONFIG.max_tokens
    }

    /// Get default temperature at compile time
    #[inline(always)]
    fn default_temperature() -> f64 {
        Self::CONFIG.temperature
    }

    /// Get default system prompt at compile time
    #[inline(always)]
    fn default_system_prompt() -> &'static str {
        Self::CONFIG.system_prompt
    }
}

/// Zero-allocation completion response wrapper
///
/// Provides a unified interface for completion responses across all providers
/// while maintaining zero-allocation semantics and lock-free access patterns.
#[derive(Debug, Clone)]
pub struct CompletionResponse<T> {
    /// The raw provider-specific response
    pub raw_response: T,
    /// Processed completion content
    pub content: ZeroOneOrMany<String>,
    /// Token usage information (if available)
    pub token_usage: Option<TokenUsage>,
    /// Response metadata
    pub metadata: ResponseMetadata,
}

impl<T> CompletionResponse<T> {
    /// Create a new completion response with zero allocations
    #[inline(always)]
    pub fn new(raw_response: T, content: ZeroOneOrMany<String>) -> Self {
        Self {
            raw_response,
            content,
            token_usage: None,
            metadata: ResponseMetadata::default(),
        }
    }

    /// Add token usage information
    #[inline(always)]
    pub fn with_token_usage(mut self, usage: TokenUsage) -> Self {
        self.token_usage = Some(usage);
        self
    }

    /// Add metadata
    #[inline(always)]
    pub fn with_metadata(mut self, metadata: ResponseMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get the primary completion text (zero-allocation)
    #[inline(always)]
    pub fn text(&self) -> Option<&str> {
        match &self.content {
            ZeroOneOrMany::Zero => None,
            ZeroOneOrMany::One(text) => Some(text),
            ZeroOneOrMany::Many(texts) => texts.first().map(|s| s.as_str()),
        }
    }

    /// Get all completion texts
    #[inline(always)]
    pub fn texts(&self) -> &[String] {
        self.content.as_slice()
    }
}

/// Zero-allocation streaming response wrapper
///
/// Provides a unified streaming interface for real-time completion responses
/// with lock-free channel communication and zero-copy token processing.
pub struct StreamingResponse<T> {
    /// The raw provider-specific streaming response
    pub raw_response: T,
    /// Stream of completion chunks
    pub stream: AsyncStream<Result<CompletionChunk, CompletionError>>,
    /// Response metadata (filled as chunks arrive)
    pub metadata: ResponseMetadata,
}

impl<T> StreamingResponse<T> {
    /// Create a new streaming response
    #[inline(always)]
    pub fn new(
        raw_response: T,
        stream: AsyncStream<Result<CompletionChunk, CompletionError>>,
    ) -> Self {
        Self {
            raw_response,
            stream,
            metadata: ResponseMetadata::default(),
        }
    }

    /// Add metadata
    #[inline(always)]
    pub fn with_metadata(mut self, metadata: ResponseMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Token usage information
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total tokens used
    pub total_tokens: u32,
}

impl TokenUsage {
    /// Create new token usage info
    #[inline(always)]
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// Response metadata for tracking request/response details
#[derive(Debug, Clone, Default)]
pub struct ResponseMetadata {
    /// Request ID (if provided by the API)
    pub request_id: Option<String>,
    /// Model used for the completion
    pub model: Option<String>,
    /// Response time in milliseconds
    pub response_time_ms: Option<u64>,
    /// Additional provider-specific metadata
    pub provider_metadata: Option<Value>,
}

impl ResponseMetadata {
    /// Create new response metadata
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set request ID
    #[inline(always)]
    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }

    /// Set model name
    #[inline(always)]
    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    /// Set response time
    #[inline(always)]
    pub fn with_response_time(mut self, response_time_ms: u64) -> Self {
        self.response_time_ms = Some(response_time_ms);
        self
    }

    /// Set provider metadata
    #[inline(always)]
    pub fn with_provider_metadata(mut self, metadata: Value) -> Self {
        self.provider_metadata = Some(metadata);
        self
    }
}
