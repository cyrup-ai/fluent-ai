//! Universal CompletionProvider trait enabling `Model::X.prompt()` syntax
//!
//! Zero-allocation streaming completions with ZeroOneOrMany defaults:
//! ```
//! Model::OpenaiGpt4o.prompt("Hello world")
//! Model::MistralMagistral.completion()?.temperature(0.8).prompt("Hello")
//! ```

use std::env;

use cyrup_sugars::prelude::{ChunkHandler, ZeroOneOrMany};
use fluent_ai_async::{AsyncStream, AsyncStreamSender};
use fluent_ai_domain::chat::Message;
use fluent_ai_domain::completion::CompletionRequestError as CompletionCoreError;
use fluent_ai_domain::completion::types::ToolDefinition;
use fluent_ai_domain::context::Document;
use fluent_ai_domain::context::chunk::CompletionChunk;
use serde_json::Value;

/// Provider-specific completion error
#[derive(Debug, thiserror::Error)]
pub enum CompletionError {
    #[error("Authentication error")]
    AuthError,
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Model error: {0}")]
    ModelError(String),
    #[error("Timeout error")]
    Timeout,
    #[error("Rate limit exceeded")]
    RateLimit,
    #[error("Parsing error: {0}")]
    ParseError(String),
    #[error("Provider unavailable: {0}")]
    ProviderUnavailable(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

// Conversion from domain error if needed
impl From<CompletionCoreError> for CompletionError {
    fn from(err: CompletionCoreError) -> Self {
        CompletionError::ModelError(format!("{:?}", err))
    }
}

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
/// All parameters use ZeroOneOrMany with ModelConfigInfo defaults
/// Enables blazing-fast zero-allocation streaming completions
pub trait CompletionProvider: Clone + Send + Sync + 'static {
    /// Create new builder with ModelConfigInfo defaults loaded at compile time
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
    ///     match chunk {
    ///         Ok(c) => { log::info!("Chunk: {:?}", c); c },
    ///         Err(e) => CompletionChunk::bad_chunk(e.to_string())
    ///     }
    /// })
    /// ```
    fn on_chunk<F>(self, handler: F) -> Self
    where
        F: Fn(Result<CompletionChunk, CompletionError>) -> CompletionChunk + Send + Sync + 'static;

    /// Terminal action - execute completion with user prompt
    /// Returns blazing-fast zero-allocation streaming
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk>;
}

/// Model trait for direct prompting syntax
///
/// Enables: `Model::MistralMagistral.prompt("What time is it in Paris?")`
/// Each model variant knows its provider and configuration at compile time
pub trait ModelPrompt: ModelConfigInfo {
    /// Associated provider type for this model
    type Provider: CompletionProvider;

    /// Direct prompt execution with ModelConfigInfo defaults
    /// Zero allocation, blazing-fast streaming
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk>
    where
        Self: Sized + ModelConfigInfo,
    {
        // Create temporary provider to get env_api_keys list
        let temp_provider = match Self::Provider::new("temp".to_string(), self.name()) {
            Ok(provider) => provider,
            Err(e) => {
                // Return error stream using AsyncStream
                return AsyncStream::with_channel(
                    move |sender: AsyncStreamSender<CompletionChunk>| {
                        let _ = sender.send(CompletionChunk::error(&e.to_string()));
                    },
                );
            }
        };

        // Discover API key using provider's env_api_keys() method
        let api_key = match discover_api_key_for_provider(&temp_provider) {
            Some(key) => key,
            None => {
                // Return error stream - discovery function already logged the error
                return AsyncStream::with_channel(
                    move |sender: AsyncStreamSender<CompletionChunk>| {
                        let _ = sender.send(CompletionChunk::error("Missing API key"));
                    },
                );
            }
        };

        // Create provider with discovered API key and execute
        match Self::Provider::new(api_key, self.name()) {
            Ok(provider) => provider.prompt(text),
            Err(e) => {
                // Return error stream
                AsyncStream::with_channel(move |sender: AsyncStreamSender<CompletionChunk>| {
                    let _ = sender.send(CompletionChunk::error(&e.to_string()));
                })
            }
        }
    }

    /// Create completion builder for advanced configuration
    /// All parameters default from ModelConfigInfo - zero allocation setup
    fn completion(self) -> Result<Self::Provider, CompletionError>
    where
        Self: Sized + ModelConfigInfo,
    {
        // Create temporary provider to get env_api_keys list
        let temp_provider = Self::Provider::new("temp".to_string(), self.name())?;

        // Discover API key using provider's env_api_keys() method
        let api_key = discover_api_key_for_provider(&temp_provider).ok_or(
            CompletionError::ProviderUnavailable("Missing API key".to_string()),
        )?;

        // Create provider with discovered API key and ModelConfigInfo defaults
        Self::Provider::new(api_key, self.name())
    }
}

/// Universal model config trait using model-info package
///
/// All model configurations come from model-info package
/// for blazing-fast zero-allocation initialization
pub trait ModelConfigInfo: model_info::common::Model {
    /// Get model name (zero allocation string literal)
    #[inline(always)]
    fn name(&self) -> &'static str {
        <Self as model_info::common::Model>::name(self)
    }

    /// Get context length from model-info
    #[inline(always)]
    fn context_length(&self) -> u64 {
        <Self as model_info::common::Model>::max_context_length(self)
    }

    /// Get pricing information from model-info
    #[inline(always)]
    fn pricing_input(&self) -> f64 {
        <Self as model_info::common::Model>::pricing_input(self).unwrap_or(0.0)
    }

    /// Get pricing information from model-info
    #[inline(always)]
    fn pricing_output(&self) -> f64 {
        <Self as model_info::common::Model>::pricing_output(self).unwrap_or(0.0)
    }

    /// Check if thinking model from model-info
    #[inline(always)]
    fn is_thinking(&self) -> bool {
        <Self as model_info::common::Model>::supports_thinking(self)
    }

    /// Get required temperature from model-info
    #[inline(always)]
    fn required_temperature(&self) -> Option<f64> {
        <Self as model_info::common::Model>::required_temperature(self)
    }
}

/// Model configuration structure
pub struct ModelConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: u32,
    /// Sampling temperature for randomness (0.0 to 2.0)
    pub temperature: f64,
    /// Nucleus sampling parameter (0.0 to 1.0)
    pub top_p: f64,
    /// Frequency penalty for repetition reduction (-2.0 to 2.0)
    pub frequency_penalty: f64,
    /// Presence penalty for topic diversity (-2.0 to 2.0)
    pub presence_penalty: f64,
    /// Maximum context length in tokens
    pub context_length: u64,
    /// Default system prompt for the model
    pub system_prompt: String,
    /// Whether the model supports function/tool calling
    pub supports_tools: bool,
    /// Whether the model supports vision/image inputs
    pub supports_vision: bool,
    /// Whether the model supports audio processing
    pub supports_audio: bool,
    /// Whether the model supports thinking/reasoning modes
    pub supports_thinking: bool,
    /// Optimal thinking budget in tokens for reasoning
    pub optimal_thinking_budget: u32,
    /// Name of the AI provider (e.g., "openai", "anthropic")
    pub provider: String,
    /// Specific model name/identifier
    pub model_name: String,
}

impl ModelConfig {
    pub fn default() -> Self {
        Self {
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            context_length: 128000,
            system_prompt: "".to_string(),
            supports_tools: true,
            supports_vision: true,
            supports_audio: false,
            supports_thinking: false,
            optimal_thinking_budget: 0,
            provider: "openai".to_string(),
            model_name: "gpt-4o".to_string(),
        }
    }
}

/// Stub for ModelInfo trait
pub trait ModelInfo {
    fn config(&self) -> ModelConfig;
}

/// Response metadata for completion operations
#[derive(Debug, Clone)]
pub struct ResponseMetadata {
    pub request_id: Option<String>,
    pub model: String,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
}

impl Default for ResponseMetadata {
    fn default() -> Self {
        Self {
            request_id: None,
            model: "unknown".to_string(),
            usage: None,
            finish_reason: None,
        }
    }
}

/// Usage information for completion operations
#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Completion response containing generated content and metadata
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub id: String,
    pub content: String,
    pub metadata: ResponseMetadata,
    pub chunks: Vec<CompletionChunk>,
}

impl CompletionResponse {
    pub fn new(id: String, content: String) -> Self {
        Self {
            id,
            content,
            metadata: ResponseMetadata::default(),
            chunks: Vec::new(),
        }
    }

    pub fn with_metadata(mut self, metadata: ResponseMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_chunks(mut self, chunks: Vec<CompletionChunk>) -> Self {
        self.chunks = chunks;
        self
    }
}
