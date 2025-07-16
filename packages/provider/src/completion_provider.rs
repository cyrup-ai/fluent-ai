//! Base CompletionProvider trait for universal provider interface
//!
//! Enables elegant syntax: `Model::MistralMagistral.prompt("What time is it in Paris?")`
//! All parameters use ZeroOneOrMany with ModelInfo defaults - nothing is Optional

use fluent_ai_domain::chunk::CompletionChunk;
use fluent_ai_domain::{Message, Document};
use fluent_ai_domain::tool::ToolDefinition;
use crate::AsyncStream;
use cyrup_sugars::ZeroOneOrMany;
use serde_json::Value;
use std::marker::PhantomData;

/// Typestate: Builder needs prompt to complete
#[derive(Debug, Clone, Copy)]
pub struct NeedsPrompt;

/// Typestate: Builder ready to execute  
#[derive(Debug, Clone, Copy)]
pub struct Ready;

/// Universal completion error (zero allocation)
#[derive(Debug, Clone, Copy)]
pub enum CompletionError {
    HttpError,
    AuthError,
    UnsupportedModel,
    RequestTooLarge,
    RateLimited,
    ParseError,
    StreamError,
}

impl CompletionError {
    #[inline(always)]
    pub const fn message(&self) -> &'static str {
        match self {
            Self::HttpError => "HTTP request failed",
            Self::AuthError => "Authentication failed",
            Self::UnsupportedModel => "Model not supported",
            Self::RequestTooLarge => "Request too large",
            Self::RateLimited => "Rate limited",
            Self::ParseError => "Parse error",
            Self::StreamError => "Stream error",
        }
    }
}

impl std::fmt::Display for CompletionError {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for CompletionError {}

/// cyrup_sugars chunk handler with pattern matching syntax
pub type ChunkHandler = Box<dyn Fn(Result<CompletionChunk, CompletionError>) + Send + Sync>;

/// Universal completion builder trait with ZeroOneOrMany defaults
/// 
/// All parameters default from ModelInfo - no Optional types
/// Enables: `Model::MistralMagistral.prompt("What time is it in Paris?")`
pub trait CompletionProvider<State = NeedsPrompt>: Clone + Send + Sync + 'static {
    /// Provider-specific model type
    type Model: Clone + Send + Sync;
    
    /// Create new completion builder with ModelInfo defaults
    fn new(api_key: String, model: Self::Model) -> Result<Self, CompletionError>;
    
    /// Set system prompt (defaults from ModelInfo)
    #[inline(always)]
    fn system_prompt(self, prompt: impl Into<String>) -> Self;
    
    /// Set temperature (defaults from ModelInfo)
    #[inline(always)] 
    fn temperature(self, temp: f64) -> Self;
    
    /// Set max tokens (defaults from ModelInfo)
    #[inline(always)]
    fn max_tokens(self, tokens: u32) -> Self;
    
    /// Set top_p (defaults from ModelInfo)
    #[inline(always)]
    fn top_p(self, p: f64) -> Self;
    
    /// Set frequency penalty (defaults from ModelInfo)
    #[inline(always)]
    fn frequency_penalty(self, penalty: f64) -> Self;
    
    /// Set presence penalty (defaults from ModelInfo)
    #[inline(always)]
    fn presence_penalty(self, penalty: f64) -> Self;
    
    /// Add chat history (ZeroOneOrMany - defaults to None)
    #[inline(always)]
    fn chat_history(self, history: ZeroOneOrMany<Message>) -> Result<Self, CompletionError>;
    
    /// Add documents for RAG (ZeroOneOrMany - defaults to None)
    #[inline(always)]
    fn documents(self, docs: ZeroOneOrMany<Document>) -> Result<Self, CompletionError>;
    
    /// Add tools for function calling (ZeroOneOrMany - defaults to None)
    #[inline(always)]
    fn tools(self, tools: ZeroOneOrMany<ToolDefinition>) -> Result<Self, CompletionError>;
    
    /// Add provider-specific parameters (defaults to None)
    #[inline(always)]
    fn additional_params(self, params: Value) -> Self;
    
    /// Set chunk handler with cyrup_sugars pattern matching
    /// ```
    /// .on_chunk(|chunk| {
    ///     Ok => log::info!("Chunk: {:?}", chunk),
    ///     Err => log::error!("Error: {:?}", chunk)
    /// })
    /// ```
    #[inline(always)]
    fn on_chunk<F>(self, handler: F) -> Self 
    where
        F: Fn(Result<CompletionChunk, CompletionError>) + Send + Sync + 'static;
}

/// Terminal action trait - converts NeedsPrompt to Ready and executes
pub trait TerminalPrompt<State> {
    /// Terminal action - execute completion with user prompt
    /// Returns blazing-fast zero-allocation streaming
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk>;
}

/// Model trait for direct prompting syntax
/// Enables: `Model::MistralMagistral.prompt("What time is it in Paris?")`
pub trait ModelPrompt {
    type Provider: CompletionProvider<NeedsPrompt> + TerminalPrompt<NeedsPrompt>;
    
    /// Direct prompt execution with ModelInfo defaults
    #[inline(always)]
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk>
    where
        Self: Sized,
    {
        // This will be implemented by each model enum variant
        // to create the appropriate provider with defaults
        unimplemented!("Each model must implement direct prompting")
    }
    
    /// Create completion builder for advanced configuration
    #[inline(always)]
    fn completion(self) -> Result<Self::Provider, CompletionError>
    where 
        Self: Sized,
    {
        // This will be implemented by each model enum variant
        unimplemented!("Each model must implement completion builder")
    }
}

/// Compile-time model configuration (zero allocation)
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
    pub provider: &'static str,
}

/// Universal model info trait for compile-time defaults
pub trait ModelInfo {
    /// Get compile-time model configuration
    const CONFIG: ModelConfig;
    
    /// Get model name (zero allocation)
    fn name(&self) -> &'static str;
    
    /// Get provider name (zero allocation)  
    fn provider(&self) -> &'static str {
        Self::CONFIG.provider
    }
    
    /// Check capabilities at compile time
    #[inline(always)]
    const fn supports_tools() -> bool {
        Self::CONFIG.supports_tools
    }
    
    #[inline(always)]
    const fn supports_vision() -> bool {
        Self::CONFIG.supports_vision
    }
    
    #[inline(always)]
    const fn supports_audio() -> bool {
        Self::CONFIG.supports_audio
    }
    
    #[inline(always)]
    const fn context_length() -> u32 {
        Self::CONFIG.context_length
    }
    
    #[inline(always)]
    const fn max_output_tokens() -> u32 {
        Self::CONFIG.max_tokens
    }
}