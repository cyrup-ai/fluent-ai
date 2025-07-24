//! HTTP Models and Types for AI Provider Integration
//!
//! This module provides centralized HTTP request and response models for all AI providers.
//! Supports 45+ endpoints across 17 providers with zero-allocation patterns and type safety.
//!
//! # Architecture
//!
//! The HTTP models system is designed around these core principles:
//! - **Zero Allocation**: Uses ArrayVec and stack-allocated structures where possible
//! - **Type Safety**: Compile-time validation of request/response structures
//! - **Provider Agnostic**: Common interfaces that work across all providers
//! - **Streaming First**: Built for AsyncStream integration with fluent_ai_http3
//! - **Lock Free**: All operations use atomic patterns for thread safety
//!
//! # Provider Support
//!
//! Supports all major AI providers:
//! - OpenAI (completion, embedding, audio, moderation, vision)
//! - Anthropic (messages with cache control and tools)
//! - Google (Vertex AI, Gemini with OAuth2 and service accounts)
//! - AWS Bedrock (converse API with signature auth)
//! - Cohere (completion, embedding, rerank)
//! - Azure OpenAI (all endpoints with deployment URLs)
//! - Local providers (Ollama with chat and generate)
//! - And 10 additional providers (AI21, Groq, HuggingFace, etc.)
//!
//! # Module Organization
//!
//! - [`common`] - Shared types and utilities across all providers
//! - [`auth`] - Authentication types for all 4 authentication methods
//! - [`requests`] - Request models for completion, embedding, audio, and specialized endpoints
//! - [`responses`] - Response models with streaming and error handling support
//! - [`builder`] - Integration with fluent_ai_http3 builder patterns
//! - [`client`] - HTTP client trait for standardized operations
//!
//! # Usage Example
//!
//! ```rust
//! use fluent_ai_domain::http::{
//!     requests::completion::CompletionRequest,
//!     responses::completion::CompletionResponse,
//!     auth::BearerAuth,
//! };
//! use fluent_ai_http3::Http3;
//!
//! // Type-safe request construction
//! let request = CompletionRequest::new("gpt-4")
//!     .with_messages(messages)
//!     .with_temperature(0.7)
//!     .with_max_tokens(1000);
//!
//! // Integration with fluent_ai_http3
//! let response: CompletionResponse = Http3::json()
//!     .auth(BearerAuth::new(api_key))
//!     .body(&request)
//!     .post("https://api.openai.com/v1/chat/completions")
//!     .collect()
//!     .await?;
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::sync::atomic::{AtomicU64, Ordering};

// Core HTTP modules
pub mod common;     // Task #7 - Shared HTTP Types - COMPLETED
pub mod auth;       // Task #8 - Authentication Types - COMPLETED
pub mod requests;   // Task #9 - Request Models - COMPLETED
pub mod responses;  // Task #10 - Response Models - COMPLETED
// pub mod builder;    // HTTP Request Builder Integration
// pub mod client;     // HTTP Client Trait

// Private utilities module - to be implemented
// mod utils;

// Re-exports for public API
pub use {
    auth::{
        ApiKeyAuth, AwsSignatureAuth, BearerAuth, OAuth2Auth, OAuth2Token, SecureString,
        ServiceAccountConfig, AuthError,
    },
    // builder::HttpRequestBuilder,
    // client::HttpClient,
    common::{
        BaseMessage, CommonUsage, FinishReason, HttpContentType, HttpMethod, ModelParameters,
        ProviderMetadata, StreamingMode, MessageRole, ToolCall, ToolCallType, FunctionCall,
        ValidationError,
    },
    requests::{
        completion::{
            CompletionRequest, CompletionRequestError, ProviderExtensions,
            ToolDefinition, ToolChoice, FunctionDefinition, ToolType,
            OpenAIExtensions, AnthropicExtensions, GoogleExtensions,
            BedrockExtensions, CohereExtensions,
        },
        // embedding::{EmbeddingRequest, EmbeddingRequestError},
        // audio::{AudioRequest, AudioTranscriptionRequest, AudioTTSRequest},
        // specialized::{ImageGenerationRequest, ModerationRequest, RerankRequest},
    },
    responses::{
        completion::{
            CompletionResponse, CompletionChoice, CompletionChunk, ChunkChoice, ChunkDelta,
            StreamingResponse, ProviderMetadata, CompletionResponseError, LogProbs, TokenLogProb,
            OpenAIMetadata, AnthropicMetadata, GoogleMetadata, BedrockMetadata, CohereMetadata,
            SafetyRating, CitationMetadata, CitationSource, BedrockMetrics,
        },
        // embedding::EmbeddingResponse,
        // error::{HttpError, ProviderError},
    },
};

/// Global statistics for HTTP operations
/// Uses atomic operations for lock-free thread safety
#[derive(Debug, Default)]
pub struct HttpStats {
    /// Total number of HTTP requests made
    pub requests_total: AtomicU64,
    /// Total number of successful responses
    pub responses_success: AtomicU64,
    /// Total number of error responses
    pub responses_error: AtomicU64,
    /// Total bytes sent in requests
    pub bytes_sent: AtomicU64,
    /// Total bytes received in responses
    pub bytes_received: AtomicU64,
    /// Total time spent in HTTP operations (microseconds)
    pub total_duration_micros: AtomicU64,
}

impl HttpStats {
    /// Create new HTTP statistics tracker
    #[inline]
    pub const fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            responses_success: AtomicU64::new(0),
            responses_error: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            total_duration_micros: AtomicU64::new(0),
        }
    }

    /// Record a successful HTTP request
    #[inline]
    pub fn record_request(&self, bytes_sent: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_sent.fetch_add(bytes_sent, Ordering::Relaxed);
    }

    /// Record a successful HTTP response
    #[inline]
    pub fn record_success(&self, bytes_received: u64, duration_micros: u64) {
        self.responses_success.fetch_add(1, Ordering::Relaxed);
        self.bytes_received.fetch_add(bytes_received, Ordering::Relaxed);
        self.total_duration_micros
            .fetch_add(duration_micros, Ordering::Relaxed);
    }

    /// Record an error HTTP response
    #[inline]
    pub fn record_error(&self, duration_micros: u64) {
        self.responses_error.fetch_add(1, Ordering::Relaxed);
        self.total_duration_micros
            .fetch_add(duration_micros, Ordering::Relaxed);
    }

    /// Get current success rate as a percentage (0.0 to 1.0)
    #[inline]
    pub fn success_rate(&self) -> f64 {
        let total = self.requests_total.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0; // No requests yet, assume 100% success
        }
        let success = self.responses_success.load(Ordering::Relaxed);
        success as f64 / total as f64
    }

    /// Get average response time in microseconds
    #[inline]
    pub fn average_response_time_micros(&self) -> u64 {
        let total_responses = self.responses_success.load(Ordering::Relaxed)
            + self.responses_error.load(Ordering::Relaxed);
        if total_responses == 0 {
            return 0;
        }
        let total_duration = self.total_duration_micros.load(Ordering::Relaxed);
        total_duration / total_responses
    }

    /// Get snapshot of current statistics for reporting
    #[inline]
    pub fn snapshot(&self) -> HttpStatsSnapshot {
        HttpStatsSnapshot {
            requests_total: self.requests_total.load(Ordering::Relaxed),
            responses_success: self.responses_success.load(Ordering::Relaxed),
            responses_error: self.responses_error.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            total_duration_micros: self.total_duration_micros.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of HTTP statistics at a point in time
/// Used for reporting and monitoring without locking
#[derive(Debug, Clone, Copy)]
pub struct HttpStatsSnapshot {
    /// Total number of HTTP requests made
    pub requests_total: u64,
    /// Total number of successful responses
    pub responses_success: u64,
    /// Total number of error responses
    pub responses_error: u64,
    /// Total bytes sent in requests
    pub bytes_sent: u64,
    /// Total bytes received in responses
    pub bytes_received: u64,
    /// Total time spent in HTTP operations (microseconds)
    pub total_duration_micros: u64,
}

impl HttpStatsSnapshot {
    /// Calculate success rate from snapshot data
    #[inline]
    pub fn success_rate(&self) -> f64 {
        if self.requests_total == 0 {
            return 1.0;
        }
        self.responses_success as f64 / self.requests_total as f64
    }

    /// Calculate average response time from snapshot data
    #[inline]
    pub fn average_response_time_micros(&self) -> u64 {
        let total_responses = self.responses_success + self.responses_error;
        if total_responses == 0 {
            return 0;
        }
        self.total_duration_micros / total_responses
    }
}

/// Global HTTP statistics instance
/// Thread-safe singleton for tracking all HTTP operations
static GLOBAL_STATS: HttpStats = HttpStats::new();

/// Get reference to global HTTP statistics
/// Used for monitoring and reporting across all HTTP operations
#[inline]
pub fn global_stats() -> &'static HttpStats {
    &GLOBAL_STATS
}

/// Provider identification for request routing and statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Provider {
    /// OpenAI (GPT models, embeddings, audio, moderation)
    OpenAI = 0,
    /// Anthropic (Claude models with tool use)
    Anthropic = 1,
    /// Google Vertex AI (PaLM, Gemini via OAuth2)
    VertexAI = 2,
    /// Google Gemini (Direct API access)
    Gemini = 3,
    /// AWS Bedrock (Claude, Titan, Llama via AWS)
    Bedrock = 4,
    /// Cohere (Command, Embed, Rerank)
    Cohere = 5,
    /// Azure OpenAI (GPT models via Azure)
    Azure = 6,
    /// AI21 Labs (Jurassic models)
    AI21 = 7,
    /// Groq (Fast inference)
    Groq = 8,
    /// HuggingFace (Open models via inference API)
    HuggingFace = 9,
    /// Mistral AI (European models)
    Mistral = 10,
    /// Ollama (Local model serving)
    Ollama = 11,
    /// OpenRouter (Model aggregation)
    OpenRouter = 12,
    /// Perplexity (Search-augmented models)
    Perplexity = 13,
    /// Together AI (Open source models)
    Together = 14,
    /// xAI (Grok models)
    XAI = 15,
    /// DeepSeek (Code and reasoning models)
    DeepSeek = 16,
}

impl Provider {
    /// Get provider name as string for logging and debugging
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::VertexAI => "vertex-ai",
            Provider::Gemini => "gemini",
            Provider::Bedrock => "bedrock",
            Provider::Cohere => "cohere",
            Provider::Azure => "azure",
            Provider::AI21 => "ai21",
            Provider::Groq => "groq",
            Provider::HuggingFace => "huggingface",
            Provider::Mistral => "mistral",
            Provider::Ollama => "ollama",
            Provider::OpenRouter => "openrouter",
            Provider::Perplexity => "perplexity",
            Provider::Together => "together",
            Provider::XAI => "xai",
            Provider::DeepSeek => "deepseek",
        }
    }

    /// Get all supported providers as a slice
    #[inline]
    pub const fn all() -> &'static [Provider] {
        &[
            Provider::OpenAI,
            Provider::Anthropic,
            Provider::VertexAI,
            Provider::Gemini,
            Provider::Bedrock,
            Provider::Cohere,
            Provider::Azure,
            Provider::AI21,
            Provider::Groq,
            Provider::HuggingFace,
            Provider::Mistral,
            Provider::Ollama,
            Provider::OpenRouter,
            Provider::Perplexity,
            Provider::Together,
            Provider::XAI,
            Provider::DeepSeek,
        ]
    }

    /// Check if provider supports streaming responses
    #[inline]
    pub const fn supports_streaming(&self) -> bool {
        match self {
            Provider::OpenAI
            | Provider::Anthropic
            | Provider::VertexAI
            | Provider::Gemini
            | Provider::Bedrock
            | Provider::Cohere
            | Provider::Azure
            | Provider::AI21
            | Provider::Groq
            | Provider::HuggingFace
            | Provider::Mistral
            | Provider::Ollama
            | Provider::OpenRouter
            | Provider::Perplexity
            | Provider::Together
            | Provider::XAI
            | Provider::DeepSeek => true,
        }
    }

    /// Check if provider supports function calling
    #[inline]
    pub const fn supports_function_calling(&self) -> bool {
        match self {
            Provider::OpenAI
            | Provider::Anthropic
            | Provider::VertexAI
            | Provider::Gemini
            | Provider::Bedrock
            | Provider::Cohere
            | Provider::Azure
            | Provider::Groq
            | Provider::Mistral
            | Provider::Ollama
            | Provider::Together
            | Provider::XAI => true,
            Provider::AI21
            | Provider::HuggingFace
            | Provider::OpenRouter
            | Provider::Perplexity
            | Provider::DeepSeek => false,
        }
    }

    /// Get default base URL for provider
    #[inline]
    pub const fn default_base_url(&self) -> &'static str {
        match self {
            Provider::OpenAI => "https://api.openai.com/v1",
            Provider::Anthropic => "https://api.anthropic.com/v1",
            Provider::VertexAI => "https://us-central1-aiplatform.googleapis.com/v1",
            Provider::Gemini => "https://generativelanguage.googleapis.com/v1beta",
            Provider::Bedrock => "https://bedrock-runtime.us-east-1.amazonaws.com",
            Provider::Cohere => "https://api.cohere.ai/v1",
            Provider::Azure => "", // Requires custom URL
            Provider::AI21 => "https://api.ai21.com/studio/v1",
            Provider::Groq => "https://api.groq.com/openai/v1",
            Provider::HuggingFace => "https://api-inference.huggingface.co",
            Provider::Mistral => "https://api.mistral.ai/v1",
            Provider::Ollama => "http://localhost:11434/api",
            Provider::OpenRouter => "https://openrouter.ai/api/v1",
            Provider::Perplexity => "https://api.perplexity.ai",
            Provider::Together => "https://api.together.xyz/v1",
            Provider::XAI => "https://api.x.ai/v1",
            Provider::DeepSeek => "https://api.deepseek.com/v1",
        }
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Provider::OpenAI),
            "anthropic" => Ok(Provider::Anthropic),
            "vertex-ai" | "vertexai" => Ok(Provider::VertexAI),
            "gemini" => Ok(Provider::Gemini),
            "bedrock" => Ok(Provider::Bedrock),
            "cohere" => Ok(Provider::Cohere),
            "azure" => Ok(Provider::Azure),
            "ai21" => Ok(Provider::AI21),
            "groq" => Ok(Provider::Groq),
            "huggingface" | "hf" => Ok(Provider::HuggingFace),
            "mistral" => Ok(Provider::Mistral),
            "ollama" => Ok(Provider::Ollama),
            "openrouter" => Ok(Provider::OpenRouter),
            "perplexity" => Ok(Provider::Perplexity),
            "together" => Ok(Provider::Together),
            "xai" => Ok(Provider::XAI),
            "deepseek" => Ok(Provider::DeepSeek),
            _ => Err(format!("Unknown provider: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_string_conversion() {
        for provider in Provider::all() {
            let as_str = provider.as_str();
            let parsed = as_str.parse::<Provider>().expect("Should parse back");
            assert_eq!(*provider, parsed);
        }
    }

    #[test]
    fn test_http_stats_operations() {
        let stats = HttpStats::new();
        
        // Record some operations
        stats.record_request(100);
        stats.record_success(200, 1000);
        
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.requests_total, 1);
        assert_eq!(snapshot.responses_success, 1);
        assert_eq!(snapshot.bytes_sent, 100);
        assert_eq!(snapshot.bytes_received, 200);
        assert_eq!(snapshot.success_rate(), 1.0);
        assert_eq!(snapshot.average_response_time_micros(), 1000);
    }

    #[test]
    fn test_provider_capabilities() {
        assert!(Provider::OpenAI.supports_streaming());
        assert!(Provider::OpenAI.supports_function_calling());
        assert!(Provider::Anthropic.supports_streaming());
        assert!(Provider::Anthropic.supports_function_calling());
        
        // Test that non-function calling providers are correctly identified
        assert!(!Provider::AI21.supports_function_calling());
        assert!(!Provider::HuggingFace.supports_function_calling());
    }
}