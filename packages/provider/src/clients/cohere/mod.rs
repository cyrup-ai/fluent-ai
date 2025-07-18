//! Cohere multi-endpoint client with zero-allocation architecture
//!
//! Provides high-performance Cohere integration supporting three distinct endpoints:
//! - Chat completions (Command-A, Command-R models)
//! - Text embeddings (Embed-V4.0, Embed-English-V3.0)
//! - Document reranking (Rerank-V3.5)
//!
//! Features:
//! - Zero allocation multi-endpoint architecture with shared connection pool
//! - Lock-free concurrent design with atomic counters
//! - Streaming chat completions with incremental JSON parsing
//! - Batch embedding processing with SIMD optimization
//! - Advanced document reranking with relevance scoring
//! - Circuit breaker fault tolerance across all endpoints
//!
//! ```
//! use fluent_ai_provider::clients::cohere::{CohereClient, CohereCompletionBuilder};
//! 
//! let client = CohereClient::new("your-api-key".to_string())?;
//! 
//! // Chat completion
//! client.completion_model("command-a-03-2025")?
//!     .system_prompt("You are a helpful assistant")
//!     .temperature(0.7)
//!     .prompt("Hello world");
//!     
//! // Embedding
//! let embeddings = client.embed_texts(&["Hello", "World"], "embed-v4.0").await?;
//! 
//! // Reranking
//! let ranked = client.rerank_documents("query", &documents, "rerank-v3.5").await?;
//! ```

use crate::completion_provider::{CompletionProvider, CompletionError};
use crate::client::{CompletionClient, ProviderClient};

// Re-export all public types
pub use client::CohereClient;
pub use completion::CohereCompletionBuilder;
pub use embedding::{CohereEmbedding, EmbeddingRequest, EmbeddingResponse, EmbeddingBatch};
pub use reranker::{CohereReranker, RerankRequest, RerankResponse, RerankResult};
pub use streaming::CohereStream;
pub use error::{CohereError, Result};

// Internal modules
mod client;
mod completion;
mod embedding;
mod error;
mod reranker;
mod streaming;

/// Cohere model constants - zero allocation static strings
pub mod models {
    /// Chat completion models
    pub const COMMAND_A_03_2025: &str = "command-a-03-2025";
    pub const COMMAND_R7B_12_2024: &str = "command-r7b-12-2024";
    
    /// Embedding models
    pub const EMBED_V4_0: &str = "embed-v4.0";
    pub const EMBED_ENGLISH_V3_0: &str = "embed-english-v3.0";
    
    /// Reranking models
    pub const RERANK_V3_5: &str = "rerank-v3.5";
    
    /// Complete model list for iteration - zero allocation
    pub const ALL_MODELS: &[&str] = &[
        COMMAND_A_03_2025,
        COMMAND_R7B_12_2024,
        EMBED_V4_0,
        EMBED_ENGLISH_V3_0,
        RERANK_V3_5,
    ];
    
    /// Chat models subset
    pub const CHAT_MODELS: &[&str] = &[
        COMMAND_A_03_2025,
        COMMAND_R7B_12_2024,
    ];
    
    /// Embedding models subset
    pub const EMBEDDING_MODELS: &[&str] = &[
        EMBED_V4_0,
        EMBED_ENGLISH_V3_0,
    ];
    
    /// Reranking models subset
    pub const RERANKING_MODELS: &[&str] = &[
        RERANK_V3_5,
    ];
    
    /// Compile-time model classification - zero allocation
    #[inline]
    pub const fn is_chat_model(model: &str) -> bool {
        matches!(model, COMMAND_A_03_2025 | COMMAND_R7B_12_2024)
    }
    
    /// Compile-time embedding model check - zero allocation
    #[inline]
    pub const fn is_embedding_model(model: &str) -> bool {
        matches!(model, EMBED_V4_0 | EMBED_ENGLISH_V3_0)
    }
    
    /// Compile-time reranking model check - zero allocation
    #[inline]
    pub const fn is_reranking_model(model: &str) -> bool {
        matches!(model, RERANK_V3_5)
    }
    
    /// Get model type for endpoint routing - compile-time optimization
    #[inline]
    pub const fn get_model_type(model: &str) -> ModelType {
        if is_chat_model(model) {
            ModelType::Chat
        } else if is_embedding_model(model) {
            ModelType::Embedding
        } else if is_reranking_model(model) {
            ModelType::Reranking
        } else {
            ModelType::Unknown
        }
    }
    
    /// Model capability validation - zero allocation
    #[inline]
    pub const fn supports_streaming(model: &str) -> bool {
        is_chat_model(model)
    }
    
    /// Model capability validation - tools support
    #[inline]
    pub const fn supports_tools(model: &str) -> bool {
        is_chat_model(model)
    }
    
    /// Maximum context length per model - compile-time constants
    #[inline]
    pub const fn context_length(model: &str) -> u32 {
        match model {
            COMMAND_A_03_2025 => 128_000,
            COMMAND_R7B_12_2024 => 32_768,
            EMBED_V4_0 => 8_192,
            EMBED_ENGLISH_V3_0 => 8_192,
            RERANK_V3_5 => 4_096,
            _ => 0,
        }
    }
    
    /// Model dimension for embeddings - compile-time constants
    #[inline]
    pub const fn embedding_dimension(model: &str) -> u16 {
        match model {
            EMBED_V4_0 => 1024,
            EMBED_ENGLISH_V3_0 => 1024,
            _ => 0,
        }
    }
}

/// Cohere API endpoints - zero allocation constants
pub mod endpoints {
    /// Base API URL
    pub const BASE_URL: &str = "https://api.cohere.ai";
    
    /// Endpoint paths
    pub const CHAT_ENDPOINT: &str = "/v1/chat";
    pub const EMBED_ENDPOINT: &str = "/v1/embed";
    pub const RERANK_ENDPOINT: &str = "/v1/rerank";
    
    /// Full endpoint URLs for performance optimization
    pub const CHAT_URL: &str = "https://api.cohere.ai/v1/chat";
    pub const EMBED_URL: &str = "https://api.cohere.ai/v1/embed";
    pub const RERANK_URL: &str = "https://api.cohere.ai/v1/rerank";
    
    /// Endpoint routing based on model type
    #[inline]
    pub const fn get_endpoint_url(model_type: ModelType) -> &'static str {
        match model_type {
            ModelType::Chat => CHAT_URL,
            ModelType::Embedding => EMBED_URL,
            ModelType::Reranking => RERANK_URL,
            ModelType::Unknown => "",
        }
    }
}

/// Model type enumeration for endpoint routing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Chat,
    Embedding,
    Reranking,
    Unknown,
}

/// Cohere provider configuration - zero allocation
pub mod config {
    use arrayvec::ArrayString;
    
    /// Default request timeout in seconds
    pub const DEFAULT_TIMEOUT_SECS: u64 = 30;
    
    /// Maximum retries for transient failures
    pub const MAX_RETRIES: u8 = 3;
    
    /// Circuit breaker failure threshold
    pub const CIRCUIT_BREAKER_THRESHOLD: u32 = 5;
    
    /// Maximum batch size for embeddings
    pub const MAX_EMBEDDING_BATCH_SIZE: usize = 96;
    
    /// Maximum documents for reranking
    pub const MAX_RERANK_DOCUMENTS: usize = 1000;
    
    /// API key validation - zero allocation
    #[inline]
    pub fn validate_api_key(api_key: &str) -> Result<(), ArrayString<64>> {
        if api_key.is_empty() {
            let mut error = ArrayString::new();
            let _ = error.try_push_str("API key cannot be empty");
            return Err(error);
        }
        
        if api_key.len() < 10 {
            let mut error = ArrayString::new();
            let _ = error.try_push_str("API key too short (minimum 10 characters)");
            return Err(error);
        }
        
        if api_key.len() > 128 {
            let mut error = ArrayString::new();
            let _ = error.try_push_str("API key too long (maximum 128 characters)");
            return Err(error);
        }
        
        Ok(())
    }
    
    /// Model validation - zero allocation
    #[inline]
    pub fn validate_model(model: &str) -> Result<(), ArrayString<64>> {
        if super::models::get_model_type(model) == super::ModelType::Unknown {
            let mut error = ArrayString::new();
            let _ = error.try_push_str("Unknown model: ");
            let _ = error.try_push_str(model);
            return Err(error);
        }
        Ok(())
    }
}

/// Cohere provider for enumeration and discovery
pub struct CohereProvider;

impl CohereProvider {
    /// Create new Cohere provider instance
    #[inline]
    pub const fn new() -> Self {
        Self
    }
    
    /// Get provider name
    #[inline]
    pub const fn name() -> &'static str {
        "cohere"
    }
    
    /// Get available models (compile-time constant)
    #[inline]
    pub const fn models() -> &'static [&'static str] {
        models::ALL_MODELS
    }
    
    /// Get chat models (compile-time constant)
    #[inline]
    pub const fn chat_models() -> &'static [&'static str] {
        models::CHAT_MODELS
    }
    
    /// Get embedding models (compile-time constant)
    #[inline]
    pub const fn embedding_models() -> &'static [&'static str] {
        models::EMBEDDING_MODELS
    }
    
    /// Get reranking models (compile-time constant)
    #[inline]
    pub const fn reranking_models() -> &'static [&'static str] {
        models::RERANKING_MODELS
    }
    
    /// Get endpoint URLs (compile-time constant)
    #[inline]
    pub const fn endpoints() -> &'static [&'static str] {
        &[
            endpoints::CHAT_URL,
            endpoints::EMBED_URL,
            endpoints::RERANK_URL,
        ]
    }
    
    /// Provider capabilities
    #[inline]
    pub const fn supports_streaming() -> bool {
        true
    }
    
    #[inline]
    pub const fn supports_tools() -> bool {
        true
    }
    
    #[inline]
    pub const fn supports_embeddings() -> bool {
        true
    }
    
    #[inline]
    pub const fn supports_reranking() -> bool {
        true
    }
}

impl Default for CohereProvider {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Performance optimization utilities
pub mod utils {
    use super::models;
    use arrayvec::ArrayString;
    
    /// Optimized model lookup with compile-time branching
    #[inline]
    pub const fn find_model_fast(model: &str) -> Option<&'static str> {
        if models::is_chat_model(model) {
            Some(model)
        } else if models::is_embedding_model(model) {
            Some(model)
        } else if models::is_reranking_model(model) {
            Some(model)
        } else {
            None
        }
    }
    
    /// Generate User-Agent header - zero allocation
    #[inline]
    pub fn user_agent() -> &'static str {
        concat!("fluent-ai-cohere/", env!("CARGO_PKG_VERSION"))
    }
    
    /// Calculate optimal batch size for embeddings
    #[inline]
    pub const fn optimal_embedding_batch_size(text_count: usize, avg_length: usize) -> usize {
        // Heuristic: balance between API efficiency and memory usage
        let base_size = if avg_length < 100 {
            super::config::MAX_EMBEDDING_BATCH_SIZE
        } else if avg_length < 1000 {
            super::config::MAX_EMBEDDING_BATCH_SIZE / 2
        } else {
            super::config::MAX_EMBEDDING_BATCH_SIZE / 4
        };
        
        if text_count < base_size {
            text_count
        } else {
            base_size
        }
    }
    
    /// Calculate optimal rerank batch size
    #[inline]
    pub const fn optimal_rerank_batch_size(doc_count: usize, avg_length: usize) -> usize {
        let base_size = if avg_length < 500 {
            super::config::MAX_RERANK_DOCUMENTS
        } else if avg_length < 2000 {
            super::config::MAX_RERANK_DOCUMENTS / 2
        } else {
            super::config::MAX_RERANK_DOCUMENTS / 4
        };
        
        if doc_count < base_size {
            doc_count
        } else {
            base_size
        }
    }
}

/// Cohere-specific constants and metadata
pub mod metadata {
    /// API version
    pub const API_VERSION: &str = "v1";
    
    /// Rate limiting information
    pub const RATE_LIMIT_CHAT_RPM: u32 = 1000;
    pub const RATE_LIMIT_EMBED_RPM: u32 = 100;
    pub const RATE_LIMIT_RERANK_RPM: u32 = 100;
    
    /// Pricing information (approximate, for optimization decisions)
    pub const CHAT_COST_PER_1K_TOKENS: f64 = 0.003;
    pub const EMBED_COST_PER_1K_TOKENS: f64 = 0.0001;
    pub const RERANK_COST_PER_1K_TOKENS: f64 = 0.0002;
    
    /// Performance characteristics
    pub const TYPICAL_CHAT_LATENCY_MS: u64 = 1500;
    pub const TYPICAL_EMBED_LATENCY_MS: u64 = 200;
    pub const TYPICAL_RERANK_LATENCY_MS: u64 = 300;
}