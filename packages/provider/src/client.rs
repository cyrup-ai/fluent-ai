//! Client trait definitions for provider implementations
//!
//! This module defines the core traits that provider clients implement
//! for completion, embedding, transcription, and other AI services.

use cyrup_sugars::{OneOrMany, ZeroOneOrMany};
use fluent_ai_domain::context::chunk::CompletionChunk;
// Note: EmbeddingChunk and VoiceChunk may be in context::chunk or may need to be created
use fluent_ai_domain::context::chunk::CompletionChunk as EmbeddingChunk;
use fluent_ai_domain::context::chunk::CompletionChunk as VoiceChunk;
use fluent_ai_async::{AsyncStream, AsyncTask};

/// Core completion client trait using async task patterns
pub trait CompletionClient: Send + Sync + Clone {
    /// Model type
    type Model;

    /// Create a completion model
    fn completion_model(&self, model: &str) -> Self::Model;
}

/// Core embeddings client trait using async task patterns
pub trait EmbeddingsClient: Send + Sync + Clone {
    /// Model type
    type Model;

    /// Create an embedding model
    fn embedding_model(&self, model: &str) -> Self::Model;

    /// Create an embedding model with specific dimensions
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::Model;
}

/// Core provider client trait
pub trait ProviderClient: Send + Sync + Clone {
    /// Provider name
    fn provider_name(&self) -> &'static str;

    /// Test connection to the provider
    fn test_connection(&self) -> AsyncTask<Result<(), Box<dyn std::error::Error + Send + Sync>>>;
}

/// Core transcription client trait
pub trait TranscriptionClient: Send + Sync + Clone {
    /// Model type
    type Model;

    /// Create a transcription model
    fn transcription_model(&self, model: &str) -> Self::Model;
}

/// Completion model trait for model-specific implementations
pub trait CompletionModel: Send + Sync + Clone {
    /// Completion response type
    type Response;
    /// Streaming response type
    type StreamingResponse;
    /// Error type
    type Error: Send + Sync + 'static;

    /// Generate completion from prompt using the domain pattern
    fn prompt(
        &self,
        prompt: crate::domain::prompt::Prompt,
    ) -> Box<dyn AsyncStream<CompletionChunk>>;

    /// Perform completion
    fn completion(
        &self,
        request: crate::domain::completion::CompletionRequest,
    ) -> AsyncTask<Result<Self::Response, Self::Error>>;

    /// Stream completion
    fn stream(
        &self,
        request: crate::domain::completion::CompletionRequest,
    ) -> Box<dyn AsyncStream<Self::StreamingResponse>>;
}

/// Embedding model trait for model-specific implementations  
pub trait EmbeddingModel: Send + Sync + Clone {
    /// Maximum number of documents per request
    const MAX_DOCUMENTS: usize = 1024;

    /// Get embedding dimensions
    fn ndims(&self) -> usize;

    /// Embed multiple texts
    fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> AsyncTask<Result<Vec<crate::embeddings::Embedding>, crate::embeddings::EmbeddingError>>;

    /// Create embeddings for a single text
    fn embed(&self, text: &str) -> Box<dyn AsyncTask<ZeroOneOrMany<f32>>>;

    /// Create embeddings for multiple texts with streaming
    fn embed_batch(&self, texts: ZeroOneOrMany<String>) -> Box<dyn AsyncStream<EmbeddingChunk>>;
}
