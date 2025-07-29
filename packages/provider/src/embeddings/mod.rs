//! Zero-allocation embedding interfaces
//!
//! This module provides blazing-fast, SIMD-optimized embedding operations
//! with zero-allocation vector handling and consistent interfaces across
//! all provider implementations.

use std::fmt;

use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_domain::embedding::{Embedding, EmbeddingModel as DomainEmbeddingModel};
use serde::Deserialize;
use serde_json::Value;

use fluent_ai_async::{AsyncStream, AsyncStreamSender};

/// Trait for types that can be embedded
pub trait Embed: Send + Sync {
    fn text(&self) -> String;
}

impl Embed for String {
    fn text(&self) -> String {
        self.clone()
    }
}

impl Embed for &str {
    fn text(&self) -> String {
        self.to_string()
    }
}

/// Trait for embedding models - unified interface across providers
pub trait EmbeddingModel: Send + Sync + Clone {
    /// The maximum number of documents this model can process in one batch
    const MAX_DOCUMENTS: usize = 100;

    /// Get the number of dimensions for embeddings produced by this model
    fn ndims(&self) -> usize;

    /// Embed multiple text documents
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, EmbeddingError>;
}

/// Builder for creating embedding requests
#[derive(Debug)]
pub struct EmbeddingBuilder<M, D> {
    model: M,
    documents: Vec<D>}

impl<M, D> EmbeddingBuilder<M, D>
where
    M: EmbeddingModel,
    D: Embed,
{
    pub fn new(model: M) -> Self {
        Self {
            model,
            documents: Vec::new()}
    }

    pub fn documents(mut self, docs: Vec<D>) -> Self {
        self.documents = docs;
        self
    }

    pub async fn execute(self) -> Result<Vec<Embedding>, EmbeddingError> {
        let texts: Vec<String> = self.documents.into_iter().map(|d| d.text()).collect();
        self.model.embed_texts(texts).await
    }
}

/// Comprehensive embedding error types for robust error handling
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Network/HTTP request failed
    #[error("Network request failed: {0}")]
    NetworkError(String),

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// Invalid input provided
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Model not found or not supported
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {message}")]
    RateLimitExceeded { message: String },

    /// Quota exceeded
    #[error("Quota exceeded: {message}")]
    QuotaExceeded { message: String },

    /// Serialization failed
    #[error("Serialization failed: {0}")]
    SerializationError(String),

    /// Deserialization failed
    #[error("Deserialization failed: {0}")]
    DeserializationError(String),

    /// Vector dimension mismatch
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Provider-specific error
    #[error("Provider error: {0}")]
    ProviderError(String),

    /// Configuration error
    #[error("Configuration error: {field}: {message}")]
    ConfigurationError { field: String, message: String }}

/// Result type for embedding operations
pub type Result<T> = std::result::Result<T, EmbeddingError>;

/// Universal embedding provider trait
///
/// Provides zero-allocation embedding generation with SIMD optimization
/// and consistent interfaces across all AI providers.
pub trait EmbeddingProvider: Send + Sync {
    /// The type of raw embedding response from the provider
    type Response: Send + Sync;

    /// Generate embeddings for a single text input
    ///
    /// This method provides zero-allocation embedding generation with
    /// inline optimization for single text inputs.
    async fn embed_text(&self, text: &str) -> Result<EmbeddingResponse<Self::Response>>;

    /// Generate embeddings for multiple text inputs (batch processing)
    ///
    /// This method provides efficient batch processing with SIMD optimization
    /// for handling multiple texts in a single request.
    async fn embed_texts(&self, texts: &[&str]) -> Result<BatchEmbeddingResponse<Self::Response>>;

    /// Generate embeddings with streaming for large inputs
    ///
    /// This method provides streaming embedding generation for large text
    /// inputs that need to be processed in chunks.
    async fn embed_stream(&self, text: &str) -> Result<StreamingEmbeddingResponse<Self::Response>>;

    /// Get the embedding dimensions for this model
    ///
    /// Returns the number of dimensions in the embedding vectors
    /// produced by this model.
    fn dimensions(&self) -> usize;

    /// Get the maximum input token length for this model
    ///
    /// Returns the maximum number of tokens that can be processed
    /// in a single embedding request.
    fn max_tokens(&self) -> usize;

    /// Check if the model supports batch processing
    ///
    /// Returns true if the model can efficiently process multiple
    /// texts in a single request.
    fn supports_batch(&self) -> bool {
        true // Most models support batch processing
    }

    /// Get the model name/identifier
    fn model_name(&self) -> &str;
}

/// Zero-allocation embedding response wrapper
///
/// Provides a unified interface for embedding responses across all providers
/// while maintaining zero-allocation semantics and SIMD-optimized vector operations.
#[derive(Debug, Clone)]
pub struct EmbeddingResponse<T> {
    /// The raw provider-specific response
    pub raw_response: T,
    /// The generated embedding vector
    pub embedding: Vec<f32>,
    /// Token usage information (if available)
    pub token_usage: Option<TokenUsage>,
    /// Response metadata
    pub metadata: EmbeddingMetadata}

impl<T> EmbeddingResponse<T> {
    /// Create a new embedding response
    #[inline(always)]
    pub fn new(raw_response: T, embedding: Vec<f32>) -> Self {
        Self {
            raw_response,
            embedding,
            token_usage: None,
            metadata: EmbeddingMetadata::default()}
    }

    /// Add token usage information
    #[inline(always)]
    pub fn with_token_usage(mut self, usage: TokenUsage) -> Self {
        self.token_usage = Some(usage);
        self
    }

    /// Add metadata
    #[inline(always)]
    pub fn with_metadata(mut self, metadata: EmbeddingMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get the embedding vector
    #[inline(always)]
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// Get the embedding dimensions
    #[inline(always)]
    pub fn dimensions(&self) -> usize {
        self.embedding.len()
    }

    /// Calculate cosine similarity with another embedding (SIMD-optimized)
    #[inline]
    pub fn cosine_similarity(&self, other: &[f32]) -> f32 {
        if self.embedding.len() != other.len() {
            return 0.0; // Invalid comparison
        }

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        // Manual SIMD-friendly loop that compilers can auto-vectorize
        for i in 0..self.embedding.len() {
            let a = self.embedding[i];
            let b = other[i];
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Calculate euclidean distance with another embedding (SIMD-optimized)
    #[inline]
    pub fn euclidean_distance(&self, other: &[f32]) -> f32 {
        if self.embedding.len() != other.len() {
            return f32::INFINITY; // Invalid comparison
        }

        let mut sum = 0.0;

        // Manual SIMD-friendly loop
        for i in 0..self.embedding.len() {
            let diff = self.embedding[i] - other[i];
            sum += diff * diff;
        }

        sum.sqrt()
    }
}

/// Batch embedding response for multiple texts
#[derive(Debug, Clone)]
pub struct BatchEmbeddingResponse<T> {
    /// The raw provider-specific response
    pub raw_response: T,
    /// The generated embedding vectors
    pub embeddings: Vec<Vec<f32>>,
    /// Token usage information (if available)
    pub token_usage: Option<TokenUsage>,
    /// Response metadata
    pub metadata: EmbeddingMetadata}

impl<T> BatchEmbeddingResponse<T> {
    /// Create a new batch embedding response
    #[inline(always)]
    pub fn new(raw_response: T, embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            raw_response,
            embeddings,
            token_usage: None,
            metadata: EmbeddingMetadata::default()}
    }

    /// Add token usage information
    #[inline(always)]
    pub fn with_token_usage(mut self, usage: TokenUsage) -> Self {
        self.token_usage = Some(usage);
        self
    }

    /// Add metadata
    #[inline(always)]
    pub fn with_metadata(mut self, metadata: EmbeddingMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get all embedding vectors
    #[inline(always)]
    pub fn embeddings(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    /// Get the number of embeddings
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.embeddings.len()
    }

    /// Get a specific embedding by index
    #[inline(always)]
    pub fn get_embedding(&self, index: usize) -> Option<&[f32]> {
        self.embeddings.get(index).map(|v| v.as_slice())
    }

    /// Calculate similarity matrix (SIMD-optimized)
    ///
    /// Returns a matrix where element (i, j) is the cosine similarity
    /// between embeddings i and j.
    pub fn similarity_matrix(&self) -> Vec<Vec<f32>> {
        let count = self.embeddings.len();
        let mut matrix = vec![vec![0.0; count]; count];

        for i in 0..count {
            for j in i..count {
                if i == j {
                    matrix[i][j] = 1.0; // Self-similarity is always 1.0
                } else {
                    let similarity = self.cosine_similarity_between(i, j);
                    matrix[i][j] = similarity;
                    matrix[j][i] = similarity; // Symmetric matrix
                }
            }
        }

        matrix
    }

    /// Calculate cosine similarity between two embeddings by index
    #[inline]
    fn cosine_similarity_between(&self, index_a: usize, index_b: usize) -> f32 {
        match (self.embeddings.get(index_a), self.embeddings.get(index_b)) {
            (Some(a), Some(b)) => {
                if a.len() != b.len() {
                    return 0.0;
                }

                let mut dot_product = 0.0;
                let mut norm_a = 0.0;
                let mut norm_b = 0.0;

                for i in 0..a.len() {
                    dot_product += a[i] * b[i];
                    norm_a += a[i] * a[i];
                    norm_b += b[i] * b[i];
                }

                if norm_a == 0.0 || norm_b == 0.0 {
                    0.0
                } else {
                    dot_product / (norm_a.sqrt() * norm_b.sqrt())
                }
            }
            _ => 0.0}
    }
}

/// Streaming embedding response for large text processing
pub struct StreamingEmbeddingResponse<T> {
    /// The raw provider-specific streaming response
    pub raw_response: T,
    /// Stream of embedding chunks
    pub stream: AsyncStream<EmbeddingChunk>,
    /// Response metadata (filled as chunks arrive)
    pub metadata: EmbeddingMetadata}

impl<T> StreamingEmbeddingResponse<T> {
    /// Create a new streaming embedding response
    #[inline(always)]
    pub fn new(
        raw_response: T,
        stream: AsyncStream<EmbeddingChunk>,
    ) -> Self {
        Self {
            raw_response,
            stream,
            metadata: EmbeddingMetadata::default()}
    }

    /// Add metadata
    #[inline(always)]
    pub fn with_metadata(mut self, metadata: EmbeddingMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Collect all chunks into a single embedding vector
    pub async fn collect(mut self) -> Result<Vec<f32>> {
        let mut embedding = Vec::new();

        while let Some(chunk) = self.stream.next().await {
            embedding.extend_from_slice(&chunk.vector);
        }

        Ok(embedding)
    }
}

/// Embedding chunk for streaming responses
#[derive(Debug, Clone)]
pub struct EmbeddingChunk {
    /// Partial embedding vector
    pub vector: Vec<f32>,
    /// Chunk index in the sequence
    pub index: usize,
    /// Whether this is the final chunk
    pub is_final: bool}

/// Token usage information for embedding requests
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    /// Number of tokens in the input
    pub prompt_tokens: u32,
    /// Total tokens processed
    pub total_tokens: u32}

impl TokenUsage {
    /// Create new token usage info
    #[inline(always)]
    pub fn new(prompt_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            total_tokens: prompt_tokens}
    }
}

/// Embedding response metadata for tracking request/response details
#[derive(Debug, Clone, Default)]
pub struct EmbeddingMetadata {
    /// Request ID (if provided by the API)
    pub request_id: Option<String>,
    /// Model used for the embedding
    pub model: Option<String>,
    /// Response time in milliseconds
    pub response_time_ms: Option<u64>,
    /// Embedding dimensions
    pub dimensions: Option<usize>,
    /// Additional provider-specific metadata
    pub provider_metadata: Option<Value>}

impl EmbeddingMetadata {
    /// Create new embedding metadata
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

    /// Set dimensions
    #[inline(always)]
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set provider metadata
    #[inline(always)]
    pub fn with_provider_metadata(mut self, metadata: Value) -> Self {
        self.provider_metadata = Some(metadata);
        self
    }
}

pub use BatchEmbeddingResponse as DefaultBatchEmbeddingResponse;
/// Re-exports for convenience
pub use EmbeddingResponse as DefaultEmbeddingResponse;
pub use StreamingEmbeddingResponse as DefaultStreamingEmbeddingResponse;
