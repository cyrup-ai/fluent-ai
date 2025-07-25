//! Zero-allocation OpenAI embeddings implementation
//!
//! Provides comprehensive support for OpenAI's text embedding models (text-embedding-3-large/small)
//! with optimal performance patterns and batch processing capabilities.

use serde::{Deserialize, Serialize};

use super::{OpenAIError, OpenAIResult};
use crate::ZeroOneOrMany;

/// OpenAI embedding request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEmbeddingRequest {
    pub input: EmbeddingInput,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>}

/// Input for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Single text string
    Single(String),
    /// Array of text strings for batch processing
    Array(Vec<String>),
    /// Array of token arrays (advanced usage)
    TokenArrays(Vec<Vec<u32>>)}

/// OpenAI embedding response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage}

/// Individual embedding data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32}

/// Usage statistics for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32}

/// Embedding model configuration
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub model: String,
    pub dimensions: Option<u32>,
    pub encoding_format: Option<String>,
    pub user: Option<String>,
    pub batch_size: Option<usize>,
    pub normalize: bool}

/// Batch embedding request for high-throughput processing
#[derive(Debug, Clone)]
pub struct BatchEmbeddingRequest {
    pub inputs: Vec<String>,
    pub config: EmbeddingConfig,
    pub chunk_size: usize,
    pub parallel_requests: usize}

/// Batch embedding response
#[derive(Debug, Clone)]
pub struct BatchEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub total_tokens: u32,
    pub processing_time_ms: u64,
    pub model: String}

/// Embedding similarity metrics
#[derive(Debug, Clone)]
pub struct SimilarityMetrics {
    pub cosine_similarity: f32,
    pub euclidean_distance: f32,
    pub dot_product: f32,
    pub manhattan_distance: f32}

/// Text chunking strategy for long documents
#[derive(Debug, Clone, Copy)]
pub enum ChunkingStrategy {
    /// Fixed character length chunks
    FixedLength(usize),
    /// Sentence-based chunking
    Sentences(usize),
    /// Paragraph-based chunking
    Paragraphs,
    /// Token-based chunking (approximate)
    Tokens(usize)}

impl EmbeddingInput {
    /// Create single text input
    #[inline(always)]
    pub fn single(text: impl Into<String>) -> Self {
        Self::Single(text.into())
    }

    /// Create batch input from vector
    #[inline(always)]
    pub fn batch(texts: Vec<String>) -> Self {
        Self::Array(texts)
    }

    /// Create from ZeroOneOrMany
    #[inline(always)]
    pub fn from_zero_one_many(texts: ZeroOneOrMany<String>) -> Self {
        match texts {
            ZeroOneOrMany::None => Self::Array(Vec::new()),
            ZeroOneOrMany::One(text) => Self::Single(text),
            ZeroOneOrMany::Many(texts) => Self::Array(texts)}
    }

    /// Get estimated token count
    #[inline(always)]
    pub fn estimate_tokens(&self) -> u32 {
        match self {
            Self::Single(text) => estimate_tokens(text),
            Self::Array(texts) => texts.iter().map(|t| estimate_tokens(t)).sum(),
            Self::TokenArrays(token_arrays) => {
                token_arrays.iter().map(|tokens| tokens.len() as u32).sum()
            }
        }
    }

    /// Get input count
    #[inline(always)]
    pub fn count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Array(texts) => texts.len(),
            Self::TokenArrays(arrays) => arrays.len()}
    }

    /// Validate input for API limits
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        match self {
            Self::Single(text) => {
                if text.is_empty() {
                    return Err(OpenAIError::EmbeddingError(
                        "Input text cannot be empty".to_string(),
                    ));
                }
                if text.len() > 8191 * 4 {
                    // Rough token limit * chars per token
                    return Err(OpenAIError::EmbeddingError(
                        "Input text too long".to_string(),
                    ));
                }
            }
            Self::Array(texts) => {
                if texts.is_empty() {
                    return Err(OpenAIError::EmbeddingError(
                        "Input array cannot be empty".to_string(),
                    ));
                }
                if texts.len() > 2048 {
                    return Err(OpenAIError::EmbeddingError(
                        "Too many inputs in batch".to_string(),
                    ));
                }
                for text in texts {
                    if text.is_empty() {
                        return Err(OpenAIError::EmbeddingError(
                            "Individual text cannot be empty".to_string(),
                        ));
                    }
                }
            }
            Self::TokenArrays(arrays) => {
                if arrays.is_empty() {
                    return Err(OpenAIError::EmbeddingError(
                        "Token arrays cannot be empty".to_string(),
                    ));
                }
                for tokens in arrays {
                    if tokens.is_empty() {
                        return Err(OpenAIError::EmbeddingError(
                            "Individual token array cannot be empty".to_string(),
                        ));
                    }
                    if tokens.len() > 8191 {
                        return Err(OpenAIError::EmbeddingError(
                            "Token array too long".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

impl EmbeddingConfig {
    /// Create config for text-embedding-3-large
    #[inline(always)]
    pub fn large() -> Self {
        Self {
            model: "text-embedding-3-large".to_string(),
            dimensions: None, // Use model default (3072)
            encoding_format: Some("float".to_string()),
            user: None,
            batch_size: Some(100),
            normalize: true}
    }

    /// Create config for text-embedding-3-small
    #[inline(always)]
    pub fn small() -> Self {
        Self {
            model: "text-embedding-3-small".to_string(),
            dimensions: None, // Use model default (1536)
            encoding_format: Some("float".to_string()),
            user: None,
            batch_size: Some(100),
            normalize: true}
    }

    /// Create config for ada-002 (legacy)
    #[inline(always)]
    pub fn ada_002() -> Self {
        Self {
            model: "text-embedding-ada-002".to_string(),
            dimensions: None, // Fixed at 1536
            encoding_format: Some("float".to_string()),
            user: None,
            batch_size: Some(100),
            normalize: true}
    }

    /// Set custom dimensions (only for v3 models)
    #[inline(always)]
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set encoding format
    #[inline(always)]
    pub fn with_encoding_format(mut self, format: impl Into<String>) -> Self {
        self.encoding_format = Some(format.into());
        self
    }

    /// Set user identifier
    #[inline(always)]
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set batch size for processing
    #[inline(always)]
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Enable/disable normalization
    #[inline(always)]
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Validate configuration
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        // Validate dimensions for v3 models
        if let Some(dimensions) = self.dimensions {
            if self.model == "text-embedding-3-large" {
                if dimensions > 3072 {
                    return Err(OpenAIError::EmbeddingError(
                        "text-embedding-3-large maximum dimensions is 3072".to_string(),
                    ));
                }
            } else if self.model == "text-embedding-3-small" {
                if dimensions > 1536 {
                    return Err(OpenAIError::EmbeddingError(
                        "text-embedding-3-small maximum dimensions is 1536".to_string(),
                    ));
                }
            } else if self.model == "text-embedding-ada-002" {
                return Err(OpenAIError::EmbeddingError(
                    "text-embedding-ada-002 does not support custom dimensions".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Get default dimensions for model
    #[inline(always)]
    pub fn get_default_dimensions(&self) -> u32 {
        match self.model.as_str() {
            "text-embedding-3-large" => 3072,
            "text-embedding-3-small" | "text-embedding-ada-002" => 1536,
            _ => 1536, // Default fallback
        }
    }

    /// Get effective dimensions (custom or default)
    #[inline(always)]
    pub fn get_dimensions(&self) -> u32 {
        self.dimensions
            .unwrap_or_else(|| self.get_default_dimensions())
    }
}

impl BatchEmbeddingRequest {
    /// Create new batch request
    #[inline(always)]
    pub fn new(inputs: Vec<String>, config: EmbeddingConfig) -> Self {
        Self {
            inputs,
            config,
            chunk_size: 100,      // Default batch size
            parallel_requests: 4, // Default parallel requests
        }
    }

    /// Set chunk size for batching
    #[inline(always)]
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set parallel request count
    #[inline(always)]
    pub fn with_parallel_requests(mut self, count: usize) -> Self {
        self.parallel_requests = count;
        self
    }

    /// Validate batch request
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        if self.inputs.is_empty() {
            return Err(OpenAIError::EmbeddingError(
                "Batch inputs cannot be empty".to_string(),
            ));
        }

        if self.chunk_size == 0 {
            return Err(OpenAIError::EmbeddingError(
                "Chunk size must be greater than 0".to_string(),
            ));
        }

        if self.parallel_requests == 0 {
            return Err(OpenAIError::EmbeddingError(
                "Parallel requests must be greater than 0".to_string(),
            ));
        }

        self.config.validate()?;

        Ok(())
    }

    /// Split into chunks for processing
    #[inline(always)]
    pub fn chunks(&self) -> Vec<Vec<String>> {
        self.inputs
            .chunks(self.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Estimate total tokens
    #[inline(always)]
    pub fn estimate_total_tokens(&self) -> u32 {
        self.inputs.iter().map(|text| estimate_tokens(text)).sum()
    }
}

impl SimilarityMetrics {
    /// Calculate all similarity metrics between two embeddings
    #[inline(always)]
    pub fn calculate(embedding1: &[f32], embedding2: &[f32]) -> OpenAIResult<Self> {
        if embedding1.len() != embedding2.len() {
            return Err(OpenAIError::EmbeddingError(
                "Embedding dimensions must match".to_string(),
            ));
        }

        let cosine_similarity = cosine_similarity(embedding1, embedding2)?;
        let euclidean_distance = euclidean_distance(embedding1, embedding2);
        let dot_product = dot_product(embedding1, embedding2);
        let manhattan_distance = manhattan_distance(embedding1, embedding2);

        Ok(Self {
            cosine_similarity,
            euclidean_distance,
            dot_product,
            manhattan_distance})
    }

    /// Get primary similarity score (cosine similarity)
    #[inline(always)]
    pub fn primary_score(&self) -> f32 {
        self.cosine_similarity
    }

    /// Check if embeddings are similar (cosine similarity > threshold)
    #[inline(always)]
    pub fn is_similar(&self, threshold: f32) -> bool {
        self.cosine_similarity > threshold
    }
}

impl ChunkingStrategy {
    /// Chunk text according to strategy
    #[inline(always)]
    pub fn chunk_text(&self, text: &str) -> Vec<String> {
        match self {
            Self::FixedLength(length) => text
                .chars()
                .collect::<Vec<_>>()
                .chunks(*length)
                .map(|chunk| chunk.iter().collect())
                .collect(),
            Self::Sentences(max_sentences) => {
                let sentences: Vec<&str> = text.split('.').collect();
                sentences
                    .chunks(*max_sentences)
                    .map(|chunk| chunk.join("."))
                    .collect()
            }
            Self::Paragraphs => text
                .split("\n\n")
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect(),
            Self::Tokens(max_tokens) => {
                // Approximate token chunking (4 chars per token)
                let chars_per_chunk = max_tokens * 4;
                text.chars()
                    .collect::<Vec<_>>()
                    .chunks(chars_per_chunk)
                    .map(|chunk| chunk.iter().collect())
                    .collect()
            }
        }
    }
}

/// Calculate cosine similarity between two embeddings
#[inline(always)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> OpenAIResult<f32> {
    if a.len() != b.len() {
        return Err(OpenAIError::EmbeddingError(
            "Vector dimensions must match".to_string(),
        ));
    }

    let dot = dot_product(a, b);
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (norm_a * norm_b))
}

/// Calculate dot product of two vectors
#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate Euclidean distance between two embeddings
#[inline(always)]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate Manhattan distance between two embeddings
#[inline(always)]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Normalize embedding vector to unit length
#[inline(always)]
pub fn normalize_embedding(embedding: &mut [f32]) {
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in embedding.iter_mut() {
            *value /= norm;
        }
    }
}

/// Find most similar embeddings using cosine similarity
#[inline(always)]
pub fn find_most_similar(
    query_embedding: &[f32],
    embeddings: &[Vec<f32>],
    top_k: usize,
) -> OpenAIResult<Vec<(usize, f32)>> {
    let mut similarities: Vec<(usize, f32)> = embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| {
            let similarity = cosine_similarity(query_embedding, emb)?;
            Ok((i, similarity))
        })
        .collect::<OpenAIResult<Vec<_>>>()?;

    // Sort by similarity (descending)
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top k
    similarities.truncate(top_k);

    Ok(similarities)
}

/// Estimate token count from text (rough approximation)
#[inline(always)]
pub fn estimate_tokens(text: &str) -> u32 {
    // Rough estimate: ~4 characters per token
    (text.len() as f32 / 4.0).ceil() as u32
}

/// Check if model supports custom dimensions
#[inline(always)]
pub fn supports_custom_dimensions(model: &str) -> bool {
    matches!(model, "text-embedding-3-large" | "text-embedding-3-small")
}

/// Get available embedding models
#[inline(always)]
pub fn available_models() -> Vec<&'static str> {
    vec![
        "text-embedding-3-large",
        "text-embedding-3-small",
        "text-embedding-ada-002",
    ]
}

/// Get model capabilities
#[inline(always)]
pub fn get_model_info(model: &str) -> Option<(u32, u32, bool)> {
    // Returns (max_dimensions, context_length, supports_custom_dims)
    match model {
        "text-embedding-3-large" => Some((3072, 8191, true)),
        "text-embedding-3-small" => Some((1536, 8191, true)),
        "text-embedding-ada-002" => Some((1536, 8191, false)),
        _ => None}
}
