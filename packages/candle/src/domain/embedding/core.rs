//! Core Embedding Types and Traits
//!
//! This module provides core types and traits for working with text embeddings.
//! All embedding-related domain objects and operations are defined here.

use serde::{Deserialize, Serialize};

use crate::AsyncTask;
use cyrup_sugars::ZeroOneOrMany;
use crate::domain::context::chunk::EmbeddingChunk;
use crate::domain::model::usage::Usage;

/// Core trait for embedding models
pub trait EmbeddingModel: Send + Sync + Clone {
    /// Create embeddings for a single text
    fn embed(&self, text: &str) -> AsyncTask<ZeroOneOrMany<f32>>;

    /// Create embeddings for multiple texts with streaming
    fn embed_batch(&self, texts: ZeroOneOrMany<String>) -> crate::AsyncStream<EmbeddingChunk>;

    /// Simple embedding with handler - STREAMING ONLY, NO FUTURES
    /// Performance: Zero allocation, direct streaming without futures
    fn on_embedding<F>(&self, text: &str, handler: F) -> crate::AsyncStream<ZeroOneOrMany<f32>>
    where
        F: Fn(ZeroOneOrMany<f32>) -> ZeroOneOrMany<f32> + Send + Sync + 'static,
    {
        // Get embedding task and process result through handler using streaming
        let embedding_task = self.embed(text);

        crate::AsyncStream::with_channel(move |sender| {
            // Use proper streams-only pattern with blocking collect (safe in thread context)
            let embedding = embedding_task.collect();
            let processed = handler(embedding);
            let _ = sender.send(processed);
        })
    }
}

/// Embedding structure containing document text and its vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The source document text that was embedded
    pub document: String,
    /// The embedding vector(s) as floating point values
    pub vec: ZeroOneOrMany<f64>,
}

/// Response format for embedding operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use = "Embedding response should be handled or logged"]
pub struct EmbeddingResponse {
    /// The resulting embedding vector(s)
    pub data: Vec<EmbeddingData>,
    /// The model used to generate the embeddings
    pub model: String,
    /// Token usage statistics
    pub usage: Option<Usage>}
/// Individual embedding data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    /// The embedding vector
    pub embedding: Vec<f64>,
    /// Index of the embedding in the input batch
    pub index: usize,
    /// Optional object type (e.g., "embedding")
    pub object: Option<String>}

impl EmbeddingResponse {
    /// Create a new embedding response
    #[inline]
    pub fn new(embeddings: Vec<Vec<f64>>, model: impl Into<String>) -> Self {
        let data = embeddings
            .into_iter()
            .enumerate()
            .map(|(idx, embedding)| EmbeddingData {
                embedding,
                index: idx,
                object: Some("embedding".to_string())})
            .collect();

        Self {
            data,
            model: model.into(),
            usage: None}
    }

    /// Create a new embedding response with usage statistics
    #[inline]
    pub fn with_usage(
        embeddings: Vec<Vec<f64>>,
        model: impl Into<String>,
        prompt_tokens: u32,
        total_tokens: u32,
    ) -> Self {
        let mut response = Self::new(embeddings, model);
        response.usage = Some(Usage::new(prompt_tokens, total_tokens - prompt_tokens));
        response
    }

    /// Get the first embedding vector if available
    #[inline]
    pub fn first_embedding(&self) -> Option<&[f64]> {
        self.data.first().map(|d| d.embedding.as_slice())
    }

    /// Get all embeddings as a slice of slices
    #[inline]
    pub fn all_embeddings(&self) -> Vec<&[f64]> {
        self.data.iter().map(|d| d.embedding.as_slice()).collect()
    }
}
