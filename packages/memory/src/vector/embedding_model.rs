//! Embedding model interface for generating vector embeddings

use std::future::Future;
use std::pin::Pin;

use crate::utils::error::Result;

/// Type alias for embedding future
pub type EmbeddingFuture<T> = Pin<Box<dyn Future<Output = Result<T>> + Send>>;

/// Trait for embedding model implementations
#[cfg_attr(test, mockall::automock)]
pub trait EmbeddingModel: Send + Sync + std::fmt::Debug {
    /// Generate an embedding for text
    fn embed(
        &self,
        text: &str,
        task: Option<String>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>>> + Send>>;

    /// Generate embeddings for multiple texts
    fn batch_embed(&self, texts: &[String], task: Option<String>)
    -> EmbeddingFuture<Vec<Vec<f32>>>;

    /// Get the dimension of the embedding vectors
    fn dimension(&self) -> usize;

    /// Get the name of the embedding model
    fn name(&self) -> &str;
}
